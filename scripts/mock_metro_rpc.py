"""Standalone TCP NDJSON mock Mini Metro server for ``bridge-rpc-v1``."""

from __future__ import annotations

import argparse
import socketserver
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import rl.bridge_protocol as bp  # noqa: E402

_SHAPES = ("CIRCLE", "TRIANGLE", "SQUARE", "CROSS", "DIAMOND", "PENTAGON", "STAR")
_OFFERS_S1 = (bp.WeeklyOffer(0, "LOCOMOTIVE", "+1 Locomotive", 1),)
_OFFERS_S2 = (
    bp.WeeklyOffer(0, "LINE", "New Line", 1),
    bp.WeeklyOffer(1, "CARRIAGE", "+1 Carriage", 1),
    bp.WeeklyOffer(2, "CROSSING", "+1 Crossing", 1),
    bp.WeeklyOffer(3, "INTERCHANGE", "+1 Interchange", 1),
)
_REWARDS = dict.fromkeys(("LOCOMOTIVE", "SHINKANSEN", "TRAM"), "locos_available")
_REWARDS |= {"CARRIAGE": "carriages_available", "CROSSING": "tunnels_available"}
_REWARDS["INTERCHANGE"] = "interchanges_available"


class MockRpcError(Exception):
    def __init__(self, error_code: bp.ErrorCode, message: str) -> None:
        super().__init__(message)
        self.error_code, self.message = error_code, message


def _err(ec: bp.ErrorCode, msg: str) -> Any:
    raise MockRpcError(ec, msg)


class MockMetroBoard:
    """Thread-safe simulated Mini Metro board supporting >20 stations and >4 lines."""

    def __init__(
        self,
        *,
        city_id: str = "london",
        initial_stations: int = 6,
        max_stations: int = 36,
        lines_total: int = 6,
        locos: int = 4,
        carriages: int = 3,
        tunnels: int = 2,
        interchanges: int = 2,
        ticks_per_day: int = 4,
        ticks_per_week: int = 28,
        ghost_ttl_ticks: int = 2,
    ) -> None:
        self._lock = threading.RLock()
        self.city_id, self.max_stations = city_id, max(initial_stations, max_stations)
        ks = (
            "lines_available",
            "lines_total",
            "locos_available",
            "carriages_available",
            "tunnels_available",
            "interchanges_available",
        )
        vs = (lines_total, lines_total, locos, carriages, tunnels, interchanges)
        self.inv: dict[str, int] = dict(zip(ks, vs))
        self.ticks_per_day, self.ticks_per_week = ticks_per_day, ticks_per_week
        self.ghost_ttl_ticks = ghost_ttl_ticks
        self.seq = self.tick_count = self.timestamp_ms = self.deliveries = 0
        self.screen = bp.ScreenKind.PLAYING
        self.is_paused, self.speed_multiplier, self.day = False, 1.0, 0
        self.weekly_stage = 0
        self._next_id = {"s": 1, "l": 1, "t": 1}
        self.stations: dict[int, SimpleNamespace] = {}
        self.lines: dict[int, SimpleNamespace] = {}
        self.trains: dict[int, SimpleNamespace] = {}
        self.weekly_offers: list[bp.WeeklyOffer] = []
        for _ in range(initial_stations):
            self._spawn_station()

    def _alloc(self, key: str) -> int:
        val = self._next_id[key]
        self._next_id[key] = val + 1
        return val

    def _spawn_station(self) -> None:
        sid = self._alloc("s")
        bank = -1 if sid % 2 else 1
        row, col = (sid - 1) // 2 % 6, (sid - 1) // 12
        x = float(bank * (180 + col * 110 + (row % 2) * 35))
        y = float((row - 2.5) * 90)
        sh = _SHAPES[(sid - 1) % len(_SHAPES)]
        self.stations[sid] = SimpleNamespace(
            id=sid,
            x=x,
            y=y,
            shape=sh,
            is_interchange=False,
            overload_ticks=0,
            waiting=[],
            peep_capacity=6,
        )

    def _pair_tunnels(self, u: int, v: int) -> int:
        return int(u != v and (self.stations[u].x < 0) != (self.stations[v].x < 0))

    def _line_tunnels(self, sids: list[int], is_looped: bool) -> int:
        edges = list(zip(sids, sids[1:]))
        if is_looped and len(sids) >= 3:
            edges.append((sids[-1], sids[0]))
        return sum(self._pair_tunnels(u, v) for u, v in edges)

    def _check_tunnels(self, new_cost: int, old_cost: int = 0) -> int:
        delta = new_cost - old_cost
        if delta > self.inv["tunnels_available"]:
            avail = self.inv["tunnels_available"]
            _err(
                bp.ErrorCode.TUNNEL_BUDGET_EXCEEDED,
                f"Need {delta} tunnel(s), have {avail}",
            )
        self.inv["tunnels_available"] -= delta
        return new_cost

    def _station(self, sid: Any) -> SimpleNamespace:
        if not isinstance(sid, int) or sid not in self.stations:
            _err(bp.ErrorCode.INVALID_ARGS, f"Unknown station_id {sid!r}")
        return self.stations[sid]

    def _line(self, lid: Any) -> SimpleNamespace:
        if not isinstance(lid, int) or lid not in self.lines:
            _err(bp.ErrorCode.INVALID_ARGS, f"Unknown line_id {lid!r}")
        ln = self.lines[lid]
        if ln.is_ghost:
            _err(bp.ErrorCode.INVALID_STATE, f"Line {lid} is a ghost")
        return ln

    def _pick_train(self, lid: int, tid: Any) -> SimpleNamespace:
        cands = [t for t in self.trains.values() if t.line_id == lid and not t.is_ghost]
        if tid is not None:
            for tr in cands:
                if tr.id == tid:
                    return tr
            _err(bp.ErrorCode.INVALID_ARGS, f"Train {tid!r} not on line {lid}")
        if not cands:
            _err(bp.ErrorCode.INVALID_STATE, f"Line {lid} has no active train")
        return cands[0]

    def snapshot(self) -> bp.Snapshot:
        """Build an immutable validated ``Snapshot`` of current board state."""
        sids = sorted(self.stations)
        cross = tuple(
            bp.CrossingCost(u, v, c)
            for i, u in enumerate(sids)
            for v in sids[i + 1 :]
            if (c := self._pair_tunnels(u, v))
        )
        sts = []
        for s in (self.stations[k] for k in sids):
            wp = tuple(
                bp.WaitingPassenger(d, 1000 * (i + 1)) for i, d in enumerate(s.waiting)
            )
            xy = (s.id, s.x, s.y, 960.0 + s.x, 540.0 + s.y, s.shape)
            ov = min(1.0, s.overload_ticks / 10.0)
            st = bp.StationSnapshot(*xy, s.is_interchange, ov, wp, s.peep_capacity)
            sts.append(st)
        lns = []
        for ln in self.lines.values():
            a = (ln.id, ln.color_index, tuple(ln.station_ids))
            lns.append(bp.LineSnapshot(*a, ln.is_looped, ln.is_ghost, ln.tunnels_used))
        trs = []
        for t in self.trains.values():
            a = (t.id, t.line_id, t.carriages, 6 * (1 + t.carriages), t.onboard_count)
            b = (t.is_unassignment_queued, t.is_ghost, t.is_waiting_for_locomotive)
            trs.append(bp.TrainSnapshot(*a, *b, t.loco_type, t.is_pending))
        h1 = (self.seq, self.timestamp_ms, self.screen, self.city_id, self.is_paused)
        h2 = (self.speed_multiplier, self.deliveries, self.day)
        cam = bp.CameraSnapshot(0.0, 0.0, 540.0, 1920, 1080, 1.0)
        inv = bp.InventorySnapshot(**self.inv)
        rest = (tuple(self.weekly_offers), tuple(sts), tuple(lns), tuple(trs), cross)
        return bp.Snapshot(*h1, *h2, cam, inv, *rest)

    def handle_request(self, req: bp.RpcRequest) -> bp.RpcResponse:
        """Execute one validated ``RpcRequest`` and return an ``RpcResponse``."""
        with self._lock:
            self.seq += 1
            try:
                if req.cmd == bp.RpcCommand.PING.value:
                    bp.parse_control_args(req.cmd, req.args)
                    d = {"pong": True, "protocol": bp.PROTOCOL_VERSION, "seq": self.seq}
                    return bp.RpcResponse.success(req.id, self.seq, d)
                if req.cmd == bp.RpcCommand.GET_STATE.value:
                    bp.parse_control_args(req.cmd, req.args)
                elif req.cmd in ("SET_PAUSED", "SET_SPEED"):
                    v = bp.parse_control_args(req.cmd, req.args)
                    self._apply_control(req.cmd, v)
                else:
                    self._apply_action(bp.action_from_request(req))
                data = {"snapshot": self.snapshot().to_dict()}
                return bp.RpcResponse.success(req.id, self.seq, data)
            except (MockRpcError, bp.ProtocolValidationError) as e:
                return bp.RpcResponse.failure(req.id, self.seq, e.error_code, str(e))

    def _apply_control(self, cmd: str, val: Any) -> None:
        if self.screen == bp.ScreenKind.GAME_OVER:
            _err(bp.ErrorCode.INVALID_STATE, "Game is already over")
        if cmd == bp.RpcCommand.SET_PAUSED.value:
            self.is_paused = val
            if self.screen in (bp.ScreenKind.PLAYING, bp.ScreenKind.PAUSED):
                self.screen = bp.ScreenKind.PAUSED if val else bp.ScreenKind.PLAYING
        else:
            self.speed_multiplier = val
            self.is_paused = False
            if self.screen == bp.ScreenKind.PAUSED:
                self.screen = bp.ScreenKind.PLAYING

    def _apply_action(self, act: Any) -> None:
        if self.screen == bp.ScreenKind.GAME_OVER:
            _err(bp.ErrorCode.INVALID_STATE, "Game is already over")
        if isinstance(act, bp.PickUpgradeAction):
            self._act_pick_upgrade(act)
            return
        if self.screen == bp.ScreenKind.WEEKLY_UPGRADE:
            _err(bp.ErrorCode.INVALID_STATE, "Pick weekly upgrade first")
        getattr(self, f"_act_{act.kind.value.lower()}")(act)

    def _act_wait(self, act: bp.WaitAction) -> None:
        for _ in range(act.ticks):
            self._step_tick()
            if self.screen in (bp.ScreenKind.WEEKLY_UPGRADE, bp.ScreenKind.GAME_OVER):
                break

    def _step_tick(self) -> None:
        self.tick_count += 1
        self.timestamp_ms += int(1000 * self.speed_multiplier)
        self.day = self.tick_count // self.ticks_per_day
        if len(self.stations) < self.max_stations:
            self._spawn_station()
        for st in self.stations.values():
            if len(st.waiting) <= st.peep_capacity:
                i = (st.id + self.tick_count) % len(_SHAPES)
                j = (i + (_SHAPES[i] == st.shape)) % len(_SHAPES)
                st.waiting.append(_SHAPES[j])
            over = len(st.waiting) > st.peep_capacity
            st.overload_ticks = st.overload_ticks + 1 if over else 0
        for tr in list(self.trains.values()):
            if tr.is_ghost or tr.is_waiting_for_locomotive:
                continue
            sids = self.lines[tr.line_id].station_ids
            shapes = {self.stations[s].shape for s in sids}
            for sid in sids:
                st = self.stations[sid]
                kept = [d for d in st.waiting if d not in shapes]
                if moved := len(st.waiting) - len(kept):
                    self.deliveries += moved
                    st.waiting = kept
                    tr.onboard_count = min(6 * (1 + tr.carriages), moved)
        for ln_id in [k for k, v in self.lines.items() if v.is_ghost]:
            ln = self.lines[ln_id]
            ln.ghost_ttl -= 1
            if ln.ghost_ttl <= 0:
                for tid in [k for k, t in self.trains.items() if t.line_id == ln_id]:
                    self._release_train(tid)
                self.inv["tunnels_available"] += ln.tunnels_used
                self.inv["lines_available"] += 1
                del self.lines[ln_id]
        if self.ticks_per_week > 0 and self.tick_count % self.ticks_per_week == 0:
            self.screen, self.is_paused, self.weekly_stage = (
                bp.ScreenKind.WEEKLY_UPGRADE,
                True,
                1,
            )
            self.weekly_offers = list(_OFFERS_S1)

    def _release_train(self, tid: int) -> None:
        tr = self.trains.pop(tid)
        self.inv["locos_available"] += 1
        self.inv["carriages_available"] += tr.carriages

    def _act_connect(self, act: bp.ConnectAction) -> None:
        if self.inv["lines_available"] < 1:
            _err(bp.ErrorCode.INSUFFICIENT_RESOURCES, "No lines available")
        sids = [self._station(s).id for s in act.station_ids]
        used = {ln.color_index for ln in self.lines.values() if not ln.is_ghost}
        color = act.color_index
        if color is None:
            color = next(c for c in range(100) if c not in used)
        elif color in used:
            _err(bp.ErrorCode.INVALID_STATE, f"Color {color} already in use")
        tunnels = self._check_tunnels(self._line_tunnels(sids, act.is_looped))
        lid = self._alloc("l")
        self.inv["lines_available"] -= 1
        d = dict(id=lid, color_index=color, station_ids=sids, is_looped=act.is_looped)
        self.lines[lid] = SimpleNamespace(
            **d, is_ghost=False, ghost_ttl=0, tunnels_used=tunnels
        )

    def _act_extend_line(self, act: bp.ExtendLineAction | bp.PrependLineAction) -> None:
        ln = self._line(act.line_id)
        if ln.is_looped:
            _err(bp.ErrorCode.INVALID_STATE, "Looped line cannot extend")
        sid = self._station(act.station_id).id
        is_ext = isinstance(act, bp.ExtendLineAction)
        close_target = ln.station_ids[0] if is_ext else ln.station_ids[-1]
        if sid in ln.station_ids:
            if sid != close_target or len(ln.station_ids) < 2:
                _err(bp.ErrorCode.INVALID_ARGS, "Station already on line")
            new_sids, new_loop = list(ln.station_ids), True
        else:
            new_sids = ln.station_ids + [sid] if is_ext else [sid] + ln.station_ids
            new_loop = act.is_looped and len(new_sids) >= 2
        cost = self._line_tunnels(new_sids, new_loop)
        ln.tunnels_used = self._check_tunnels(cost, ln.tunnels_used)
        ln.station_ids, ln.is_looped = new_sids, new_loop

    _act_prepend_line = _act_extend_line

    def _act_add_to_line(self, act: bp.AddToLineAction) -> None:
        ln = self._line(act.line_id)
        sid = self._station(act.station_id).id
        pos = act.position
        if pos < 0 or pos > len(ln.station_ids):
            _err(bp.ErrorCode.INVALID_ARGS, f"ADD_TO_LINE position out of range: {pos}")
        if ln.is_looped:
            if sid in ln.station_ids:
                _err(bp.ErrorCode.INVALID_ARGS, "Station already on line")
            new_sids = ln.station_ids[:pos] + [sid] + ln.station_ids[pos:]
            new_loop = True
        elif pos in (0, len(ln.station_ids)):
            is_ext = pos == len(ln.station_ids)
            close_target = ln.station_ids[0] if is_ext else ln.station_ids[-1]
            if sid in ln.station_ids:
                if sid != close_target or len(ln.station_ids) < 3:
                    _err(bp.ErrorCode.INVALID_ARGS, "Station already on line")
                new_sids, new_loop = list(ln.station_ids), True
            else:
                new_sids = ln.station_ids + [sid] if is_ext else [sid] + ln.station_ids
                new_loop = act.is_looped and len(new_sids) >= 3
        else:
            if act.is_looped:
                _err(
                    bp.ErrorCode.INVALID_ARGS,
                    "Cannot close loop via interior ADD_TO_LINE",
                )
            if sid in ln.station_ids:
                _err(bp.ErrorCode.INVALID_ARGS, "Station already on line")
            new_sids = ln.station_ids[:pos] + [sid] + ln.station_ids[pos:]
            new_loop = False
        cost = self._line_tunnels(new_sids, new_loop)
        ln.tunnels_used = self._check_tunnels(cost, ln.tunnels_used)
        ln.station_ids, ln.is_looped = new_sids, new_loop

    def _act_insert_station(self, act: bp.InsertStationAction) -> None:
        ln = self._line(act.line_id)
        sid = self._station(act.station_id).id
        after_id = self._station(act.after_station_id).id
        if sid in ln.station_ids or after_id not in ln.station_ids:
            _err(bp.ErrorCode.INVALID_ARGS, "Invalid insert_station args")
        idx = ln.station_ids.index(after_id)
        new_sids = ln.station_ids[: idx + 1] + [sid] + ln.station_ids[idx + 1 :]
        cost = self._line_tunnels(new_sids, ln.is_looped)
        ln.tunnels_used = self._check_tunnels(cost, ln.tunnels_used)
        ln.station_ids = new_sids

    def _act_replace_line(self, act: bp.ReplaceLineAction) -> None:
        ln = self._line(act.line_id)
        new_sids = [self._station(s).id for s in act.station_ids]
        cost = self._line_tunnels(new_sids, act.is_looped)
        ln.tunnels_used = self._check_tunnels(cost, ln.tunnels_used)
        ln.station_ids, ln.is_looped = new_sids, act.is_looped

    def _act_remove_line(self, act: bp.RemoveLineAction) -> None:
        ln = self._line(act.line_id)
        trains = [t for t in self.trains.values() if t.line_id == ln.id]
        if not trains:
            self.inv["tunnels_available"] += ln.tunnels_used
            self.inv["lines_available"] += 1
            del self.lines[ln.id]
            return
        ln.is_ghost, ln.ghost_ttl = True, self.ghost_ttl_ticks
        for tr in trains:
            tr.is_ghost = True

    def _act_assign_loco(self, act: bp.AssignLocoAction) -> None:
        ln = self._line(act.line_id)
        if self.inv["locos_available"] < 1:
            _err(bp.ErrorCode.INSUFFICIENT_RESOURCES, "No locos available")
        self.inv["locos_available"] -= 1
        tid = self._alloc("t")
        d = dict(
            id=tid,
            line_id=ln.id,
            carriages=0,
            onboard_count=0,
            is_ghost=False,
            is_pending=False,
        )
        self.trains[tid] = SimpleNamespace(
            **d,
            is_unassignment_queued=False,
            is_waiting_for_locomotive=False,
            loco_type="LOCOMOTIVE",
        )

    def _act_relocate_loco(self, act: bp.RelocateLocoAction) -> None:
        src, dst = self._line(act.source_line_id), self._line(act.target_line_id)
        tr = self._pick_train(src.id, act.train_id)
        tr.line_id, tr.onboard_count, tr.is_pending = dst.id, 0, False

    def _act_attach_carriage(self, act: bp.AttachCarriageAction) -> None:
        ln = self._line(act.line_id)
        if self.inv["carriages_available"] < 1:
            _err(bp.ErrorCode.INSUFFICIENT_RESOURCES, "No carriages left")
        tr = self._pick_train(ln.id, act.train_id)
        self.inv["carriages_available"] -= 1
        tr.carriages += 1

    def _act_place_interchange(self, act: bp.PlaceInterchangeAction) -> None:
        st = self._station(act.station_id)
        if st.is_interchange:
            _err(bp.ErrorCode.INVALID_STATE, "Already an interchange")
        if self.inv["interchanges_available"] < 1:
            _err(bp.ErrorCode.INSUFFICIENT_RESOURCES, "No interchanges left")
        self.inv["interchanges_available"] -= 1
        st.is_interchange, st.peep_capacity = True, 18

    def _act_pick_upgrade(self, act: bp.PickUpgradeAction) -> None:
        if self.screen != bp.ScreenKind.WEEKLY_UPGRADE:
            _err(bp.ErrorCode.INVALID_STATE, "Not on WEEKLY_UPGRADE screen")
        offer = next(
            (o for o in self.weekly_offers if o.index == act.offer_index), None
        )
        if offer is None:
            _err(bp.ErrorCode.INVALID_ARGS, f"Invalid offer {act.offer_index}")
        if offer.kind == "LINE":
            self.inv["lines_total"] += offer.count
            self.inv["lines_available"] += offer.count
        elif key := _REWARDS.get(offer.kind):
            self.inv[key] += offer.count
        if self.weekly_stage == 1:
            self.weekly_stage = 2
            self.weekly_offers = list(_OFFERS_S2)
            return
        self.weekly_stage = 0
        self.weekly_offers.clear()
        self.is_paused, self.screen = False, bp.ScreenKind.PLAYING


class _RpcHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        board: MockMetroBoard = self.server.board  # type: ignore[attr-defined]
        for raw_bytes in self.rfile:
            resp = None
            try:
                req = bp.RpcRequest.from_json(raw_bytes.decode("utf-8"))
                resp = board.handle_request(req)
            except UnicodeDecodeError as e:
                ec, rid, msg = bp.ErrorCode.INVALID_JSON, None, str(e)
            except bp.ProtocolValidationError as e:
                ec, rid, msg = e.error_code, e.request_id, str(e)
            except Exception as e:
                ec, rid, msg = bp.ErrorCode.INTERNAL_ERROR, None, str(e)
            if resp is None:
                resp = bp.RpcResponse.failure(rid, board.seq, ec, msg)
            self.wfile.write((resp.to_json() + "\n").encode("utf-8"))
            self.wfile.flush()


class _ThreadedRpcServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address, daemon_threads = True, True


class MockMetroRpcServer:
    """Context-managed TCP NDJSON mock server wrapping ``MockMetroBoard``."""

    def __init__(self, host: str = bp.DEFAULT_HOST, port: int = 0, board: Any = None):
        self.board = board or MockMetroBoard()
        self._srv = _ThreadedRpcServer((host, port), _RpcHandler)
        self._srv.board = self.board  # type: ignore[attr-defined]
        addr = self._srv.server_address
        self.host, self.port = str(addr[0]), int(addr[1])
        self.address = (self.host, self.port)
        self._thread: threading.Thread | None = None

    def start(self) -> MockMetroRpcServer:
        if self._thread is None:
            self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._srv.shutdown()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._srv.server_close()

    def __enter__(self) -> MockMetroRpcServer:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mock bridge-rpc-v1 TCP NDJSON server")
    p.add_argument("--host", default=bp.DEFAULT_HOST)
    p.add_argument("--port", type=int, default=bp.DEFAULT_PORT)
    p.add_argument("--initial-stations", type=int, default=6)
    p.add_argument("--max-stations", type=int, default=36)
    p.add_argument("--tunnels", type=int, default=2)
    a = p.parse_args(argv)
    b = MockMetroBoard(
        initial_stations=a.initial_stations,
        max_stations=a.max_stations,
        tunnels=a.tunnels,
    )
    with MockMetroRpcServer(a.host, a.port, board=b) as srv:
        print(f"Listening on {srv.host}:{srv.port}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
