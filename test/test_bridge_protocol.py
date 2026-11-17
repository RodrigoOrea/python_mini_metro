"""Tests for the frozen ``bridge-rpc-v1`` protocol contract and mock RPC server."""

from __future__ import annotations

import os
import socket
import sys
import unittest
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import rl.bridge_protocol as bp  # noqa: E402
from scripts.mock_metro_rpc import MockMetroBoard, MockMetroRpcServer  # noqa: E402


def _make_snapshot(n_stations: int = 24, n_lines: int = 6) -> dict[str, Any]:
    shapes = ("CIRCLE", "TRIANGLE", "SQUARE", "STAR", "PENTAGON", "DIAMOND")
    stations = [
        {
            "id": i,
            "x": float((-1 if i % 2 else 1) * (100 + i * 10)),
            "y": float((i % 5) * 30),
            "screen_x": 960.0 + i * 5,
            "screen_y": 540.0 - i * 3,
            "shape": shapes[i % len(shapes)],
            "is_interchange": i == 1,
            "overload_progress": 0.25 if i == 1 else 0.0,
            "waiting_passengers": [{"dest_shape": "CIRCLE", "wait_ms": 1500}]
            if i == 1
            else [],
            "peep_capacity": 18 if i == 1 else 6,
        }
        for i in range(1, n_stations + 1)
    ]
    lines = [
        {
            "id": ln_id,
            "color_index": ln_id - 1,
            "station_ids": [ln_id * 2 - 1, ln_id * 2, ln_id * 2 + 1],
            "is_looped": ln_id == 1,
            "is_ghost": False,
            "tunnels_used": 1,
        }
        for ln_id in range(1, n_lines + 1)
    ]
    trains = [
        {
            "id": 100 + k,
            "line_id": k,
            "carriages": 1,
            "capacity": 12,
            "onboard_count": 3,
            "is_unassignment_queued": False,
            "is_ghost": False,
            "is_waiting_for_locomotive": k == 2,
            "loco_type": "SHINKANSEN" if k == 1 else "LOCOMOTIVE",
        }
        for k in range(1, n_lines + 1)
    ]
    return {
        "protocol": bp.PROTOCOL_VERSION,
        "seq": 42,
        "timestamp_ms": 123456,
        "screen": "PLAYING",
        "city_id": "tokyo",
        "is_paused": False,
        "speed_multiplier": 2.0,
        "deliveries": 99,
        "day": 14,
        "camera": {
            "center_x": 0.0,
            "center_y": 0.0,
            "ortho_size": 540.0,
            "screen_width": 1920,
            "screen_height": 1080,
            "world_to_ref_scale": 1.0,
        },
        "inventory": {
            "lines_available": 1,
            "lines_total": 7,
            "locos_available": 2,
            "carriages_available": 3,
            "tunnels_available": 4,
            "interchanges_available": 1,
        },
        "weekly_offers": [{"index": 0, "kind": "new_line", "label": "New Line"}],
        "stations": stations,
        "lines": lines,
        "trains": trains,
        "crossing_costs": [
            {"u": 1, "v": 2, "tunnels": 1},
            {"u": 3, "v": 4, "tunnels": 2},
        ],
    }


class TestBridgeProtocol(unittest.TestCase):
    def test_constants_and_uppercase_enums(self) -> None:
        self.assertEqual(bp.PROTOCOL_VERSION, "bridge-rpc-v1")
        self.assertEqual(bp.DEFAULT_HOST, "127.0.0.1")
        self.assertEqual(bp.DEFAULT_PORT, 17842)
        self.assertEqual(
            bp.ErrorCode.TUNNEL_BUDGET_EXCEEDED.value, "TUNNEL_BUDGET_EXCEEDED"
        )
        self.assertEqual(bp.OpenActionKind.RELOCATE_LOCO.value, "RELOCATE_LOCO")
        self.assertEqual(bp.OpenActionKind.PLACE_INTERCHANGE.value, "PLACE_INTERCHANGE")
        self.assertEqual(bp.LocoType.SHINKANSEN.value, "SHINKANSEN")
        self.assertIn("SET_SPEED", {c.value for c in bp.RpcCommand})

    def test_large_snapshot_roundtrip_and_crossing_lookup(self) -> None:
        raw = _make_snapshot(n_stations=26, n_lines=6)
        snap = bp.Snapshot.from_dict(raw)
        self.assertEqual(len(snap.stations), 26)
        self.assertEqual(len(snap.lines), 6)
        self.assertEqual(snap.stations[0].peep_capacity, 18)
        self.assertTrue(snap.trains[1].is_waiting_for_locomotive)
        self.assertEqual(snap.trains[0].loco_type, "SHINKANSEN")
        self.assertEqual(snap.crossing_cost(1, 2), 1)
        self.assertEqual(snap.crossing_cost(2, 1), 1)
        self.assertEqual(snap.crossing_cost(4, 3), 2)
        self.assertEqual(snap.crossing_cost(1, 1), 0)
        self.assertEqual(snap.crossing_cost(1, 26), 0)

        wire = snap.to_json()
        self.assertNotIn("\n", wire)
        reloaded = bp.Snapshot.from_json(wire)
        self.assertEqual(reloaded, snap)
        self.assertEqual(reloaded.crossing_cost(2, 1), 1)

    def test_rejects_malformed_non_finite_and_duplicate_json(self) -> None:
        bad_lines = [
            "not json",
            "[1, 2, 3]",
            '{"a": NaN}',
            '{"a": Infinity}',
            '{"a": -Infinity}',
            '{"a": 1, "a": 2}',
            '{"a": 1}\n{"b": 2}',
            "",
        ]
        for line in bad_lines:
            with self.assertRaises(bp.ProtocolValidationError) as ctx:
                bp.loads_json_object(line)
            self.assertEqual(ctx.exception.error_code, bp.ErrorCode.INVALID_JSON)

    def test_snapshot_and_envelope_validation_errors(self) -> None:
        bad_snap = _make_snapshot(4, 1)
        bad_snap["stations"][0]["overload_progress"] = 1.5
        with self.assertRaises(bp.ProtocolValidationError):
            bp.Snapshot.from_dict(bad_snap)

        bad_snap = _make_snapshot(4, 1)
        bad_snap["camera"]["center_x"] = float("nan")
        with self.assertRaises(bp.ProtocolValidationError):
            bp.Snapshot.from_dict(bad_snap)

        bad_snap = _make_snapshot(4, 1)
        bad_snap["protocol"] = "wrong-v0"
        with self.assertRaises(bp.ProtocolValidationError) as ctx:
            bp.Snapshot.from_dict(bad_snap)
        self.assertEqual(ctx.exception.error_code, bp.ErrorCode.UNSUPPORTED_PROTOCOL)

        bad_snap = _make_snapshot(4, 1)
        bad_snap["trains"][0]["loco_type"] = "ROCKET"
        with self.assertRaises(bp.ProtocolValidationError):
            bp.Snapshot.from_dict(bad_snap)

        with self.assertRaises(bp.ProtocolValidationError):
            bp.WaitAction(ticks=True)  # type: ignore[arg-type]
        with self.assertRaises(bp.ProtocolValidationError):
            bp.CrossingCost(u=1, v=1, tunnels=1)
        with self.assertRaises(bp.ProtocolValidationError):
            bp.RpcResponse(id="1", ok=True, seq=1, error="should fail")
        with self.assertRaises(bp.ProtocolValidationError):
            bp.RpcResponse(id="1", ok=False, seq=1)

    def test_open_actions_control_helpers_and_speed_validation(self) -> None:
        actions: list[bp.OpenAction] = [
            bp.WaitAction(ticks=3),
            bp.ConnectAction(station_ids=(1, 2, 3), is_looped=True, color_index=0),
            bp.ExtendLineAction(line_id=1, station_id=4, is_looped=True),
            bp.PrependLineAction(line_id=1, station_id=5, is_looped=False),
            bp.InsertStationAction(line_id=1, station_id=6, after_station_id=2),
            bp.ReplaceLineAction(line_id=1, station_ids=(1, 4, 2), is_looped=False),
            bp.RemoveLineAction(line_id=1),
            bp.AssignLocoAction(line_id=2),
            bp.UnassignLocoAction(line_id=2, train_id=10),
            bp.RelocateLocoAction(source_line_id=2, target_line_id=3, train_id=10),
            bp.AttachCarriageAction(line_id=3),
            bp.PlaceInterchangeAction(station_id=2),
            bp.PickUpgradeAction(offer_index=1),
        ]
        for act in actions:
            req = bp.action_request(act, rid="req-1")
            parsed = bp.action_from_request(bp.RpcRequest.from_json(req.to_json()))
            self.assertEqual(parsed, act)

        loop_req = bp.RpcRequest(
            id="x",
            cmd="EXTEND_LINE",
            args={"line_id": 1, "station_id": 2, "loop": True},
        )
        self.assertEqual(
            bp.action_from_request(loop_req), bp.ExtendLineAction(1, 2, True)
        )

        with self.assertRaises(bp.ProtocolValidationError):
            bp.RelocateLocoAction(source_line_id=2, target_line_id=2)
        with self.assertRaises(bp.ProtocolValidationError):
            bp.InsertStationAction(line_id=1, station_id=2, after_station_id=2)

        for spd in (1, 2, 4, 2.0):
            req = bp.set_speed_request(spd, rid=9)
            self.assertEqual(bp.parse_control_args("SET_SPEED", req.args), float(spd))
        for bad_spd in (0, 3, -1, True, float("inf")):
            with self.assertRaises(bp.ProtocolValidationError):
                bp.set_speed_request(bad_spd)  # type: ignore[arg-type]

        req = bp.ping_request("p1")
        self.assertEqual(bp.RpcRequest.from_json(req.to_json()), req)
        self.assertEqual(
            bp.parse_control_args("GET_STATE", bp.get_state_request().args), {}
        )
        self.assertTrue(
            bp.parse_control_args("SET_PAUSED", bp.set_paused_request(True).args)
        )


class TestMockMetroRpcServer(unittest.TestCase):
    def _send(self, rw: Any, raw: str) -> bp.RpcResponse:
        rw.write(raw.rstrip("\n") + "\n")
        rw.flush()
        reply = rw.readline()
        self.assertTrue(reply.endswith("\n"))
        return bp.RpcResponse.from_json(reply)

    def _call(self, rw: Any, req: bp.RpcRequest) -> bp.RpcResponse:
        return self._send(rw, req.to_json())

    def test_end_to_end_tcp_rpc_and_game_mechanics(self) -> None:
        board = MockMetroBoard(
            initial_stations=8,
            max_stations=26,
            lines_total=5,
            locos=2,
            carriages=2,
            tunnels=1,
            interchanges=1,
            ticks_per_week=20,
            ghost_ttl_ticks=2,
        )
        with MockMetroRpcServer(port=0, board=board) as srv:
            with socket.create_connection(srv.address, timeout=3.0) as sock:
                rw = sock.makefile("rw", encoding="utf-8", newline="\n")
                try:
                    ping = self._call(rw, bp.ping_request("p0"))
                    self.assertTrue(ping.ok and ping.data["pong"])

                    st0 = bp.Snapshot.from_dict(
                        self._call(rw, bp.get_state_request("s0")).data["snapshot"]
                    )
                    self.assertEqual(len(st0.stations), 8)
                    self.assertEqual(st0.crossing_cost(1, 2), 1)
                    self.assertEqual(st0.crossing_cost(2, 1), 1)
                    self.assertEqual(st0.crossing_cost(1, 3), 0)

                    paused = self._call(rw, bp.set_paused_request(True, "pa1"))
                    self.assertEqual(paused.data["snapshot"]["screen"], "PAUSED")
                    self._call(rw, bp.set_paused_request(False, "pa2"))
                    spd = self._call(rw, bp.set_speed_request(2, "sp1"))
                    self.assertEqual(spd.data["snapshot"]["speed_multiplier"], 2.0)

                    # Connect west-bank line [1, 3] (0 tunnels), extend to 5, prepend 7
                    r = self._call(
                        rw, bp.action_request(bp.ConnectAction((1, 3)), "c1")
                    )
                    self.assertTrue(r.ok)
                    line1 = bp.Snapshot.from_dict(r.data["snapshot"]).lines[0]
                    self.assertEqual(line1.tunnels_used, 0)

                    r = self._call(
                        rw,
                        bp.action_request(
                            bp.ExtendLineAction(line1.id, 5, is_looped=True), "e1"
                        ),
                    )
                    snap = bp.Snapshot.from_dict(r.data["snapshot"])
                    self.assertTrue(snap.lines[0].is_looped)
                    self.assertEqual(snap.lines[0].station_ids, (1, 3, 5))

                    # Assign loco, attach carriage, then insert station 7 without demolition
                    self._call(
                        rw, bp.action_request(bp.AssignLocoAction(line1.id), "a1")
                    )
                    self._call(
                        rw, bp.action_request(bp.AttachCarriageAction(line1.id), "ac1")
                    )
                    r = self._call(
                        rw,
                        bp.action_request(bp.InsertStationAction(line1.id, 7, 3), "i1"),
                    )
                    snap = bp.Snapshot.from_dict(r.data["snapshot"])
                    self.assertEqual(snap.lines[0].station_ids, (1, 3, 7, 5))
                    self.assertEqual(len(snap.trains), 1)
                    self.assertEqual(snap.trains[0].carriages, 1)

                    # Cross-river line [2, 1] spends the 1 available tunnel; 2nd fails
                    r = self._call(
                        rw, bp.action_request(bp.ConnectAction((2, 1)), "c2")
                    )
                    self.assertTrue(r.ok)
                    r = self._call(
                        rw, bp.action_request(bp.ConnectAction((4, 3)), "c_bad")
                    )
                    self.assertFalse(r.ok)
                    self.assertEqual(r.error_code, bp.ErrorCode.TUNNEL_BUDGET_EXCEEDED)

                    # Relocate loco from line 1 to line 2 (no extra loco consumed)
                    r = self._call(
                        rw,
                        bp.action_request(bp.RelocateLocoAction(line1.id, 2), "rel1"),
                    )
                    snap = bp.Snapshot.from_dict(r.data["snapshot"])
                    self.assertEqual(snap.trains[0].line_id, 2)
                    self.assertEqual(snap.inventory.locos_available, 1)

                    # Place interchange -> peep_capacity=18; 2nd fails INSUFFICIENT_RESOURCES
                    r = self._call(
                        rw, bp.action_request(bp.PlaceInterchangeAction(1), "ic1")
                    )
                    snap = bp.Snapshot.from_dict(r.data["snapshot"])
                    self.assertTrue(snap.stations[0].is_interchange)
                    self.assertEqual(snap.stations[0].peep_capacity, 18)
                    r_ic2 = self._call(
                        rw, bp.action_request(bp.PlaceInterchangeAction(2), "ic2")
                    )
                    self.assertEqual(
                        r_ic2.error_code, bp.ErrorCode.INSUFFICIENT_RESOURCES
                    )

                    # Ghost line/train lifecycle on removing served line 2 (uses 1 tunnel, 1 train, 1 carriage)
                    r = self._call(rw, bp.action_request(bp.RemoveLineAction(2), "rm2"))
                    snap = bp.Snapshot.from_dict(r.data["snapshot"])
                    l2 = next(ln for ln in snap.lines if ln.id == 2)
                    self.assertTrue(l2.is_ghost)
                    self.assertTrue(snap.trains[0].is_ghost)
                    self.assertEqual(snap.inventory.tunnels_available, 0)
                    r_edit_ghost = self._call(
                        rw, bp.action_request(bp.AssignLocoAction(2), "ag")
                    )
                    self.assertEqual(
                        r_edit_ghost.error_code, bp.ErrorCode.INVALID_STATE
                    )

                    # Wait 2 ticks -> ghost line settles, returns loco, carriage, tunnel, line slot
                    r = self._call(rw, bp.action_request(bp.WaitAction(ticks=2), "w1"))
                    snap = bp.Snapshot.from_dict(r.data["snapshot"])
                    self.assertFalse(any(ln.id == 2 for ln in snap.lines))
                    self.assertEqual(snap.inventory.tunnels_available, 1)
                    self.assertEqual(snap.inventory.locos_available, 2)
                    self.assertEqual(snap.inventory.carriages_available, 2)

                    # Grow past 20 stations and hit WEEKLY_UPGRADE at tick 20
                    r = self._call(rw, bp.action_request(bp.WaitAction(ticks=18), "w2"))
                    snap = bp.Snapshot.from_dict(r.data["snapshot"])
                    self.assertGreater(len(snap.stations), 20)
                    self.assertEqual(snap.screen, bp.ScreenKind.WEEKLY_UPGRADE)
                    self.assertGreaterEqual(len(snap.weekly_offers), 3)

                    blocked = self._call(
                        rw, bp.action_request(bp.ConnectAction((4, 6)), "blk")
                    )
                    self.assertEqual(blocked.error_code, bp.ErrorCode.INVALID_STATE)

                    picked = self._call(
                        rw, bp.action_request(bp.PickUpgradeAction(0), "up0")
                    )
                    snap = bp.Snapshot.from_dict(picked.data["snapshot"])
                    self.assertEqual(snap.screen, bp.ScreenKind.PLAYING)
                    self.assertEqual(snap.inventory.lines_total, 6)

                    # Create >4 lines (5 distinct active lines) & test prepend/replace/unassign
                    for idx, (u, v) in enumerate(
                        ((2, 4), (4, 6), (6, 8), (8, 10)), start=1
                    ):
                        res = self._call(
                            rw, bp.action_request(bp.ConnectAction((u, v)), f"ml{idx}")
                        )
                        self.assertTrue(res.ok)
                    prep = self._call(
                        rw,
                        bp.action_request(
                            bp.PrependLineAction(line_id=3, station_id=10), "pr3"
                        ),
                    )
                    self.assertEqual(
                        bp.Snapshot.from_dict(prep.data["snapshot"])
                        .lines[1]
                        .station_ids[0],
                        10,
                    )
                    repl = self._call(
                        rw,
                        bp.action_request(
                            bp.ReplaceLineAction(
                                line_id=3, station_ids=(2, 4, 6), is_looped=True
                            ),
                            "rp3",
                        ),
                    )
                    self.assertTrue(
                        bp.Snapshot.from_dict(repl.data["snapshot"]).lines[1].is_looped
                    )
                    self._call(
                        rw, bp.action_request(bp.AssignLocoAction(line_id=3), "al3")
                    )
                    self._call(rw, bp.action_request(bp.WaitAction(ticks=2), "wd1"))
                    un = self._call(
                        rw, bp.action_request(bp.UnassignLocoAction(line_id=3), "un3")
                    )
                    final_snap = bp.Snapshot.from_dict(un.data["snapshot"])
                    self.assertGreater(final_snap.deliveries, 0)
                    self.assertGreater(len(final_snap.lines), 4)
                    self.assertGreater(len(final_snap.stations), 20)
                finally:
                    rw.close()

    def test_wire_error_recovery_over_tcp(self) -> None:
        with MockMetroRpcServer(port=0) as srv:
            with socket.create_connection(srv.address, timeout=3.0) as sock:
                sock.sendall(b"\xff\xfe\n")
                rw = sock.makefile("rw", encoding="utf-8", newline="\n")
                try:
                    raw_bad = bp.RpcResponse.from_json(rw.readline())
                    self.assertFalse(raw_bad.ok)
                    self.assertEqual(raw_bad.error_code, bp.ErrorCode.INVALID_JSON)
                    bad = [
                        (
                            '{"protocol":"bridge-rpc-v1","id":"e1","cmd":"PING","args":{"x":NaN}}',
                            bp.ErrorCode.INVALID_JSON,
                        ),
                        (
                            '{"protocol":"bad","id":"e2","cmd":"PING","args":{}}',
                            bp.ErrorCode.UNSUPPORTED_PROTOCOL,
                        ),
                        (
                            '{"protocol":"bridge-rpc-v1","id":"e3","cmd":"TELEPORT","args":{}}',
                            bp.ErrorCode.UNKNOWN_COMMAND,
                        ),
                        (
                            '{"protocol":"bridge-rpc-v1","id":"e4","cmd":"CONNECT","args":{"station_ids":[1]}}',
                            bp.ErrorCode.INVALID_ARGS,
                        ),
                    ]
                    for payload, expected in bad:
                        resp = self._send(rw, payload)
                        self.assertFalse(resp.ok)
                        self.assertEqual(resp.error_code, expected)
                    self.assertTrue(self._call(rw, bp.ping_request("ok1")).ok)
                finally:
                    rw.close()


if __name__ == "__main__":
    unittest.main()
