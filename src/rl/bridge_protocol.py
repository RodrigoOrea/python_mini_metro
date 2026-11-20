"""Frozen ``bridge-rpc-v1`` protocol contract, schema, and validation layer."""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, make_dataclass
from enum import Enum, StrEnum
from functools import reduce
from typing import Any, Self

PROTOCOL_VERSION = "bridge-rpc-v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17842
ALLOWED_SPEED_MULTIPLIERS = frozenset({1.0, 2.0})
EntityId = int | str


def _se(name: str, words: str) -> Any:
    return StrEnum(name, {w: w for w in words.split()})


_OAK = (
    "WAIT CONNECT EXTEND_LINE PREPEND_LINE ADD_TO_LINE INSERT_STATION REPLACE_LINE REMOVE_LINE "
    "ASSIGN_LOCO RELOCATE_LOCO ATTACH_CARRIAGE PLACE_INTERCHANGE PICK_UPGRADE"
)
ErrorCode = _se(
    "ErrorCode",
    "INVALID_JSON UNSUPPORTED_PROTOCOL UNKNOWN_COMMAND INVALID_ARGS INVALID_STATE "
    "INSUFFICIENT_RESOURCES TUNNEL_BUDGET_EXCEEDED OUT_OF_MAIN_THREAD_TIMEOUT "
    "INTERNAL_ERROR",
)
ScreenKind = _se("ScreenKind", "MENU PLAYING PAUSED WEEKLY_UPGRADE GAME_OVER OTHER")
Screen = ScreenKind
LocoType = _se("LocoType", "LOCOMOTIVE SHINKANSEN TRAM")
StationShape = _se(
    "StationShape",
    "CIRCLE TRIANGLE SQUARE CROSS DIAMOND EGG GEM PENTAGON STAR WEDGE",
)
OfferAssetKind = _se(
    "OfferAssetKind",
    "LINE LOCOMOTIVE SHINKANSEN TRAM FERRY CARRIAGE CROSSING INTERCHANGE",
)
OpenActionKind = _se("OpenActionKind", _OAK)
RpcCommand = _se("RpcCommand", f"PING GET_STATE SET_PAUSED SET_SPEED {_OAK}")


class ProtocolValidationError(ValueError):
    """Raised when a wire message, snapshot, or action violates the contract."""

    def __init__(self, msg: str, ec: Any = ErrorCode.INVALID_ARGS, rid: Any = None):
        super().__init__(msg)
        self.error_code, self.request_id = ec, rid


_EN = {"scr": (ScreenKind, 1), "loco": (LocoType, 0), "ec": (ErrorCode, 1)}
_EN |= {"cmd": (RpcCommand, 0), "oak": (OpenActionKind, 1)}
_EN |= {"shp": (StationShape, 0), "oac": (OfferAssetKind, 0)}


def _eid(v: Any) -> bool:
    is_i = isinstance(v, int) and not isinstance(v, bool)
    return (is_i and v >= 0) or (isinstance(v, str) and bool(v.strip()))


def _chk(t: str, v: Any, n: str) -> Any:
    if t == "b" and isinstance(v, bool):
        return v
    if t in ("i0", "i1", "c1") and isinstance(v, int) and not isinstance(v, bool):
        if v >= (t != "i0") and (t != "c1" or v == 1):
            return v
    if t in ("f", "f0", "fp", "fu", "sp") and not isinstance(v, bool):
        if isinstance(v, (int, float)) and math.isfinite(x := float(v)):
            lo = 1e-12 if t == "fp" else (0.0 if t in ("f0", "fu") else x)
            if lo <= x <= (1.0 if t == "fu" else x):
                if t != "sp" or x in ALLOWED_SPEED_MULTIPLIERS:
                    return x
    if (t == "s" and isinstance(v, str) and v.strip()) or (t == "id" and _eid(v)):
        return v
    if t == "sq" and isinstance(v, (list, tuple)):
        return tuple(v)
    if t == "sids" and isinstance(v, (list, tuple)):
        xs = tuple(_chk("id", x, n) for x in v)
        if len(xs) >= 2 and len(set(xs)) == len(xs):
            return xs
    if t == "js":
        if v is None or isinstance(v, (bool, int, str)):
            return v
        if isinstance(v, float) and math.isfinite(v):
            return v
        if isinstance(v, (list, tuple)):
            return [_chk("js", x, n) for x in v]
        if isinstance(v, Mapping):
            return {_chk("s", k, n): _chk("js", w, n) for k, w in v.items()}
    if entry := _EN.get(t):
        cls, keep = entry
        ok = isinstance(v, cls) or (isinstance(v, str) and v in cls._value2member_map_)
        if ok:
            return cls(v) if keep else (v.value if isinstance(v, cls) else v)
    raise ProtocolValidationError(f"{n} invalid ({t}): {v!r}")


def _map(
    v: Any, n: str, req: set[str] = frozenset(), allow: set[str] | None = None
) -> Mapping[str, Any]:
    if not isinstance(v, Mapping) or not all(isinstance(k, str) for k in v):
        raise ProtocolValidationError(f"{n} must be dict")
    ks = set(v)
    if (m := req - ks) or (allow is not None and (e := ks - allow)):
        raise ProtocolValidationError(f"{n} keys: {sorted(m or e)}")
    return v


def loads_json_object(raw: str) -> dict[str, Any]:
    """Decode one NDJSON line into a strict JSON dict (no NaN/Inf/duplicates)."""
    ec = ErrorCode.INVALID_JSON
    s = raw.rstrip("\r\n") if isinstance(raw, str) and raw.endswith("\n") else raw
    if not isinstance(s, str) or not s or "\n" in s or "\r" in s or raw.count("\n") > 1:
        raise ProtocolValidationError("Need 1 JSON line", ec)

    def _dups(ps: list[tuple[str, Any]]) -> dict[str, Any]:
        if len(out := dict(ps)) != len(ps):
            raise ProtocolValidationError("Duplicate JSON key", ec)
        return out

    def _bad(t: str) -> Any:
        raise ProtocolValidationError(f"Bad JSON {t!r}", ec)

    try:
        obj = json.loads(s, parse_constant=_bad, object_pairs_hook=_dups)
    except json.JSONDecodeError as exc:
        raise ProtocolValidationError(f"JSON: {exc}", ec) from exc
    if not isinstance(obj, dict):
        raise ProtocolValidationError("Root must be dict", ec)
    return obj


def dumps_json_line(payload: Mapping[str, Any]) -> str:
    """Serialize one mapping to compact UTF-8 JSON without NaN/Inf."""
    return json.dumps(
        payload, allow_nan=False, separators=(",", ":"), ensure_ascii=False
    )


_TAGS = {
    fn: ln.split(":", 1)[0]
    for ln in """f:center_x center_y x y screen_x screen_y
fp:ortho_size world_to_ref_scale
fu:overload_progress
sp:speed_multiplier
c1:tunnels
i1:screen_width screen_height ticks count peep_capacity
i0:lines_available lines_total locos_available carriages_available tunnels_available interchanges_available index position wait_ms color_index tunnels_used carriages capacity onboard_count seq timestamp_ms deliveries day offer_index
s:label city_id error
b:is_interchange is_looped is_ghost is_unassignment_queued is_pending is_paused is_waiting_for_locomotive ok
id:id line_id source_line_id target_line_id u v station_id after_station_id train_id
sids:station_ids""".splitlines()
    for fn in ln.split(":", 1)[1].split()
}
_TAGS |= {"screen": "scr", "loco_type": "loco", "cmd": "cmd", "error_code": "ec"}
_TAGS |= {"shape": "shp", "dest_shape": "shp", "kind": "oac"}
_SUB: dict[str, Any] = {}
_OPT_R = frozenset({"id", "error_code", "error", "data"})
_DIFF = {
    "CrossingCost": ("u", "v"),
    "InsertStationAction": ("station_id", "after_station_id"),
}
_DIFF["RelocateLocoAction"] = ("source_line_id", "target_line_id")


class _BaseModel:
    __slots__ = ()

    def __post_init__(self) -> None:
        p = type(self).__name__
        for f in [x for x in fields(self) if x.init]:  # type: ignore[arg-type]
            v, fn = getattr(self, f.name), f.name
            if fn == "protocol":
                if v != PROTOCOL_VERSION:
                    ec = ErrorCode.UNSUPPORTED_PROTOCOL
                    raise ProtocolValidationError("Bad protocol", ec)
            elif v is None and (
                fn == "train_id"
                or (p, fn) == ("ConnectAction", "color_index")
                or (p == "RpcResponse" and fn in _OPT_R)
            ):
                continue
            elif t := _TAGS.get(fn):
                try:
                    object.__setattr__(self, fn, _chk(t, v, f"{p}.{fn}"))
                except ProtocolValidationError as exc:
                    if fn == "cmd" and isinstance(v, str) and v.strip():
                        exc.error_code = ErrorCode.UNKNOWN_COMMAND
                    raise
            elif fn in ("args", "data"):
                object.__setattr__(self, fn, _chk("js", dict(_map(v, fn)), fn))
            elif s := _SUB.get(fn):
                sq = _chk("sq", v, fn) if isinstance(s, list) else (v,)
                cls = s[0] if isinstance(s, list) else s
                if not all(isinstance(x, cls) for x in sq):
                    raise ProtocolValidationError(f"{p}.{fn} bad type")
                if isinstance(s, list):
                    object.__setattr__(self, fn, sq)
        if (pr := _DIFF.get(p)) and getattr(self, pr[0]) == getattr(self, pr[1]):
            raise ProtocolValidationError(f"{p}.{pr[0]} == {pr[1]}")
        if p == "InventorySnapshot" and self.lines_available > self.lines_total:  # type: ignore[attr-defined]
            raise ProtocolValidationError("lines_available > lines_total")

    @staticmethod
    def _plain(v: Any) -> Any:
        if isinstance(v, (Enum, _BaseModel)):
            return v.value if isinstance(v, Enum) else v.to_dict()
        return [_BaseModel._plain(x) for x in v] if isinstance(v, tuple) else v

    def to_dict(self) -> dict[str, Any]:
        p = self._plain
        return {f.name: p(getattr(self, f.name)) for f in fields(self) if f.init}  # type: ignore[arg-type]

    def to_json(self) -> str:
        return dumps_json_line(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        inits = [f for f in fields(cls) if f.init]  # type: ignore[arg-type]
        allow = {f.name for f in inits}
        req = {f.name for f in inits if f.default is f.default_factory is MISSING}
        d = dict(_map(data, cls.__name__, req=req, allow=allow))
        for k in allow & _SUB.keys():
            s = _SUB[k]
            d[k] = (
                tuple(s[0].from_dict(x) for x in _chk("sq", d[k], k))
                if isinstance(s, list)
                else s.from_dict(d[k])
            )
        return cls(**d)

    @classmethod
    def from_json(cls, raw: str) -> Self:
        return cls.from_dict(loads_json_object(raw))

    @staticmethod
    def _env(data: Any, nm: str, req: set[str], allow: set[str]) -> tuple[Any, Any]:
        if not isinstance(data, Mapping):
            raise ProtocolValidationError(f"{nm} not dict", ErrorCode.INVALID_JSON)
        rid = data.get("id")
        req_id = rid if _eid(rid) else None
        if data.get("protocol") != PROTOCOL_VERSION:
            ec = ErrorCode.UNSUPPORTED_PROTOCOL
            raise ProtocolValidationError("Bad protocol", ec, req_id)
        try:
            return _map(data, nm, req=req, allow=allow), req_id
        except ProtocolValidationError as exc:
            exc.request_id = req_id if exc.request_id is None else exc.request_id
            raise


_FT: dict[str, Any] = {"i": int, "b": bool, "f": float, "s": str, "id": EntityId}
_FT |= {"ids": tuple[EntityId, ...], "oi": int | None, "oid": EntityId | None}
_DF: dict[str, Any] = {"0": 0, "1": 1, "6": 6, "F": False, "n": None, "f0": 0.0}
_DF |= {"emp": (), "L": "LOCOMOTIVE"}


def _dc(name: str, spec_str: str, ns: dict[str, Any] | None = None) -> Any:
    spec = [
        (fn, _FT[tp], _DF[dv]) if eq else (fn, _FT[tp])
        for it in spec_str.split()
        for fn, rest in (it.split(":", 1),)
        for tp, eq, dv in (rest.partition("="),)
    ]
    return make_dataclass(
        name, spec, bases=(_BaseModel,), frozen=True, slots=True, namespace=ns
    )


CameraSnapshot = _dc(
    "CameraSnapshot",
    "center_x:f center_y:f ortho_size:f screen_width:i screen_height:i world_to_ref_scale:f",
)
WeeklyOffer = _dc("WeeklyOffer", "index:i kind:s label:s count:i=1")
WaitingPassenger = _dc("WaitingPassenger", "dest_shape:s wait_ms:i")
_FT["wp"] = tuple[WaitingPassenger, ...]
CrossingCost = _dc("CrossingCost", "u:id v:id tunnels:i")
InventorySnapshot = _dc(
    "InventorySnapshot",
    "lines_available:i lines_total:i locos_available:i carriages_available:i tunnels_available:i interchanges_available:i",
)
StationSnapshot = _dc(
    "StationSnapshot",
    "id:id x:f y:f screen_x:f screen_y:f shape:s is_interchange:b=F overload_progress:f=f0 waiting_passengers:wp=emp peep_capacity:i=6",
)
LineSnapshot = _dc(
    "LineSnapshot",
    "id:id color_index:i station_ids:ids is_looped:b=F is_ghost:b=F tunnels_used:i=0",
)
LineSnapshot.__doc__ = "LineSnapshot: id is unique per instance (color_index may overlap ghosts); station_ids omits loop repeat & preserves ghost route."
TrainSnapshot = _dc(
    "TrainSnapshot",
    "id:id line_id:id carriages:i capacity:i onboard_count:i is_unassignment_queued:b=F is_ghost:b=F is_waiting_for_locomotive:b=F loco_type:s=L is_pending:b=F",
)


@dataclass(frozen=True, slots=True)
class Snapshot(_BaseModel):
    """Board snapshot (day is 0-based; LineSnapshot.id is unique per instance)."""

    seq: int
    timestamp_ms: int
    screen: ScreenKind
    city_id: str
    is_paused: bool
    speed_multiplier: float
    deliveries: int
    day: int
    camera: CameraSnapshot
    inventory: InventorySnapshot
    weekly_offers: tuple[WeeklyOffer, ...] = ()
    stations: tuple[StationSnapshot, ...] = ()
    lines: tuple[LineSnapshot, ...] = ()
    trains: tuple[TrainSnapshot, ...] = ()
    crossing_costs: tuple[CrossingCost, ...] = ()
    protocol: str = PROTOCOL_VERSION
    _cmap: dict[tuple[Any, Any], int] = field(init=False, repr=False, compare=False)

    @staticmethod
    def _pair(u: EntityId, v: EntityId) -> tuple[Any, Any]:
        a = (0 if isinstance(u, int) else 1, u)
        b = (0 if isinstance(v, int) else 1, v)
        return (a, b) if a <= b else (b, a)

    def __post_init__(self) -> None:
        super().__post_init__()
        sids, lids = {s.id for s in self.stations}, {ln.id for ln in self.lines}
        dups = len({o.index for o in self.weekly_offers}) != len(self.weekly_offers)
        dups = dups or len({t.id for t in self.trains}) != len(self.trains)
        if dups or len(sids) != len(self.stations) or len(lids) != len(self.lines):
            raise ProtocolValidationError("Duplicate entity id or offer index")
        bad_s = any(sid not in sids for ln in self.lines for sid in ln.station_ids)
        if bad_s or any(tr.line_id not in lids for tr in self.trains):
            raise ProtocolValidationError("Unknown station or line ref")
        cmap: dict[tuple[Any, Any], int] = {}
        for cc in self.crossing_costs:
            k = self._pair(cc.u, cc.v)
            if cc.u not in sids or cc.v not in sids or k in cmap:
                raise ProtocolValidationError("Invalid crossing_cost")
            cmap[k] = cc.tunnels
        object.__setattr__(self, "_cmap", cmap)

    def crossing_cost(self, u: EntityId, v: EntityId) -> int:
        """Return tunnel cost for crossing between station IDs u and v (O(1))."""
        if _chk("id", u, "u") == _chk("id", v, "v"):
            return 0
        return self._cmap.get(self._pair(u, v), 0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Snapshot:
        if isinstance(data, Mapping) and data.get("protocol") != PROTOCOL_VERSION:
            ec = ErrorCode.UNSUPPORTED_PROTOCOL
            raise ProtocolValidationError("Bad protocol", ec)
        return super().from_dict(data)


_SUB.update(camera=CameraSnapshot, inventory=InventorySnapshot)
_SUB.update(weekly_offers=[WeeklyOffer], waiting_passengers=[WaitingPassenger])
_SUB.update(stations=[StationSnapshot], lines=[LineSnapshot], trains=[TrainSnapshot])
_SUB["crossing_costs"] = [CrossingCost]
_ACTION_MAP: dict[OpenActionKind, type[_BaseModel]] = {}


def _act(k: str, fs: str) -> Any:
    kind = OpenActionKind(k)
    cls = _dc(f"{k.title().replace('_', '')}Action", fs, ns={"kind": kind})
    _ACTION_MAP[kind] = cls
    return cls


WaitAction = _act("WAIT", "ticks:i=1")
ConnectAction = _act("CONNECT", "station_ids:ids is_looped:b=F color_index:oi=n")
ExtendLineAction = _act("EXTEND_LINE", "line_id:id station_id:id is_looped:b=F")
PrependLineAction = _act("PREPEND_LINE", "line_id:id station_id:id is_looped:b=F")
AddToLineAction = _act(
    "ADD_TO_LINE", "line_id:id station_id:id position:i is_looped:b=F"
)
InsertStationAction = _act(
    "INSERT_STATION", "line_id:id station_id:id after_station_id:id"
)
ReplaceLineAction = _act("REPLACE_LINE", "line_id:id station_ids:ids is_looped:b=F")
RemoveLineAction = _act("REMOVE_LINE", "line_id:id")
AssignLocoAction = _act("ASSIGN_LOCO", "line_id:id")
RelocateLocoAction = _act(
    "RELOCATE_LOCO", "source_line_id:id target_line_id:id train_id:oid=n"
)
AttachCarriageAction = _act("ATTACH_CARRIAGE", "line_id:id train_id:oid=n")
PlaceInterchangeAction = _act("PLACE_INTERCHANGE", "station_id:id")
PickUpgradeAction = _act("PICK_UPGRADE", "offer_index:i")
OpenAction = reduce(operator.or_, _ACTION_MAP.values())


@dataclass(frozen=True, slots=True)
class RpcRequest(_BaseModel):
    """Single NDJSON request envelope for ``bridge-rpc-v1``."""

    id: EntityId
    cmd: str
    args: dict[str, Any] = field(default_factory=dict)
    protocol: str = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RpcRequest:
        allow = {"protocol", "id", "cmd", "args"}
        d, rid = cls._env(data, "request", {"protocol", "id", "cmd"}, allow)
        try:
            return cls(**d)
        except ProtocolValidationError as exc:
            exc.request_id = rid if exc.request_id is None else exc.request_id
            raise

    @classmethod
    def from_action(cls, a: OpenAction, rid: EntityId = "1") -> RpcRequest:
        d = {k: v for k, v in a.to_dict().items() if v is not None}
        return cls(rid, a.kind.value, d)

    @classmethod
    def ping(cls, rid: EntityId = "1") -> RpcRequest:
        return cls(rid, "PING")

    @classmethod
    def get_state(cls, rid: EntityId = "1") -> RpcRequest:
        return cls(rid, "GET_STATE")

    @classmethod
    def set_paused(cls, paused: bool, rid: EntityId = "1") -> RpcRequest:
        return cls(rid, "SET_PAUSED", {"paused": _chk("b", paused, "p")})

    @classmethod
    def set_speed(cls, spd: float, rid: EntityId = "1") -> RpcRequest:
        return cls(rid, "SET_SPEED", {"speed_multiplier": _chk("sp", spd, "speed")})


@dataclass(frozen=True, slots=True)
class RpcResponse(_BaseModel):
    """Single NDJSON response envelope for ``bridge-rpc-v1``."""

    id: EntityId | None
    ok: bool
    seq: int
    error_code: ErrorCode | None = None
    error: str | None = None
    data: dict[str, Any] | None = None
    protocol: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.error_code is None) != self.ok or (self.error is None) != self.ok:
            raise ProtocolValidationError("Inconsistent response error fields")

    @classmethod
    def success(cls, rid: Any, seq: int, data: Any = None) -> RpcResponse:
        return cls(id=rid, ok=True, seq=seq, data=None if data is None else dict(data))

    @classmethod
    def failure(cls, rid: Any, seq: int, ec: ErrorCode, err: str) -> RpcResponse:
        return cls(id=rid, ok=False, seq=seq, error_code=ec, error=err)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RpcResponse:
        allow = {"protocol", "id", "ok", "seq", "error_code", "error", "data"}
        d, _ = cls._env(data, "response", {"protocol", "ok", "seq"}, allow)
        return cls(**d)


def action_from_request(request: RpcRequest) -> OpenAction:
    """Parse and validate a typed OpenAction from an RpcRequest."""
    kind = _chk("oak", request.cmd, "request.cmd")
    cls = _ACTION_MAP[kind]
    inits = [f for f in fields(cls) if f.init]  # type: ignore[arg-type]
    allow = {f.name for f in inits}
    req = {f.name for f in inits if f.default is f.default_factory is MISSING}
    args = dict(request.args)
    if "is_looped" in allow and "loop" in args:
        if "is_looped" in args:
            raise ProtocolValidationError("Specify either is_looped or loop")
        args["is_looped"] = args.pop("loop")
    return cls(**_map(args, f"{kind.value}.args", req, allow))  # type: ignore[return-value]


def parse_control_args(cmd: str, args: Mapping[str, Any]) -> Any:
    """Validate arguments for non-action RPC commands (PING/GET_STATE/SET_*)."""
    if cmd in ("PING", "GET_STATE"):
        return _map(args, cmd, allow=set())
    if cmd == "SET_PAUSED":
        return _chk("b", _map(args, cmd, {"paused"}, {"paused"})["paused"], "paused")
    if cmd == "SET_SPEED":
        v = _map(args, cmd, {"speed_multiplier"}, {"speed_multiplier"})
        return _chk("sp", v["speed_multiplier"], "speed_multiplier")
    raise ProtocolValidationError(f"Not control cmd: {cmd!r}")


action_request, ping_request = RpcRequest.from_action, RpcRequest.ping
get_state_request, set_paused_request = RpcRequest.get_state, RpcRequest.set_paused
set_speed_request = RpcRequest.set_speed
