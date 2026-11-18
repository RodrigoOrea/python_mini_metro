"""State-dependent network objective H_theta(s, G) and learnable parameter specification."""

from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl.heuristic import _distance
from rl.heuristic_global_replan import compute_global_objective

D_REF = 250.0
UNREACH_PENALTY = 30.0

PARAM_NAMES: tuple[str, ...] = (
    "alpha_dist",
    "beta_transfer",
    "gamma_load",
    "balance_weight",
    "w_shape_rarity",
    "w_station_pax",
    "w_station_wait",
    "w_line_overload",
    "single_min_rel",
    "multi_min_rel",
    "demo_risk_weight",
    "w_fleet_wait",
    "w_fleet_load",
    "w_rebalance",
    "hub_share_bonus",
)

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "alpha_dist": (0.3, 4.0),
    "beta_transfer": (0.1, 5.0),
    "gamma_load": (1.0, 35.0),
    "balance_weight": (0.0, 6.0),
    "w_shape_rarity": (0.0, 3.0),
    "w_station_pax": (0.0, 6.0),
    "w_station_wait": (0.0, 15.0),
    "w_line_overload": (0.0, 5.0),
    "single_min_rel": (0.02, 0.35),
    "multi_min_rel": (0.02, 0.40),
    "demo_risk_weight": (0.0, 300.0),
    "w_fleet_wait": (0.0, 12.0),
    "w_fleet_load": (0.0, 8.0),
    "w_rebalance": (0.0, 10.0),
    "hub_share_bonus": (0.0, 12.0),
}


@dataclass(frozen=True)
class ParametricPolicyParams:
    """Learnable parameter vector anchored cleanly at `heuristic_global_replan`."""

    alpha_dist: float = 1.5
    beta_transfer: float = 1.5
    gamma_load: float = 12.0
    balance_weight: float = 1.8
    w_shape_rarity: float = 0.0
    w_station_pax: float = 0.0
    w_station_wait: float = 0.0
    w_line_overload: float = 0.0
    single_min_rel: float = 0.12
    multi_min_rel: float = 0.15
    demo_risk_weight: float = 0.0
    w_fleet_wait: float = 0.0
    w_fleet_load: float = 0.0
    w_rebalance: float = 0.0
    hub_share_bonus: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in PARAM_NAMES], dtype=np.float64)

    @classmethod
    def from_array(
        cls, values: np.ndarray | list[float] | tuple[float, ...]
    ) -> ParametricPolicyParams:
        if len(values) != len(PARAM_NAMES):
            raise ValueError(
                f"Expected {len(PARAM_NAMES)} parameters, got {len(values)}"
            )
        kwargs = {
            name: float(np.clip(val, *PARAM_BOUNDS[name]))
            for name, val in zip(PARAM_NAMES, values, strict=True)
        }
        return cls(**kwargs)

    def clamped(self) -> ParametricPolicyParams:
        return self.from_array(self.to_array())

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, raw: dict[str, float]) -> ParametricPolicyParams:
        defaults = asdict(cls())
        merged = {name: float(raw.get(name, defaults[name])) for name in PARAM_NAMES}
        return cls(**merged).clamped()

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ParametricPolicyParams:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "params" in raw and isinstance(raw["params"], dict):
            raw = raw["params"]
        return cls.from_dict(raw)

    def uses_live_demand(self) -> bool:
        return (
            self.w_shape_rarity != 0.0
            or self.w_station_pax != 0.0
            or self.w_station_wait != 0.0
            or self.w_line_overload != 0.0
        )

    def uses_dynamic_fleet(self) -> bool:
        return (
            self.w_fleet_wait != 0.0
            or self.w_fleet_load != 0.0
            or self.w_rebalance != 0.0
        )


ANCHOR_PARAMS = ParametricPolicyParams()
FLEET_ABLATION_PARAMS = ParametricPolicyParams(
    w_fleet_wait=4.0, w_fleet_load=2.0, w_rebalance=2.5
)
RISK_FLEET_ABLATION_PARAMS = ParametricPolicyParams(
    demo_risk_weight=80.0, w_fleet_wait=4.0, w_fleet_load=2.0, w_rebalance=2.5
)
BEST_PHASE1_PARAMS = ParametricPolicyParams(
    alpha_dist=1.4188691680249734,
    beta_transfer=1.6191614061389707,
    gamma_load=11.598474102358638,
    balance_weight=2.0251007016506133,
    w_shape_rarity=0.2075868031402226,
    w_station_pax=0.5485723873474961,
    w_station_wait=0.2703927988828928,
    w_line_overload=0.2213541050131369,
    single_min_rel=0.11024860473357267,
    multi_min_rel=0.15640666750453808,
    demo_risk_weight=3.4590248264435637,
    w_fleet_wait=3.2725003587589185,
    w_fleet_load=1.9866530408591565,
    w_rebalance=2.555048203170028,
    hub_share_bonus=0.43094441040311865,
)


@dataclass(frozen=True)
class LiveDemandSnapshot:
    """Lightweight read-only demand and queue telemetry extracted once per decision."""

    station_pax_norm: tuple[float, ...]
    station_wait_sq: tuple[float, ...]
    od_pax_norm: tuple[dict[str, float], ...]
    od_wait_sq: tuple[dict[str, float], ...]
    shape_rarity: dict[str, float]
    line_load: tuple[float, ...]
    line_max_wait_sq: tuple[float, ...]
    line_queue_norm: tuple[float, ...]
    line_onboard_ratio: tuple[float, ...]
    line_px: tuple[float, ...]
    max_station_wait_norm: float


def extract_live_demand(mediator) -> LiveDemandSnapshot:
    """Extract normalized queue, wait-time, OD demand, and line load snapshots."""
    stations = mediator.stations
    idx_by_id = {s.id: i for i, s in enumerate(stations)}
    st_cap = float(max(1, getattr(mediator, "station_capacity", 12)))
    wait_limit = float(max(1, getattr(mediator, "passenger_max_wait_time_ms", 40_000)))

    shape_counts: dict[str, int] = {}
    st_pax_norm: list[float] = []
    st_wait_sq: list[float] = []
    od_pax_norm: list[dict[str, float]] = []
    od_wait_sq: list[dict[str, float]] = []
    max_wait_norm = 0.0

    for st in stations:
        sh_name = st.shape.type.value
        shape_counts[sh_name] = shape_counts.get(sh_name, 0) + 1
        pax = st.passengers
        st_pax_norm.append(len(pax) / st_cap)
        mw = 0.0
        dest_cnt: dict[str, float] = {}
        dest_wsq: dict[str, float] = {}
        for px in pax:
            wn = px.wait_ms / wait_limit
            if wn > mw:
                mw = wn
            d_shape = px.destination_shape.type.value
            dest_cnt[d_shape] = dest_cnt.get(d_shape, 0.0) + 1.0 / st_cap
            w2 = wn * wn
            if w2 > dest_wsq.get(d_shape, 0.0):
                dest_wsq[d_shape] = w2
        if mw > max_wait_norm:
            max_wait_norm = mw
        st_wait_sq.append(mw * mw)
        od_pax_norm.append(dest_cnt)
        od_wait_sq.append(dest_wsq)

    n_st = max(1, len(stations))
    n_unique = max(1, len(shape_counts))
    avg_count = n_st / n_unique
    shape_rarity = {
        sh: max(0.0, avg_count / max(1, cnt) - 1.0) for sh, cnt in shape_counts.items()
    }

    line_load: list[float] = []
    line_max_wsq: list[float] = []
    line_q_norm: list[float] = []
    line_onboard_r: list[float] = []
    line_px_list: list[float] = []
    for path in mediator.paths:
        l_indices = [idx_by_id[s.id] for s in path.stations if s.id in idx_by_id]
        waiting_pax = sum(len(stations[i].passengers) for i in l_indices)
        l_wsq = max((st_wait_sq[i] for i in l_indices), default=0.0)
        l_qnorm = sum(st_pax_norm[i] for i in l_indices) / max(1, len(l_indices))
        onboard = sum(len(m.passengers) for m in path.metros)
        cap = max(6.0, float(sum(m.capacity for m in path.metros)))
        l_load = (waiting_pax + onboard) / cap
        l_px = 0.0
        for i in range(len(l_indices) - 1):
            p1, p2 = (
                stations[l_indices[i]].position,
                stations[l_indices[i + 1]].position,
            )
            l_px += float(((p1.left - p2.left) ** 2 + (p1.top - p2.top) ** 2) ** 0.5)
        line_load.append(l_load)
        line_max_wsq.append(l_wsq)
        line_q_norm.append(l_qnorm)
        line_onboard_r.append(onboard / cap)
        line_px_list.append(l_px)

    return LiveDemandSnapshot(
        station_pax_norm=tuple(st_pax_norm),
        station_wait_sq=tuple(st_wait_sq),
        od_pax_norm=tuple(od_pax_norm),
        od_wait_sq=tuple(od_wait_sq),
        shape_rarity=shape_rarity,
        line_load=tuple(line_load),
        line_max_wait_sq=tuple(line_max_wsq),
        line_queue_norm=tuple(line_q_norm),
        line_onboard_ratio=tuple(line_onboard_r),
        line_px=tuple(line_px_list),
        max_station_wait_norm=max_wait_norm,
    )


def compute_ml_objective(
    positions: list[tuple[float, float]],
    station_shapes: list[str],
    routes: list[list[int]],
    params: ParametricPolicyParams = ANCHOR_PARAMS,
    demand: LiveDemandSnapshot | None = None,
) -> float:
    """Compute demand-, wait-, and overload-aware H_theta(s, G).

    Reduces identically to `compute_global_objective` when live weights are zero.
    """
    if demand is None or not params.uses_live_demand():
        return compute_global_objective(
            positions,
            station_shapes,
            routes,
            alpha_dist=params.alpha_dist,
            beta_transfer=params.beta_transfer,
            gamma_load=params.gamma_load,
            balance_weight=params.balance_weight,
        )

    n_stations = len(positions)
    if n_stations == 0:
        return 0.0
    unique_shapes = set(station_shapes)
    station_lines: list[list[int]] = [[] for _ in range(n_stations)]
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}

    for line_idx, route in enumerate(routes):
        for s in route:
            station_lines[s].append(line_idx)
            adj.setdefault((s, line_idx), [])
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            dist_px = _distance(positions[u], positions[v])
            step_cost = 1.0 + params.alpha_dist * (dist_px / D_REF)
            adj[(u, line_idx)].append(((v, line_idx), step_cost))
            adj[(v, line_idx)].append(((u, line_idx), step_cost))

    for s in range(n_stations):
        lines_at_s = station_lines[s]
        if len(lines_at_s) > 1:
            for i in range(len(lines_at_s)):
                for j in range(i + 1, len(lines_at_s)):
                    l1, l2 = lines_at_s[i], lines_at_s[j]
                    adj[(s, l1)].append(((s, l2), params.beta_transfer))
                    adj[(s, l2)].append(((s, l1), params.beta_transfer))

    total_access = 0.0
    for target_shape in unique_shapes:
        dist_to_s = [UNREACH_PENALTY] * n_stations
        pq: list[tuple[float, int, int]] = []
        best_state: dict[tuple[int, int], float] = {}
        for u in range(n_stations):
            if station_shapes[u] == target_shape:
                dist_to_s[u] = 0.0
                for l_idx in station_lines[u]:
                    best_state[(u, l_idx)] = 0.0
                    heapq.heappush(pq, (0.0, u, l_idx))
        while pq:
            d, u, l_idx = heapq.heappop(pq)
            if d > best_state.get((u, l_idx), UNREACH_PENALTY):
                continue
            if d < dist_to_s[u]:
                dist_to_s[u] = d
            for (nxt_u, nxt_l), w in adj.get((u, l_idx), ()):
                nd = d + w
                if nd < best_state.get((nxt_u, nxt_l), UNREACH_PENALTY):
                    best_state[(nxt_u, nxt_l)] = nd
                    heapq.heappush(pq, (nd, nxt_u, nxt_l))

        rarity_t = demand.shape_rarity.get(target_shape, 0.0)
        base_t_mult = 1.0 + params.w_shape_rarity * rarity_t
        for s in range(n_stations):
            w_s = (
                base_t_mult
                + params.w_station_pax
                * (
                    demand.station_pax_norm[s]
                    + demand.od_pax_norm[s].get(target_shape, 0.0)
                )
                + params.w_station_wait
                * (
                    demand.station_wait_sq[s]
                    + demand.od_wait_sq[s].get(target_shape, 0.0)
                )
            )
            total_access += w_s * dist_to_s[s]

    load_penalty = 0.0
    balance_penalty = 0.0
    for line_idx, route in enumerate(routes):
        line_px = sum(
            _distance(positions[route[i]], positions[route[i + 1]])
            for i in range(len(route) - 1)
        )
        norm_len = line_px / D_REF
        norm_stops = len(route) / 4.0
        overload_mult = 1.0
        if params.w_line_overload != 0.0 and line_idx < len(demand.line_load):
            overload_mult += params.w_line_overload * max(
                0.0, demand.line_load[line_idx] - 1.0
            )
        load_penalty += params.gamma_load * norm_len
        balance_penalty += (
            params.balance_weight
            * overload_mult
            * (norm_len**2 + 2.0 * (norm_stops**2))
        )
    return total_access + load_penalty + balance_penalty


def demolition_risk_cost(
    line_idx: int,
    params: ParametricPolicyParams,
    demand: LiveDemandSnapshot | None,
    *,
    current_cost: float = 0.0,
    n_metros: int = 1,
    n_stations: int = 3,
) -> float:
    """Penalty scaling inversely with remaining slack before 40s overdue and fleet disruption."""
    if (
        params.demo_risk_weight <= 0.0
        or demand is None
        or line_idx >= len(demand.line_load)
    ):
        return 0.0
    wait_norm = min(0.98, demand.line_max_wait_sq[line_idx] ** 0.5)
    urgency_barrier = (1.0 / max(0.02, 1.0 - wait_norm) - 1.0) ** 2
    raw_risk = (
        0.25 * urgency_barrier
        + demand.line_max_wait_sq[line_idx]
        + 0.35 * demand.line_queue_norm[line_idx]
        + 0.25 * demand.line_onboard_ratio[line_idx]
    )
    fleet_mult = 1.0 + 0.35 * max(0, n_metros - 1) + 0.15 * max(0, n_stations - 3)
    scale_mult = 1.0 + 0.004 * max(0.0, current_cost)
    return params.demo_risk_weight * raw_risk * fleet_mult * scale_mult
