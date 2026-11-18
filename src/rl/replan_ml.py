"""Combinatorial route search and risk-gated rebuild selection for `policy_parametric_cem`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.heuristic import _distance
from rl.heuristic_global_replan import (
    RebuildPlan,
    _clears_threshold,
    _eligible_lines_balanced,
    _lookup_action,
    _routes_equivalent,
    next_action_for_plan,
)
from rl.objective_ml import (
    LiveDemandSnapshot,
    ParametricPolicyParams,
    compute_ml_objective,
    demolition_risk_cost,
)
from rl.semantic_env import ActionKind

REBUILD_COOLDOWN = 500
CRISIS_CHECK_INTERVAL = 25


@dataclass
class MLPlannerState:
    """Per-episode mutable planner hysteresis and active rebuild state."""

    mediator_id: int
    last_decision: int = -1
    active_plan: RebuildPlan | None = None
    last_rebuild_decision: int = -10_000
    last_replan_topology: tuple[int, int, tuple[tuple[int, ...], ...]] | None = None
    last_rebalance_decision: int = -10_000
    last_crisis_check_decision: int = -10_000
    pending_rebalance_target_id: str | None = None
    pending_rebalance_until: int = -10_000


def optimize_line_order_ml(
    positions: list[tuple[float, float]],
    station_shapes: list[str],
    routes: list[list[int]],
    line_idx: int,
    params: ParametricPolicyParams,
    demand: LiveDemandSnapshot | None,
    *,
    max_passes: int = 4,
) -> list[int]:
    """Improve intra-line stop ordering under `compute_ml_objective`."""
    cur_routes = [list(r) for r in routes]
    route = cur_routes[line_idx]
    m = len(route)
    if m < 3:
        return route
    best_cost = compute_ml_objective(
        positions, station_shapes, cur_routes, params, demand
    )

    for _ in range(max_passes):
        best_cand: list[int] | None = None
        for i in range(m - 1):
            for j in range(i + 2, m + 1):
                if i == 0 and j == m:
                    continue
                cand = route[:i] + list(reversed(route[i:j])) + route[j:]
                cur_routes[line_idx] = cand
                cost = compute_ml_objective(
                    positions, station_shapes, cur_routes, params, demand
                )
                if cost + 1e-9 < best_cost:
                    best_cost = cost
                    best_cand = cand
        for i in range(m):
            station = route[i]
            trimmed = route[:i] + route[i + 1 :]
            for j in range(m):
                if j == i:
                    continue
                cand = trimmed[:j] + [station] + trimmed[j:]
                cur_routes[line_idx] = cand
                cost = compute_ml_objective(
                    positions, station_shapes, cur_routes, params, demand
                )
                if cost + 1e-9 < best_cost:
                    best_cost = cost
                    best_cand = cand
        cur_routes[line_idx] = route
        if best_cand is None:
            break
        route = best_cand
        cur_routes[line_idx] = route
    return route


def optimize_network_ml(
    positions: list[tuple[float, float]],
    station_shapes: list[str],
    routes: list[list[int]],
    params: ParametricPolicyParams,
    demand: LiveDemandSnapshot | None,
    *,
    max_outer: int = 3,
) -> list[list[int]]:
    """Optimize multi-line network topology via transfers, swaps, and optional hubs."""
    if not routes:
        return []
    cur = [list(r) for r in routes]
    for line_idx in range(len(cur)):
        cur[line_idx] = optimize_line_order_ml(
            positions, station_shapes, cur, line_idx, params, demand
        )
    if len(cur) < 2:
        return cur

    best_cost = compute_ml_objective(positions, station_shapes, cur, params, demand)
    for _ in range(max_outer):
        best_move: list[list[int]] | None = None
        best_move_cost = best_cost

        for a in range(len(cur)):
            if len(cur[a]) < 3:
                continue
            for idx_a, station in enumerate(cur[a]):
                donor_trimmed = cur[a][:idx_a] + cur[a][idx_a + 1 :]
                for b in range(len(cur)):
                    if b == a or station in cur[b]:
                        continue
                    for ins_pos in range(len(cur[b]) + 1):
                        cand = [list(r) for r in cur]
                        cand[a] = donor_trimmed
                        cand[b] = cur[b][:ins_pos] + [station] + cur[b][ins_pos:]
                        cost = compute_ml_objective(
                            positions, station_shapes, cand, params, demand
                        )
                        if cost + 1e-9 < best_move_cost:
                            best_move_cost = cost
                            best_move = cand

        for a in range(len(cur)):
            for b in range(a + 1, len(cur)):
                set_a, set_b = set(cur[a]), set(cur[b])
                for idx_a, s_a in enumerate(cur[a]):
                    if s_a in set_b:
                        continue
                    for idx_b, s_b in enumerate(cur[b]):
                        if s_b in set_a:
                            continue
                        cand = [list(r) for r in cur]
                        cand[a][idx_a] = s_b
                        cand[b][idx_b] = s_a
                        cost = compute_ml_objective(
                            positions, station_shapes, cand, params, demand
                        )
                        if cost + 1e-9 < best_move_cost:
                            best_move_cost = cost
                            best_move = cand

        if params.hub_share_bonus > 0.0:
            cov: dict[int, int] = {}
            for r in cur:
                for st_i in r:
                    cov[st_i] = cov.get(st_i, 0) + 1
            for s in range(len(positions)):
                if cov.get(s, 0) != 1:
                    continue
                for b in range(len(cur)):
                    if s in cur[b] or len(cur[b]) >= 6:
                        continue
                    route_b = cur[b]
                    ins_detours = []
                    for pos in range(len(route_b) + 1):
                        if pos == 0:
                            det = _distance(positions[s], positions[route_b[0]])
                        elif pos == len(route_b):
                            det = _distance(positions[route_b[-1]], positions[s])
                        else:
                            det = (
                                _distance(positions[route_b[pos - 1]], positions[s])
                                + _distance(positions[s], positions[route_b[pos]])
                                - _distance(
                                    positions[route_b[pos - 1]], positions[route_b[pos]]
                                )
                            )
                        ins_detours.append((det, pos))
                    ins_detours.sort()
                    for _, pos in ins_detours[:2]:
                        cand = [list(r) for r in cur]
                        cand[b] = route_b[:pos] + [s] + route_b[pos:]
                        cost = (
                            compute_ml_objective(
                                positions, station_shapes, cand, params, demand
                            )
                            - params.hub_share_bonus
                        )
                        if cost + 1e-9 < best_move_cost:
                            best_move_cost = cost
                            best_move = cand

        if best_move is None:
            break
        for line_idx in range(len(best_move)):
            best_move[line_idx] = optimize_line_order_ml(
                positions,
                station_shapes,
                best_move,
                line_idx,
                params,
                demand,
                max_passes=2,
            )
        polished_cost = compute_ml_objective(
            positions, station_shapes, best_move, params, demand
        )
        if polished_cost + 1e-9 < best_cost:
            cur = best_move
            best_cost = polished_cost
        else:
            break
    return cur


def choose_unserved_or_rebuild_ml(
    env,
    state: MLPlannerState,
    mediator,
    mask: np.ndarray,
    legal: dict[ActionKind, list[tuple[int, int, int]]],
    positions: list[tuple[float, float]],
    shapes: list[str],
    routes: list[list[int]],
    unserved: list[int],
    params: ParametricPolicyParams,
    demand: LiveDemandSnapshot | None,
) -> int | None:
    """Compare zero-downtime extension vs risk-weighted interior rebuild."""
    eligible_lines = set(_eligible_lines_balanced(mediator))
    best_inc_cost: float | None = None
    best_inc_idx: int | None = None
    for kind in (ActionKind.EXTEND_LINE, ActionKind.PREPEND_LINE):
        for index, line, station in legal.get(kind, ()):
            if (
                station not in unserved
                or line >= len(routes)
                or line not in eligible_lines
            ):
                continue
            cand_routes = [list(r) for r in routes]
            if kind is ActionKind.EXTEND_LINE:
                cand_routes[line].append(station)
            else:
                cand_routes[line].insert(0, station)
            cost = compute_ml_objective(positions, shapes, cand_routes, params, demand)
            if best_inc_cost is None or cost < best_inc_cost:
                best_inc_cost = cost
                best_inc_idx = index

    all_crewed = all(len(p.metros) > 0 for p in mediator.paths)
    if (
        all_crewed
        and best_inc_cost is not None
        and env._decision - state.last_rebuild_decision >= REBUILD_COOLDOWN
    ):
        best_reb_cost = best_inc_cost
        best_reb_line: int | None = None
        best_reb_route: list[int] | None = None

        for line in range(len(routes)):
            if line not in eligible_lines:
                continue
            if _lookup_action(mask, ActionKind.REMOVE_LINE, line, 0) is None:
                continue
            risk_cost = demolition_risk_cost(
                line,
                params,
                demand,
                current_cost=best_inc_cost,
                n_metros=len(mediator.paths[line].metros),
                n_stations=len(routes[line]),
            )
            base_route = routes[line]
            for station in unserved:
                if station in base_route:
                    continue
                for pos in range(1, len(base_route)):
                    cand_routes = [list(r) for r in routes]
                    cand_routes[line] = base_route[:pos] + [station] + base_route[pos:]
                    cand_routes[line] = optimize_line_order_ml(
                        positions,
                        shapes,
                        cand_routes,
                        line,
                        params,
                        demand,
                        max_passes=2,
                    )
                    cost = (
                        compute_ml_objective(
                            positions, shapes, cand_routes, params, demand
                        )
                        + risk_cost
                    )
                    if cost < best_reb_cost:
                        best_reb_cost = cost
                        best_reb_line = line
                        best_reb_route = cand_routes[line]

        if (
            best_reb_line is not None
            and best_reb_route is not None
            and _clears_threshold(
                best_inc_cost, best_reb_cost, 15.0, params.single_min_rel
            )
        ):
            plan = RebuildPlan.create(mediator.paths[best_reb_line].id, best_reb_route)
            action = next_action_for_plan(mediator, mask, plan)
            if action is not None:
                state.active_plan = plan
                return action
    return best_inc_idx


def _eval_rebuild_candidate_cost(
    positions: list[tuple[float, float]],
    shapes: list[str],
    routes: list[list[int]],
    target_route: list[int],
    line: int,
    params: ParametricPolicyParams,
    demand: LiveDemandSnapshot | None,
    *,
    current_cost: float,
    n_metros: int,
) -> float:
    cand = [list(r) for r in routes]
    cand[line] = target_route
    return compute_ml_objective(
        positions, shapes, cand, params, demand
    ) + demolition_risk_cost(
        line,
        params,
        demand,
        current_cost=current_cost,
        n_metros=n_metros,
        n_stations=len(routes[line]),
    )


def maybe_start_global_rebuild_ml(
    env,
    state: MLPlannerState,
    mediator,
    mask: np.ndarray,
    positions: list[tuple[float, float]],
    shapes: list[str],
    routes: list[list[int]],
    params: ParametricPolicyParams,
    demand: LiveDemandSnapshot | None,
) -> int | None:
    """Evaluate full-network replan when served & crewed, gated by demolition risk."""
    if (
        not routes
        or any(len(p.metros) == 0 for p in mediator.paths)
        or env._decision - state.last_rebuild_decision < REBUILD_COOLDOWN
    ):
        return None
    crisis_wake = (
        params.uses_live_demand()
        and demand is not None
        and demand.max_station_wait_norm >= 0.625
        and env._decision - state.last_crisis_check_decision >= CRISIS_CHECK_INTERVAL
    )
    sig = (len(positions), len(routes), tuple(tuple(r) for r in routes))
    if sig == state.last_replan_topology and not crisis_wake:
        return None
    state.last_replan_topology = sig
    if crisis_wake:
        state.last_crisis_check_decision = env._decision

    current_cost = compute_ml_objective(positions, shapes, routes, params, demand)
    target_routes = optimize_network_ml(positions, shapes, routes, params, demand)
    joint_cost = compute_ml_objective(positions, shapes, target_routes, params, demand)
    if not _clears_threshold(current_cost, joint_cost, 15.0, params.single_min_rel):
        return None

    changed_lines = [
        i
        for i in range(len(routes))
        if not _routes_equivalent(routes[i], target_routes[i])
    ]
    if not changed_lines:
        return None

    if params.hub_share_bonus > 0.0:
        for line in changed_lines:
            cur_r, tgt_r = routes[line], target_routes[line]
            if len(tgt_r) == len(cur_r) + 1:
                if tgt_r[:-1] == cur_r:
                    direct = _lookup_action(
                        mask, ActionKind.EXTEND_LINE, line, tgt_r[-1]
                    )
                    if direct is not None:
                        return direct
                if tgt_r[1:] == cur_r:
                    direct = _lookup_action(
                        mask, ActionKind.PREPEND_LINE, line, tgt_r[0]
                    )
                    if direct is not None:
                        return direct

    covered_counts: dict[int, int] = {}
    for r in routes:
        for s in r:
            covered_counts[s] = covered_counts.get(s, 0) + 1

    best_line: int | None = None
    best_cost = current_cost
    for line in changed_lines:
        if _lookup_action(mask, ActionKind.REMOVE_LINE, line, 0) is None:
            continue
        dropped = set(routes[line]) - set(target_routes[line])
        if any(covered_counts.get(s, 0) <= 1 for s in dropped):
            continue
        single_cost = _eval_rebuild_candidate_cost(
            positions,
            shapes,
            routes,
            target_routes[line],
            line,
            params,
            demand,
            current_cost=current_cost,
            n_metros=len(mediator.paths[line].metros),
        )
        if single_cost < best_cost and _clears_threshold(
            current_cost, single_cost, 15.0, params.single_min_rel
        ):
            best_cost = single_cost
            best_line = line

    if best_line is None and _clears_threshold(
        current_cost, joint_cost, 22.0, params.multi_min_rel
    ):
        best_bridge_cost: float | None = None
        for line in changed_lines:
            if _lookup_action(mask, ActionKind.REMOVE_LINE, line, 0) is None:
                continue
            dropped = set(routes[line]) - set(target_routes[line])
            added = set(target_routes[line]) - set(routes[line])
            if added and not dropped:
                bridge_cost = _eval_rebuild_candidate_cost(
                    positions,
                    shapes,
                    routes,
                    target_routes[line],
                    line,
                    params,
                    demand,
                    current_cost=current_cost,
                    n_metros=len(mediator.paths[line].metros),
                )
                if bridge_cost <= current_cost + (
                    current_cost - joint_cost
                ) * 0.25 and (
                    best_bridge_cost is None or bridge_cost < best_bridge_cost
                ):
                    best_bridge_cost = bridge_cost
                    best_line = line

    if best_line is None:
        return None
    plan = RebuildPlan.create(mediator.paths[best_line].id, target_routes[best_line])
    action = next_action_for_plan(mediator, mask, plan)
    if action is not None:
        state.active_plan = plan
        return action
    return None
