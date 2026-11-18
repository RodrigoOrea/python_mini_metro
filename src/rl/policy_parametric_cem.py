"""Parameterized state-dependent global planner (Phase 1 ML policy) over SemanticMetroEnv."""

from __future__ import annotations

import numpy as np

from rl.heuristic import _served
from rl.heuristic_global_replan import next_action_for_plan
from rl.heuristic_load_objective import _extract_graph_state
from rl.objective_ml import (
    ANCHOR_PARAMS,
    BEST_PHASE1_PARAMS,
    FLEET_ABLATION_PARAMS,
    PARAM_BOUNDS,
    PARAM_NAMES,
    RISK_FLEET_ABLATION_PARAMS,
    LiveDemandSnapshot,
    ParametricPolicyParams,
    compute_ml_objective,
    extract_live_demand,
)
from rl.replan_ml import (
    CRISIS_CHECK_INTERVAL,
    REBUILD_COOLDOWN,
    MLPlannerState,
    choose_unserved_or_rebuild_ml,
    maybe_start_global_rebuild_ml,
)
from rl.semantic_env import ACTION_TABLE, ActionKind

REBALANCE_COOLDOWN = 350
REBALANCE_PENDING_WINDOW = 400

__all__ = [
    "ANCHOR_PARAMS",
    "BEST_PHASE1_PARAMS",
    "FLEET_ABLATION_PARAMS",
    "PARAM_BOUNDS",
    "PARAM_NAMES",
    "RISK_FLEET_ABLATION_PARAMS",
    "LiveDemandSnapshot",
    "ParametricPolicyParams",
    "choose",
    "compute_ml_objective",
    "extract_live_demand",
    "play",
]


def _get_ml_state(env) -> MLPlannerState:
    mediator = env._mediator
    state: MLPlannerState | None = getattr(env, "_parametric_cem_state", None)
    if (
        state is None
        or state.mediator_id != id(mediator)
        or env._decision < state.last_decision
    ):
        state = MLPlannerState(mediator_id=id(mediator), last_decision=env._decision)
        env._parametric_cem_state = state
    else:
        state.last_decision = env._decision
    return state


def _line_fleet_raw(
    mediator, line_idx: int, params: ParametricPolicyParams, demand: LiveDemandSnapshot
) -> float:
    path = mediator.paths[line_idx]
    return params.w_fleet_wait * demand.line_max_wait_sq[
        line_idx
    ] + params.w_fleet_load * (
        demand.line_queue_norm[line_idx]
        + demand.line_onboard_ratio[line_idx]
        + len(path.stations) / 6.0
    )


def _line_eff_units(mediator, line_idx: int) -> float:
    path = mediator.paths[line_idx]
    metros = len(path.metros)
    carriages = sum(len(m.carriages) for m in path.metros)
    return max(0.5, metros + 0.55 * carriages)


def _line_fleet_pressure(
    mediator, line_idx: int, params: ParametricPolicyParams, demand: LiveDemandSnapshot
) -> float:
    return _line_fleet_raw(mediator, line_idx, params, demand) / _line_eff_units(
        mediator, line_idx
    )


def _maybe_rebalance_locomotives(
    env,
    state: MLPlannerState,
    mediator,
    params: ParametricPolicyParams,
    demand: LiveDemandSnapshot,
) -> None:
    """Safely shift 1 surplus carriage-free locomotive via minimax pressure reduction."""
    if state.pending_rebalance_target_id is not None:
        still_exists = any(
            p.id == state.pending_rebalance_target_id for p in mediator.paths
        )
        if env._decision > state.pending_rebalance_until or not still_exists:
            state.pending_rebalance_target_id = None
        else:
            return

    if (
        params.w_rebalance <= 0.0
        or state.active_plan is not None
        or mediator.available_locomotives > 0
        or len(mediator.paths) < 2
        or (env._decision - state.last_rebalance_decision) < REBALANCE_COOLDOWN
        or any(len(p.metros) == 0 for p in mediator.paths)
        or any(getattr(m, "is_unassignment_queued", False) for m in mediator.metros)
    ):
        return

    n_paths = len(mediator.paths)
    raws = [_line_fleet_raw(mediator, i, params, demand) for i in range(n_paths)]
    effs = [_line_eff_units(mediator, i) for i in range(n_paths)]
    pressures = [raws[i] / effs[i] for i in range(n_paths)]

    best_target = max(range(n_paths), key=lambda i: (pressures[i], -i))
    target_stressed = (
        demand.line_max_wait_sq[best_target] > 0.12
        or demand.line_onboard_ratio[best_target] > 0.50
        or demand.line_queue_norm[best_target] > 0.40
    )
    if not target_stressed:
        return
    extreme_crisis = demand.line_max_wait_sq[best_target] >= 0.45

    best_donor: int | None = None
    best_key: tuple[float, int, int] | None = None
    for i, path in enumerate(mediator.paths):
        if (
            i == best_target
            or len(path.metros) < 2
            or demand.line_max_wait_sq[i] >= 0.35
            or not mediator.can_queue_locomotive_unassignment(path)
        ):
            continue
        cand_metro = mediator._fleet.unassignment_candidate(mediator, path)
        if cand_metro is None:
            continue
        lost_car = len(cand_metro.carriages)
        if lost_car > 0 and not (extreme_crisis and len(path.metros) >= 3):
            continue

        donor_after = raws[i] / max(0.5, effs[i] - 1.0 - 0.55 * lost_car)
        target_after = raws[best_target] / (effs[best_target] + 1.0)
        minimax_drop = max(pressures[i], pressures[best_target]) - max(
            donor_after, target_after
        )
        if params.w_rebalance * minimax_drop <= 2.5:
            continue
        cand_key = (minimax_drop, len(path.metros), -i)
        if best_key is None or cand_key > best_key:
            best_key = cand_key
            best_donor = i

    if best_donor is not None:
        if mediator.queue_locomotive_unassignment(mediator.paths[best_donor]):
            state.last_rebalance_decision = env._decision
            state.pending_rebalance_target_id = mediator.paths[best_target].id
            state.pending_rebalance_until = env._decision + REBALANCE_PENDING_WINDOW


def choose(env, params: ParametricPolicyParams | None = None) -> int:
    """Pick the action optimizing live `H_theta(s, G)` and smart fleet control, or WAIT."""
    cfg = BEST_PHASE1_PARAMS if params is None else params
    mediator = env._mediator
    state = _get_ml_state(env)

    if state.active_plan is not None:
        mask = env.action_masks()
        action = next_action_for_plan(mediator, mask, state.active_plan)
        if action is not None:
            return action
        if state.active_plan.is_complete(mediator):
            state.last_rebuild_decision = env._decision
        state.active_plan = None

    served = _served(mediator)
    purchasable = mediator.get_next_path_button_idx_to_purchase()
    can_buy = purchasable is not None and mediator.can_purchase_path_button_idx(
        purchasable
    )
    positions, shapes, routes, _ = _extract_graph_state(mediator)
    unserved = [i for i in range(len(positions)) if i not in served]
    topo_sig = (len(positions), len(routes), tuple(tuple(r) for r in routes))

    periodic_due = (cfg.uses_dynamic_fleet() or cfg.uses_live_demand()) and (
        env._decision - state.last_crisis_check_decision
    ) >= CRISIS_CHECK_INTERVAL
    if (
        not can_buy
        and not unserved
        and mediator.available_locomotives == 0
        and mediator.available_carriages == 0
        and all(len(p.metros) > 0 for p in mediator.paths)
        and not periodic_due
        and (
            topo_sig == state.last_replan_topology
            or env._decision - state.last_rebuild_decision < REBUILD_COOLDOWN
        )
    ):
        return 0

    demand: LiveDemandSnapshot | None = None
    if cfg.uses_live_demand() or cfg.uses_dynamic_fleet() or cfg.demo_risk_weight > 0.0:
        demand = extract_live_demand(mediator)

    if periodic_due and demand is not None:
        _maybe_rebalance_locomotives(env, state, mediator, cfg, demand)
        if (
            topo_sig == state.last_replan_topology
            and demand.max_station_wait_norm < 0.625
            and mediator.available_locomotives == 0
            and mediator.available_carriages == 0
            and not can_buy
            and not unserved
            and all(len(p.metros) > 0 for p in mediator.paths)
        ):
            state.last_crisis_check_decision = env._decision
            return 0

    mask = env.action_masks()
    legal: dict[ActionKind, list[tuple[int, int, int]]] = {}
    for index in np.flatnonzero(mask):
        kind, first, second = ACTION_TABLE[index]
        legal.setdefault(kind, []).append((int(index), int(first), int(second)))

    if ActionKind.PURCHASE_LINE in legal:
        return legal[ActionKind.PURCHASE_LINE][0][0]

    uncrewed = [i for i, p in enumerate(mediator.paths) if len(p.metros) == 0]
    if uncrewed:
        if ActionKind.ASSIGN_LOCOMOTIVE in legal:
            for index, line_idx, _ in legal[ActionKind.ASSIGN_LOCOMOTIVE]:
                if line_idx in uncrewed:
                    return index
        elif not any(
            getattr(m, "is_unassignment_queued", False) for m in mediator.metros
        ):
            candidates = [
                p
                for p in mediator.paths
                if len(p.metros) >= 2 and mediator.can_queue_locomotive_unassignment(p)
            ]
            if candidates:
                donor = max(candidates, key=lambda p: len(p.metros))
                mediator.queue_locomotive_unassignment(donor)

    if (
        ActionKind.CONNECT in legal
        and len(mediator.paths) < mediator.get_unlocked_num_paths()
        and unserved
    ):
        best_cost: float | None = None
        best_idx: int | None = None
        for index, first, second in legal[ActionKind.CONNECT]:
            if first in unserved or second in unserved:
                cost = compute_ml_objective(
                    positions, shapes, routes + [[first, second]], cfg, demand
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = index
        if best_idx is not None:
            return best_idx

    if unserved and mediator.paths:
        chosen = choose_unserved_or_rebuild_ml(
            env,
            state,
            mediator,
            mask,
            legal,
            positions,
            shapes,
            routes,
            unserved,
            cfg,
            demand,
        )
        if chosen is not None:
            return chosen

    if len(mediator.paths) == 0 and ActionKind.CONNECT in legal:
        best_cost = None
        best_idx = None
        for index, first, second in legal[ActionKind.CONNECT]:
            if first in unserved and second in unserved:
                cost = compute_ml_objective(
                    positions, shapes, [[first, second]], cfg, demand
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = index
        if best_idx is not None:
            return best_idx

    if not unserved and mediator.paths:
        replan_action = maybe_start_global_rebuild_ml(
            env, state, mediator, mask, positions, shapes, routes, cfg, demand
        )
        if replan_action is not None:
            return replan_action

    if ActionKind.ASSIGN_LOCOMOTIVE in legal:
        has_uncrewed = any(
            line_idx < len(mediator.paths) and len(mediator.paths[line_idx].metros) == 0
            for _, line_idx, _ in legal[ActionKind.ASSIGN_LOCOMOTIVE]
        )
        if state.pending_rebalance_target_id is not None:
            if env._decision > state.pending_rebalance_until:
                state.pending_rebalance_target_id = None
            elif not has_uncrewed:
                for index, line_idx, _ in legal[ActionKind.ASSIGN_LOCOMOTIVE]:
                    if (
                        line_idx < len(mediator.paths)
                        and mediator.paths[line_idx].id
                        == state.pending_rebalance_target_id
                    ):
                        state.pending_rebalance_target_id = None
                        return index
        valid = []
        for index, line_idx, _ in legal[ActionKind.ASSIGN_LOCOMOTIVE]:
            if line_idx >= len(mediator.paths):
                continue
            n_metros = len(mediator.paths[line_idx].metros)
            if demand is not None and cfg.uses_dynamic_fleet():
                score = (
                    _line_fleet_pressure(mediator, line_idx, cfg, demand)
                    - 0.35 * n_metros
                )
                key = (0 if n_metros == 0 else 1, -score, index)
            else:
                key = (n_metros, -len(mediator.paths[line_idx].stations), index)
            valid.append((key, index))
        if valid:
            valid.sort(key=lambda item: item[0])
            return valid[0][1]

    if ActionKind.ATTACH_CARRIAGE in legal:
        if demand is not None and cfg.uses_dynamic_fleet():
            candidates_car = [
                (
                    _line_fleet_pressure(mediator, line_idx, cfg, demand),
                    len(mediator.paths[line_idx].stations),
                    -line_idx,
                    index,
                )
                for index, line_idx, _ in legal[ActionKind.ATTACH_CARRIAGE]
                if line_idx < len(mediator.paths)
            ]
            if candidates_car:
                candidates_car.sort(reverse=True)
                return candidates_car[0][3]
        return legal[ActionKind.ATTACH_CARRIAGE][0][0]

    return 0


def play(
    seed: int,
    params: ParametricPolicyParams | None = None,
    max_decisions: int = 200_000,
) -> dict:
    """Run one full episode, returning deliveries, decisions survived, and actions."""
    from rl.semantic_env import SemanticMetroEnv

    env = SemanticMetroEnv(max_decisions=max_decisions)
    env.reset(seed=seed)
    delivered = 0.0
    decisions = 0
    acted = 0
    try:
        while decisions < max_decisions:
            action = choose(env, params)
            if action != 0:
                acted += 1
            _, reward, terminated, truncated, _ = env.step(action)
            delivered += float(reward)
            decisions += 1
            if terminated or truncated:
                break
        mediator = env._mediator
        return {
            "seed": seed,
            "deliveries": int(delivered),
            "decisions": decisions,
            "actions_taken": acted,
            "lines": len(mediator.paths),
            "stations": len(mediator.stations),
        }
    finally:
        env.close()
