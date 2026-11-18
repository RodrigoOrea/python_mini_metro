"""Global replanning combinatorial optimization heuristic (global_replan) by Rodrigo Orea.

Evolution:
1. Manual rules (`heuristic_multiline.py`): greedy endpoint attachment & fleet balancing.
2. Objective-guided incremental (`heuristic_load_objective.py`): minimizes H(G) over
   endpoint-only actions (`CONNECT` / `EXTEND_LINE` / `PREPEND_LINE`).
3. Global replanning (`heuristic_global_replan.py`): solves the Transit Network Design
   Problem (TNDP) over the full station set via intra-line 2-opt/relocate and
   inter-line station transfers/swaps minimizing a globally balanced objective
   H_global(G) = H(G) + BalancePenalty(G), executing beneficial line redesigns
   safely line-by-line via discrete `SemanticMetroEnv` actions when savings clear
   a hysteresis threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.heuristic import _distance, _served
from rl.heuristic_load_objective import _extract_graph_state, compute_network_cost
from rl.semantic_env import ACTION_TABLE, ActionKind

_DEFAULT_REBUILD_COOLDOWN = 500
_DEFAULT_SINGLE_MIN_ABS_SAVING = 15.0
_DEFAULT_SINGLE_MIN_REL_SAVING = 0.12
_DEFAULT_MULTI_MIN_ABS_SAVING = 22.0
_DEFAULT_MULTI_MIN_REL_SAVING = 0.15


def compute_global_objective(
    positions: list[tuple[float, float]],
    station_shapes: list[str],
    routes: list[list[int]],
    *,
    alpha_dist: float = 1.5,
    d_ref: float = 250.0,
    beta_transfer: float = 1.5,
    gamma_load: float = 12.0,
    unreach_penalty: float = 30.0,
    balance_weight: float = 1.8,
) -> float:
    """Compute H(G) plus a convex per-line lap-time & load balance regularizer.

    A linear route-length penalty treats a lopsided `[5-stop, 2-stop]` network as
    identical to a balanced `[4-stop, 3-stop]` network of the same total length.
    Penalizing the square of normalized route length/stops directly favors
    balanced line lap times without relying on brittle creation-order indices.
    """
    base = compute_network_cost(
        positions,
        station_shapes,
        routes,
        alpha_dist=alpha_dist,
        d_ref=d_ref,
        beta_transfer=beta_transfer,
        gamma_load=gamma_load,
        unreach_penalty=unreach_penalty,
    )
    balance_penalty = 0.0
    for route in routes:
        line_px = sum(
            _distance(positions[route[i]], positions[route[i + 1]])
            for i in range(len(route) - 1)
        )
        norm_len = line_px / d_ref
        norm_stops = len(route) / 4.0
        balance_penalty += balance_weight * (norm_len**2 + 2.0 * (norm_stops**2))
    return base + balance_penalty


def _normalize_route_for_connect(route: list[int]) -> tuple[list[int], int]:
    """Return `(normalized_route, pivot)` where `normalized_route[pivot] < [pivot + 1]`.

    `SemanticMetroEnv`'s `CONNECT(first, second)` entries always enforce `first < second`.
    Since an unlooped metro line serves `[s_0, ..., s_{m-1}]` identically to its
    reverse, reversing a strictly decreasing route guarantees a valid adjacent
    `CONNECT` seed pair while preserving the exact physical path.
    """
    if len(route) < 2:
        raise ValueError("a metro line must have at least 2 stations")
    if len(set(route)) != len(route):
        raise ValueError("route stations must be distinct")

    for pivot in range(len(route) - 1):
        if route[pivot] < route[pivot + 1]:
            return list(route), pivot

    reversed_route = list(reversed(route))
    for pivot in range(len(reversed_route) - 1):
        if reversed_route[pivot] < reversed_route[pivot + 1]:
            return reversed_route, pivot

    raise ValueError("could not find ascending adjacent pair in route")


@dataclass
class RebuildPlan:
    """Tracks staged demolition and reconstruction of a single metro line."""

    old_path_id: str | None
    target_route: list[int]
    pivot: int
    phase: str = "REMOVE"
    built_path_id: str | None = None
    known_path_ids: frozenset[str] = frozenset()

    @classmethod
    def create(cls, old_path_id: str | None, target_route: list[int]) -> RebuildPlan:
        normalized, pivot = _normalize_route_for_connect(target_route)
        phase = "REMOVE" if old_path_id is not None else "CONNECT"
        return cls(
            old_path_id=old_path_id,
            target_route=normalized,
            pivot=pivot,
            phase=phase,
        )

    def is_complete(self, mediator) -> bool:
        return self.phase == "DONE"


@dataclass
class _PlannerState:
    mediator_id: int
    last_decision: int = -1
    active_plan: RebuildPlan | None = None
    last_rebuild_decision: int = -10_000
    last_replan_topology: tuple[int, int, tuple[tuple[int, ...], ...]] | None = None


def _get_planner_state(env) -> _PlannerState:
    mediator = env._mediator
    state: _PlannerState | None = getattr(env, "_global_replan_state", None)
    if (
        state is None
        or state.mediator_id != id(mediator)
        or env._decision < state.last_decision
    ):
        state = _PlannerState(mediator_id=id(mediator), last_decision=env._decision)
        env._global_replan_state = state
    else:
        state.last_decision = env._decision
    return state


def _lookup_action(
    mask: np.ndarray, kind: ActionKind, first: int, second: int = 0
) -> int | None:
    try:
        idx = ACTION_TABLE.index((kind, first, second))
    except ValueError:
        return None
    return idx if bool(mask[idx]) else None


def next_action_for_plan(mediator, mask: np.ndarray, plan: RebuildPlan) -> int | None:
    """Return the next legal action index advancing `plan`, or None when done/blocked."""
    if plan.phase == "DONE":
        return None

    if plan.phase == "REMOVE":
        old_idx = next(
            (i for i, p in enumerate(mediator.paths) if p.id == plan.old_path_id),
            None,
        )
        if old_idx is None:
            plan.phase = "CONNECT"
        else:
            action = _lookup_action(mask, ActionKind.REMOVE_LINE, old_idx, 0)
            if action is None:
                return None
            plan.known_path_ids = frozenset(
                p.id for p in mediator.paths if p.id != plan.old_path_id
            )
            plan.phase = "CONNECT"
            return action

    if plan.phase == "CONNECT":
        if plan.old_path_id is not None and any(
            p.id == plan.old_path_id for p in mediator.paths
        ):
            return None
        plan.known_path_ids = frozenset(p.id for p in mediator.paths)
        first = plan.target_route[plan.pivot]
        second = plan.target_route[plan.pivot + 1]
        action = _lookup_action(mask, ActionKind.CONNECT, first, second)
        if action is None:
            return None
        plan.phase = "GROW"
        return action

    idx_by_id = {s.id: i for i, s in enumerate(mediator.stations)}
    if plan.phase == "GROW":
        if plan.built_path_id is None:
            new_paths = [p for p in mediator.paths if p.id not in plan.known_path_ids]
            if not new_paths:
                return None
            plan.built_path_id = new_paths[-1].id

        built = next(
            (
                (i, p)
                for i, p in enumerate(mediator.paths)
                if p.id == plan.built_path_id
            ),
            None,
        )
        if built is None:
            return None
        line_idx, path = built
        cur = [idx_by_id[s.id] for s in path.stations if s.id in idx_by_id]

        sub_len = len(cur)
        start_pos = None
        for pos in range(len(plan.target_route) - sub_len + 1):
            if plan.target_route[pos : pos + sub_len] == cur:
                start_pos = pos
                break
        if start_pos is None:
            return None

        end_pos = start_pos + sub_len - 1
        if end_pos + 1 < len(plan.target_route):
            nxt = plan.target_route[end_pos + 1]
            action = _lookup_action(mask, ActionKind.EXTEND_LINE, line_idx, nxt)
            if action is None:
                return None
            return action
        if start_pos - 1 >= 0:
            prv = plan.target_route[start_pos - 1]
            action = _lookup_action(mask, ActionKind.PREPEND_LINE, line_idx, prv)
            if action is None:
                return None
            return action

        plan.phase = "CREW"

    if plan.phase == "CREW":
        built = next(
            (
                (i, p)
                for i, p in enumerate(mediator.paths)
                if p.id == plan.built_path_id
            ),
            None,
        )
        if built is None:
            plan.phase = "DONE"
            return None
        line_idx, path = built
        if len(path.metros) == 0:
            action = _lookup_action(mask, ActionKind.ASSIGN_LOCOMOTIVE, line_idx, 0)
            if action is not None:
                plan.phase = "DONE"
                return action
        plan.phase = "DONE"
        return None

    return None


def optimize_line_order(
    positions: list[tuple[float, float]],
    station_shapes: list[str],
    routes: list[list[int]],
    line_idx: int,
    *,
    max_passes: int = 4,
) -> list[int]:
    """Improve station order on `routes[line_idx]` via 2-opt and relocate."""
    cur_routes = [list(r) for r in routes]
    route = cur_routes[line_idx]
    m = len(route)
    if m < 3:
        return route

    best_cost = compute_global_objective(positions, station_shapes, cur_routes)

    for _ in range(max_passes):
        best_cand: list[int] | None = None

        # 2-opt segment reversals (excluding full-route reversal i=0, j=m).
        for i in range(m - 1):
            for j in range(i + 2, m + 1):
                if i == 0 and j == m:
                    continue
                cand = route[:i] + list(reversed(route[i:j])) + route[j:]
                cur_routes[line_idx] = cand
                cost = compute_global_objective(positions, station_shapes, cur_routes)
                if cost + 1e-9 < best_cost:
                    best_cost = cost
                    best_cand = cand

        # Intra-line node relocate.
        for i in range(m):
            station = route[i]
            trimmed = route[:i] + route[i + 1 :]
            for j in range(m):
                if j == i:
                    continue
                cand = trimmed[:j] + [station] + trimmed[j:]
                cur_routes[line_idx] = cand
                cost = compute_global_objective(positions, station_shapes, cur_routes)
                if cost + 1e-9 < best_cost:
                    best_cost = cost
                    best_cand = cand

        cur_routes[line_idx] = route
        if best_cand is None:
            break
        route = best_cand
        cur_routes[line_idx] = route

    return route


def optimize_network(
    positions: list[tuple[float, float]],
    station_shapes: list[str],
    routes: list[list[int]],
    *,
    max_outer: int = 3,
) -> list[list[int]]:
    """Globally optimize multi-line routes via intra-line 2-opt and inter-line transfers/swaps."""
    if not routes:
        return []

    cur = [list(r) for r in routes]
    for line_idx in range(len(cur)):
        cur[line_idx] = optimize_line_order(positions, station_shapes, cur, line_idx)

    if len(cur) < 2:
        return cur

    best_cost = compute_global_objective(positions, station_shapes, cur)

    for _ in range(max_outer):
        best_move: list[list[int]] | None = None
        best_move_cost = best_cost

        # 1. Inter-line single-station transfer (preserve min length 2 on donor).
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
                        cost = compute_global_objective(positions, station_shapes, cand)
                        if cost + 1e-9 < best_move_cost:
                            best_move_cost = cost
                            best_move = cand

        # 2. Inter-line single-station swap.
        for a in range(len(cur)):
            for b in range(a + 1, len(cur)):
                set_a = set(cur[a])
                set_b = set(cur[b])
                for idx_a, s_a in enumerate(cur[a]):
                    if s_a in set_b:
                        continue
                    for idx_b, s_b in enumerate(cur[b]):
                        if s_b in set_a:
                            continue
                        cand = [list(r) for r in cur]
                        cand[a][idx_a] = s_b
                        cand[b][idx_b] = s_a
                        cost = compute_global_objective(positions, station_shapes, cand)
                        if cost + 1e-9 < best_move_cost:
                            best_move_cost = cost
                            best_move = cand

        if best_move is None:
            break

        for line_idx in range(len(best_move)):
            best_move[line_idx] = optimize_line_order(
                positions, station_shapes, best_move, line_idx, max_passes=2
            )
        polished_cost = compute_global_objective(positions, station_shapes, best_move)
        if polished_cost + 1e-9 < best_cost:
            cur = best_move
            best_cost = polished_cost
        else:
            break

    return cur


def _routes_equivalent(a: list[int], b: list[int]) -> bool:
    return a == b or a == list(reversed(b))


def _clears_threshold(
    cost_before: float, cost_after: float, min_abs: float, min_rel: float
) -> bool:
    saving = cost_before - cost_after
    return saving >= min_abs and saving / max(cost_before, 1.0) >= min_rel


def _eligible_lines_balanced(mediator) -> list[int]:
    """Return eligible lines, protecting 2-station newborn/stub lines from starvation."""
    paths = mediator.paths
    if not paths:
        return []
    lengths = [len(p.stations) for p in paths]
    if max(lengths) >= 4 and min(lengths) <= 2:
        return [i for i, length in enumerate(lengths) if length <= 2]
    return list(range(len(paths)))


def _choose_unserved_action_or_rebuild(
    env,
    state: _PlannerState,
    mediator,
    mask: np.ndarray,
    legal: dict[ActionKind, list[tuple[int, int, int]]],
    positions: list[tuple[float, float]],
    shapes: list[str],
    routes: list[list[int]],
    unserved: list[int],
    *,
    min_abs_saving: float,
    min_rel_saving: float,
    rebuild_cooldown: int,
) -> int | None:
    """Compare zero-downtime endpoint extension against interior insertion + 2-opt rebuild."""
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
            cost = compute_global_objective(positions, shapes, cand_routes)
            if best_inc_cost is None or cost < best_inc_cost:
                best_inc_cost = cost
                best_inc_idx = index

    # Only compare against line rebuild when every active line is crewed.
    all_crewed = all(len(p.metros) > 0 for p in mediator.paths)
    if (
        all_crewed
        and best_inc_cost is not None
        and env._decision - state.last_rebuild_decision >= rebuild_cooldown
    ):
        best_reb_cost = best_inc_cost
        best_reb_line: int | None = None
        best_reb_route: list[int] | None = None

        for line in range(len(routes)):
            if line not in eligible_lines:
                continue
            if _lookup_action(mask, ActionKind.REMOVE_LINE, line, 0) is None:
                continue
            base_route = routes[line]
            for station in unserved:
                if station in base_route:
                    continue
                for pos in range(1, len(base_route)):
                    cand_routes = [list(r) for r in routes]
                    cand_routes[line] = base_route[:pos] + [station] + base_route[pos:]
                    cand_routes[line] = optimize_line_order(
                        positions, shapes, cand_routes, line, max_passes=2
                    )
                    cost = compute_global_objective(positions, shapes, cand_routes)
                    if cost < best_reb_cost:
                        best_reb_cost = cost
                        best_reb_line = line
                        best_reb_route = cand_routes[line]

        if (
            best_reb_line is not None
            and best_reb_route is not None
            and _clears_threshold(
                best_inc_cost, best_reb_cost, min_abs_saving, min_rel_saving
            )
        ):
            plan = RebuildPlan.create(mediator.paths[best_reb_line].id, best_reb_route)
            action = next_action_for_plan(mediator, mask, plan)
            if action is not None:
                state.active_plan = plan
                return action

    return best_inc_idx


def _maybe_start_global_rebuild(
    env,
    state: _PlannerState,
    mediator,
    mask: np.ndarray,
    positions: list[tuple[float, float]],
    shapes: list[str],
    routes: list[list[int]],
    *,
    single_min_abs: float,
    single_min_rel: float,
    multi_min_abs: float,
    multi_min_rel: float,
    rebuild_cooldown: int,
) -> int | None:
    """When all stations are served & crewed, rebuild a line only for large global gains."""
    if (
        not routes
        or any(len(p.metros) == 0 for p in mediator.paths)
        or env._decision - state.last_rebuild_decision < rebuild_cooldown
    ):
        return None

    sig = (len(positions), len(routes), tuple(tuple(r) for r in routes))
    if sig == state.last_replan_topology:
        return None
    state.last_replan_topology = sig

    current_cost = compute_global_objective(positions, shapes, routes)
    target_routes = optimize_network(positions, shapes, routes)
    joint_cost = compute_global_objective(positions, shapes, target_routes)
    if not _clears_threshold(current_cost, joint_cost, single_min_abs, single_min_rel):
        return None

    changed_lines = [
        i
        for i in range(len(routes))
        if not _routes_equivalent(routes[i], target_routes[i])
    ]
    if not changed_lines:
        return None

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
        cand = [list(r) for r in routes]
        cand[line] = target_routes[line]
        single_cost = compute_global_objective(positions, shapes, cand)
        # Never allow an intermediate single-line step that worsens or disconnects H(G).
        if single_cost < best_cost and _clears_threshold(
            current_cost, single_cost, single_min_abs, single_min_rel
        ):
            best_cost = single_cost
            best_line = line

    # For profitable 2-line station transfers, rebuild a purely additive receiving line
    # first (dropping zero stations so it cannot break any transfer bridge).
    if best_line is None and _clears_threshold(
        current_cost, joint_cost, multi_min_abs, multi_min_rel
    ):
        best_bridge_cost: float | None = None
        for line in changed_lines:
            if _lookup_action(mask, ActionKind.REMOVE_LINE, line, 0) is None:
                continue
            dropped = set(routes[line]) - set(target_routes[line])
            added = set(target_routes[line]) - set(routes[line])
            if added and not dropped:
                cand = [list(r) for r in routes]
                cand[line] = target_routes[line]
                bridge_cost = compute_global_objective(positions, shapes, cand)
                # Ensure bridge doesn't break reachability before completing second line.
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


def choose(
    env,
    *,
    single_min_abs: float = _DEFAULT_SINGLE_MIN_ABS_SAVING,
    single_min_rel: float = _DEFAULT_SINGLE_MIN_REL_SAVING,
    multi_min_abs: float = _DEFAULT_MULTI_MIN_ABS_SAVING,
    multi_min_rel: float = _DEFAULT_MULTI_MIN_REL_SAVING,
    rebuild_cooldown: int = _DEFAULT_REBUILD_COOLDOWN,
) -> int:
    """Pick the action minimizing balanced global H(G) with hysteresis rebuilds, or WAIT."""
    mediator = env._mediator
    state = _get_planner_state(env)

    # Continue any active multi-step line reconstruction before making fresh choices.
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

    if (
        not can_buy
        and not unserved
        and mediator.available_locomotives == 0
        and mediator.available_carriages == 0
        and all(len(p.metros) > 0 for p in mediator.paths)
        and (
            topo_sig == state.last_replan_topology
            or env._decision - state.last_rebuild_decision < rebuild_cooldown
        )
    ):
        return 0

    mask = env.action_masks()
    legal: dict[ActionKind, list[tuple[int, int, int]]] = {}
    for index in np.flatnonzero(mask):
        kind, first, second = ACTION_TABLE[index]
        legal.setdefault(kind, []).append((int(index), int(first), int(second)))

    # 1. Buy a line slot whenever affordable.
    if ActionKind.PURCHASE_LINE in legal:
        return legal[ActionKind.PURCHASE_LINE][0][0]

    # 2. Keep every active line crewed with >= 1 locomotive (dynamically unassign from
    #    the richest donor line when the reserve pool is empty).
    uncrewed = [i for i, p in enumerate(mediator.paths) if len(p.metros) == 0]
    if uncrewed:
        if ActionKind.ASSIGN_LOCOMOTIVE in legal:
            for index, line_idx, _ in legal[ActionKind.ASSIGN_LOCOMOTIVE]:
                if line_idx in uncrewed:
                    return index
        else:
            any_queued = any(
                getattr(m, "is_unassignment_queued", False) for m in mediator.metros
            )
            if not any_queued:
                candidates = [
                    p
                    for p in mediator.paths
                    if len(p.metros) >= 2
                    and mediator.can_queue_locomotive_unassignment(p)
                ]
                if candidates:
                    donor = max(candidates, key=lambda p: len(p.metros))
                    mediator.queue_locomotive_unassignment(donor)

    # 3. Bootstrap a newly unlocked line slot minimizing balanced global objective.
    if (
        ActionKind.CONNECT in legal
        and len(mediator.paths) < mediator.get_unlocked_num_paths()
        and unserved
    ):
        best_cost: float | None = None
        best_idx: int | None = None
        for index, first, second in legal[ActionKind.CONNECT]:
            if first in unserved or second in unserved:
                cand_routes = routes + [[first, second]]
                cost = compute_global_objective(positions, shapes, cand_routes)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = index
        if best_idx is not None:
            return best_idx

    # 4. Place unserved stations: compare endpoint extension vs interior insertion + 2-opt rebuild.
    if unserved and mediator.paths:
        chosen = _choose_unserved_action_or_rebuild(
            env,
            state,
            mediator,
            mask,
            legal,
            positions,
            shapes,
            routes,
            unserved,
            min_abs_saving=single_min_abs,
            min_rel_saving=single_min_rel,
            rebuild_cooldown=rebuild_cooldown,
        )
        if chosen is not None:
            return chosen

    # 5. Bootstrap the very first line minimizing global objective.
    if len(mediator.paths) == 0 and ActionKind.CONNECT in legal:
        best_cost = None
        best_idx = None
        for index, first, second in legal[ActionKind.CONNECT]:
            if first in unserved and second in unserved:
                cost = compute_global_objective(positions, shapes, [[first, second]])
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = index
        if best_idx is not None:
            return best_idx

    # 6. Check global full-map replanning when all stations are served.
    if not unserved and mediator.paths:
        replan_action = _maybe_start_global_rebuild(
            env,
            state,
            mediator,
            mask,
            positions,
            shapes,
            routes,
            single_min_abs=single_min_abs,
            single_min_rel=single_min_rel,
            multi_min_abs=multi_min_abs,
            multi_min_rel=multi_min_rel,
            rebuild_cooldown=rebuild_cooldown,
        )
        if replan_action is not None:
            return replan_action

    # 7. Fleet allocation: assign spare locomotives to the line with the fewest metros
    #    (breaking ties in favor of the longest route).
    if ActionKind.ASSIGN_LOCOMOTIVE in legal:
        valid = [
            (
                len(mediator.paths[line_idx].metros),
                -len(mediator.paths[line_idx].stations),
                index,
            )
            for index, line_idx, _ in legal[ActionKind.ASSIGN_LOCOMOTIVE]
            if line_idx < len(mediator.paths)
        ]
        if valid:
            valid.sort()
            return valid[0][2]

    if ActionKind.ATTACH_CARRIAGE in legal:
        return legal[ActionKind.ATTACH_CARRIAGE][0][0]

    return 0


def play(seed: int, max_decisions: int = 200_000) -> dict:
    """Run one full episode, returning its score, network metrics, and final H(G)."""
    from rl.semantic_env import SemanticMetroEnv

    env = SemanticMetroEnv()
    env.reset(seed=seed)
    delivered = 0.0
    decisions = 0
    acted = 0
    try:
        while decisions < max_decisions:
            action = choose(env)
            if action != 0:
                acted += 1
            _, reward, terminated, truncated, _ = env.step(action)
            delivered += float(reward)
            decisions += 1
            if terminated or truncated:
                break
        mediator = env._mediator
        positions, shapes, routes, metros = _extract_graph_state(mediator)
        final_h = compute_network_cost(positions, shapes, routes, metros)
        return {
            "seed": seed,
            "deliveries": int(delivered),
            "decisions": decisions,
            "actions_taken": acted,
            "lines": len(mediator.paths),
            "station_counts": [len(p.stations) for p in mediator.paths],
            "metros_per_line": [len(p.metros) for p in mediator.paths],
            "longest_line": max((len(p.stations) for p in mediator.paths), default=0),
            "stations": len(mediator.stations),
            "final_objective_h": round(final_h, 2),
        }
    finally:
        env.close()
