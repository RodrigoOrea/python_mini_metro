"""Combinatorial optimization heuristic guided by a network objective H(G) by Rodrigo Orea.

Evolution from manual rules (`heuristic_rorea.py`) to objective-guided search:
Instead of relying solely on greedy nearest-endpoint distance, every candidate
network modification `a` (creating a line with CONNECT, or growing a line via
EXTEND_LINE / PREPEND_LINE, or allocating a locomotive) is evaluated by
computing the resulting network graph `G_a` and minimizing a global objective
function `H(G_a)`:

    h(s) = sum_{t in ShapeTypes} min_{u in V_t} d(s, u)
    H(G) = sum_{s in V} h(s) + LoadPenalty(G)

where:
- `d(s, u)` is the minimum effective stop-distance in the metro graph from
  station `s` to any station `u` of shape type `t` (0 if `s` already has shape `t`),
  combining:
    * 1.0 base cost per stop (hop),
    * `alpha_dist * (euclidean_px / d_ref)` physical travel-time penalty per segment,
    * `beta_transfer` penalty per line transfer at an interchange station.
- `LoadPenalty(G)` penalizes excessive route length per assigned locomotive.
"""

from __future__ import annotations

import heapq

import numpy as np

from rl.heuristic import _distance, _served, _station_positions
from rl.heuristic_rorea import _eligible_lines_for_extension
from rl.semantic_env import ACTION_TABLE, ActionKind


def _extract_graph_state(
    mediator,
) -> tuple[list[tuple[float, float]], list[str], list[list[int]], list[int]]:
    """Extract station coordinates, shape types, line station-index lists, and fleet counts."""
    positions = _station_positions(mediator)
    shapes = [s.shape.type.value for s in mediator.stations]
    idx_by_id = {s.id: i for i, s in enumerate(mediator.stations)}
    routes = [
        [idx_by_id[s.id] for s in p.stations if s.id in idx_by_id]
        for p in mediator.paths
    ]
    metros = [len(p.metros) for p in mediator.paths]
    return positions, shapes, routes, metros


def station_accessibility_scores(
    positions: list[tuple[float, float]],
    station_shapes: list[str],
    routes: list[list[int]],
    *,
    alpha_dist: float = 1.5,
    d_ref: float = 250.0,
    beta_transfer: float = 1.5,
    unreach_penalty: float = 30.0,
) -> list[float]:
    """Compute h(s) for each station s: sum of minimum distances to each shape type."""
    n_stations = len(positions)
    if n_stations == 0:
        return []
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
            step_cost = 1.0 + alpha_dist * (dist_px / d_ref)
            adj[(u, line_idx)].append(((v, line_idx), step_cost))
            adj[(v, line_idx)].append(((u, line_idx), step_cost))

    # Transfer edges between lines sharing a station
    for s in range(n_stations):
        lines_at_s = station_lines[s]
        if len(lines_at_s) > 1:
            for i in range(len(lines_at_s)):
                for j in range(i + 1, len(lines_at_s)):
                    l1, l2 = lines_at_s[i], lines_at_s[j]
                    adj[(s, l1)].append(((s, l2), beta_transfer))
                    adj[(s, l2)].append(((s, l1), beta_transfer))

    scores = [0.0] * n_stations
    for target_shape in unique_shapes:
        dist_to_s = [unreach_penalty] * n_stations
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
            if d > best_state.get((u, l_idx), unreach_penalty):
                continue
            if d < dist_to_s[u]:
                dist_to_s[u] = d
            for (nxt_u, nxt_l), w in adj.get((u, l_idx), ()):
                nd = d + w
                if nd < best_state.get((nxt_u, nxt_l), unreach_penalty):
                    best_state[(nxt_u, nxt_l)] = nd
                    heapq.heappush(pq, (nd, nxt_u, nxt_l))

        for s in range(n_stations):
            scores[s] += dist_to_s[s]

    return scores


def compute_network_cost(
    positions: list[tuple[float, float]],
    station_shapes: list[str],
    routes: list[list[int]],
    metros_per_line: list[int] | None = None,
    *,
    alpha_dist: float = 1.5,
    d_ref: float = 250.0,
    beta_transfer: float = 1.5,
    gamma_load: float = 12.0,
    unreach_penalty: float = 30.0,
) -> float:
    """Compute the global map objective H(G) = sum_s h(s) + LoadPenalty(G)."""
    scores = station_accessibility_scores(
        positions,
        station_shapes,
        routes,
        alpha_dist=alpha_dist,
        d_ref=d_ref,
        beta_transfer=beta_transfer,
        unreach_penalty=unreach_penalty,
    )
    total_station_cost = sum(scores)

    # Penalize permanent physical line length directly (without dividing by
    # temporary early-game fleet counts, which would under-penalize zig-zags
    # on Line 0 while it temporarily holds all 4 initial locomotives).
    load_penalty = 0.0
    for route in routes:
        line_px = sum(
            _distance(positions[route[i]], positions[route[i + 1]])
            for i in range(len(route) - 1)
        )
        load_penalty += gamma_load * (line_px / d_ref)

    return total_station_cost + load_penalty


def choose(env) -> int:
    """Pick the action that minimizes H(G), or WAIT."""
    mediator = env._mediator

    # Fast-path: if no line can be bought, all stations are served, no fleet is
    # unassigned, and every active line has at least 1 locomotive, return WAIT.
    served = _served(mediator)
    purchasable = mediator.get_next_path_button_idx_to_purchase()
    can_buy = purchasable is not None and mediator.can_purchase_path_button_idx(
        purchasable
    )
    if (
        not can_buy
        and len(served) == len(mediator.stations)
        and mediator.available_locomotives == 0
        and mediator.available_carriages == 0
        and all(len(p.metros) > 0 for p in mediator.paths)
    ):
        return 0

    mask = env.action_masks()
    positions, shapes, routes, metros = _extract_graph_state(mediator)
    unserved = [i for i in range(len(positions)) if i not in served]

    legal: dict[ActionKind, list[tuple[int, int, int]]] = {}
    for index in np.flatnonzero(mask):
        kind, first, second = ACTION_TABLE[index]
        legal.setdefault(kind, []).append((int(index), int(first), int(second)))

    # 1. Buy a line slot whenever affordable.
    if ActionKind.PURCHASE_LINE in legal:
        return legal[ActionKind.PURCHASE_LINE][0][0]

    # 2. Ensure every active line has at least 1 locomotive (dynamically unassign
    #    from the line with the most locomotives if pool is empty).
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

    # 3. If a line slot is unlocked and an unserved station exists, choose the
    #    CONNECT action (transfer hub + unserved station) that minimizes H(G).
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
                cand_metros = metros + [1]
                cost = compute_network_cost(positions, shapes, cand_routes, cand_metros)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = index
        if best_idx is not None:
            return best_idx

    # 4. Attach unserved stations to eligible lines by minimizing H(G).
    if unserved:
        eligible_lines = _eligible_lines_for_extension(mediator)
        best_cost = None
        best_idx = None
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
                cost = compute_network_cost(positions, shapes, cand_routes, metros)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = index
        if best_idx is not None:
            return best_idx

    # 5. Bootstrap the very first line minimizing H(G).
    if len(mediator.paths) == 0 and ActionKind.CONNECT in legal:
        best_cost = None
        best_idx = None
        for index, first, second in legal[ActionKind.CONNECT]:
            if first in unserved and second in unserved:
                cost = compute_network_cost(positions, shapes, [[first, second]], [1])
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = index
        if best_idx is not None:
            return best_idx

    # 6. Fleet assignment: allocate spare locomotives to minimize H(G) (or fewest metros).
    if ActionKind.ASSIGN_LOCOMOTIVE in legal:
        valid = [
            (len(mediator.paths[line_idx].metros), index)
            for index, line_idx, _ in legal[ActionKind.ASSIGN_LOCOMOTIVE]
            if line_idx < len(mediator.paths)
        ]
        if valid:
            valid.sort()
            return valid[0][1]

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
