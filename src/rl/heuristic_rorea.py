"""A multi-line balancing heuristic for the semantic lane by Rodrigo Orea.

Core design:
1. Line purchase: Buy a line slot as soon as affordable.
2. Locomotive priority & dynamic unassignment: Any active line with 0 locomotives
   immediately gets one. If no spare locomotive is available in the global pool,
   queue an unassignment from the line that currently has the most locomotives.
3. New line bootstrap: When a line slot is unlocked, start it as soon as a new
   (or unserved) station arrives, connecting it to the nearest station (hub).
4. Catch-up balancing: When a new line is created, all subsequent connections
   (extensions/prepends) are dedicated to the newest line until its station count
   equals that of the previous line.
5. Balanced fleet allocation: Spare locomotives are assigned to the line with
   the fewest locomotives.
"""

from __future__ import annotations

import numpy as np

from rl.heuristic import _distance, _served, _station_positions
from rl.semantic_env import ACTION_TABLE, ActionKind


def _eligible_lines_for_extension(mediator) -> list[int]:
    """Return lines eligible for new station attachments.

    If the newest line has fewer stations than the previous line, it has
    exclusive priority for new connections until it catches up.
    """
    paths = mediator.paths
    if not paths:
        return []
    if len(paths) >= 2:
        newest = len(paths) - 1
        previous = len(paths) - 2
        if len(paths[newest].stations) < len(paths[previous].stations):
            return [newest]
    return list(range(len(paths)))


def choose(env) -> int:
    """Pick one action index for the current state, or WAIT."""
    mediator = env._mediator
    mask = env.action_masks()
    positions = _station_positions(mediator)
    served = _served(mediator)
    unserved = [i for i in range(len(positions)) if i not in served]

    legal: dict[ActionKind, list[tuple[int, int, int]]] = {}
    for index in np.flatnonzero(mask):
        kind, first, second = ACTION_TABLE[index]
        legal.setdefault(kind, []).append((int(index), int(first), int(second)))

    # 1. Buy a line slot whenever affordable.
    if ActionKind.PURCHASE_LINE in legal:
        return legal[ActionKind.PURCHASE_LINE][0][0]

    # 2. Critical: ensure every active line has at least 1 locomotive.
    #    If a line has 0 locomotives and none are available in inventory,
    #    unassign one from the line with the most locomotives.
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

    # 3. If a line slot is unlocked but not created, start it as soon as an
    #    unserved station arrives. Connect it to the nearest station (transfer hub).
    if (
        ActionKind.CONNECT in legal
        and len(mediator.paths) < mediator.get_unlocked_num_paths()
    ):
        if unserved:
            pick = None
            for index, first, second in legal[ActionKind.CONNECT]:
                if first in unserved or second in unserved:
                    gap = _distance(positions[first], positions[second])
                    both_unserved = first in unserved and second in unserved
                    score = (0 if both_unserved else 1, gap)
                    if pick is None or score < pick[0]:
                        pick = (score, index)
            if pick is not None:
                return pick[1]

    # 4. Attach unserved stations to eligible lines.
    #    If the newest line is catching up to the previous line, only it receives stations.
    eligible_lines = _eligible_lines_for_extension(mediator)
    best = None
    for kind in (ActionKind.EXTEND_LINE, ActionKind.PREPEND_LINE):
        for index, line, station in legal.get(kind, ()):
            if station not in unserved or line >= len(mediator.paths):
                continue
            if line not in eligible_lines:
                continue
            route = mediator.paths[line].stations
            if not route:
                continue
            end = route[-1] if kind is ActionKind.EXTEND_LINE else route[0]
            gap = _distance(positions[station], (end.position.left, end.position.top))
            if best is None or gap < best[0]:
                best = (gap, index)
    if best is not None:
        return best[1]

    # 5. Bootstrap the very first line if no lines exist yet.
    if len(mediator.paths) == 0 and ActionKind.CONNECT in legal:
        pick = None
        for index, first, second in legal[ActionKind.CONNECT]:
            if first in unserved and second in unserved:
                gap = _distance(positions[first], positions[second])
                if pick is None or gap < pick[0]:
                    pick = (gap, index)
        if pick is not None:
            return pick[1]

    # 6. Fleet assignment: assign spare locomotives to the line with the fewest
    #    locomotives, and attach carriages as available.
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

    # 7. Otherwise WAIT.
    return 0


def play(seed: int, max_decisions: int = 200_000) -> dict:
    """Run one full episode, returning its score and what it did."""
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
        }
    finally:
        env.close()
