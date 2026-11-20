"""Optimal-map recalculating planner for Unity using ``ADD_TO_LINE(l, s, p)``."""

from __future__ import annotations

from dataclasses import dataclass, field

import rl.bridge_protocol as bp
import rl.objective_ml as oml
from rl.heuristic import _distance as _d
from rl.open_topology_planner import (
    _CCI,
    _RBD,
    OpenLine,
    OpenMetroView,
    OpenPlannerState,
    _Ctx,
    _det,
    _eq,
    _opt_network,
    _rt,
    _with,
    compute_open_ml_objective,
    extract_open_demand,
)

_BP = oml.BEST_PHASE1_PARAMS


@dataclass
class OptimalReplanState:
    """Mutable state for event-driven optimal-map replanning without cooldowns."""

    last_decision: int = -1
    pending_actions: list[bp.OpenAction] = field(default_factory=list)
    pending_rebuild: tuple[bp.EntityId, int, int] | None = None
    last_rebalance_decision: int = -10_000
    last_crisis_check_decision: int = -10_000
    last_trigger_sig: tuple | None = None
    target_routes: list[tuple[list[int], bool]] | None = None

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]


def _trigger_signature(
    view: OpenMetroView, active_lines: tuple[OpenLine, ...]
) -> tuple:
    return (
        tuple(s.id for s in view.stations),
        view.lines_available,
        view.locos_available,
        view.carriages_available,
        view.tunnels_available,
        view.interchanges_available,
        len(active_lines),
    )


def _first_add_to_line_step(
    cur_r: list[int],
    cur_lp: bool,
    tgt_r: list[int],
    tgt_lp: bool,
) -> tuple[int, int, int, bool] | None:
    """Return `(num_steps, station_idx, insert_pos, close_loop)` if `cur_r` -> `tgt_r` needs only ADD_TO_LINE."""
    if len(tgt_r) <= len(cur_r):
        return None
    if cur_lp != tgt_lp:
        if cur_lp or not tgt_lp or len(tgt_r) != len(cur_r) + 1:
            return None

    orientations: list[list[int]] = []
    if tgt_lp:
        n = len(tgt_r)
        for base in (tgt_r, tgt_r[::-1]):
            for shift in range(n):
                orientations.append(base[shift:] + base[:shift])
    else:
        orientations.append(tgt_r)
        orientations.append(tgt_r[::-1])

    for orient in orientations:
        ci = 0
        for val in orient:
            if ci < len(cur_r) and val == cur_r[ci]:
                ci += 1
        if ci == len(cur_r):
            # Find the first missing element index `pos` in `orient`.
            for pos, val in enumerate(orient):
                if pos >= len(cur_r) or val != cur_r[pos]:
                    if (not cur_lp) and tgt_lp and pos not in (0, len(cur_r)):
                        break
                    close_loop = (not cur_lp) and tgt_lp
                    return (len(tgt_r) - len(cur_r), val, pos, close_loop)
    return None


def _line_edit_cost(
    cur_r: list[int], cur_lp: bool, tgt_r: list[int], tgt_lp: bool
) -> tuple[int, int]:
    """Return `(action_count, removal_count)` to transform `(cur_r, cur_lp)` into `(tgt_r, tgt_lp)`."""
    if cur_lp == tgt_lp and _eq(cur_r, tgt_r, cur_lp):
        return (0, 0)
    add_info = _first_add_to_line_step(cur_r, cur_lp, tgt_r, tgt_lp)
    if add_info is not None:
        return (add_info[0], 0)
    return (2, 1)


def match_lines_min_steps(
    cur_routes: list[tuple[list[int], bool]],
    tgt_routes: list[tuple[list[int], bool]],
) -> tuple[dict[int, int], list[int], list[int]]:
    """Find the exact minimum-step matching between current and target lines.

    Returns `(cur_to_tgt, unmatched_cur, unmatched_tgt)`.
    Tie-breaks on fewer line removals.
    """
    m, n = len(cur_routes), len(tgt_routes)
    costs = [
        [_line_edit_cost(*cur_routes[i], *tgt_routes[j]) for j in range(n)]
        for i in range(m)
    ]

    best_score: tuple[int, int] | None = None
    best_assign: dict[int, int] = {}

    def _search(
        i: int,
        used_tgt: int,
        act_sum: int,
        rem_sum: int,
        assign: dict[int, int],
    ) -> None:
        nonlocal best_score, best_assign
        if best_score is not None and (act_sum, rem_sum) >= best_score:
            return
        if i == m:
            unmatched_t = n - bin(used_tgt).count("1")
            total = (act_sum + unmatched_t, rem_sum)
            if best_score is None or total < best_score:
                best_score = total
                best_assign = dict(assign)
            return

        # Option 1: match line i to an unused target line j (only if better than remove+connect=2).
        for j in range(n):
            if used_tgt & (1 << j):
                continue
            c_act, c_rem = costs[i][j]
            if c_act < 2 or (c_act == 2 and c_rem <= 1):
                assign[i] = j
                _search(
                    i + 1, used_tgt | (1 << j), act_sum + c_act, rem_sum + c_rem, assign
                )
                del assign[i]

        # Option 2: leave line i unmatched (REMOVE_LINE -> 1 action, 1 removal).
        _search(i + 1, used_tgt, act_sum + 1, rem_sum + 1, assign)

    _search(0, 0, 0, 0, {})
    matched_tgt = set(best_assign.values())
    unmatched_cur = [i for i in range(m) if i not in best_assign]
    unmatched_tgt = [j for j in range(n) if j not in matched_tgt]
    return best_assign, unmatched_cur, unmatched_tgt


def _greedy_complete(
    ctx: _Ctx,
    seed_rts: list[list[int]],
    seed_lp: list[bool],
    max_lines: int,
    cap_t: int,
) -> tuple[list[list[int]], list[bool]]:
    """Greedily place all remaining unserved stations onto `seed_rts` or new lines."""
    rts = [list(r) for r in seed_rts]
    lp = list(seed_lp)
    pos = ctx.pos
    n = len(pos)
    served = {s for r in rts for s in r}
    unserved = [i for i in range(n) if i not in served]

    if not rts and len(unserved) >= 2 and max_lines >= 1:
        best_pair: tuple[float, list[int]] | None = None
        for i in range(len(unserved)):
            for j in range(i + 1, len(unserved)):
                a, b = unserved[i], unserved[j]
                if ctx.rtun([a, b], False) <= cap_t:
                    c = compute_open_ml_objective(
                        pos, ctx.sh, [[a, b]], [False], ctx.p, ctx.d
                    )
                    if best_pair is None or c < best_pair[0]:
                        best_pair = (c, [a, b])
        if best_pair is not None:
            rts.append(best_pair[1])
            lp.append(False)
            served.update(best_pair[1])
            unserved = [i for i in range(n) if i not in served]

    while unserved:
        used_t = sum(ctx.rtun(r, lp[i]) for i, r in enumerate(rts))
        rem_t = cap_t - used_t
        best_cost: float | None = None
        best_cand: tuple[list[list[int]], list[bool], int] | None = None

        for li, rt in enumerate(rts):
            lp3 = lp[li] and len(rt) >= 3
            base_t = ctx.rtun(rt, lp[li])
            for s in unserved:
                ins = [p for _, p in _det(pos, rt, s, lp3)]
                for p in ins[: (4 if len(rt) > 8 else len(ins))]:
                    nr = rt[:p] + [s] + rt[p:]
                    if ctx.rtun(nr, lp[li]) - base_t <= rem_t:
                        cand_rts = _with(rts, li, nr)
                        c = compute_open_ml_objective(
                            pos, ctx.sh, cand_rts, lp, ctx.p, ctx.d
                        )
                        if best_cost is None or c < best_cost:
                            best_cost = c
                            best_cand = (cand_rts, lp, s)

        if len(rts) < max_lines:
            for s in unserved[:6]:
                partners = sorted(
                    (k for k in range(n) if k != s),
                    key=lambda k: _d(pos[s], pos[k]),
                )[:4]
                for u in partners:
                    if ctx.rtun([s, u], False) <= rem_t:
                        cand_rts = [*rts, [s, u]]
                        cand_lp = [*lp, False]
                        c = compute_open_ml_objective(
                            pos, ctx.sh, cand_rts, cand_lp, ctx.p, ctx.d
                        )
                        if best_cost is None or c < best_cost:
                            best_cost = c
                            best_cand = (cand_rts, cand_lp, s)

        if best_cand is None:
            break
        rts, lp, _ = best_cand
        served = {s for r in rts for s in r}
        unserved = [i for i in range(n) if i not in served]

    return rts, lp


def _polish_network(
    ctx: _Ctx,
    rts: list[list[int]],
    lp: list[bool],
    cap_t: int,
    *,
    max_outer: int = 3,
) -> tuple[list[list[int]], list[bool]]:
    """Run loop-aware 2-opt, relocate, inter-line transfers/swaps on `(rts, lp)`."""
    if not rts:
        return [], []
    tmp_ctx = _Ctx(ctx.v, OpenPlannerState(), ctx.p, ctx.d)
    tmp_ctx.rts = [list(r) for r in rts]
    tmp_ctx.lp = list(lp)
    tmp_ctx.bt = [tmp_ctx.rtun(r, lp[i]) for i, r in enumerate(rts)]
    # Override tunnels_available in a shallow copy of view? No, _opt_network reads
    # sum(ctx.bt) + ctx.v.tunnels_available == cap_t, which already matches if
    # total tunnels of rts <= cap_t, wait: sum(tmp_ctx.bt) could be < cap_t!
    # So pass a lightweight view proxy with adjusted tunnels_available if needed.
    rem = max(0, cap_t - sum(tmp_ctx.bt))
    if rem != ctx.v.tunnels_available:
        v = ctx.v
        tmp_ctx.v = OpenMetroView(
            decision=v.decision,
            stations=v.stations,
            lines=v.lines,
            lines_available=v.lines_available,
            locos_available=v.locos_available,
            carriages_available=v.carriages_available,
            tunnels_available=rem,
            interchanges_available=v.interchanges_available,
            crossing_cost_fn=v.crossing_cost_fn,
            max_locos_per_line=v.max_locos_per_line,
            supports_insert_station=v.supports_insert_station,
            supports_relocate_loco=v.supports_relocate_loco,
            can_remove_tracks=v.can_remove_tracks,
        )
    opt_rts = _opt_network(tmp_ctx, max_outer=max_outer)
    return opt_rts, list(lp)


def compute_optimal_network(ctx: _Ctx) -> list[tuple[list[int], bool]]:
    """Compute the globally optimal multi-line network under `compute_open_ml_objective`."""
    n = len(ctx.pos)
    if n < 2:
        return []
    cap_t = sum(ctx.bt) + ctx.v.tunnels_available
    max_lines = min(len(ctx.rts) + ctx.v.lines_available, max(1, n // 2))
    if max_lines == 0:
        return [(list(r), ctx.lp[i]) for i, r in enumerate(ctx.rts)]

    candidates: list[tuple[list[list[int]], list[bool]]] = []

    # Candidate 1: warm start from current routes + greedy placement of unserved stations.
    w_rts, w_lp = _greedy_complete(ctx, ctx.rts, ctx.lp, max_lines, cap_t)
    if not ctx.v.can_remove_tracks:
        return [(list(r), w_lp[i]) for i, r in enumerate(w_rts)]
    if w_rts:
        candidates.append((w_rts, w_lp))
        candidates.append(_polish_network(ctx, w_rts, w_lp, cap_t))

    # Candidate 2: if a free line slot exists and an active line has >= 4 stops, try splitting it.
    if len(w_rts) < max_lines:
        for li, rt in enumerate(w_rts):
            if len(rt) >= 4:
                mid = len(rt) // 2
                s1 = rt[: mid + 1]
                s2 = rt[mid:]
                sp_rts = [*w_rts[:li], s1, s2, *w_rts[li + 1 :]]
                sp_lp = [*w_lp[:li], False, False, *w_lp[li + 1 :]]
                if sum(ctx.rtun(r, sp_lp[k]) for k, r in enumerate(sp_rts)) <= cap_t:
                    candidates.append(_polish_network(ctx, sp_rts, sp_lp, cap_t))

    # Candidate 3: fresh multi-line seed (farthest-first) when >= 2 lines are allowed.
    if max_lines >= 2 and n >= 4:
        for k_lines in {min(max_lines, max(2, len(ctx.rts))), max_lines}:
            if n < k_lines * 2:
                continue
            # Pick k_lines spread-out seed stations, pair each with its nearest neighbor.
            seeds = [0]
            while len(seeds) < k_lines:
                nxt = max(
                    (s for s in range(n) if s not in seeds),
                    key=lambda s: min(_d(ctx.pos[s], ctx.pos[x]) for x in seeds),
                )
                seeds.append(nxt)
            f_rts: list[list[int]] = []
            used: set[int] = set()
            for s in seeds:
                if s in used:
                    continue
                partner = min(
                    (u for u in range(n) if u != s and u not in used),
                    key=lambda u: _d(ctx.pos[s], ctx.pos[u]),
                    default=None,
                )
                if partner is None:
                    continue
                if ctx.rtun([s, partner], False) <= cap_t - sum(
                    ctx.rtun(r, False) for r in f_rts
                ):
                    f_rts.append([s, partner])
                    used.update((s, partner))
            if f_rts:
                f_lp = [False] * len(f_rts)
                f_rts, f_lp = _greedy_complete(ctx, f_rts, f_lp, max_lines, cap_t)
                candidates.append(_polish_network(ctx, f_rts, f_lp, cap_t))

    candidates = [
        (c_rts, c_lp)
        for c_rts, c_lp in candidates
        if c_rts and sum(ctx.rtun(r, c_lp[i]) for i, r in enumerate(c_rts)) <= cap_t
    ]
    if not candidates:
        return [(list(r), ctx.lp[i]) for i, r in enumerate(ctx.rts)]

    cur_served = len({s for r in ctx.rts for s in r})
    cur_obj = compute_open_ml_objective(ctx.pos, ctx.sh, ctx.rts, ctx.lp, ctx.p, ctx.d)

    def _cand_key(item: tuple[list[list[int]], list[bool]]) -> tuple[int, float, int]:
        c_rts, c_lp = item
        served_cnt = len({s for r in c_rts for s in r})
        obj = compute_open_ml_objective(ctx.pos, ctx.sh, c_rts, c_lp, ctx.p, ctx.d)
        return (-served_cnt, obj, len(c_rts))

    best_rts, best_lp = min(candidates, key=_cand_key)
    best_served = len({s for r in best_rts for s in r})
    best_obj = compute_open_ml_objective(
        ctx.pos, ctx.sh, best_rts, best_lp, ctx.p, ctx.d
    )

    if best_served > cur_served or best_obj + 1e-6 < cur_obj:
        return [(list(r), best_lp[i]) for i, r in enumerate(best_rts)]
    return [(list(r), ctx.lp[i]) for i, r in enumerate(ctx.rts)]


def _next_transition_action(
    ctx: _Ctx,
    state: OptimalReplanState,
    targets: list[tuple[list[int], bool]],
) -> bp.OpenAction | None:
    """Emit the next shortest-path step toward `targets` using CONNECT, ADD_TO_LINE, REMOVE_LINE."""
    cur_pairs = [(list(ctx.rts[i]), ctx.lp[i]) for i in range(len(ctx.rts))]
    cur_to_tgt, unmatched_cur, unmatched_tgt = match_lines_min_steps(cur_pairs, targets)

    # 1. Non-destructive ADD_TO_LINE steps on matched lines.
    for ci, tj in cur_to_tgt.items():
        cur_r, cur_lp = cur_pairs[ci]
        tgt_r, tgt_lp = targets[tj]
        step = _first_add_to_line_step(cur_r, cur_lp, tgt_r, tgt_lp)
        if step is None:
            continue
        _, st_idx, pos_i, close_loop = step
        # Re-derive exact new_r for tunnel check.
        new_r = cur_r[:pos_i] + [st_idx] + cur_r[pos_i:]
        new_lp = cur_lp or close_loop
        if ctx.rtun(new_r, new_lp) - ctx.bt[ci] <= ctx.v.tunnels_available:
            return bp.AddToLineAction(
                line_id=ctx.lns[ci].id,
                station_id=ctx.ids[st_idx],
                position=pos_i,
                is_looped=close_loop,
            )

    # 2. Non-destructive CONNECT for unmatched target lines in free slots.
    if ctx.v.lines_available > 0 and unmatched_tgt:
        for tj in unmatched_tgt:
            tgt_r, tgt_lp = targets[tj]
            if ctx.rtun(tgt_r, tgt_lp) <= ctx.v.tunnels_available:
                sids = tuple(ctx.ids[s] for s in tgt_r)
                return bp.ConnectAction(station_ids=sids, is_looped=tgt_lp)

    # 3. Destructive rebuilds (only if track removal is allowed): REMOVE_LINE -> CONNECT.
    if not ctx.v.can_remove_tracks:
        return None

    cov = {s: sum(s in r for r in ctx.rts) for s in range(len(ctx.pos))}
    rebuild_cands: list[
        tuple[int, int, bp.EntityId, tuple[list[int], bool] | None]
    ] = []

    for ci, tj in cur_to_tgt.items():
        cur_r, cur_lp = cur_pairs[ci]
        tgt_r, tgt_lp = targets[tj]
        if _line_edit_cost(cur_r, cur_lp, tgt_r, tgt_lp) == (2, 1):
            need_t = ctx.rtun(tgt_r, tgt_lp)
            avail_after = ctx.v.tunnels_available + max(
                ctx.lns[ci].tunnels_used, ctx.bt[ci]
            )
            if need_t <= avail_after:
                dropped = set(cur_r) - set(tgt_r)
                lost_unique = sum(1 for s in dropped if cov.get(s, 0) <= 1)
                rebuild_cands.append((lost_unique, 0, ctx.lns[ci].id, (tgt_r, tgt_lp)))

    for ci in unmatched_cur:
        lost_unique = sum(1 for s in cur_pairs[ci][0] if cov.get(s, 0) <= 1)
        rebuild_cands.append((lost_unique, 1, ctx.lns[ci].id, None))

    if not rebuild_cands:
        return None

    rebuild_cands.sort(key=lambda item: (item[0], item[1]))
    _, _, old_id, tgt_opt = rebuild_cands[0]
    if tgt_opt is not None:
        tgt_r, tgt_lp = tgt_opt
        old_ln = next(ln for ln in ctx.lns if ln.id == old_id)
        sids = tuple(ctx.ids[s] for s in tgt_r)
        state.pending_actions = [bp.ConnectAction(sids, tgt_lp, old_ln.color_index)]
        state.pending_rebuild = (old_id, ctx.v.decision, ctx.v.decision + 200)
    return bp.RemoveLineAction(old_id)


def choose_optimal_replan(
    view: OpenMetroView,
    state: OptimalReplanState,
    params: oml.ParametricPolicyParams = _BP,
) -> bp.OpenAction:
    """Pick the next `OpenAction` using optimal-map recalculation and shortest-step execution."""
    v, st = view, state
    if v.decision < st.last_decision:
        st.reset()
    st.last_decision = v.decision
    ctx = _Ctx(v, OpenPlannerState(), params)

    # 0. Finish any queued ConnectAction after a RemoveLineAction (wait for ghost to clear).
    if st.pending_actions:
        act = st.pending_actions[0]
        if not isinstance(act, bp.ConnectAction):
            return st.pending_actions.pop(0)
        oid, sd, ud = st.pending_rebuild or (None, -1, -1)
        o_act = oid is not None and any(ln.id == oid for ln in ctx.lns)
        o_gh = oid is not None and any(ln.id == oid and ln.is_ghost for ln in v.lines)
        bad = not set(act.station_ids).issubset(ctx.ids) or (o_act and v.decision > sd)
        need_t = _rt(act.station_ids, act.is_looped, ctx.fn)
        blk = v.lines_available < 1 or need_t > v.tunnels_available
        if not bad and blk and (o_gh or o_act) and v.decision <= ud:
            return bp.WaitAction(ticks=1)
        st.pending_actions = [] if (bad or blk) else st.pending_actions[1:]
        if not st.pending_actions:
            st.pending_rebuild = None
        if not (bad or blk):
            used = {ln.color_index for ln in ctx.lns}
            c_idx = None if act.color_index in used else act.color_index
            return bp.ConnectAction(act.station_ids, act.is_looped, c_idx)

    lns, rts, mx = ctx.lns, ctx.rts, v.max_locos_per_line
    tc = [x.active_train_count for x in lns]
    rdy = [i for i, x in enumerate(lns) if not x.has_pending_train]
    unc = [i for i in rdy if len(rts[i]) > 1 and tc[i] == 0]
    don = [i for i in rdy if tc[i] > 1 and lns[i].can_relocate_donor]

    # 1. Keep every active line crewed with >= 1 locomotive.
    if unc:
        if v.locos_available > 0 and tc[unc[0]] < mx:
            return bp.AssignLocoAction(lns[unc[0]].id)
        if v.supports_relocate_loco and don:
            bi = max(don, key=lambda i: (tc[i], -i))
            return bp.RelocateLocoAction(lns[bi].id, lns[unc[0]].id)

    # 2. Extract demand and check event-driven triggers (new station or new asset/slot).
    need_d = (
        params.uses_live_demand()
        or params.uses_dynamic_fleet()
        or v.interchanges_available > 0
    )
    ctx.d = d = extract_open_demand(v) if need_d else None

    sig = _trigger_signature(v, lns)
    if st.target_routes is None or sig != st.last_trigger_sig or ctx.uns:
        st.last_trigger_sig = sig
        st.target_routes = compute_optimal_network(ctx)

    # 3. Advance shortest-step transition toward `st.target_routes`.
    if st.target_routes:
        step_act = _next_transition_action(ctx, st, st.target_routes)
        if step_act is not None:
            return step_act

    # 4. Reuse minimax locomotive rebalancing, interchanges, and fleet allocation.
    live = params.uses_dynamic_fleet() or params.uses_live_demand()
    if live and d is not None and v.decision - st.last_crisis_check_decision >= _CCI:
        st.last_crisis_check_decision = v.decision
        cool = v.decision - st.last_rebalance_decision < _RBD
        if (
            v.supports_relocate_loco
            and params.w_rebalance > 0
            and not (v.locos_available or len(lns) < 2 or cool or unc)
        ):
            met = [ctx.fstat(i) for i in range(len(lns))]
            tgt = max(rdy, key=lambda i: met[i][2] if tc[i] < mx else -1.0, default=-1)
            ws, qs, os = d.line_max_wait_sq, d.line_queue_norm, d.line_onboard_ratio
            wt = ws[tgt] if (tgt >= 0 and tc[tgt] < mx) else -1.0
            busy = wt > 0.12 or (wt >= 0 and (os[tgt] > 0.5 or qs[tgt] > 0.4))
            bd, bk = None, None
            for i in (x for x in don if busy and x != tgt and ws[x] < 0.35):
                lost = lns[i].carriage_count // max(1, tc[i])
                if lost == 0 or (wt >= 0.45 and tc[i] >= 3):
                    da = met[i][0] / max(0.5, met[i][1] - 1.0 - 0.55 * lost)
                    ta = met[tgt][0] / (met[tgt][1] + 1.0)
                    drop = max(met[i][2], met[tgt][2]) - max(da, ta)
                    k = (drop, tc[i], -i)
                    if params.w_rebalance * drop > 2.5 and (bk is None or k > bk):
                        bk, bd = k, i
            if bd is not None:
                st.last_rebalance_decision = v.decision
                return bp.RelocateLocoAction(lns[bd].id, lns[tgt].id)

    if v.interchanges_available > 0:
        cov, ic = [sum(i in r for r in rts) for i in range(len(v.stations))], []
        for i, s_o in enumerate(v.stations):
            pn = len(s_o.waiting_dest_shapes) / float(max(1, s_o.peep_capacity))
            fb = max(s_o.waiting_wait_ms, default=0) / 40_000.0
            w_n = d.station_wait_sq[i] ** 0.5 if d else fb
            urg = max(w_n, float(s_o.overload_progress))
            q_hi = len(s_o.waiting_dest_shapes) >= max(4, s_o.peep_capacity - 1)
            if not s_o.is_interchange and (urg >= 0.35 or pn >= 0.75 or q_hi):
                ic.append(
                    ((int(cov[i] > 0), int(cov[i] >= 2), urg, pn, cov[i], -i), s_o.id)
                )
        if ic:
            return bp.PlaceInterchangeAction(max(ic)[1])

    dyn = d is not None and params.uses_dynamic_fleet()
    fu = [ctx.fstat(i)[2] if (dyn and d) else 0.0 for i in range(len(lns))]
    el = [i for i in range(len(lns)) if len(rts[i]) > 1 and tc[i] < mx]
    if v.locos_available > 0 and el:
        ks = [(tc[i], -len(rts[i]), i) for i in el]
        if dyn and d:
            hp = [x.has_pending_train for x in lns]
            ks = [(bool(c or hp[i]), 0.35 * c - fu[i], i) for c, _, i in ks]
        return bp.AssignLocoAction(lns[min(ks)[2]].id)

    ec = [i for i in range(len(lns)) if len(rts[i]) > 1 and tc[i] > 0]
    if v.carriages_available > 0 and ec:
        bi = max(ec, key=lambda i: (fu[i], len(rts[i]), -i)) if (dyn and d) else ec[0]
        return bp.AttachCarriageAction(lns[bi].id)

    return bp.WaitAction(ticks=1)
