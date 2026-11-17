"""Open-topology, tunnel-aware, non-mutating `rorea_ml` planner over `OpenAction`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import rl.bridge_protocol as bp
import rl.objective_ml as oml
from rl.heuristic import _distance as _d
from rl.heuristic_rorea_global import _clears_threshold, _routes_equivalent
from rl.objective_ml import demolition_risk_cost as _drc
from rl.replan_ml import CRISIS_CHECK_INTERVAL, REBUILD_COOLDOWN

ANCHOR_PARAMS, BEST_PHASE1_PARAMS = oml.ANCHOR_PARAMS, oml.BEST_PHASE1_PARAMS
_AP, _BP, _ct = ANCHOR_PARAMS, BEST_PHASE1_PARAMS, _clears_threshold
REBALANCE_COOLDOWN, WAIT_LIMIT_MS = 350, 40_000.0
_CCI, _RCD, _RBD = CRISIS_CHECK_INTERVAL, REBUILD_COOLDOWN, REBALANCE_COOLDOWN
CrossingFn = Callable[[bp.EntityId, bp.EntityId], int] | None


@dataclass(frozen=True, slots=True)
class OpenStation:
    id: bp.EntityId
    x: float
    y: float
    shape: str
    waiting_dest_shapes: tuple[str, ...] = ()
    waiting_wait_ms: tuple[int, ...] = ()
    peep_capacity: int = 6
    is_interchange: bool = False
    overload_progress: float = 0.0


@dataclass(frozen=True, slots=True)
class OpenLine:
    id: bp.EntityId
    color_index: int
    station_ids: tuple[bp.EntityId, ...]
    is_looped: bool = False
    is_ghost: bool = False
    tunnels_used: int = 0
    active_train_count: int = 0
    carriage_count: int = 0
    total_capacity: int = 0
    onboard_count: int = 0
    can_relocate_donor: bool = True
    has_pending_train: bool = False


@dataclass(frozen=True, slots=True)
class OpenMetroView:
    decision: int
    stations: tuple[OpenStation, ...]
    lines: tuple[OpenLine, ...]
    lines_available: int
    locos_available: int
    carriages_available: int
    tunnels_available: int
    interchanges_available: int = 0
    crossing_cost_fn: CrossingFn = None
    max_locos_per_line: int = 4
    supports_insert_station: bool = True
    supports_relocate_loco: bool = True
    can_remove_tracks: bool = True


@dataclass
class OpenPlannerState:
    last_decision: int = -1
    pending_actions: list[bp.OpenAction] = field(default_factory=list)
    last_rebuild_decision: int = -10_000
    last_replan_topology: tuple[int, int, tuple[tuple[int, ...], ...]] | None = None
    last_rebalance_decision: int = -10_000
    last_crisis_check_decision: int = -10_000
    pending_rebuild: tuple[bp.EntityId, int, int] | None = None

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]


def _rt(sids: tuple | list, loop: bool, fn: CrossingFn) -> int:
    if fn is None or len(sids) < 2:
        return 0
    es = [(sids[i], sids[i + 1]) for i in range(len(sids) - 1)]
    if loop and len(sids) >= 3:
        es.append((sids[-1], sids[0]))
    return sum(0 if u == v else max(0, int(fn(u, v))) for u, v in es)


def _eq(a: list[int], b: list[int], loop: bool) -> bool:
    dbl, rev, n = a + a, b[::-1], len(a)
    ok = not a or any(dbl[i : i + n] in (b, rev) for i in range(n))
    return (len(a) == len(b) and ok) if loop else _routes_equivalent(a, b)


def _with(rts: list[list[int]], idx: int, nr: list[int]) -> list[list[int]]:
    return [*rts[:idx], nr, *rts[idx + 1 :]]


def _det(pos: list[tuple[float, float]], rt: list[int], st: int, lp: bool):
    m, pst, lp3, out = len(rt), pos[st], lp and len(rt) >= 3, []
    for p in range(1 if lp3 else 0, m + 1):
        u, v = rt[(p - 1) % m], rt[p % m]
        de = _d(pst, pos[rt[0] if p == 0 else rt[-1]])
        dm = _d(pos[u], pst) + _d(pst, pos[v]) - _d(pos[u], pos[v])
        out.append(((de if (not lp3 and p in (0, m)) else dm) / oml.D_REF, p))
    return sorted(out)


def compute_open_ml_objective(pos, shapes, routes, looped, params=_AP, demand=None):
    """Loop-aware `H_theta(s, G)` matching `compute_ml_objective` when unlooped."""
    eff = [
        r + [r[0]] if i < len(looped) and looped[i] and len(r) > 2 else r
        for i, r in enumerate(routes)
    ]
    val = oml.compute_ml_objective(pos, shapes, eff, params, demand)
    loads = demand.line_load if (demand and params.uses_live_demand()) else ()
    for i, r in enumerate(routes):
        if i < len(looped) and looped[i] and len(r) >= 3:
            ex = max(0.0, loads[i] - 1.0) if i < len(loads) else 0.0
            mult = 1.0 + params.w_line_overload * ex
            val -= params.balance_weight * mult * ((2 * len(r) + 1) / 8.0)
    return val


def extract_open_demand(view: OpenMetroView) -> oml.LiveDemandSnapshot:
    """Extract normalized queue, wait-time, OD demand, and line load snapshots."""
    sts, by_id = view.stations, {s.id: i for i, s in enumerate(view.stations)}
    sh_c, s_rows, l_rows, mwn = {}, [], [], 0.0
    for s in sts:
        sh_c[s.shape], cap = sh_c.get(s.shape, 0) + 1, float(max(1, s.peep_capacity))
        mw, dc, dw, ws = max(0.0, float(s.overload_progress)), {}, {}, s.waiting_wait_ms
        for i, ds in enumerate(s.waiting_dest_shapes):
            w = max(0.0, float(ws[i] if i < len(ws) else 0)) / WAIT_LIMIT_MS
            mw, dc[ds] = max(mw, w), dc.get(ds, 0.0) + 1.0 / cap
            dw[ds] = max(dw.get(ds, 0.0), w * w)
        mwn = max(mwn, mw)
        s_rows.append((len(s.waiting_dest_shapes) / cap, mw * mw, dc, dw))
    s_cols = tuple(zip(*s_rows)) if s_rows else ((), (), (), ())
    sp, sw = s_cols[0], s_cols[1]
    avg, pos = max(1, len(sts)) / max(1, len(sh_c)), [(s.x, s.y) for s in sts]
    rar = {sh: max(0.0, avg / max(1, c) - 1.0) for sh, c in sh_c.items()}
    for x in (ln for ln in view.lines if not ln.is_ghost):
        li = [by_id[s] for s in x.station_ids if s in by_id]
        raw_c = x.total_capacity or 6 * (x.active_train_count + x.carriage_count)
        cap, n_li = max(6.0, float(raw_c)), max(1, len(li))
        px = sum(_d(pos[li[i]], pos[li[i + 1]]) for i in range(len(li) - 1))
        px += _d(pos[li[-1]], pos[li[0]]) if (x.is_looped and len(li) >= 3) else 0.0
        wp = sum(len(sts[i].waiting_dest_shapes) for i in li)
        mw_l, q_l = max((sw[i] for i in li), default=0.0), sum(sp[i] for i in li) / n_li
        ob = x.onboard_count
        l_rows.append(((wp + ob) / cap, mw_l, q_l, ob / cap, px))
    l_cols = tuple(zip(*l_rows)) if l_rows else ((), (), (), (), ())
    return oml.LiveDemandSnapshot(*s_cols, rar, *l_cols, mwn)  # type: ignore[arg-type]


class _Ctx:
    def __init__(self, v: OpenMetroView, st: OpenPlannerState, p, d=None) -> None:
        self.v, self.st, self.p, self.d, self.fn = v, st, p, d, v.crossing_cost_fn
        ss, ls = v.stations, tuple(x for x in v.lines if not x.is_ghost)
        by_id = {s.id: i for i, s in enumerate(ss)}
        self.lns, self.ids = ls, tuple(s.id for s in ss)
        self.pos, self.sh = [(s.x, s.y) for s in ss], [s.shape for s in ss]
        self.lp = [x.is_looped for x in ls]
        self.rts = [[by_id[s] for s in x.station_ids if s in by_id] for x in ls]
        self.bt = [self.rtun(r, self.lp[i]) for i, r in enumerate(self.rts)]
        self.uns = [i for i in range(len(ss)) if not any(i in r for r in self.rts)]

    def obj(self, r: list[list[int]]) -> float:
        return compute_open_ml_objective(self.pos, self.sh, r, self.lp, self.p, self.d)

    def rtun(self, r: list[int], loop: bool) -> int:
        return _rt([self.ids[i] for i in r], loop, self.fn)

    def fstat(self, i: int) -> tuple[float, float, float]:
        ln, d, p = self.lns[i], self.d, self.p
        assert d is not None
        ld = d.line_queue_norm[i] + d.line_onboard_ratio[i] + len(self.rts[i]) / 6.0
        raw = p.w_fleet_wait * d.line_max_wait_sq[i] + p.w_fleet_load * ld
        eff = max(0.5, ln.active_train_count + 0.55 * ln.carriage_count)
        return raw, eff, raw / eff

    def ins(self, ln: OpenLine, st: int, rt: list[int], p: int) -> bp.OpenAction:
        sid, lp3 = self.ids[st], ln.is_looped and len(rt) >= 3
        if not lp3 and p in (0, len(rt)):
            cls = bp.PrependLineAction if p == 0 else bp.ExtendLineAction
            return cls(ln.id, sid, False)
        return bp.InsertStationAction(ln.id, sid, self.ids[rt[(p - 1) % len(rt)]])

    def reb(self, rc: list[int], li: int, base: float) -> float | None:
        ln, r, nt = self.lns[li], self.rts[li], self.lns[li].active_train_count
        dt = self.rtun(rc, self.lp[li]) - max(ln.tunnels_used, self.bt[li])
        kw = {"current_cost": base, "n_metros": nt, "n_stations": len(r)}
        c = self.obj(_with(self.rts, li, rc)) + _drc(li, self.p, self.d, **kw)
        return None if dt > self.v.tunnels_available else c

    def start_reb(self, li: int, tgt: list[int]) -> bp.RemoveLineAction:
        old, d, sids = self.lns[li], self.v.decision, tuple(self.ids[i] for i in tgt)
        act = bp.ConnectAction(sids, old.is_looped and len(sids) >= 3, old.color_index)
        self.st.pending_actions, self.st.pending_rebuild = [act], (old.id, d, d + 200)
        return bp.RemoveLineAction(old.id)

    def conn(self, both: bool) -> bp.ConnectAction | None:
        uset, n, lg, pos = set(self.uns), len(self.pos), len(self.pos) > 25, self.pos
        seen: set[tuple[int, int]] = set()
        for u in self.uns if (self.v.lines_available > 0 and n >= 2) else ():
            cv = [k for k in range(n) if k != u and (not both or k in uset)]
            if lg:
                cv = sorted(cv, key=lambda k: _d(pos[u], pos[k]))[:6]
            seen.update((min(u, x), max(u, x)) for x in cv)
        rk = sorted(seen, key=lambda p: (not uset >= set(p), _d(pos[p[0]], pos[p[1]])))
        cands, av, ids = [], self.v.tunnels_available, self.ids
        for a, b in rk[:40] if lg else rk:
            if self.rtun([a, b], False) <= av:
                cands.append((self.obj([*self.rts, [a, b]]), (ids[a], ids[b])))
        return bp.ConnectAction(min(cands)[1]) if cands else None


def _opt_line(ctx: _Ctx, rts, li: int, cap: int | None, passes: int = 4) -> list[int]:
    cur, rt, lp, m = [list(r) for r in rts], list(rts[li]), ctx.lp[li], len(rts[li])
    lg = m > 10 or len(ctx.pos) > 25
    bc, lim = (ctx.obj(cur) if m >= 3 else 0.0), (5 if lg else m + 1)
    for _ in range((min(passes, 2) if lg else passes) if m >= 3 else 0):
        best, cands = None, []
        for i in range(m - 1):
            js = [j for j in range(i + 2, min(m + 1, i + lim)) if (i, j) != (0, m)]
            cands.extend(rt[:i] + rt[i:j][::-1] + rt[j:] for j in js)
        for i, st in enumerate(rt):
            tr = rt[:i] + rt[i + 1 :]
            ins = _det(ctx.pos, tr, st, lp)[: (2 if lg else m)]
            cands.extend(tr[:j] + [st] + tr[j:] for _, j in ins if j != i)
        for cand in (c for c in cands if cap is None or ctx.rtun(c, lp) <= cap):
            if (c := ctx.obj(_with(cur, li, cand))) + 1e-9 < bc:
                bc, best = c, cand
        if best is None:
            break
        rt = cur[li] = best
    return rt


def _opt_network(ctx: _Ctx, max_outer: int = 3) -> list[list[int]]:
    pos, lp, n, av = ctx.pos, ctx.lp, len(ctx.pos), ctx.v.tunnels_available
    lg, cap_t = n > 25, sum(ctx.bt) + av
    cur = [_opt_line(ctx, ctx.rts, i, ctx.bt[i] + av) for i in range(len(ctx.rts))]
    bc, k_ins = (ctx.obj(cur) if len(cur) >= 2 else 0.0), (1 if lg else 99)
    for _ in range((1 if lg else max_outer) if len(cur) >= 2 else 0):
        bm, bmc, cands, near = None, bc, [], [range(len(cur))] * n
        for s in range(n if lg else 0):
            ds = [(min(_d(pos[s], pos[t]) for t in r), b) for b, r in enumerate(cur)]
            near[s] = [b for _, b in sorted(ds)[:2]]
        moves = []
        for a, ra in enumerate(cur):
            for ia, st in enumerate(ra) if len(ra) >= 3 else ():
                ca = _with(cur, a, ra[:ia] + ra[ia + 1 :])
                moves.append((st, ca, k_ins, 99, 0.0))
        if (hb := ctx.p.hub_share_bonus) > 0:
            hubs = [s for s in range(n) if sum(s in r for r in cur) == 1]
            if lg and (d := ctx.d):
                hubs.sort(key=lambda s: -(d.station_pax_norm[s] + d.station_wait_sq[s]))
            moves.extend((s, cur, 2, 6, hb) for s in (hubs[:8] if lg else hubs))
        for st, ca, k, ml, bon in moves:
            bs = [x for x in near[st] if ca[x] == cur[x] and st not in cur[x]]
            for b in (x for x in bs if len(cur[x]) < ml):
                for _, p in _det(pos, cur[b], st, lp[b])[:k]:
                    cands.append((_with(ca, b, cur[b][:p] + [st] + cur[b][p:]), bon))
        for a, ra in enumerate(cur):
            for ia, sa in enumerate(ra):
                for b in (x for x in near[sa] if x > a and sa not in cur[x]):
                    bcs = [(ib, x) for ib, x in enumerate(cur[b]) if x not in ra]
                    if lg and len(bcs) > 1:
                        bcs = [min(bcs, key=lambda t, ps=pos[sa]: _d(ps, pos[t[1]]))]
                    for ib, sb in bcs:
                        ca = _with(cur, a, ra[:ia] + [sb] + ra[ia + 1 :])
                        nr_b = cur[b][:ib] + [sa] + cur[b][ib + 1 :]
                        cands.append((_with(ca, b, nr_b), 0.0))
        for ca, bon in cands:
            t_ca = sum(ctx.rtun(r, lp[k]) for k, r in enumerate(ca)) if ctx.fn else 0
            if t_ca <= cap_t and (c := ctx.obj(ca) - bon) + 1e-9 < bmc:
                bmc, bm = c, ca
        if bm is None:
            break
        bm = [_opt_line(ctx, bm, i, ctx.bt[i] + av, 2) for i in range(len(bm))]
        t_bm = sum(ctx.rtun(r, lp[i]) for i, r in enumerate(bm)) if ctx.fn else 0
        if t_bm > cap_t or (nb := ctx.obj(bm)) + 1e-9 >= bc:
            break
        cur, bc = bm, nb
    return cur


def _choose_unserved(ctx: _Ctx, due: bool, crewed: bool) -> bp.OpenAction | None:
    v, rts, lns, pos, d = ctx.v, ctx.rts, ctx.lns, ctx.pos, ctx.d
    lens = [len(ln.station_ids) for ln in lns]
    catch = bool(lens) and max(lens) >= 4 and min(lens) <= 2
    elig = [i for i, k in enumerate(lens) if not catch or k <= 2]
    cs = ctx.uns
    if len(pos) > 25 and len(cs) > 10:
        sv = [pos[t] for r in rts for t in r]
        ug = {s: -d.station_pax_norm[s] - d.station_wait_sq[s] for s in cs} if d else {}
        md = {s: min((_d(pos[s], p) for p in sv), default=0.0) for s in cs}
        cs = sorted(cs, key=lambda s: (ug.get(s, 0.0), md[s]))[:10]
    bic, bia = float("inf"), None
    for li in elig:
        ln, rt = lns[li], rts[li]
        lp3, bt = ln.is_looped and len(rt) >= 3, ctx.bt[li]
        for s in cs if len(rt) >= 2 else ():
            ins = [p for _, p in _det(pos, rt, s, lp3) if 1 <= p < len(rt) + int(lp3)]
            ps = [] if lp3 else [0, len(rt)]
            if v.supports_insert_station:
                ps.extend(ins[: (3 if len(rt) > 8 else None)])
            for p in ps:
                nr = rt[:p] + [s] + rt[p:]
                if ctx.rtun(nr, ln.is_looped) - bt <= v.tunnels_available:
                    if (c := ctx.obj(_with(rts, li, nr))) < bic:
                        bic, bia = c, ctx.ins(ln, s, rt, p)
    brc, bli, brr = bic, None, None
    do_reb = v.can_remove_tracks and crewed and bia is not None and due
    for li in elig if do_reb else ():
        ln, base = lns[li], rts[li]
        cap = max(ln.tunnels_used, ctx.bt[li]) + v.tunnels_available
        for s in cs if len(base) >= 2 else ():
            ins_p = [p for _, p in _det(pos, base, s, ctx.lp[li]) if 1 <= p < len(base)]
            if ins_p:
                nr = _with(rts, li, base[: ins_p[0]] + [s] + base[ins_p[0] :])
                ro = _opt_line(ctx, nr, li, cap, 2)
                if (cost := ctx.reb(ro, li, bic)) is not None and cost < brc:
                    brc, bli, brr = cost, li, ro
    ok = bli is not None and _ct(bic, brc, 15, ctx.p.single_min_rel)
    return ctx.start_reb(bli, brr) if (ok and brr) else bia  # type: ignore[arg-type]


def _maybe_global_rebuild(ctx: _Ctx, sig, due: bool, crewed: bool):
    v, p, d, st, rts = ctx.v, ctx.p, ctx.d, ctx.st, ctx.rts
    if not (rts and due and crewed):
        return None
    w_hi = p.uses_live_demand() and d is not None and d.max_station_wait_norm >= 0.625
    crisis = w_hi and v.decision - st.last_crisis_check_decision >= _CCI
    if sig == st.last_replan_topology and not crisis:
        return None
    st.last_replan_topology = sig
    if crisis:
        st.last_crisis_check_decision = v.decision
    cur_c, tgts, rel = ctx.obj(rts), _opt_network(ctx), p.single_min_rel
    j_c, av = ctx.obj(tgts), v.tunnels_available
    if not _ct(cur_c, j_c, 15.0, rel):
        return None
    changed = [i for i in range(len(rts)) if not _eq(rts[i], tgts[i], ctx.lp[i])]
    ok_m, step_c = _ct(cur_c, j_c, 22.0, p.multi_min_rel), cur_c + (cur_c - j_c) * 0.25
    cov = {s: sum(s in r for r in rts) for s in range(len(ctx.pos))}
    bli, bc, bbi, bbc = None, cur_c, None, None
    for li in changed:
        ln, cur, tgt, lp = ctx.lns[li], rts[li], tgts[li], ctx.lp[li]
        if len(tgt) == len(cur) + 1 and set(cur) < set(tgt):
            ns, lp3 = next(iter(set(tgt) - set(cur))), lp and len(cur) >= 3
            ps = () if lp3 else (0, len(cur))
            if v.supports_insert_station:
                ps = range(1 if lp3 else 0, len(cur) + 1)
            for pos_i in ps:
                nr = cur[:pos_i] + [ns] + cur[pos_i:]
                if _eq(nr, tgt, lp) and ctx.rtun(nr, lp) - ctx.bt[li] <= av:
                    sc = ctx.obj(_with(rts, li, tgt))
                    ok_1 = p.hub_share_bonus > 0 or _ct(cur_c, sc, 15, rel)
                    if ok_1 or (ok_m and sc <= step_c):
                        return ctx.ins(ln, ns, cur, pos_i)
        if v.can_remove_tracks and (rc := ctx.reb(tgt, li, cur_c)) is not None:
            drp, add = set(cur) - set(tgt), set(tgt) - set(cur)
            safe = not any(cov.get(s, 0) <= 1 for s in drp)
            if safe and rc < bc and _ct(cur_c, rc, 15, rel):
                bc, bli = rc, li
            if add and not drp and rc <= step_c and (bbc is None or rc < bbc):
                bbc, bbi = rc, li
    pick = bli if bli is not None else (bbi if ok_m else None)
    return None if pick is None else ctx.start_reb(pick, tgts[pick])


def optimize_network_open(view, params=_BP, demand=None, *, max_outer=3):
    """Optimize multi-line routes under loop-aware objective and tunnel budget."""
    return _opt_network(_Ctx(view, OpenPlannerState(), params, demand), max_outer)


def choose_open(view: OpenMetroView, state: OpenPlannerState, params=_BP):
    """Pick the next open `OpenAction` without mutating `view`."""
    v, st = view, state
    if v.decision < st.last_decision:
        st.reset()
    st.last_decision = v.decision
    ctx = _Ctx(v, st, params)
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
            st.last_rebuild_decision = v.decision
            used = {ln.color_index for ln in ctx.lns}
            c_idx = None if act.color_index in used else act.color_index
            return bp.ConnectAction(act.station_ids, act.is_looped, c_idx)
    lns, rts, mx = ctx.lns, ctx.rts, v.max_locos_per_line
    tc = [x.active_train_count for x in lns]
    rdy = [i for i, x in enumerate(lns) if not x.has_pending_train]
    unc = [i for i in rdy if len(rts[i]) > 1 and tc[i] == 0]
    don = [i for i in rdy if tc[i] > 1 and lns[i].can_relocate_donor]
    sig = (len(ctx.pos), len(rts), tuple(tuple(r) for r in rts))
    crewed, live = all(tc), params.uses_dynamic_fleet() or params.uses_live_demand()
    per = live and v.decision - st.last_crisis_check_decision >= _CCI
    idl = not (v.locos_available or v.carriages_available or v.interchanges_available)
    due = v.decision - st.last_rebuild_decision >= _RCD
    calm_b = not (ctx.uns or unc) and idl
    calm_s = calm_b and sig == st.last_replan_topology
    if calm_b and not per and (calm_s or not due):
        return bp.WaitAction(ticks=1)
    need_d = live or params.demo_risk_weight > 0.0 or v.interchanges_available > 0
    ctx.d = d = extract_open_demand(v) if need_d else None
    if per and d is not None:
        cool = v.decision - st.last_rebalance_decision < _RBD
        can_rb = v.supports_relocate_loco and params.w_rebalance > 0
        blk_rb = st.pending_actions or v.locos_available or len(lns) < 2 or cool or unc
        if can_rb and not blk_rb:
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
        if calm_s and d.max_station_wait_norm < 0.625:
            st.last_crisis_check_decision = v.decision
            return bp.WaitAction(ticks=1)
    if unc:
        if v.locos_available > 0 and tc[unc[0]] < mx:
            return bp.AssignLocoAction(lns[unc[0]].id)
        if v.supports_relocate_loco and don:
            bi = max(don, key=lambda i: (tc[i], -i))
            return bp.RelocateLocoAction(lns[bi].id, lns[unc[0]].id)
    if lns and ctx.uns and v.lines_available > 0 and (a := ctx.conn(False)):
        return a
    if ctx.uns and lns and (a := _choose_unserved(ctx, due, crewed)):
        return a
    if not lns and v.lines_available > 0 and (a := ctx.conn(True)):
        return a
    if not ctx.uns and lns and (a := _maybe_global_rebuild(ctx, sig, due, crewed)):
        return a
    if v.interchanges_available > 0:
        cov, ic = [sum(i in r for r in rts) for i in range(len(v.stations))], []
        for i, s_o in enumerate(v.stations):
            pn = len(s_o.waiting_dest_shapes) / float(max(1, s_o.peep_capacity))
            fb = max(s_o.waiting_wait_ms, default=0) / WAIT_LIMIT_MS
            w_n = d.station_wait_sq[i] ** 0.5 if d else fb
            urg = max(w_n, float(s_o.overload_progress))
            q_hi = len(s_o.waiting_dest_shapes) >= max(4, s_o.peep_capacity - 1)
            if not s_o.is_interchange and (urg >= 0.35 or pn >= 0.75 or q_hi):
                k_ic = (int(cov[i] > 0), int(cov[i] >= 2), urg, pn, cov[i], -i)
                ic.append((k_ic, s_o.id))
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
