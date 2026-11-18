"""Unit tests for open-topology, tunnel-aware `open_topology_planner` planner (`OpenAction`)."""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rl.bridge_protocol as bp
from rl.objective_ml import ANCHOR_PARAMS, BEST_PHASE1_PARAMS, compute_ml_objective
from rl.open_topology_planner import (
    OpenLine,
    OpenMetroView,
    OpenPlannerState,
    OpenStation,
    choose_open,
    compute_open_ml_objective,
    extract_open_demand,
    optimize_network_open,
)
from rl.policy_parametric_cem import choose
from rl.semantic_env import ACTION_TABLE, ActionKind, SemanticMetroEnv


def _st(
    sid: int | str,
    x: float,
    y: float,
    shape: str = "CIRCLE",
    *,
    dests: tuple[str, ...] = (),
    waits: tuple[int, ...] = (),
    cap: int = 6,
    ic: bool = False,
    ov: float = 0.0,
) -> OpenStation:
    return OpenStation(sid, x, y, shape, dests, waits, cap, ic, ov)


def _ln(lid: int | str, c_idx: int, sids: tuple[int | str, ...], **kw) -> OpenLine:
    return OpenLine(
        lid,
        c_idx,
        sids,
        kw.get("loop", False),
        kw.get("ghost", False),
        kw.get("tunnels", 0),
        kw.get("trains", 1),
        kw.get("cars", 0),
        kw.get("cap", 0),
        kw.get("onboard", 0),
        kw.get("donor", True),
        kw.get("pending", False),
    )


def _view_from_env(env: SemanticMetroEnv) -> OpenMetroView:
    m = env._mediator
    sts = tuple(
        OpenStation(
            id=s.id,
            x=float(s.position.left),
            y=float(s.position.top),
            shape=s.shape.type.value,
            waiting_dest_shapes=tuple(
                p.destination_shape.type.value for p in s.passengers
            ),
            waiting_wait_ms=tuple(int(p.wait_ms) for p in s.passengers),
            peep_capacity=int(getattr(m, "station_passenger_capacity", 6)),
        )
        for s in m.stations
    )
    lns = tuple(
        OpenLine(
            id=p.id,
            color_index=i,
            station_ids=tuple(s.id for s in p.stations),
            is_looped=bool(getattr(p, "is_looped", False)),
            active_train_count=len(p.metros),
            carriage_count=sum(max(0, int(t.capacity // 6) - 1) for t in p.metros),
            total_capacity=sum(int(t.capacity) for t in p.metros),
            onboard_count=sum(len(t.passengers) for t in p.metros),
            can_relocate_donor=bool(m.can_queue_locomotive_unassignment(p)),
            has_pending_train=any(
                getattr(t, "is_unassignment_queued", False) for t in p.metros
            ),
        )
        for i, p in enumerate(m.paths)
    )
    return OpenMetroView(
        decision=env._decision,
        stations=sts,
        lines=lns,
        lines_available=max(0, m.get_unlocked_num_paths() - len(m.paths)),
        locos_available=int(m.available_locomotives),
        carriages_available=int(m.available_carriages),
        tunnels_available=99,
        supports_insert_station=False,
        supports_relocate_loco=True,
        can_remove_tracks=True,
    )


class TestOpenTopologyPlannerScaleAndLatency(unittest.TestCase):
    def test_32_stations_6_lines_decision_and_bounded_latency(self) -> None:
        shapes = ("CIRCLE", "TRIANGLE", "SQUARE", "STAR", "PENTAGON", "DIAMOND")
        sts = tuple(
            _st(
                i,
                float((i % 8) * 140 + 60),
                float((i // 8) * 160 + 80),
                shapes[i % len(shapes)],
                dests=(shapes[(i + 1) % len(shapes)],) * (i % 4),
                waits=(8_000 * (i % 4),) * (i % 4),
            )
            for i in range(32)
        )
        lns = tuple(
            _ln(
                f"L{k}",
                k,
                tuple(range(k * 4, k * 4 + 4)),
                loop=(k == 0),
                trains=2 if k < 2 else 1,
                cars=1 if k == 0 else 0,
            )
            for k in range(6)
        )
        view = OpenMetroView(
            decision=600,
            stations=sts,
            lines=lns,
            lines_available=1,
            locos_available=1,
            carriages_available=1,
            tunnels_available=4,
            interchanges_available=1,
        )
        state = OpenPlannerState()
        t0 = time.perf_counter()
        act = choose_open(view, state)
        dt_choose = time.perf_counter() - t0
        self.assertLess(dt_choose, 0.5)
        self.assertIsInstance(
            act,
            (
                bp.ConnectAction,
                bp.ExtendLineAction,
                bp.PrependLineAction,
                bp.InsertStationAction,
            ),
        )
        if isinstance(
            act, (bp.ExtendLineAction, bp.PrependLineAction, bp.InsertStationAction)
        ):
            self.assertGreaterEqual(int(act.station_id), 24)
        elif isinstance(act, bp.ConnectAction):
            self.assertTrue(any(int(s) >= 24 for s in act.station_ids))

        full_lns = tuple(
            _ln(f"L{k}", k, tuple(range(k * 5, min(32, k * 5 + 6))), trains=1)
            for k in range(6)
        )
        full_view = OpenMetroView(
            decision=1200,
            stations=sts,
            lines=full_lns,
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=4,
        )
        t1 = time.perf_counter()
        routes = optimize_network_open(full_view, max_outer=2)
        dt_opt = time.perf_counter() - t1
        self.assertEqual(len(routes), 6)
        self.assertLess(dt_opt, 1.0)


class TestOpenTopologyPlannerTunnelGating(unittest.TestCase):
    def test_zero_tunnels_blocks_water_crossings_and_allows_same_bank(self) -> None:
        sts = (
            _st("N1", 100.0, 100.0, "CIRCLE"),
            _st("N2", 220.0, 100.0, "TRIANGLE"),
            _st("S1", 150.0, 300.0, "SQUARE"),
            _st("S2", 260.0, 300.0, "STAR"),
        )
        cross = lambda u, v: 0 if str(u)[0] == str(v)[0] else 1  # noqa: E731
        v_conn = OpenMetroView(
            decision=10,
            stations=(sts[0], sts[2]),
            lines=(),
            lines_available=1,
            locos_available=1,
            carriages_available=0,
            tunnels_available=0,
            crossing_cost_fn=cross,
        )
        self.assertIsInstance(choose_open(v_conn, OpenPlannerState()), bp.WaitAction)

        v_same = OpenMetroView(
            decision=11,
            stations=sts[:3],
            lines=(),
            lines_available=1,
            locos_available=1,
            carriages_available=0,
            tunnels_available=0,
            crossing_cost_fn=cross,
        )
        act_c = choose_open(v_same, OpenPlannerState())
        self.assertIsInstance(act_c, bp.ConnectAction)
        assert isinstance(act_c, bp.ConnectAction)
        self.assertEqual(set(act_c.station_ids), {"N1", "N2"})

        v_blk_ext = OpenMetroView(
            decision=12,
            stations=sts[:3],
            lines=(_ln("L0", 0, ("N1", "N2"), trains=1),),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=0,
            crossing_cost_fn=cross,
        )
        self.assertIsInstance(choose_open(v_blk_ext, OpenPlannerState()), bp.WaitAction)

        v_ok_ext = OpenMetroView(
            decision=13,
            stations=sts[:3],
            lines=(_ln("L0", 0, ("N1", "N2"), trains=1),),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=1,
            crossing_cost_fn=cross,
        )
        self.assertIsInstance(
            choose_open(v_ok_ext, OpenPlannerState()),
            (bp.ExtendLineAction, bp.PrependLineAction, bp.InsertStationAction),
        )

    def test_net_zero_tunnel_delta_allows_interior_insert_with_zero_spare_tunnels(
        self,
    ) -> None:
        sts = (
            _st("N1", 100.0, 100.0, "CIRCLE"),
            _st("N2", 200.0, 110.0, "TRIANGLE"),
            _st("S1", 300.0, 300.0, "SQUARE"),
        )
        cross = lambda u, v: 0 if str(u)[0] == str(v)[0] else 1  # noqa: E731
        view = OpenMetroView(
            decision=20,
            stations=sts,
            lines=(_ln("L0", 0, ("N1", "S1"), tunnels=1, trains=1),),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=0,
            crossing_cost_fn=cross,
            supports_insert_station=True,
            can_remove_tracks=False,
        )
        act = choose_open(view, OpenPlannerState())
        self.assertIn(
            act,
            (
                bp.InsertStationAction("L0", "N2", "N1"),
                bp.PrependLineAction("L0", "N2", False),
            ),
        )


class TestOpenTopologyPlannerActionsAndNonMutation(unittest.TestCase):
    def test_relocate_loco_uncrewed_and_minimax_rebalance_and_donor_guards(
        self,
    ) -> None:
        sts = (
            _st(
                "A",
                100.0,
                100.0,
                "CIRCLE",
                dests=("TRIANGLE",) * 5,
                waits=(32_000,) * 5,
                ov=0.8,
            ),
            _st(
                "B",
                240.0,
                100.0,
                "TRIANGLE",
                dests=("CIRCLE",) * 4,
                waits=(28_000,) * 4,
                ov=0.7,
            ),
            _st("C", 100.0, 260.0, "SQUARE"),
            _st("D", 240.0, 260.0, "STAR"),
        )
        v_unc = OpenMetroView(
            decision=50,
            stations=sts,
            lines=(
                _ln("L0", 0, ("A", "B"), trains=0),
                _ln("L1", 1, ("C", "D"), trains=2, donor=True),
            ),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=2,
        )
        self.assertEqual(
            choose_open(v_unc, OpenPlannerState()),
            bp.RelocateLocoAction("L1", "L0"),
        )

        v_blocked = OpenMetroView(
            decision=51,
            stations=sts,
            lines=(
                _ln("L0", 0, ("A", "B"), trains=0),
                _ln("L1", 1, ("C", "D"), trains=2, donor=False),
                _ln("L2", 2, ("A", "C"), trains=3, donor=True, pending=True),
            ),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=2,
        )
        self.assertIsInstance(choose_open(v_blocked, OpenPlannerState()), bp.WaitAction)

        v_rebal = OpenMetroView(
            decision=500,
            stations=sts,
            lines=(
                _ln("L_starved", 0, ("A", "B"), trains=1, cap=6, onboard=5),
                _ln("L_calm", 1, ("C", "D"), trains=3, cap=18, onboard=0, donor=True),
            ),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=2,
        )
        self.assertEqual(
            choose_open(v_rebal, OpenPlannerState()),
            bp.RelocateLocoAction("L_calm", "L_starved"),
        )

    def test_insert_station_preferred_over_remove_and_can_remove_tracks_false(
        self,
    ) -> None:
        sts = (
            _st("S0", 100.0, 100.0, "CIRCLE"),
            _st("S1", 250.0, 105.0, "STAR"),
            _st("S2", 400.0, 100.0, "TRIANGLE"),
            _st("S3", 550.0, 100.0, "SQUARE"),
        )
        v_ins = OpenMetroView(
            decision=600,
            stations=sts,
            lines=(_ln("L0", 0, ("S0", "S2", "S3"), trains=1),),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=2,
            supports_insert_station=True,
            can_remove_tracks=True,
        )
        self.assertEqual(
            choose_open(v_ins, OpenPlannerState()),
            bp.InsertStationAction("L0", "S1", "S0"),
        )

        zig = (
            _st(0, 100.0, 100.0, "CIRCLE"),
            _st(1, 500.0, 100.0, "TRIANGLE"),
            _st(2, 200.0, 100.0, "SQUARE"),
            _st(3, 600.0, 100.0, "STAR"),
            _st(4, 300.0, 100.0, "PENTAGON"),
        )
        v_no_rm = OpenMetroView(
            decision=700,
            stations=zig,
            lines=(_ln("L0", 0, (0, 3, 1, 4, 2), trains=1),),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=2,
            supports_insert_station=False,
            can_remove_tracks=False,
        )
        act = choose_open(v_no_rm, OpenPlannerState())
        self.assertNotIsInstance(act, bp.RemoveLineAction)

    def test_place_interchange_on_congested_hub_station(self) -> None:
        sts = (
            _st("S0", 100.0, 100.0, "CIRCLE"),
            _st(
                "HUB",
                250.0,
                200.0,
                "TRIANGLE",
                dests=("SQUARE", "STAR", "CIRCLE", "PENTAGON", "DIAMOND"),
                waits=(25_000, 22_000, 20_000, 18_000, 16_000),
                ov=0.65,
            ),
            _st("S2", 400.0, 100.0, "SQUARE"),
            _st("S3", 250.0, 350.0, "STAR"),
        )
        view = OpenMetroView(
            decision=100,
            stations=sts,
            lines=(
                _ln("L0", 0, ("S0", "HUB", "S2"), trains=1),
                _ln("L1", 1, ("HUB", "S3"), trains=1),
            ),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=2,
            interchanges_available=1,
        )
        st = OpenPlannerState(
            last_replan_topology=(4, 2, ((0, 1, 2), (1, 3))),
            last_rebuild_decision=90,
        )
        self.assertEqual(choose_open(view, st), bp.PlaceInterchangeAction("HUB"))

    def test_loop_objective_and_input_immutability(self) -> None:
        pos = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
        shapes = ["CIRCLE", "TRIANGLE", "SQUARE", "STAR"]
        routes = [[0, 1, 2, 3]]
        c_unloop = compute_open_ml_objective(
            pos, shapes, routes, [False], ANCHOR_PARAMS
        )
        self.assertAlmostEqual(
            c_unloop,
            compute_ml_objective(pos, shapes, routes, ANCHOR_PARAMS),
            places=9,
        )
        c_loop = compute_open_ml_objective(pos, shapes, routes, [True], ANCHOR_PARAMS)
        self.assertNotAlmostEqual(c_loop, c_unloop)

        sts = tuple(_st(i, p[0], p[1], shapes[i]) for i, p in enumerate(pos))
        lns = (_ln("L0", 0, (0, 1, 2), loop=True, trains=1),)
        view = OpenMetroView(10, sts, lns, 1, 1, 1, 2, 1)
        before_s, before_l = view.stations, view.lines
        _ = choose_open(view, OpenPlannerState())
        self.assertEqual(view.stations, before_s)
        self.assertEqual(view.lines, before_l)


class TestOpenTopologyPlannerParityWithParametricCEM(unittest.TestCase):
    def test_objective_and_bootstrap_decision_parity_on_standard_boards(self) -> None:
        env = SemanticMetroEnv()
        try:
            for seed in (0, 1, 2):
                env.reset(seed=seed)
                view = _view_from_env(env)
                d_open = extract_open_demand(view)
                pos = [(s.x, s.y) for s in view.stations]
                sh = [s.shape for s in view.stations]
                rts = [[0, 1]]
                self.assertAlmostEqual(
                    compute_open_ml_objective(
                        pos, sh, rts, [False], BEST_PHASE1_PARAMS, d_open
                    ),
                    compute_ml_objective(pos, sh, rts, BEST_PHASE1_PARAMS, d_open),
                    places=9,
                )
                idx = choose(env, params=BEST_PHASE1_PARAMS)
                kind, a, b = ACTION_TABLE[idx]
                act = choose_open(view, OpenPlannerState(), BEST_PHASE1_PARAMS)
                self.assertEqual(kind, ActionKind.CONNECT)
                self.assertIsInstance(act, bp.ConnectAction)
                assert isinstance(act, bp.ConnectAction)
                sids = {env._mediator.stations[a].id, env._mediator.stations[b].id}
                self.assertEqual(set(act.station_ids), sids)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
