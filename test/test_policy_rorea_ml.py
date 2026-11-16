"""Contract tests for the parameterized Phase 1 global policy (`rl.policy_rorea_ml`)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from entity.passenger import Passenger  # noqa: E402
from rl.heuristic_rorea_global import choose as choose_rorea_global  # noqa: E402
from rl.policy_rorea_ml import (  # noqa: E402
    ANCHOR_PARAMS,
    FLEET_ABLATION_PARAMS,
    PARAM_NAMES,
    RoreaMLParams,
    choose,
    compute_ml_objective,
)
from rl.semantic_env import ACTION_TABLE, ActionKind, SemanticMetroEnv  # noqa: E402


def _unlock_stations(mediator, count: int) -> None:
    mediator.unlocked_num_stations = count
    mediator.stations = mediator.all_stations[:count]


def _unlock_two_paths(mediator) -> None:
    if mediator.get_unlocked_num_paths() < 2:
        mediator._progression.grant_free_path()
        mediator.update_unlocked_num_paths()


class TestPolicyRoreaMLContract(unittest.TestCase):
    def test_params_array_dict_roundtrip_and_clamping(self):
        arr = ANCHOR_PARAMS.to_array()
        self.assertEqual(len(arr), len(PARAM_NAMES))
        restored = RoreaMLParams.from_array(arr)
        self.assertEqual(restored, ANCHOR_PARAMS)

        extreme = RoreaMLParams(
            alpha_dist=99.0,
            beta_transfer=-5.0,
            w_station_wait=-1.0,
            demo_risk_weight=1000.0,
        ).clamped()
        self.assertLessEqual(extreme.alpha_dist, 4.0)
        self.assertGreaterEqual(extreme.beta_transfer, 0.1)
        self.assertGreaterEqual(extreme.w_station_wait, 0.0)
        self.assertLessEqual(extreme.demo_risk_weight, 300.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.json"
            extreme.save(path)
            loaded = RoreaMLParams.load(path)
            self.assertEqual(loaded, extreme)

    def test_anchor_params_match_rorea_global_step_by_step(self):
        """Zero-drift check: at ANCHOR_THETA every decision matches rorea_global."""
        for seed in (42, 9000):
            env_ref = SemanticMetroEnv(max_decisions=2500)
            env_ml = SemanticMetroEnv(max_decisions=2500)
            try:
                env_ref.reset(seed=seed)
                env_ml.reset(seed=seed)
                deliv_ref = 0.0
                deliv_ml = 0.0
                for step_idx in range(2500):
                    a_ref = choose_rorea_global(env_ref)
                    a_ml = choose(env_ml, ANCHOR_PARAMS)
                    self.assertEqual(
                        a_ml,
                        a_ref,
                        f"seed={seed} step={step_idx}: ANCHOR action {a_ml} "
                        f"({ACTION_TABLE[a_ml]}) != rorea_global {a_ref} ({ACTION_TABLE[a_ref]})",
                    )
                    _, r_ref, term_ref, trunc_ref, _ = env_ref.step(a_ref)
                    _, r_ml, term_ml, trunc_ml, _ = env_ml.step(a_ml)
                    deliv_ref += float(r_ref)
                    deliv_ml += float(r_ml)
                    self.assertEqual(term_ml, term_ref)
                    self.assertEqual(trunc_ml, trunc_ref)
                    if term_ref or trunc_ref:
                        break
                self.assertEqual(deliv_ml, deliv_ref)
            finally:
                env_ref.close()
                env_ml.close()

    def test_returned_actions_always_legal_under_active_params(self):
        active = RoreaMLParams(
            w_shape_rarity=0.8,
            w_station_pax=1.5,
            w_station_wait=4.0,
            w_line_overload=1.0,
            demo_risk_weight=80.0,
            w_fleet_wait=4.0,
            w_fleet_load=2.0,
            w_rebalance=2.5,
            hub_share_bonus=2.0,
        )
        env = SemanticMetroEnv(max_decisions=2000)
        try:
            env.reset(seed=1002)
            for _ in range(2000):
                mask = env.action_masks()
                action = choose(env, active)
                self.assertTrue(
                    bool(mask[action]),
                    f"Illegal action {action} {ACTION_TABLE[action]} returned",
                )
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_carriage_selection_targets_congested_line_over_line_zero(self):
        env = SemanticMetroEnv()
        try:
            env.reset(seed=1000)
            m = env._mediator
            # Ensure at least 4 stations exist and create 2 distinct paths.
            # Advance or add stations if needed, or connect pairs cleanly.
            _unlock_stations(m, 4)
            _unlock_two_paths(m)
            env.step(ACTION_TABLE.index((ActionKind.CONNECT, 0, 1)))
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 0, 0)))
            env.step(ACTION_TABLE.index((ActionKind.CONNECT, 2, 3)))
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 1, 0)))

            # Give an extra spare carriage and overload station 2 on Line 1 with slow wait.
            m.num_carriages = max(m.num_carriages, m.assigned_carriages + 1)
            target_shape = m.stations[0].shape
            for _ in range(8):
                px = Passenger(target_shape)
                px.wait_ms = 31_000
                m.stations[2].passengers.append(px)

            # Clear waiting passengers on Line 0 so Line 0 is calm, and drain spare locos.
            m.stations[0].passengers.clear()
            m.stations[1].passengers.clear()
            m.num_metros = len(m.metros)
            m.line_credits = 0

            base_action = choose_rorea_global(env)
            ml_action = choose(env, FLEET_ABLATION_PARAMS)

            self.assertEqual(
                ACTION_TABLE[base_action][:2], (ActionKind.ATTACH_CARRIAGE, 0)
            )
            self.assertEqual(
                ACTION_TABLE[ml_action][:2], (ActionKind.ATTACH_CARRIAGE, 1)
            )
        finally:
            env.close()

    def test_demolition_risk_veto_blocks_remove_line_under_high_wait(self):
        env = SemanticMetroEnv()
        try:
            env.reset(seed=9000)
            m = env._mediator
            # Create a deliberately bad zigzag on Line 0 that satisfies static rebuild gain.
            _unlock_stations(m, 5)
            pts = sorted(range(5), key=lambda i: m.stations[i].position.left)
            bad_order = [pts[0], pts[4], pts[1], pts[3]]
            env.step(
                ACTION_TABLE.index(
                    (ActionKind.CONNECT, min(bad_order[:2]), max(bad_order[:2]))
                )
            )
            for st_idx in bad_order[2:]:
                idx = ACTION_TABLE.index((ActionKind.EXTEND_LINE, 0, st_idx))
                env.step(idx)
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 0, 0)))
            # Advance past remove_min_age so REMOVE_LINE is legal.
            for _ in range(env.remove_min_age + 5):
                env.step(0)

            # Attach an urgent waiting passenger on a station served by Line 0.
            dest_shape = m.stations[pts[2]].shape
            urgent_px = Passenger(dest_shape)
            urgent_px.wait_ms = 35_000
            m.stations[bad_order[1]].passengers.append(urgent_px)

            # Static anchor wants to REMOVE_LINE(0) when placing unserved pts[2];
            # high demo_risk_weight must avoid REMOVE_LINE and pick zero-downtime action.
            m.line_credits = 0
            a_anchor = choose(env, ANCHOR_PARAMS)
            self.assertEqual(ACTION_TABLE[a_anchor][0], ActionKind.REMOVE_LINE)
            env._rorea_ml_state = None
            risk_params = RoreaMLParams(demo_risk_weight=150.0)
            a_risk = choose(env, risk_params)
            self.assertNotEqual(ACTION_TABLE[a_risk][0], ActionKind.REMOVE_LINE)
        finally:
            env.close()

    def test_dynamic_locomotive_rebalance_moves_surplus_to_starved_line(self):
        env = SemanticMetroEnv()
        try:
            env.reset(seed=1002)
            m = env._mediator
            _unlock_stations(m, 4)
            _unlock_two_paths(m)
            env.step(ACTION_TABLE.index((ActionKind.CONNECT, 0, 1)))
            env.step(ACTION_TABLE.index((ActionKind.CONNECT, 2, 3)))
            # Put 2 locomotives on Line 0 and 1 on Line 1, exhausting reserve pool (4 metros: give 3 to L0 or adjust).
            m.num_metros = 3
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 0, 0)))
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 0, 0)))
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 1, 0)))
            m.num_carriages = m.assigned_carriages  # no spare carriages
            # Keep wait calm on Line 0 and overload Line 1.
            for st_i in (0, 1):
                m.stations[st_i].passengers.clear()
            dest = m.stations[0].shape
            for _ in range(9):
                px = Passenger(dest)
                px.wait_ms = 28_000
                m.stations[2].passengers.append(px)

            rebal_params = RoreaMLParams(
                w_fleet_wait=6.0, w_fleet_load=3.0, w_rebalance=4.0
            )
            _ = choose(env, rebal_params)
            self.assertTrue(
                any(
                    getattr(metro, "is_unassignment_queued", False)
                    for metro in m.paths[0].metros
                )
                or m.available_locomotives > 0,
                "Surplus Line 0 metro should be queued/returned for starved Line 1",
            )
            self.assertGreaterEqual(len(m.paths[1].metros), 1)
        finally:
            env.close()

    def test_shared_hub_objective_reduction(self):
        positions = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (200.0, 100.0)]
        shapes = ["RECT", "CIRCLE", "TRIANGLE", "DIAMOND"]
        disjoint = [[0, 1], [2, 3]]
        connected = [[0, 1], [1, 2, 3]]
        c_disjoint = compute_ml_objective(positions, shapes, disjoint, ANCHOR_PARAMS)
        c_connected = compute_ml_objective(positions, shapes, connected, ANCHOR_PARAMS)
        self.assertLess(c_connected, c_disjoint)

    def test_planner_state_resets_cleanly_on_env_reset(self):
        env = SemanticMetroEnv(max_decisions=500)
        try:
            env.reset(seed=42)
            for _ in range(20):
                a = choose(env, ANCHOR_PARAMS)
                env.step(a)
            env.reset(seed=99)
            a2 = choose(env, ANCHOR_PARAMS)
            self.assertTrue(bool(env.action_masks()[a2]))
            state = getattr(env, "_rorea_ml_state")
            self.assertEqual(state.last_decision, 0)
            self.assertIsNone(state.active_plan)
        finally:
            env.close()

    def test_rebalance_skips_carriage_metro_and_pins_target(self):
        env = SemanticMetroEnv()
        try:
            env.reset(seed=1002)
            m = env._mediator
            _unlock_stations(m, 4)
            _unlock_two_paths(m)
            env.step(ACTION_TABLE.index((ActionKind.CONNECT, 0, 1)))
            env.step(ACTION_TABLE.index((ActionKind.CONNECT, 2, 3)))
            m.num_metros = 3
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 0, 0)))
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 0, 0)))
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 1, 0)))
            m.num_carriages = 2
            env.step(ACTION_TABLE.index((ActionKind.ATTACH_CARRIAGE, 0, 0)))
            env.step(ACTION_TABLE.index((ActionKind.ATTACH_CARRIAGE, 0, 0)))
            for st_i in (0, 1):
                m.stations[st_i].passengers.clear()
            dest = m.stations[0].shape
            for _ in range(9):
                px = Passenger(dest)
                px.wait_ms = 18_000
                m.stations[2].passengers.append(px)

            cand = m._fleet.unassignment_candidate(m, m.paths[0])
            self.assertIsNotNone(cand)
            self.assertGreater(len(cand.carriages), 0)
            rebal_params = RoreaMLParams(
                w_fleet_wait=6.0, w_fleet_load=3.0, w_rebalance=4.0
            )
            _ = choose(env, rebal_params)
            self.assertFalse(
                any(
                    getattr(metro, "is_unassignment_queued", False)
                    for metro in m.paths[0].metros
                ),
                "Moderate rebalance must not strip carriages from donor metro",
            )

            # Without carriages on donor candidate, rebalance fires and pins target.
            cand.carriages.clear()
            for _ in range(5):
                px = Passenger(dest)
                px.wait_ms = 26_000
                m.stations[2].passengers.append(px)
            env._rorea_ml_state = None
            _ = choose(env, rebal_params)
            state = env._rorea_ml_state
            self.assertEqual(state.pending_rebalance_target_id, m.paths[1].id)

            # Even if Line 0 looks busy when locomotive lands, it routes to pinned target.
            m.num_metros += 1
            for _ in range(12):
                px = Passenger(m.stations[2].shape)
                px.wait_ms = 32_000
                m.stations[0].passengers.append(px)
            action = choose(env, rebal_params)
            self.assertEqual(ACTION_TABLE[action], (ActionKind.ASSIGN_LOCOMOTIVE, 1, 0))
            self.assertIsNone(state.pending_rebalance_target_id)
        finally:
            env.close()

    def test_crisis_blocks_discretionary_global_rebuild_demolition(self):
        env = SemanticMetroEnv()
        try:
            env.reset(seed=9000)
            m = env._mediator
            _unlock_stations(m, 5)
            pts = sorted(range(5), key=lambda i: m.stations[i].position.left)
            bad_order = [pts[0], pts[4], pts[1], pts[3], pts[2]]
            env.step(
                ACTION_TABLE.index(
                    (ActionKind.CONNECT, min(bad_order[:2]), max(bad_order[:2]))
                )
            )
            for st_idx in bad_order[2:]:
                env.step(ACTION_TABLE.index((ActionKind.EXTEND_LINE, 0, st_idx)))
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 0, 0)))
            for _ in range(env.remove_min_age + 5):
                env.step(0)
            for st in m.stations:
                st.passengers.clear()

            px = Passenger(m.stations[pts[1]].shape)
            px.wait_ms = 32_000
            m.stations[pts[0]].passengers.append(px)

            a_anchor = choose(env, ANCHOR_PARAMS)
            self.assertEqual(ACTION_TABLE[a_anchor][0], ActionKind.REMOVE_LINE)
            env._rorea_ml_state = None
            a_risk = choose(env, RoreaMLParams(demo_risk_weight=80.0))
            self.assertNotEqual(ACTION_TABLE[a_risk][0], ActionKind.REMOVE_LINE)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
