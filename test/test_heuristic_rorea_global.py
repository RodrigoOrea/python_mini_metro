"""Contract tests for the global replanning combinatorial heuristic (rorea_global)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../scripts")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rl.heuristic_rorea_obj import compute_network_cost  # noqa: E402
from rl.semantic_env import ACTION_TABLE, ActionKind, SemanticMetroEnv  # noqa: E402


class RoreaGlobalHeuristicTest(unittest.TestCase):
    def test_optimize_line_order_untangles_backtracking_detour(self) -> None:
        from rl.heuristic_rorea_global import optimize_line_order

        positions = [
            (100.0, 500.0),
            (300.0, 500.0),
            (500.0, 500.0),
            (700.0, 500.0),
        ]
        shapes = ["CIRCLE", "TRIANGLE", "CIRCLE", "RECT"]
        # Station 1 (300, 500) was appended at the tail instead of inserted in the middle.
        tangled = [[0, 2, 3, 1]]
        cost_before = compute_network_cost(positions, shapes, tangled)

        improved = optimize_line_order(positions, shapes, tangled, line_idx=0)
        cost_after = compute_network_cost(positions, shapes, [improved])

        self.assertLess(cost_after, cost_before)
        self.assertIn(improved, ([0, 1, 2, 3], [3, 2, 1, 0]))

    def test_optimize_network_rebalances_stations_across_lines(self) -> None:
        from rl.heuristic_rorea_global import optimize_network

        # Horizontal Line 0 around y=300; Station 3 sits at (500, 700) right next to Line 1,
        # while Line 0 has a massive detour down to (500, 700).
        positions = [
            (200.0, 300.0),  # 0: CIRCLE (hub candidate)
            (500.0, 300.0),  # 1: TRIANGLE (Line 0)
            (800.0, 300.0),  # 2: RECT (Line 0)
            (500.0, 700.0),  # 3: CIRCLE (misplaced on Line 0, sits on Line 1 corridor)
            (200.0, 700.0),  # 4: RECT (Line 1)
            (800.0, 700.0),  # 5: TRIANGLE (Line 1)
        ]
        shapes = ["CIRCLE", "TRIANGLE", "RECT", "CIRCLE", "RECT", "TRIANGLE"]
        routes = [[0, 1, 3, 2], [0, 4, 5]]
        cost_before = compute_network_cost(positions, shapes, routes)

        optimized = optimize_network(positions, shapes, routes)
        cost_after = compute_network_cost(positions, shapes, optimized)

        self.assertLess(cost_after, cost_before)
        served_after = {s for r in optimized for s in r}
        self.assertEqual(served_after, {0, 1, 2, 3, 4, 5})
        # Station 3 should have moved off Line 0 onto Line 1.
        self.assertNotIn(3, optimized[0])
        self.assertIn(3, optimized[1])

    def test_rebuild_executor_handles_strictly_decreasing_target_route(self) -> None:
        from rl.heuristic_rorea_global import RebuildPlan, next_action_for_plan

        env = SemanticMetroEnv()
        try:
            env.reset(seed=9000)
            # Connect initial 3 stations and crew a metro so deliveries unlock station 4.
            env.step(ACTION_TABLE.index((ActionKind.CONNECT, 0, 1)))
            env.step(ACTION_TABLE.index((ActionKind.EXTEND_LINE, 0, 2)))
            env.step(ACTION_TABLE.index((ActionKind.ASSIGN_LOCOMOTIVE, 0, 0)))
            for _ in range(3000):
                if len(env._mediator.stations) >= 4:
                    break
                _, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break

            self.assertGreaterEqual(len(env._mediator.stations), 4)
            env.step(ACTION_TABLE.index((ActionKind.EXTEND_LINE, 0, 3)))

            target_route = [3, 2, 1, 0]  # strictly decreasing!
            target_id = env._mediator.paths[0].id
            plan = RebuildPlan.create(
                old_path_id=target_id,
                target_route=[3, 0, 2, 1],
            )
            plan_dec = RebuildPlan.create(
                old_path_id=target_id, target_route=target_route
            )
            self.assertEqual(set(plan_dec.target_route), set(target_route))

            # Execute plan using strictly legal masked actions in env.
            steps = 0
            while not plan.is_complete(env._mediator) and steps < 20:
                mask = env.action_masks()
                action = next_action_for_plan(env._mediator, mask, plan)
                self.assertIsNotNone(action)
                self.assertTrue(bool(mask[action]))
                _, _, terminated, truncated, info = env.step(action)
                self.assertTrue(info["applied"])
                self.assertFalse(terminated or truncated)
                steps += 1

            self.assertTrue(plan.is_complete(env._mediator))
            built_path = env._mediator.paths[-1]
            idx_by_id = {s.id: i for i, s in enumerate(env._mediator.stations)}
            actual = [idx_by_id[s.id] for s in built_path.stations]
            self.assertIn(actual, ([3, 0, 2, 1], [1, 2, 0, 3]))

            # Now execute strictly decreasing target route [3, 2, 1, 0] via rebuild too!
            plan2 = RebuildPlan.create(
                old_path_id=built_path.id, target_route=target_route
            )
            steps = 0
            while not plan2.is_complete(env._mediator) and steps < 20:
                mask = env.action_masks()
                action = next_action_for_plan(env._mediator, mask, plan2)
                self.assertIsNotNone(action)
                self.assertTrue(bool(mask[action]))
                _, _, terminated, truncated, info = env.step(action)
                self.assertTrue(info["applied"])
                self.assertFalse(terminated or truncated)
                steps += 1

            self.assertTrue(plan2.is_complete(env._mediator))
            built_path2 = env._mediator.paths[-1]
            actual2 = [idx_by_id[s.id] for s in built_path2.stations]
            self.assertIn(actual2, ([3, 2, 1, 0], [0, 1, 2, 3]))
        finally:
            env.close()

    def test_choose_acts_legally_and_resets_across_episodes(self) -> None:
        from rl.heuristic_rorea_global import choose

        env = SemanticMetroEnv(remove_min_age=50)
        try:
            for seed in (9000, 9001):
                env.reset(seed=seed)
                decisions = 0
                actions_taken = 0
                while decisions < 600:
                    mask = env.action_masks()
                    action = choose(env)
                    self.assertTrue(
                        bool(mask[action]),
                        f"choose returned illegal action {ACTION_TABLE[action]}",
                    )
                    if action != 0:
                        actions_taken += 1
                    _, _, terminated, truncated, info = env.step(action)
                    self.assertTrue(info["applied"])
                    decisions += 1
                    if terminated or truncated:
                        break
                self.assertGreater(actions_taken, 0)
        finally:
            env.close()

    def test_registered_in_evaluate_semantic(self) -> None:
        import evaluate_semantic

        self.assertIn("rorea_global", evaluate_semantic.CONTROLS)

    def test_global_rebuild_never_orphans_bridge_line(self) -> None:
        from rl.heuristic_rorea_global import _clears_threshold

        # Threshold gate must reject tiny (<12%) or negligible improvements.
        self.assertFalse(_clears_threshold(209.1, 199.1, min_abs=15.0, min_rel=0.12))
        self.assertTrue(_clears_threshold(200.0, 165.0, min_abs=15.0, min_rel=0.12))

    def test_eligible_lines_balanced_protects_short_stub_at_any_index(self) -> None:
        from types import SimpleNamespace

        from rl.heuristic_rorea_global import _eligible_lines_balanced

        mediator = SimpleNamespace(
            paths=[
                SimpleNamespace(stations=[0, 1, 2, 3, 4]),
                SimpleNamespace(stations=[5, 6]),  # stub in middle index!
                SimpleNamespace(stations=[7, 8, 9, 10]),
            ]
        )
        self.assertEqual(_eligible_lines_balanced(mediator), [1])


if __name__ == "__main__":
    unittest.main()
