"""Tests for `ADD_TO_LINE`, `optimal_replan_open`, and selectable `RealGameAgent`."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rl.bridge_protocol as bp
from rl.open_topology_planner import OpenLine, OpenMetroView, OpenStation
from rl.optimal_replan_open import (
    OptimalReplanState,
    _first_add_to_line_step,
    choose_optimal_replan,
    match_lines_min_steps,
)
from rl.real_game_adapter import PLANNER_NAMES, RealGameAgent
from scripts.mock_metro_rpc import MockMetroBoard, MockMetroRpcServer
from scripts.play_real_metro import main as play_real_metro_main
from scripts.play_real_metro import run_replay_mock


def _st(sid: int | str, x: float, y: float, shape: str = "CIRCLE") -> OpenStation:
    return OpenStation(sid, x, y, shape)


def _ln(
    lid: int | str,
    c_idx: int,
    sids: tuple[int | str, ...],
    *,
    loop: bool = False,
    ghost: bool = False,
    tunnels: int = 0,
    trains: int = 1,
) -> OpenLine:
    return OpenLine(
        id=lid,
        color_index=c_idx,
        station_ids=sids,
        is_looped=loop,
        is_ghost=ghost,
        tunnels_used=tunnels,
        active_train_count=trains,
    )


class TestAddToLineProtocolAndMock(unittest.TestCase):
    def test_add_to_line_protocol_roundtrip_and_validation(self) -> None:
        act = bp.AddToLineAction(line_id=1, station_id=5, position=2, is_looped=False)
        req = bp.action_request(act, rid="a2l")
        parsed = bp.action_from_request(bp.RpcRequest.from_json(req.to_json()))
        self.assertEqual(parsed, act)

        with self.assertRaises(bp.ProtocolValidationError):
            bp.AddToLineAction(line_id=1, station_id=5, position=-1)

    def test_mock_server_add_to_line_head_middle_tail_and_loop(self) -> None:
        board = MockMetroBoard(
            initial_stations=12,
            max_stations=20,
            lines_total=3,
            locos=2,
            tunnels=4,
        )
        with MockMetroRpcServer(port=0, board=board) as srv:
            from rl.bridge_client import BridgeRpcClient, BridgeRpcError

            with BridgeRpcClient(srv.host, srv.port, timeout_s=3.0) as client:
                s0 = client.execute_action(bp.ConnectAction((1, 3)))
                lid = s0.lines[0].id

                # Insert at position 0 (head)
                s1 = client.execute_action(bp.AddToLineAction(lid, 5, position=0))
                self.assertEqual(s1.lines[0].station_ids, (5, 1, 3))

                # Insert at position 1 (interior)
                s2 = client.execute_action(bp.AddToLineAction(lid, 7, position=1))
                self.assertEqual(s2.lines[0].station_ids, (5, 7, 1, 3))

                # Insert at tail and close loop
                s3 = client.execute_action(
                    bp.AddToLineAction(lid, 2, position=4, is_looped=True)
                )
                self.assertTrue(s3.lines[0].is_looped)
                self.assertEqual(s3.lines[0].station_ids, (5, 7, 1, 3, 2))

                # Insert into looped line at position 0
                s4 = client.execute_action(bp.AddToLineAction(lid, 4, position=0))
                self.assertTrue(s4.lines[0].is_looped)
                self.assertEqual(s4.lines[0].station_ids, (4, 5, 7, 1, 3, 2))

                # Out-of-range position fails
                with self.assertRaises(BridgeRpcError) as ctx:
                    client.execute_action(bp.AddToLineAction(lid, 6, position=99))
                self.assertEqual(ctx.exception.error_code, bp.ErrorCode.INVALID_ARGS)

                # Interior is_looped=True on non-looped line fails; duplicate fails; 2-station loop fails
                s_open = client.execute_action(bp.ConnectAction((6, 8)))
                lid2 = [ln.id for ln in s_open.lines if ln.id != lid][0]
                with self.assertRaises(BridgeRpcError) as ctx2:
                    client.execute_action(bp.AddToLineAction(lid2, 6, position=2))
                self.assertEqual(ctx2.exception.error_code, bp.ErrorCode.INVALID_ARGS)

                s_open2 = client.execute_action(
                    bp.AddToLineAction(lid2, 10, position=2)
                )
                with self.assertRaises(BridgeRpcError) as ctx3:
                    client.execute_action(
                        bp.AddToLineAction(lid2, 12, position=1, is_looped=True)
                    )
                self.assertEqual(ctx3.exception.error_code, bp.ErrorCode.INVALID_ARGS)

                # Duplicate station on looped line fails
                with self.assertRaises(BridgeRpcError) as ctx4:
                    client.execute_action(bp.AddToLineAction(lid, 5, position=0))
                self.assertEqual(ctx4.exception.error_code, bp.ErrorCode.INVALID_ARGS)
                self.assertEqual(len(s_open2.lines), 2)


class TestShortestStepTransition(unittest.TestCase):
    def test_subsequence_forward_reverse_and_loop_use_add_to_line(self) -> None:
        # Forward subsequence
        step = _first_add_to_line_step([0, 2], False, [4, 0, 1, 2], False)
        self.assertEqual(step, (2, 4, 0, False))

        # Reversed subsequence: [0, 1] -> [3, 1, 0, 2] == reversed([2, 0, 1, 3])
        step_rev = _first_add_to_line_step([0, 1], False, [3, 1, 0, 2], False)
        self.assertEqual(step_rev, (2, 2, 0, False))

        # Looped rotation subsequence: [1, 2, 3] -> [4, 1, 2, 3] (rotation of [2, 3, 4, 1])
        step_lp = _first_add_to_line_step([1, 2, 3], True, [2, 3, 4, 1], True)
        self.assertEqual(step_lp, (1, 4, 0, False))

        # Open -> looped: interior insertion cannot close loop in 1 ADD_TO_LINE
        self.assertIsNone(_first_add_to_line_step([0, 1, 2], False, [0, 3, 1, 2], True))
        # Open -> looped: endpoint insertion via cyclic rotation CAN close loop in 1 ADD_TO_LINE
        step_close = _first_add_to_line_step([0, 1], False, [1, 2, 0], True)
        self.assertEqual(step_close, (1, 2, 0, True))

    def test_match_lines_prefers_minimal_steps_and_fewer_removals(self) -> None:
        cur = [([0, 1], False), ([2, 3], False)]
        tgt = [([2, 4, 3], False), ([5, 0, 1], False)]
        assign, un_c, un_t = match_lines_min_steps(cur, tgt)
        self.assertEqual(assign, {0: 1, 1: 0})
        self.assertEqual(un_c, [])
        self.assertEqual(un_t, [])


class TestOptimalReplanAgent(unittest.TestCase):
    def test_uses_only_add_to_line_connect_remove_and_recalculates_immediately(
        self,
    ) -> None:
        sts = (
            _st("A", 100.0, 100.0, "CIRCLE"),
            _st("B", 300.0, 100.0, "TRIANGLE"),
            _st("C", 200.0, 110.0, "SQUARE"),
        )
        # No cooldown: even at decision=1 (right after decision=0), spawning C triggers optimal placement.
        v = OpenMetroView(
            decision=1,
            stations=sts,
            lines=(_ln("L0", 0, ("A", "B"), trains=1),),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=3,
        )
        st = OptimalReplanState()
        act = choose_optimal_replan(v, st)
        self.assertIsInstance(act, bp.AddToLineAction)
        assert isinstance(act, bp.AddToLineAction)
        self.assertEqual(act.line_id, "L0")
        self.assertEqual(act.station_id, "C")
        self.assertEqual(act.position, 1)

    def test_recalculates_when_new_line_slot_unlocked(self) -> None:
        sts = (
            _st("A", 100.0, 100.0, "CIRCLE"),
            _st("B", 180.0, 100.0, "TRIANGLE"),
            _st("C", 450.0, 250.0, "SQUARE"),
            _st("D", 780.0, 400.0, "CIRCLE"),
            _st("E", 860.0, 400.0, "TRIANGLE"),
            _st("F", 940.0, 400.0, "CROSS"),
        )
        state = OptimalReplanState()
        v1 = OpenMetroView(
            decision=10,
            stations=sts,
            lines=(_ln("L0", 0, ("A", "B", "C", "D", "E", "F"), trains=1),),
            lines_available=0,
            locos_available=1,
            carriages_available=0,
            tunnels_available=3,
        )
        a1 = choose_optimal_replan(v1, state)
        self.assertNotIsInstance(
            a1, (bp.ExtendLineAction, bp.PrependLineAction, bp.InsertStationAction)
        )

        # Now a new line slot is unlocked! Even with no new station, it should recalculate & split/connect.
        v2 = OpenMetroView(
            decision=11,
            stations=sts,
            lines=(_ln("L0", 0, ("A", "B", "C", "D", "E", "F"), trains=1),),
            lines_available=1,
            locos_available=1,
            carriages_available=0,
            tunnels_available=3,
        )
        a2 = choose_optimal_replan(v2, state)
        self.assertIsInstance(a2, (bp.ConnectAction, bp.RemoveLineAction))

    def test_extreme_mode_and_tunnel_budget_conservation(self) -> None:
        sts = (
            _st("N1", 100.0, 100.0, "CIRCLE"),
            _st("N2", 300.0, 100.0, "TRIANGLE"),
            _st("N3", 200.0, 120.0, "SQUARE"),
            _st("S1", 200.0, 300.0, "CROSS"),
        )
        cross = lambda u, v: 0 if str(u)[0] == str(v)[0] else 1  # noqa: E731
        v_ext = OpenMetroView(
            decision=5,
            stations=sts,
            lines=(_ln("L0", 0, ("N1", "N2"), trains=1),),
            lines_available=0,
            locos_available=0,
            carriages_available=0,
            tunnels_available=0,
            crossing_cost_fn=cross,
            can_remove_tracks=False,
        )
        act = choose_optimal_replan(v_ext, OptimalReplanState())
        self.assertIsInstance(act, bp.AddToLineAction)
        assert isinstance(act, bp.AddToLineAction)
        self.assertEqual(act.station_id, "N3")

    def test_selectable_planner_and_end_to_end_mock_replay(self) -> None:
        self.assertIn("open_topology", PLANNER_NAMES)
        self.assertIn("optimal_replan", PLANNER_NAMES)
        with self.assertRaises(ValueError):
            RealGameAgent(planner="nonexistent")

        board = MockMetroBoard(initial_stations=6, max_stations=24, tunnels=4)
        summary = run_replay_mock(
            agent=RealGameAgent(planner="optimal_replan"),
            max_decisions=60,
            wait_ticks=2,
            board=board,
        )
        self.assertEqual(summary["decisions"], 60)
        self.assertGreater(summary["deliveries"], 0)
        counts = summary["action_counts"]
        self.assertEqual(counts.get("EXTEND_LINE", 0), 0)
        self.assertEqual(counts.get("PREPEND_LINE", 0), 0)
        self.assertEqual(counts.get("INSERT_STATION", 0), 0)
        self.assertGreater(counts.get("ADD_TO_LINE", 0) + counts.get("CONNECT", 0), 0)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = play_real_metro_main(
                ["--replay-mock", "--agent", "optimal_replan", "--max-decisions", "40"]
            )
        self.assertEqual(rc, 0)
        parsed = json.loads(buf.getvalue().strip())
        self.assertEqual(parsed["decisions"], 40)
        self.assertGreater(parsed["deliveries"], 0)
