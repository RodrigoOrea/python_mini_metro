"""Unit and TCP integration tests for bridge client, real-game adapter, and CLI runner."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import rl.bridge_protocol as bp  # noqa: E402
from rl.bridge_client import BridgeRpcClient, BridgeRpcError  # noqa: E402
from rl.real_game_adapter import (  # noqa: E402
    SHAPE_TO_ROREA,
    RealGameAgent,
    choose_weekly_upgrade,
    map_shape_to_rorea,
    snapshot_to_open_view,
)
from scripts.mock_metro_rpc import MockMetroBoard, MockMetroRpcServer  # noqa: E402
from scripts.play_real_metro import (  # noqa: E402
    main as play_real_metro_main,
)
from scripts.play_real_metro import (
    run_dry_run,
    run_live_session,
    run_replay_mock,
)


def _sample_snapshot_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocol": bp.PROTOCOL_VERSION,
        "seq": 10,
        "timestamp_ms": 5000,
        "screen": "PLAYING",
        "city_id": "london",
        "is_paused": False,
        "speed_multiplier": 1.0,
        "deliveries": 12,
        "day": 2,
        "camera": {
            "center_x": 0.0,
            "center_y": 0.0,
            "ortho_size": 540.0,
            "screen_width": 1920,
            "screen_height": 1080,
            "world_to_ref_scale": 1.5,
        },
        "inventory": {
            "lines_available": 2,
            "lines_total": 5,
            "locos_available": 1,
            "carriages_available": 2,
            "tunnels_available": 3,
            "interchanges_available": 1,
        },
        "weekly_offers": [],
        "stations": [
            {
                "id": 1,
                "x": -100.0,
                "y": 40.0,
                "screen_x": 860.0,
                "screen_y": 580.0,
                "shape": "SQUARE",
                "is_interchange": True,
                "overload_progress": 0.3,
                "waiting_passengers": [
                    {"dest_shape": "SQUARE", "wait_ms": 2500},
                    {"dest_shape": "STAR", "wait_ms": 4000},
                ],
                "peep_capacity": 18,
            },
            {
                "id": 2,
                "x": 200.0,
                "y": -60.0,
                "screen_x": 1160.0,
                "screen_y": 480.0,
                "shape": "CIRCLE",
                "is_interchange": False,
                "overload_progress": 0.0,
                "waiting_passengers": [
                    {"dest_shape": "TRIANGLE", "wait_ms": 1200},
                ],
                "peep_capacity": 6,
            },
            {
                "id": 3,
                "x": 300.0,
                "y": 80.0,
                "screen_x": 1260.0,
                "screen_y": 620.0,
                "shape": "TRIANGLE",
                "is_interchange": False,
                "overload_progress": 0.1,
                "waiting_passengers": [],
                "peep_capacity": 6,
            },
        ],
        "lines": [
            {
                "id": 10,
                "color_index": 0,
                "station_ids": [1, 2],
                "is_looped": False,
                "is_ghost": False,
                "tunnels_used": 1,
            },
            {
                "id": 20,
                "color_index": 1,
                "station_ids": [2, 3],
                "is_looped": False,
                "is_ghost": False,
                "tunnels_used": 0,
            },
            {
                "id": 30,
                "color_index": 2,
                "station_ids": [1, 3],
                "is_looped": False,
                "is_ghost": True,
                "tunnels_used": 1,
            },
        ],
        "trains": [
            {
                "id": 101,
                "line_id": 10,
                "carriages": 1,
                "capacity": 12,
                "onboard_count": 4,
                "is_unassignment_queued": False,
                "is_ghost": False,
                "is_waiting_for_locomotive": False,
                "loco_type": "LOCOMOTIVE",
                "is_pending": False,
            },
            {
                "id": 102,
                "line_id": 10,
                "carriages": 2,
                "capacity": 18,
                "onboard_count": 5,
                "is_unassignment_queued": False,
                "is_ghost": False,
                "is_waiting_for_locomotive": False,
                "loco_type": "SHINKANSEN",
                "is_pending": False,
            },
            {
                "id": 103,
                "line_id": 10,
                "carriages": 3,
                "capacity": 24,
                "onboard_count": 9,
                "is_unassignment_queued": False,
                "is_ghost": True,
                "is_waiting_for_locomotive": False,
                "loco_type": "LOCOMOTIVE",
                "is_pending": False,
            },
            {
                "id": 201,
                "line_id": 20,
                "carriages": 0,
                "capacity": 6,
                "onboard_count": 2,
                "is_unassignment_queued": False,
                "is_ghost": False,
                "is_waiting_for_locomotive": False,
                "loco_type": "LOCOMOTIVE",
                "is_pending": False,
            },
            {
                "id": 202,
                "line_id": 20,
                "carriages": 1,
                "capacity": 12,
                "onboard_count": 0,
                "is_unassignment_queued": False,
                "is_ghost": False,
                "is_waiting_for_locomotive": True,
                "loco_type": "LOCOMOTIVE",
                "is_pending": True,
            },
        ],
        "crossing_costs": [
            {"u": 1, "v": 2, "tunnels": 1},
            {"u": 1, "v": 3, "tunnels": 1},
        ],
    }
    base.update(overrides)
    return base


class RealGameAdapterUnitTests(unittest.TestCase):
    def test_shape_to_rorea_maps_square_to_rect_and_preserves_others(self) -> None:
        self.assertEqual(SHAPE_TO_ROREA["SQUARE"], "RECT")
        self.assertEqual(map_shape_to_rorea("SQUARE"), "RECT")
        for shape in (
            "CIRCLE",
            "TRIANGLE",
            "CROSS",
            "DIAMOND",
            "EGG",
            "GEM",
            "PENTAGON",
            "STAR",
            "WEDGE",
        ):
            self.assertEqual(SHAPE_TO_ROREA[shape], shape)
            self.assertEqual(map_shape_to_rorea(shape), shape)

    def test_snapshot_to_open_view_scales_coords_filters_ghosts_and_wires_trains(
        self,
    ) -> None:
        snap = bp.Snapshot.from_dict(_sample_snapshot_dict())
        view = snapshot_to_open_view(snap, decision=17)

        self.assertEqual(view.decision, 17)
        self.assertEqual(len(view.stations), 3)
        st1 = view.stations[0]
        self.assertEqual(st1.id, 1)
        self.assertAlmostEqual(st1.x, -150.0)
        self.assertAlmostEqual(st1.y, 60.0)
        self.assertEqual(st1.shape, "RECT")
        self.assertEqual(st1.waiting_dest_shapes, ("RECT", "STAR"))
        self.assertEqual(st1.waiting_wait_ms, (2500, 4000))
        self.assertEqual(st1.peep_capacity, 18)
        self.assertTrue(st1.is_interchange)
        self.assertAlmostEqual(st1.overload_progress, 0.3)

        # Ghost line 30 is filtered out by default
        self.assertEqual([ln.id for ln in view.lines], [10, 20])
        ln10, ln20 = view.lines[0], view.lines[1]

        # Line 10 has 2 active non-ghost trains and 1 ignored ghost train
        self.assertEqual(ln10.active_train_count, 2)
        self.assertEqual(ln10.carriage_count, 3)
        self.assertEqual(ln10.total_capacity, 30)
        self.assertEqual(ln10.onboard_count, 9)
        self.assertFalse(ln10.has_pending_train)
        self.assertTrue(ln10.can_relocate_donor)

        # Line 20 has 1 active train + 1 pending/waiting train -> has_pending_train=True, donor=False
        self.assertEqual(ln20.active_train_count, 1)
        self.assertEqual(ln20.carriage_count, 1)
        self.assertEqual(ln20.total_capacity, 18)
        self.assertEqual(ln20.onboard_count, 2)
        self.assertTrue(ln20.has_pending_train)
        self.assertFalse(ln20.can_relocate_donor)

        # Crossing cost fn wired accurately
        self.assertIsNotNone(view.crossing_cost_fn)
        assert view.crossing_cost_fn is not None
        self.assertEqual(view.crossing_cost_fn(1, 2), 1)
        self.assertEqual(view.crossing_cost_fn(2, 1), 1)
        self.assertEqual(view.crossing_cost_fn(2, 3), 0)

    def test_choose_weekly_upgrade_priorities_and_two_stage_resolution(self) -> None:
        # Priority 1: LINE when lines_total < max_lines_cap
        s_line = bp.Snapshot.from_dict(
            _sample_snapshot_dict(
                screen="WEEKLY_UPGRADE",
                weekly_offers=[
                    {"index": 0, "kind": "CARRIAGE", "label": "Carriage", "count": 1},
                    {"index": 1, "kind": "LINE", "label": "Line", "count": 1},
                    {"index": 2, "kind": "CROSSING", "label": "Tunnel", "count": 1},
                ],
            )
        )
        self.assertEqual(choose_weekly_upgrade(s_line, max_lines_cap=7).offer_index, 1)

        # Priority 2: CROSSING when lines capped and tunnels_available <= 1
        s_tunnel = bp.Snapshot.from_dict(
            _sample_snapshot_dict(
                screen="WEEKLY_UPGRADE",
                inventory={
                    "lines_available": 0,
                    "lines_total": 7,
                    "locos_available": 1,
                    "carriages_available": 1,
                    "tunnels_available": 1,
                    "interchanges_available": 0,
                },
                weekly_offers=[
                    {"index": 0, "kind": "LINE", "label": "Line", "count": 1},
                    {"index": 1, "kind": "CARRIAGE", "label": "Carriage", "count": 1},
                    {"index": 2, "kind": "CROSSING", "label": "Tunnel", "count": 2},
                ],
            )
        )
        self.assertEqual(
            choose_weekly_upgrade(s_tunnel, max_lines_cap=7).offer_index, 2
        )

        # Priority 3: SHINKANSEN / LOCOMOTIVE / TRAM when tunnels > 1 and lines capped
        s_loco = bp.Snapshot.from_dict(
            _sample_snapshot_dict(
                screen="WEEKLY_UPGRADE",
                inventory={
                    "lines_available": 0,
                    "lines_total": 7,
                    "locos_available": 0,
                    "carriages_available": 0,
                    "tunnels_available": 3,
                    "interchanges_available": 0,
                },
                weekly_offers=[
                    {"index": 0, "kind": "CARRIAGE", "label": "Carriage", "count": 1},
                    {"index": 1, "kind": "SHINKANSEN", "label": "Bullet", "count": 1},
                ],
            )
        )
        self.assertEqual(choose_weekly_upgrade(s_loco, max_lines_cap=7).offer_index, 1)

        # Priority 4 & 5 & 6: CARRIAGE -> INTERCHANGE -> fallback[0]
        s_car = bp.Snapshot.from_dict(
            _sample_snapshot_dict(
                screen="WEEKLY_UPGRADE",
                inventory={
                    "lines_available": 0,
                    "lines_total": 7,
                    "locos_available": 0,
                    "carriages_available": 0,
                    "tunnels_available": 3,
                    "interchanges_available": 0,
                },
                weekly_offers=[
                    {"index": 4, "kind": "INTERCHANGE", "label": "IC", "count": 1},
                    {"index": 7, "kind": "CARRIAGE", "label": "Carriage", "count": 1},
                ],
            )
        )
        self.assertEqual(choose_weekly_upgrade(s_car, max_lines_cap=7).offer_index, 7)

    def test_real_game_agent_screen_handling(self) -> None:
        agent = RealGameAgent()
        for inactive in ("MENU", "GAME_OVER", "OTHER"):
            snap = bp.Snapshot.from_dict(_sample_snapshot_dict(screen=inactive))
            self.assertIsNone(agent.choose_for_snapshot(snap))

        up_snap = bp.Snapshot.from_dict(
            _sample_snapshot_dict(
                screen="WEEKLY_UPGRADE",
                weekly_offers=[
                    {"index": 0, "kind": "LOCOMOTIVE", "label": "+1 Loco", "count": 1}
                ],
            )
        )
        act = agent.choose_for_snapshot(up_snap)
        self.assertIsInstance(act, bp.PickUpgradeAction)


class BridgeClientAndRunnerIntegrationTests(unittest.TestCase):
    def test_bridge_rpc_client_methods_and_error_handling(self) -> None:
        board = MockMetroBoard(initial_stations=6, ticks_per_week=10)
        with MockMetroRpcServer(board=board) as srv:
            with BridgeRpcClient(srv.host, srv.port, timeout_s=3.0) as client:
                ping = client.ping()
                self.assertTrue(ping["pong"])

                s0 = client.get_state()
                self.assertEqual(s0.screen, bp.ScreenKind.PLAYING)

                s_paused = client.set_paused(True)
                self.assertTrue(s_paused.is_paused)
                self.assertEqual(s_paused.screen, bp.ScreenKind.PAUSED)

                s_speed = client.set_speed(2.0)
                self.assertEqual(s_speed.speed_multiplier, 2.0)
                self.assertFalse(s_speed.is_paused)

                # Invalid PickUpgradeAction while PLAYING raises BridgeRpcError
                with self.assertRaises(BridgeRpcError) as ctx:
                    client.execute_action(bp.PickUpgradeAction(0))
                self.assertEqual(ctx.exception.error_code, bp.ErrorCode.INVALID_STATE)
                self.assertIsNotNone(ctx.exception.response)

    def test_dry_run_and_replay_mock_over_60_decisions(self) -> None:
        board = MockMetroBoard(initial_stations=6, max_stations=36, tunnels=3)
        with MockMetroRpcServer(board=board) as srv:
            with BridgeRpcClient(srv.host, srv.port, timeout_s=3.0) as client:
                dry = run_dry_run(client)
                self.assertEqual(dry["mode"], "dry_run")
                self.assertTrue(dry["ping_ok"])
                self.assertEqual(dry["actions_taken"], 0)
                self.assertEqual(dry["lines"], 0)
                self.assertEqual(dry["chosen_action"]["kind"], "CONNECT")

                live = run_live_session(
                    client,
                    max_decisions=65,
                    wait_ticks=2,
                    speed=2.0,
                )
                self.assertGreaterEqual(live["decisions"], 60)
                self.assertGreater(live["deliveries"], 0)
                self.assertGreaterEqual(live["lines"], 2)
                self.assertGreaterEqual(live["connected_stations"], 6)
                self.assertGreaterEqual(live["upgrades_picked"], 2)
                counts = live["action_counts"]
                self.assertGreater(counts.get("CONNECT", 0), 0)
                self.assertGreater(
                    counts.get("EXTEND_LINE", 0)
                    + counts.get("PREPEND_LINE", 0)
                    + counts.get("INSERT_STATION", 0),
                    0,
                )
                self.assertGreaterEqual(counts.get("PICK_UPGRADE", 0), 2)

        # Also verify CLI entry point with --replay-mock
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = play_real_metro_main(["--replay-mock", "--max-decisions", "60"])
        self.assertEqual(rc, 0)
        parsed = json.loads(buf.getvalue().strip())
        self.assertEqual(parsed["mode"], "replay_mock")
        self.assertEqual(parsed["decisions"], 60)
        self.assertGreater(parsed["deliveries"], 0)
        self.assertGreaterEqual(parsed["upgrades_picked"], 2)

        # Verify direct helper run_replay_mock as well
        direct = run_replay_mock(max_decisions=20, wait_ticks=2)
        self.assertEqual(direct["decisions"], 20)
        self.assertGreater(direct["deliveries"], 0)


if __name__ == "__main__":
    unittest.main()
