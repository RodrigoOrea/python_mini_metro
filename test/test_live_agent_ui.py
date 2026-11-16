"""Track B tests: live AI player selector, MediatorEnvAdapter, LiveAgentRunner, and HUD."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np
import pygame

import main
from app_controller import AppController, AppScreen
from config import WEEK_LENGTH_STEPS, screen_color, screen_height, screen_width
from event.keyboard import KeyboardEvent
from event.mouse import MouseEvent
from event.type import KeyboardEventType, MouseEventType
from game_session import GameSession
from geometry.point import Point
from mediator import Mediator
from offers import Offer, OfferKind
from rendering.game_renderer import GameRenderer
from rl.live_agent_runner import (
    LIVE_PLAYER_IDS,
    LIVE_PLAYER_LABELS,
    LiveAgentRunner,
    MediatorEnvAdapter,
    available_player_ids,
    next_player_id,
)
from rl.semantic_env import ACTION_TABLE, ActionKind, SemanticMetroEnv
from ui.menu_screens import (
    draw_live_agent_hud,
    draw_title_screen,
    pause_menu_layout,
    title_layout,
)


def _fresh_surface() -> pygame.Surface:
    surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA, 32)
    surface.fill(screen_color)
    return surface


def _bytes(surface: pygame.Surface) -> bytes:
    return pygame.image.tobytes(surface, "RGBA")


def _region_bytes(surface: pygame.Surface, rect: pygame.Rect) -> bytes:
    return _bytes(surface.subsurface(rect).copy())


class TestLiveAgentTitleAndHudChrome(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_title_layout_with_player_appends_disjoint_onscreen_rect(self):
        base = title_layout(screen_width, screen_height)
        with_player = title_layout(screen_width, screen_height, include_player=True)
        self.assertEqual(
            tuple(base.keys()),
            ("new_game", "continue", "exit", "settings", "tutorial", "map"),
        )
        for key, rect in base.items():
            self.assertEqual(
                with_player[key], rect, f"{key!r} must stay byte-identical"
            )
        player_rect = with_player["player"]
        self.assertGreater(player_rect.top, with_player["map"].bottom)
        self.assertLessEqual(player_rect.bottom, screen_height)
        self.assertGreaterEqual(player_rect.left, 0)
        self.assertLessEqual(player_rect.right, screen_width)
        for key in base:
            self.assertFalse(player_rect.colliderect(with_player[key]))

    def test_draw_title_screen_supports_default_and_custom_player_labels(self):
        default_a = _fresh_surface()
        default_b = _fresh_surface()
        custom = _fresh_surface()
        draw_title_screen(default_a)
        draw_title_screen(default_b, player_label="Human")
        draw_title_screen(custom, player_label="Rorea ML")
        self.assertEqual(_bytes(default_a), _bytes(default_b))
        layout = title_layout(screen_width, screen_height, include_player=True)
        self.assertNotEqual(
            _region_bytes(default_a, layout["player"]),
            _region_bytes(custom, layout["player"]),
        )
        for key in ("new_game", "exit", "map"):
            self.assertEqual(
                _region_bytes(default_a, layout[key]),
                _region_bytes(custom, layout[key]),
            )

    def test_draw_live_agent_hud_only_renders_for_non_human_ai(self):
        blank = _fresh_surface()
        human_surface = _fresh_surface()
        draw_live_agent_hud(human_surface, None)
        draw_live_agent_hud(human_surface, {"player_id": "human", "label": "Human"})
        self.assertEqual(_bytes(human_surface), _bytes(blank))

        ai_a = _fresh_surface()
        ai_b = _fresh_surface()
        hud = {
            "player_id": "rorea_ml",
            "label": "Rorea ML",
            "deliveries": 14,
            "decisions": 25,
            "infer_ms": 0.42,
            "last_action": "CONNECT 0-1",
        }
        draw_live_agent_hud(ai_a, hud)
        draw_live_agent_hud(ai_b, hud)
        self.assertEqual(_bytes(ai_a), _bytes(ai_b))
        self.assertNotEqual(_bytes(ai_a), _bytes(blank))
        # Top-right corner painted; top-left HUD exclusion zone untouched.
        self.assertEqual(
            _region_bytes(ai_a, pygame.Rect(0, 0, 840, 250)),
            _region_bytes(blank, pygame.Rect(0, 0, 840, 250)),
        )
        self.assertNotEqual(
            _region_bytes(ai_a, pygame.Rect(screen_width - 500, 0, 500, 200)),
            _region_bytes(blank, pygame.Rect(screen_width - 500, 0, 500, 200)),
        )


class TestLiveAgentRunnerAndAdapter(unittest.TestCase):
    def test_player_ids_labels_and_optional_phase2_registration(self):
        self.assertEqual(
            LIVE_PLAYER_IDS,
            ("human", "heuristic", "rorea", "rorea_obj", "rorea_global", "rorea_ml"),
        )
        for pid in LIVE_PLAYER_IDS:
            self.assertIn(pid, LIVE_PLAYER_LABELS)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best_val.json"
            ckpt.write_text("{}", encoding="utf-8")
            with (
                patch(
                    "rl.live_agent_runner.importlib.util.find_spec",
                    return_value=object(),
                ),
                patch(
                    "rl.live_agent_runner._find_phase2_checkpoint", return_value=ckpt
                ),
            ):
                ids = available_player_ids()
                self.assertEqual(ids, (*LIVE_PLAYER_IDS, "rorea_phase2"))
                self.assertEqual(next_player_id("rorea_ml"), "rorea_phase2")
                self.assertEqual(next_player_id("rorea_phase2"), "human")

    def test_adapter_masks_and_remove_min_age_match_semantic_env(self):
        mediator = Mediator(seed=77)
        adapter = MediatorEnvAdapter(mediator, remove_min_age=60)
        sem_env = SemanticMetroEnv(remove_min_age=60)
        sem_env.reset(seed=77)
        np.testing.assert_array_equal(adapter.action_masks(), sem_env.action_masks())

        connect_idx = ACTION_TABLE.index((ActionKind.CONNECT, 0, 1))
        self.assertTrue(adapter.apply_action(connect_idx))
        adapter.advance_decision()
        remove_idx = ACTION_TABLE.index((ActionKind.REMOVE_LINE, 0, 0))
        self.assertFalse(adapter.action_masks()[remove_idx])
        for _ in range(59):
            adapter.advance_decision()
        self.assertFalse(adapter.action_masks()[remove_idx])
        adapter.advance_decision()
        self.assertTrue(adapter.action_masks()[remove_idx])

    def test_live_driving_rorea_ml_and_rorea_global_over_180_ticks(self):
        for player_id in (
            "rorea_ml",
            "rorea_global",
            "heuristic",
            "rorea",
            "rorea_obj",
        ):
            with self.subTest(player_id=player_id):
                mediator = Mediator(seed=42)
                runner = LiveAgentRunner(player_id, mediator)
                for _ in range(180):
                    mediator.increment_time(16)
                    runner.step(1)
                self.assertEqual(runner.decisions, 30)
                self.assertGreaterEqual(runner.actions_taken, 2)
                self.assertGreaterEqual(len(mediator.paths), 1)
                self.assertNotEqual(runner.last_action_name, "WAIT")
                summary = runner.hud_summary()
                self.assertEqual(
                    set(summary.keys()),
                    {
                        "player_id",
                        "label",
                        "deliveries",
                        "decisions",
                        "infer_ms",
                        "last_action",
                        "actions_taken",
                    },
                )
                self.assertEqual(summary["player_id"], player_id)
                self.assertGreaterEqual(float(summary["infer_ms"]), 0.0)

    def test_auto_resolve_offers_clears_week_boundary(self):
        mediator = Mediator(seed=42)
        mediator.week_calendar = True
        runner = LiveAgentRunner("rorea_ml", mediator)
        for _ in range(WEEK_LENGTH_STEPS):
            mediator.increment_time(16)
            runner.step(1)
        self.assertTrue(mediator.is_week_boundary_pending)
        self.assertTrue(mediator.current_offers)
        # Blocked until offers are resolved.
        before_decisions = runner.decisions
        self.assertEqual(runner.step(6), 0)
        self.assertEqual(runner.decisions, before_decisions)

        chosen = runner.auto_resolve_offers()
        self.assertIsInstance(chosen, Offer)
        self.assertIn(chosen.kind, set(OfferKind))
        self.assertEqual(mediator.current_offers, ())
        self.assertFalse(mediator.is_week_boundary_pending)
        self.assertFalse(mediator.is_paused)


class TestAppControllerLiveAgentIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def _build_real_game(self, map_id: str = "classic"):
        mediator = Mediator(seed=42)
        renderer = GameRenderer()
        session = GameSession(mediator, step_observer=renderer)
        session.prepare_layout(pygame.Surface((screen_width, screen_height)))
        return mediator, renderer, session

    def _click_title(self, controller: AppController, key: str) -> None:
        rect = title_layout(screen_width, screen_height, include_player=True)[key]
        controller.handle_event(
            MouseEvent(MouseEventType.MOUSE_UP, Point(rect.centerx, rect.centery))
        )

    def test_title_cycles_player_and_attaches_ai_on_new_game_and_restart(self):
        controller = AppController(self._build_real_game, start_state=AppScreen.TITLE)
        self.assertEqual(controller.current_player_id, "human")
        self.assertIsNone(controller.live_agent)
        self.assertIsNone(controller.live_agent_hud())

        seen = [controller.current_player_id]
        for _ in range(len(available_player_ids())):
            self._click_title(controller, "player")
            seen.append(controller.current_player_id)
        self.assertEqual(seen[:-1], list(available_player_ids()))
        self.assertEqual(seen[-1], "human")

        while controller.current_player_id != "rorea_ml":
            self._click_title(controller, "player")
        self._click_title(controller, "new_game")
        self.assertIs(controller.state, AppScreen.PLAYING)
        self.assertIsNotNone(controller.live_agent)
        self.assertEqual(controller.live_agent.player_id, "rorea_ml")

        # Pause menu -> Restart creates a fresh runner on the new mediator.
        first_runner = controller.live_agent
        controller.handle_event(
            KeyboardEvent(KeyboardEventType.KEY_UP, pygame.K_ESCAPE)
        )
        self.assertIs(controller.state, AppScreen.PAUSE_MENU)
        restart_rect = pause_menu_layout(screen_width, screen_height)["restart"]
        pos = Point(restart_rect.centerx, restart_rect.centery)
        controller.handle_event(MouseEvent(MouseEventType.MOUSE_DOWN, pos))
        controller.handle_event(MouseEvent(MouseEventType.MOUSE_UP, pos))
        self.assertIs(controller.state, AppScreen.PLAYING)
        self.assertIsNotNone(controller.live_agent)
        self.assertIsNot(controller.live_agent, first_runner)

    def test_ai_mode_blocks_board_drags_but_allows_speed_pause_and_auto_offers(self):
        controller = AppController(self._build_real_game, start_state=AppScreen.TITLE)
        while controller.current_player_id != "rorea_global":
            self._click_title(controller, "player")
        self._click_title(controller, "new_game")

        # Drag attempt on a real station must be ignored.
        station_pos = controller.mediator.stations[0].position
        controller.handle_event(MouseEvent(MouseEventType.MOUSE_DOWN, station_pos))
        self.assertFalse(controller.mediator.is_mouse_down)
        self.assertFalse(controller.mediator.is_creating_path)

        # Speed keyboard & button controls remain usable.
        controller.handle_event(KeyboardEvent(KeyboardEventType.KEY_UP, pygame.K_3))
        self.assertEqual(controller.mediator.game_speed_multiplier, 4)
        speed_2_btn = controller.mediator.speed_buttons[2]
        controller.handle_event(
            MouseEvent(MouseEventType.MOUSE_UP, speed_2_btn.position)
        )
        self.assertEqual(controller.mediator.game_speed_multiplier, 2)

        # Pending week offer auto-resolves and returns to PLAYING.
        controller.mediator.week_calendar = True
        for _ in range(WEEK_LENGTH_STEPS // 2 + 2):
            controller.mediator.increment_time(16)
        controller.reconcile_week_boundary()
        self.assertIs(controller.state, AppScreen.OFFER)
        controller.advance_live_agent(0)
        self.assertIs(controller.state, AppScreen.PLAYING)
        self.assertEqual(controller.mediator.current_offers, ())

    def test_human_mode_keeps_board_drags_and_offer_modal_unchanged(self):
        controller = AppController(self._build_real_game, start_state=AppScreen.TITLE)
        self._click_title(controller, "new_game")
        self.assertIsNone(controller.live_agent)
        station_pos = controller.mediator.stations[0].position
        controller.handle_event(MouseEvent(MouseEventType.MOUSE_DOWN, station_pos))
        self.assertTrue(controller.mediator.is_mouse_down)
        self.assertTrue(controller.mediator.is_creating_path)
        controller.handle_event(MouseEvent(MouseEventType.MOUSE_UP, Point(-1, -1)))
        # Human OFFER screen does NOT auto-resolve.
        controller.mediator.week_calendar = True
        for _ in range(WEEK_LENGTH_STEPS):
            controller.mediator.increment_time(16)
        controller.reconcile_week_boundary()
        controller.advance_live_agent(0)
        self.assertIs(controller.state, AppScreen.OFFER)
        self.assertTrue(controller.mediator.current_offers)

    def test_end_to_end_run_game_selects_ai_and_renders_live_hud(self):
        layout = title_layout(screen_width, screen_height, include_player=True)
        player_pt = layout["player"].center
        new_game_pt = layout["new_game"].center
        frames = [
            [
                SimpleNamespace(type=pygame.MOUSEBUTTONUP, pos=player_pt, button=1)
            ],  # human -> heuristic
            [SimpleNamespace(type=pygame.MOUSEBUTTONUP, pos=new_game_pt, button=1)],
            *([[]] * 15),
            [SimpleNamespace(type=pygame.QUIT)],
        ]
        built: list[Mediator] = []

        def _make_mediator(map_definition=None):
            m = Mediator(seed=42, map_definition=map_definition)
            built.append(m)
            return m

        hud_calls: list[dict] = []
        with contextlib.ExitStack() as stack:
            pygame_mock = stack.enter_context(patch("main.pygame"))
            stack.enter_context(patch("main.Mediator", side_effect=_make_mediator))
            stack.enter_context(patch("main.write_autosave", lambda m: None))
            stack.enter_context(patch("main.delete_autosave", lambda: None))
            stack.enter_context(patch("main.peek_autosave", lambda: False))
            stack.enter_context(patch("main.record_highscore", lambda m: None))
            stack.enter_context(
                patch(
                    "main.draw_live_agent_hud",
                    side_effect=lambda surface, hud: hud_calls.append(dict(hud)),
                )
            )
            pygame_mock.QUIT = pygame.QUIT
            pygame_mock.RESIZABLE = pygame.RESIZABLE
            pygame_mock.MOUSEBUTTONDOWN = pygame.MOUSEBUTTONDOWN
            pygame_mock.MOUSEBUTTONUP = pygame.MOUSEBUTTONUP
            pygame_mock.MOUSEMOTION = pygame.MOUSEMOTION
            pygame_mock.KEYUP = pygame.KEYUP
            window = pygame.Surface((screen_width, screen_height))
            game_surface = pygame.Surface((screen_width, screen_height))
            pygame_mock.display.set_mode.return_value = window
            pygame_mock.Surface.return_value = game_surface
            pygame_mock.time.Clock.return_value = SimpleNamespace(tick=lambda fps: 17)
            pygame_mock.transform = pygame.transform
            pygame_mock.draw = pygame.draw
            pygame_mock.font = pygame.font
            pygame_mock.event.get.side_effect = frames
            with self.assertRaises(SystemExit):
                main.run_game(start_state=AppScreen.TITLE)

        self.assertTrue(hud_calls, "live AI HUD must render during AI PLAYING frames")
        self.assertEqual(hud_calls[-1]["player_id"], "heuristic")
        self.assertGreaterEqual(hud_calls[-1]["decisions"], 2)
        self.assertGreaterEqual(len(built[-1].paths), 1)


if __name__ == "__main__":
    unittest.main()
