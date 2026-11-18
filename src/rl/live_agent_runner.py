"""Live interactive AI player bridge over a running Mediator."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import time
import types
from pathlib import Path

from event.mouse import MouseEvent
from event.type import MouseEventType
from offers import Offer, OfferKind
from rl.semantic_env import (
    ACTION_TABLE,
    TICKS_PER_DECISION,
    ActionKind,
    SemanticMetroEnv,
)

LIVE_PLAYER_LABELS = {
    "human": "Human",
    "heuristic": "Heuristic",
    "multiline_balancing": "Multiline Balancing",
    "load_objective": "Load Objective",
    "global_replan": "Global Replan",
    "parametric_cem": "Parametric CEM",
    "contextual_residual": "Contextual Residual",
}
LIVE_PLAYER_IDS = tuple(k for k in LIVE_PLAYER_LABELS if k != "contextual_residual")
_CHOOSER_MODULES = {
    "heuristic": "rl.heuristic",
    "multiline_balancing": "rl.heuristic_multiline",
    "load_objective": "rl.heuristic_load_objective",
    "global_replan": "rl.heuristic_global_replan",
    "parametric_cem": "rl.policy_parametric_cem",
}
_PHASE2_PATHS = (
    Path("output/phase2_rl/best_val.json"),
    Path("output/phase2_il/model.json"),
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_phase2_checkpoint() -> Path | None:
    for rel in _PHASE2_PATHS:
        for base in (Path.cwd(), _REPO_ROOT):
            if (candidate := base / rel).is_file():
                return candidate
    return None


def available_player_ids() -> tuple[str, ...]:
    """Return available player IDs, adding Phase 2 ML when module and weights exist."""
    has_mod = importlib.util.find_spec("rl.contextual_ml") is not None
    if has_mod and _find_phase2_checkpoint() is not None:
        return (*LIVE_PLAYER_IDS, "contextual_residual")
    return LIVE_PLAYER_IDS


def player_label(player_id: str) -> str:
    """Return the human-readable UI label for a player ID."""
    return LIVE_PLAYER_LABELS.get(player_id, player_id)


def next_player_id(current_id: str) -> str:
    """Cycle to the next available player ID, wrapping cleanly."""
    ids = available_player_ids()
    idx = ids.index(current_id) if current_id in ids else -1
    return ids[(idx + 1) % len(ids)]


def format_action_name(action_index: int) -> str:
    """Format a SemanticMetroEnv action index into a compact HUD label."""
    kind, first, second = ACTION_TABLE[int(action_index)]
    labels = {
        ActionKind.WAIT: "WAIT",
        ActionKind.CONNECT: f"CONNECT {first}-{second}",
        ActionKind.ASSIGN_LOCOMOTIVE: f"LOCO L{first + 1}",
        ActionKind.ATTACH_CARRIAGE: f"CARRIAGE L{first + 1}",
        ActionKind.PURCHASE_LINE: "BUY LINE",
        ActionKind.EXTEND_LINE: f"EXTEND L{first + 1}->{second}",
        ActionKind.PREPEND_LINE: f"PREPEND {second}->L{first + 1}",
        ActionKind.REMOVE_LINE: f"REMOVE L{first + 1}",
    }
    return labels.get(kind, str(kind.name))


class MediatorEnvAdapter:
    """SemanticMetroEnv-compatible adapter wrapping a live UI Mediator."""

    action_masks = SemanticMetroEnv.action_masks

    def __init__(self, mediator: object, *, remove_min_age: int = 60) -> None:
        self._mediator = mediator
        self._decision = 0
        self.remove_min_age = int(remove_min_age)
        self._line_born: dict[str, int] = {}
        self._mask_cache: tuple[tuple, object] | None = None
        self.sync_lines()

    def __getattr__(self, name: str):
        raw = inspect.getattr_static(SemanticMetroEnv, name)
        if isinstance(raw, staticmethod):
            return raw.__func__
        return types.MethodType(raw, self) if callable(raw) else raw

    def sync_lines(self) -> None:
        """Record newly observed line IDs and prune removed lines."""
        paths = getattr(self._mediator, "paths", ())
        live = {path.id for path in paths}
        self._line_born = {k: v for k, v in self._line_born.items() if k in live}
        for path in paths:
            self._line_born.setdefault(path.id, self._decision)

    def advance_decision(self) -> None:
        """Advance decision counter and update line-birth ages like SemanticMetroEnv."""
        self._decision += 1
        self.sync_lines()

    def apply_action(self, index: int) -> bool:
        """Apply an ACTION_TABLE entry to the live Mediator."""
        return bool(self._apply(index))


def _resolve_chooser(player_id: str):
    if player_id == "contextual_residual":
        if (checkpoint := _find_phase2_checkpoint()) is None:
            raise ValueError("Phase 2 checkpoint not found for 'contextual_residual'")
        mod = importlib.import_module("rl.contextual_ml")
        for name in ("load_policy", "load_model"):
            if callable(loader := getattr(mod, name, None)):
                loaded = loader(checkpoint)
                if callable(getattr(loaded, "choose", None)):
                    return loaded.choose
                return lambda env: mod.choose(env, loaded)
        return mod.choose
    if (module_name := _CHOOSER_MODULES.get(player_id)) is None:
        raise ValueError(f"Unknown AI player ID: {player_id!r}")
    return importlib.import_module(module_name).choose


class LiveAgentRunner:
    """Drives a live UI Mediator with a semantic AI policy at 6-tick cadence."""

    def __init__(
        self, player_id: str, mediator: object, *, remove_min_age: int = 60
    ) -> None:
        if player_id == "human":
            raise ValueError("LiveAgentRunner requires an AI player ID, not 'human'")
        self.player_id = player_id
        self.label = player_label(player_id)
        self.mediator = mediator
        self.adapter = MediatorEnvAdapter(mediator, remove_min_age=remove_min_age)
        self._choose = _resolve_chooser(player_id)
        self.decisions = self.actions_taken = 0
        self.last_infer_ms = 0.0
        self.last_action_name = "WAIT"
        self._step_accum = TICKS_PER_DECISION - 1

    def _blocked(self) -> bool:
        m = self.mediator
        return bool(
            getattr(m, "is_game_over", False)
            or getattr(m, "is_paused", False)
            or getattr(m, "current_offers", ())
        )

    def step(self, steps_advanced: int) -> int:
        """Advance accumulated sim ticks and invoke the policy when due."""
        if steps_advanced <= 0 or self._blocked():
            return 0
        self._step_accum += int(steps_advanced)
        fired = 0
        while self._step_accum >= TICKS_PER_DECISION and not self._blocked():
            self._step_accum -= TICKS_PER_DECISION
            t0 = time.perf_counter()
            action = int(self._choose(self.adapter))
            self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
            applied = self.adapter.apply_action(action)
            if action != 0 and applied:
                self.actions_taken += 1
                self.last_action_name = format_action_name(action)
            elif action != 0:
                for attr in ("_parametric_cem_state", "_global_replan_state"):
                    if (st := getattr(self.adapter, attr, None)) is not None:
                        st.active_plan = None
            self.decisions += 1
            self.adapter.advance_decision()
            fired += 1
        return fired

    on_session_steps = step

    def auto_resolve_offers(self) -> Offer | None:
        """Pick and apply a strategic weekly offer so AI runs cross week boundaries."""
        if not (offers := tuple(getattr(self.mediator, "current_offers", ()))):
            return None
        by_kind = {offer.kind: offer for offer in offers}
        tunnels = getattr(self.mediator, "available_tunnels", None)
        prefs: list[OfferKind] = []
        if self.mediator.get_unlocked_num_paths() < getattr(
            self.mediator, "num_paths", 4
        ):
            prefs.append(OfferKind.NEW_LINE)
        if tunnels is not None and tunnels <= 1:
            prefs.append(OfferKind.TUNNEL)
        prefs.extend((OfferKind.LOCOMOTIVE, OfferKind.CARRIAGE))
        chosen = next((by_kind[k] for k in prefs if k in by_kind), offers[0])
        self.mediator.resolve_week_boundary(chosen)
        return chosen

    def hud_summary(self) -> dict[str, object]:
        """Return compact metrics for the live AI HUD overlay."""
        return {
            "player_id": self.player_id,
            "label": self.label,
            "deliveries": int(getattr(self.mediator, "deliveries", 0)),
            "decisions": self.decisions,
            "infer_ms": round(self.last_infer_ms, 2),
            "last_action": self.last_action_name,
            "actions_taken": self.actions_taken,
        }


def filter_ai_playing_event(mediator: object, event: object) -> bool:
    """Allow speed/pause keys and speed-button mouse clicks; block board drags."""
    if not isinstance(event, MouseEvent):
        return True
    speed_buttons = getattr(mediator, "speed_buttons", ())
    if any(btn.contains(event.position) for btn in speed_buttons):
        return True
    if event.event_type == MouseEventType.MOUSE_UP and getattr(
        mediator, "is_mouse_down", False
    ):
        mediator.is_mouse_down = False
    if event.event_type == MouseEventType.MOUSE_MOTION:
        for btn in speed_buttons:
            btn.on_exit()
    return False


def advance_controller(controller: object, screen_enum: object, steps: int) -> None:
    """Auto-resolve weekly offers and drive gameplay when AI player is active."""
    if (runner := getattr(controller, "live_agent", None)) is None:
        return
    if controller.state is screen_enum.OFFER:
        if runner.auto_resolve_offers() is not None:
            controller.state = screen_enum.PLAYING
    if controller.state is screen_enum.PLAYING:
        runner.step(steps)
