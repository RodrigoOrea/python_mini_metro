"""CLI runner connecting ``BridgeRpcClient`` and ``RealGameAgent`` to live or mock Mini Metro."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl.bridge_client import BridgeRpcClient, BridgeRpcError  # noqa: E402
from rl.bridge_protocol import (  # noqa: E402
    ALLOWED_SPEED_MULTIPLIERS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    RemoveLineAction,
    ScreenKind,
    Snapshot,
    WaitAction,
)
from rl.real_game_adapter import RealGameAgent  # noqa: E402
from scripts.mock_metro_rpc import MockMetroBoard, MockMetroRpcServer  # noqa: E402


def _build_summary(
    *,
    mode: str,
    snapshot: Snapshot,
    decisions: int,
    actions_taken: int,
    action_counts: Counter[str],
    upgrades_picked: int,
    chosen_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_lines = [ln for ln in snapshot.lines if not ln.is_ghost]
    connected_stations = {sid for ln in active_lines for sid in ln.station_ids}
    summary: dict[str, Any] = {
        "mode": mode,
        "city_id": snapshot.city_id,
        "screen": snapshot.screen.value,
        "day": snapshot.day,
        "deliveries": snapshot.deliveries,
        "decisions": decisions,
        "actions_taken": actions_taken,
        "upgrades_picked": upgrades_picked,
        "lines": len(active_lines),
        "stations": len(snapshot.stations),
        "connected_stations": len(connected_stations),
        "action_counts": dict(action_counts),
    }
    if chosen_action is not None:
        summary["chosen_action"] = chosen_action
    return summary


def run_dry_run(
    client: BridgeRpcClient,
    agent: RealGameAgent | None = None,
) -> dict[str, Any]:
    """Connect, ping, fetch state, compute one action without mutating the game."""
    active_agent = agent or RealGameAgent()
    ping_info = client.ping()
    snapshot = client.get_state()
    action = active_agent.choose_for_snapshot(snapshot, decision=0)
    action_dict: dict[str, Any] | None = None
    if action is not None:
        action_dict = {"kind": action.kind.value, **action.to_dict()}
    summary = _build_summary(
        mode="dry_run",
        snapshot=snapshot,
        decisions=1 if action is not None else 0,
        actions_taken=0,
        action_counts=Counter(),
        upgrades_picked=0,
        chosen_action=action_dict,
    )
    summary["ping_ok"] = bool(ping_info.get("pong", True))
    return summary


def run_live_session(
    client: BridgeRpcClient,
    *,
    agent: RealGameAgent | None = None,
    max_decisions: int = 80,
    decision_interval_ms: float = 0.0,
    wait_ticks: int = 2,
    speed: float = 1.0,
    mode_label: str = "live",
) -> dict[str, Any]:
    """Run the closed-loop ``RealGameAgent`` controller against ``client``."""
    active_agent = agent or RealGameAgent()
    client.ping()
    snapshot = client.get_state()

    action_counts: Counter[str] = Counter()
    decisions = 0
    actions_taken = 0
    upgrades_picked = 0

    if (
        speed in ALLOWED_SPEED_MULTIPLIERS
        and snapshot.screen in (ScreenKind.PLAYING, ScreenKind.PAUSED)
        and snapshot.speed_multiplier != speed
    ):
        snapshot = client.set_speed(speed)

    for step in range(max_decisions):
        if snapshot.screen in (
            ScreenKind.GAME_OVER,
            ScreenKind.MENU,
            ScreenKind.OTHER,
        ):
            break

        upgrade_guard = 0
        while snapshot.screen == ScreenKind.WEEKLY_UPGRADE and upgrade_guard < 25:
            upgrade_guard += 1
            if not snapshot.weekly_offers:
                time.sleep(0.1)
                snapshot = client.get_state()
                continue
            up_action = active_agent.choose_for_snapshot(snapshot, decision=step)
            if up_action is None:
                break
            try:
                snapshot = client.execute_action(up_action)
                actions_taken += 1
                upgrades_picked += 1
                action_counts[up_action.kind.value] += 1
            except BridgeRpcError:
                time.sleep(0.1)
                snapshot = client.get_state()

        if snapshot.screen not in (ScreenKind.PLAYING, ScreenKind.PAUSED):
            break

        action = active_agent.choose_for_snapshot(snapshot, decision=step)
        decisions += 1
        if action is None:
            break

        if isinstance(action, WaitAction):
            try:
                if snapshot.is_paused and not active_agent.state.pending_actions:
                    snapshot = client.set_paused(False)
                eff_wait = WaitAction(ticks=max(1, int(wait_ticks)))
                snapshot = client.execute_action(eff_wait)
                actions_taken += 1
                action_counts[eff_wait.kind.value] += 1
            except BridgeRpcError:
                time.sleep(max(0.05, decision_interval_ms / 1000.0))
                snapshot = client.get_state()
                continue
            if decision_interval_ms > 0.0:
                time.sleep(decision_interval_ms / 1000.0)
            continue

        if (
            isinstance(action, RemoveLineAction)
            and active_agent.state.pending_actions
            and not snapshot.is_paused
        ):
            snapshot = client.set_paused(True)

        try:
            snapshot = client.execute_action(action)
            actions_taken += 1
            action_counts[action.kind.value] += 1
        except BridgeRpcError:
            snapshot = client.get_state()
            continue

        if (
            snapshot.is_paused
            and not active_agent.state.pending_actions
            and snapshot.screen == ScreenKind.PAUSED
        ):
            snapshot = client.set_paused(False)

    return _build_summary(
        mode=mode_label,
        snapshot=snapshot,
        decisions=decisions,
        actions_taken=actions_taken,
        action_counts=action_counts,
        upgrades_picked=upgrades_picked,
    )


def run_replay_mock(
    *,
    max_decisions: int = 80,
    decision_interval_ms: float = 0.0,
    wait_ticks: int = 2,
    speed: float = 1.0,
    timeout_s: float = 5.0,
    board: MockMetroBoard | None = None,
) -> dict[str, Any]:
    """Start an in-process ``MockMetroRpcServer`` on an ephemeral port and run ``RealGameAgent``."""
    with MockMetroRpcServer(host=DEFAULT_HOST, port=0, board=board) as srv:
        with BridgeRpcClient(
            host=srv.host,
            port=srv.port,
            timeout_s=timeout_s,
        ) as client:
            return run_live_session(
                client,
                max_decisions=max_decisions,
                decision_interval_ms=decision_interval_ms,
                wait_ticks=wait_ticks,
                speed=speed,
                mode_label="replay_mock",
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RealGameAgent over TCP NDJSON bridge-rpc-v1."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--decision-interval-ms", type=float, default=0.0)
    parser.add_argument("--wait-ticks", type=int, default=2)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        choices=sorted(ALLOWED_SPEED_MULTIPLIERS),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--replay-mock", action="store_true")
    mode.add_argument("--live", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.replay_mock:
        summary = run_replay_mock(
            max_decisions=args.max_decisions,
            decision_interval_ms=args.decision_interval_ms,
            wait_ticks=args.wait_ticks,
            speed=args.speed,
            timeout_s=args.timeout,
        )
    elif args.dry_run:
        with BridgeRpcClient(
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
        ) as client:
            summary = run_dry_run(client)
    else:
        with BridgeRpcClient(
            host=args.host,
            port=args.port,
            timeout_s=args.timeout,
        ) as client:
            summary = run_live_session(
                client,
                max_decisions=args.max_decisions,
                decision_interval_ms=args.decision_interval_ms,
                wait_ticks=args.wait_ticks,
                speed=args.speed,
                mode_label="live",
            )
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
