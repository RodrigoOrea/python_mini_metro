"""Record a semantic-lane playthrough as a watchable animation.

The semantic environment is headless -- it hands the policy a 574-float vector,
never pixels -- so unlike the pixel lane there is no observation tensor to write
out. What is worth watching here is the *game*, because the interesting claim is
about play: search scores 803 on seed 9000 where the heuristic scores 275, and
the visible difference is that it commits to an 11-station line instead of
spreading across short ones.

So this renders the live game the same way the human client does, driven by
whichever player is chosen. Frames are sampled rather than captured every
decision -- an episode runs 8,000 or more decisions and almost all of them are
WAIT, so a frame per decision would be a mostly-static hour.

Players:
  heuristic  the scripted rule, ~262 deliveries
  search     rollout lookahead, ~408 -- slow, since each decision point costs
             several full-episode simulations
  model      a trained policy from --model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from collections import Counter  # noqa: E402

import pygame  # noqa: E402
from PIL import Image  # noqa: E402
from search_policy import (  # noqa: E402
    STRUCTURAL,
    _restore,
    _signature,
    expected_value,
    shortlist_for,
)

from config import screen_height, screen_width  # noqa: E402
from rl.heuristic import choose  # noqa: E402
from rl.heuristic_rorea import choose as choose_rorea  # noqa: E402
from rl.heuristic_rorea_obj import choose as choose_rorea_obj  # noqa: E402
from rl.semantic_env import ACTION_TABLE, SemanticMetroEnv  # noqa: E402
from save_game import serialize_game  # noqa: E402

PLAYERS = ("heuristic", "rorea", "rorea_obj", "search", "model")

SHAPE_LABELS = {
    "RECT": "Square / Cuadrado",
    "CIRCLE": "Circle / Círculo",
    "TRIANGLE": "Triangle / Triángulo",
    "CROSS": "Cross / Cruz",
    "DIAMOND": "Diamond / Rombo",
    "PENTAGON": "Pentagon / Pentágono",
    "STAR": "Star / Estrella",
}


def _overbooking_details(mediator) -> dict | None:
    """Identify the station that triggered game over or has the highest wait pressure."""
    if not mediator.stations:
        return None

    candidates = []
    max_wait_ms = getattr(mediator, "passenger_max_wait_time_ms", 40_000)
    for station in mediator.stations:
        overdue = [p for p in station.passengers if p.wait_ms >= max_wait_ms]
        station_max_wait = max((p.wait_ms for p in station.passengers), default=0)
        candidates.append(
            (station, len(overdue), station_max_wait, len(station.passengers))
        )

    candidates.sort(key=lambda c: (c[1], c[2], c[3]), reverse=True)
    if not candidates or candidates[0][3] == 0:
        return None

    station, overdue_count, max_wait, total_passengers = candidates[0]
    shape_name = getattr(station.shape.type, "name", str(station.shape.type))
    shape_label = SHAPE_LABELS.get(shape_name, shape_name)

    counts: Counter[str] = Counter()
    for p in station.passengers:
        p_name = getattr(
            p.destination_shape.type, "name", str(p.destination_shape.type)
        )
        counts[SHAPE_LABELS.get(p_name, p_name)] += 1

    lines = []
    for idx, path in enumerate(mediator.paths):
        if station in path.stations:
            lines.append(f"Line {idx}")

    return {
        "station": station,
        "shape": shape_label,
        "position": (int(station.position.left), int(station.position.top)),
        "total_passengers": total_passengers,
        "overdue_count": overdue_count,
        "max_wait_s": max_wait / 1000.0,
        "limit_s": max_wait_ms / 1000.0,
        "breakdown": dict(counts),
        "lines": lines if lines else ["None (Unserved)"],
    }


def _annotate_final_frame(
    surface: pygame.Surface, mediator, details: dict | None
) -> Image.Image:
    """Render the game over state annotated with the overcrowded station details."""
    surface.fill((255, 255, 255))
    mediator.render(surface)
    if details is None:
        raw = pygame.image.tobytes(surface, "RGB")
        return Image.frombytes("RGB", surface.get_size(), raw)

    pos = details["position"]
    # Reticle and rings around the overcrowded station
    pygame.draw.circle(surface, (230, 35, 35), pos, radius=50, width=5)
    pygame.draw.circle(surface, (255, 95, 95), pos, radius=62, width=2)
    pygame.draw.line(
        surface, (230, 35, 35), (pos[0] - 75, pos[1]), (pos[0] - 55, pos[1]), width=3
    )
    pygame.draw.line(
        surface, (230, 35, 35), (pos[0] + 55, pos[1]), (pos[0] + 75, pos[1]), width=3
    )
    pygame.draw.line(
        surface, (230, 35, 35), (pos[0], pos[1] - 75), (pos[0], pos[1] - 55), width=3
    )
    pygame.draw.line(
        surface, (230, 35, 35), (pos[0], pos[1] + 55), (pos[0], pos[1] + 75), width=3
    )

    # Info card at bottom
    sw, sh = surface.get_size()
    card_w = min(1200, sw - 80)
    card_h = 220
    card_x = (sw - card_w) // 2
    card_y = sh - card_h - 40

    card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(card, (20, 22, 28, 235), (0, 0, card_w, card_h), border_radius=16)
    pygame.draw.rect(
        card, (230, 45, 45), (0, 0, card_w, card_h), width=3, border_radius=16
    )

    title_font = pygame.font.Font(None, 42)
    text_font = pygame.font.Font(None, 34)

    t1 = title_font.render(
        "ESTACION CON OVERBOOKING (FIN DEL JUEGO)", True, (255, 75, 75)
    )
    lines_str = ", ".join(details["lines"])
    t2 = text_font.render(
        f"Estacion: {details['shape']} en {details['position']}  |  Lineas conectadas: {lines_str}",
        True,
        (240, 240, 240),
    )
    t3 = text_font.render(
        f"Pasajeros esperando: {details['total_passengers']}  |  "
        f"Retrasados (>{details['limit_s']:.0f}s): {details['overdue_count']}  |  "
        f"Espera maxima: {details['max_wait_s']:.1f}s",
        True,
        (240, 240, 240),
    )
    comp_items = [f"{v} {k}" for k, v in details["breakdown"].items()]
    comp_str = ", ".join(comp_items) if comp_items else "Ninguno"
    t4 = text_font.render(
        f"Composicion de pasajeros: {comp_str}",
        True,
        (255, 215, 80),
    )

    card.blit(t1, (30, 20))
    card.blit(t2, (30, 68))
    card.blit(t3, (30, 112))
    card.blit(t4, (30, 158))

    surface.blit(card, (card_x, card_y))
    raw = pygame.image.tobytes(surface, "RGB")
    return Image.frombytes("RGB", surface.get_size(), raw)


def _frame(mediator, surface) -> Image.Image:
    """Render the live game and hand back an RGB image."""
    surface.fill((255, 255, 255))
    mediator.render(surface)
    raw = pygame.image.tobytes(surface, "RGB")
    return Image.frombytes("RGB", surface.get_size(), raw)


def play(
    player: str,
    seed: int,
    every: int,
    candidates: int,
    cap: int,
    futures: int,
    model=None,
    fps: int = 20,
    freeze_seconds: float = 4.0,
) -> tuple[list[Image.Image], dict]:
    """`futures` is REQUIRED for the same reason it is in `search_policy`.

    Zero means clairvoyant: every candidate scored against the one future
    that actually happens, because the serialised state carries the RNG.
    It has no function default here so a programmatic caller cannot get
    the oracle by omission, which is exactly how it was got before.
    """
    pygame.init()
    surface = pygame.Surface((screen_width, screen_height))
    env = SemanticMetroEnv()
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    frames: list[Image.Image] = []
    delivered = 0.0
    decisions = 0
    last_signature: frozenset | None = None

    while True:
        if decisions % every == 0:
            frames.append(_frame(env._mediator, surface))

        if player == "heuristic":
            action = choose(env)
        elif player == "rorea":
            action = choose_rorea(env)
        elif player == "rorea_obj":
            action = choose_rorea_obj(env)
        elif player == "model":
            predicted, _ = model.predict(
                observation, action_masks=env.action_masks(), deterministic=True
            )
            action = int(np.asarray(predicted).ravel()[0])
        else:
            mask = env.action_masks()
            structural = [
                i for i in np.flatnonzero(mask) if ACTION_TABLE[i][0] in STRUCTURAL
            ]
            preferred = choose(env)
            signature = _signature(mask)
            if structural and (preferred != 0 or signature != last_signature):
                last_signature = signature
                shortlist = shortlist_for(rng, structural, preferred, candidates)
                document = serialize_game(env._mediator)
                at = env._decision
                # Through `expected_value` with RESAMPLED futures, not
                # `_rollout` on the unmodified document. The serialised
                # state carries the RNG, so a bare rollout scores every
                # candidate against the one future that actually happens --
                # an oracle no observation-conditioned policy could
                # reproduce. `search_policy` was fixed for exactly this and
                # this recorder kept the old call, so every playthrough it
                # has produced was the clairvoyant player rather than the
                # search this repo runs.
                keys = [int(k) for k in rng.integers(0, 2**31 - 1, size=futures)]
                scored = [
                    (expected_value(env, document, at, entry, cap, keys), entry)
                    for entry in shortlist
                ]
                _restore(env, document, at)
                action = max(scored)[1]
            else:
                action = preferred

        observation, reward, terminated, truncated, _ = env.step(action)
        delivered += float(reward)
        decisions += 1
        if terminated or truncated:
            break

    details = _overbooking_details(env._mediator)
    final_frame = _annotate_final_frame(surface, env._mediator, details)
    freeze_frames = max(1, int(fps * freeze_seconds))
    for _ in range(freeze_frames):
        frames.append(final_frame)

    mediator = env._mediator
    summary = {
        "deliveries": int(delivered),
        "decisions": decisions,
        "frames": len(frames),
        "lines": len(mediator.paths),
        "longest_line": max((len(p.stations) for p in mediator.paths), default=0),
        "stations": len(mediator.stations),
        "overbooking": details,
    }
    env.close()
    pygame.quit()
    return frames, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=PLAYERS, default="heuristic")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--every", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--cap", type=int, default=15_000)
    parser.add_argument(
        "--futures",
        type=int,
        default=4,
        help="sampled futures per candidate; 0 is the clairvoyant oracle",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--freeze-seconds",
        type=float,
        default=4.0,
        help="seconds to hold annotated final frame",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/semantic/playthrough.gif")
    )
    args = parser.parse_args(argv)

    model = None
    if args.player == "model":
        if args.model is None:
            raise SystemExit(
                "--player model needs --model pointing at a saved policy, e.g. "
                "output/semantic/distilled"
            )
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(str(args.model), device="cpu")

    print(f"recording {args.player} on seed {args.seed}, a frame every {args.every}")
    frames, summary = play(
        args.player,
        args.seed,
        args.every,
        args.candidates,
        args.cap,
        args.futures,
        model,
        fps=args.fps,
        freeze_seconds=args.freeze_seconds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".mp4":
        saved = False
        try:
            import imageio

            writer = imageio.get_writer(str(args.output), fps=args.fps)
            for frame in frames:
                writer.append_data(np.asarray(frame))
            writer.close()
            saved = True
        except ImportError:
            pass

        if not saved:
            import shutil
            import subprocess

            if shutil.which("ffmpeg"):
                proc = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-vcodec",
                        "rawvideo",
                        "-s",
                        f"{screen_width}x{screen_height}",
                        "-pix_fmt",
                        "rgb24",
                        "-r",
                        str(args.fps),
                        "-i",
                        "-",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(args.output),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                for frame in frames:
                    proc.stdin.write(frame.tobytes())  # type: ignore[union-attr]
                proc.stdin.close()  # type: ignore[union-attr]
                proc.wait()
                saved = True

        if not saved:
            raise SystemExit(
                "Exporting to .mp4 requires 'imageio' (pip install imageio imageio-ffmpeg) "
                "or system 'ffmpeg'. Alternatively, export to .gif."
            )
    else:
        frames[0].save(
            args.output,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, 1000 // args.fps),
            loop=0,
            optimize=True,
        )
    size = args.output.stat().st_size
    print(
        f"  {summary['deliveries']} deliveries over {summary['decisions']} decisions, "
        f"{summary['lines']} lines, longest {summary['longest_line']} of "
        f"{summary['stations']} stations"
    )
    # Frames captured and frames written differ: GIF collapses identical
    # consecutive frames, and this game is static between decisions.
    print(
        f"  {summary['frames']} frames captured -> {args.output} ({size / 1e6:.1f} MB)"
    )
    if summary.get("overbooking"):
        ob = summary["overbooking"]
        print("-" * 64)
        print("ESTACION CON OVERBOOKING (FIN DEL JUEGO):")
        print(f"  Forma: {ob['shape']}")
        print(f"  Posición: {ob['position']}")
        print(f"  Líneas conectadas: {', '.join(ob['lines'])}")
        print(f"  Pasajeros esperando: {ob['total_passengers']}")
        print(f"  Pasajeros retrasados (>{ob['limit_s']:.0f}s): {ob['overdue_count']}")
        print(f"  Tiempo máximo de espera: {ob['max_wait_s']:.1f}s")
        print("  Composición de pasajeros esperando:")
        for shape, count in ob["breakdown"].items():
            print(f"    - {shape}: {count}")
        print("-" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
