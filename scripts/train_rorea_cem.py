"""Paired Common-Random-Numbers (CRN) CEM trainer for Phase 1 `RoreaMLParams`."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rl.heuristic_rorea_global import play as play_rorea_global  # noqa: E402
from rl.policy_rorea_ml import (  # noqa: E402
    ANCHOR_PARAMS,
    FLEET_ABLATION_PARAMS,
    PARAM_BOUNDS,
    PARAM_NAMES,
    RISK_FLEET_ABLATION_PARAMS,
    RoreaMLParams,
)
from rl.policy_rorea_ml import (
    play as play_rorea_ml,
)

DEFAULT_EVAL_EPISODES = 20
DEFAULT_TRAIN_EPISODES = 12
Z95 = 1.959963985
MDE80 = 2.801585

INIT_STDS: dict[str, float] = {
    "alpha_dist": 0.30,
    "beta_transfer": 0.35,
    "gamma_load": 2.50,
    "balance_weight": 0.45,
    "w_shape_rarity": 0.70,
    "w_station_pax": 1.20,
    "w_station_wait": 3.00,
    "w_line_overload": 0.90,
    "single_min_rel": 0.04,
    "multi_min_rel": 0.05,
    "demo_risk_weight": 45.0,
    "w_fleet_wait": 2.50,
    "w_fleet_load": 1.50,
    "w_rebalance": 2.00,
    "hub_share_bonus": 2.00,
}

INIT_PRESETS: dict[str, RoreaMLParams] = {
    "anchor": ANCHOR_PARAMS,
    "fleet": FLEET_ABLATION_PARAMS,
    "fleet_risk": RISK_FLEET_ABLATION_PARAMS,
    "prior": RoreaMLParams(
        w_shape_rarity=0.8,
        w_station_pax=1.5,
        w_station_wait=4.0,
        w_line_overload=1.0,
        demo_risk_weight=80.0,
        w_fleet_wait=4.0,
        w_fleet_load=2.0,
        w_rebalance=2.5,
        hub_share_bonus=2.0,
    ),
}


def paired_stats(
    cand_scores: list[float], base_scores: list[float]
) -> dict[str, float | int]:
    """Compute paired gap, 95% CI, 80% MDE, and W/L/T against `base_scores`."""
    cand = np.asarray(cand_scores, dtype=np.float64)
    base = np.asarray(base_scores, dtype=np.float64)
    diff = cand - base
    n = len(cand)
    sd = float(np.std(cand, ddof=1)) if n > 1 else 0.0
    diff_sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    ci95 = Z95 * diff_sd / math.sqrt(n) if n > 1 else float("inf")
    mde80 = MDE80 * diff_sd / math.sqrt(n) if n > 1 else float("inf")
    return {
        "n": n,
        "mean": float(cand.mean()),
        "sd": sd,
        "vs_base": float(diff.mean()),
        "ci95": ci95,
        "mde80": mde80,
        "won": int(np.sum(diff > 0)),
        "lost": int(np.sum(diff < 0)),
        "tied": int(np.sum(diff == 0)),
    }


def verify_anchor_equivalence(seeds: list[int], max_decisions: int = 200_000) -> None:
    """Abort before Generation 0 if `ANCHOR_PARAMS` diverges from `rorea_global`."""
    for seed in seeds:
        ref = play_rorea_global(seed, max_decisions=max_decisions)
        anc = play_rorea_ml(seed, ANCHOR_PARAMS, max_decisions=max_decisions)
        if (
            anc["deliveries"] != ref["deliveries"]
            or anc["decisions"] != ref["decisions"]
        ):
            raise AssertionError(
                f"Anchor drift detected on seed={seed}: ANCHOR={anc} vs rorea_global={ref}"
            )


def _play_ml_job(job: tuple[int, int, np.ndarray, int]) -> dict:
    cand_idx, seed, arr, max_decisions = job
    t0 = time.perf_counter()
    res = play_rorea_ml(
        seed, RoreaMLParams.from_array(arr), max_decisions=max_decisions
    )
    res["cand_idx"] = cand_idx
    res["elapsed"] = time.perf_counter() - t0
    return res


def _play_base_job(job: tuple[int, int]) -> dict:
    seed, max_decisions = job
    t0 = time.perf_counter()
    res = play_rorea_global(seed, max_decisions=max_decisions)
    res["elapsed"] = time.perf_counter() - t0
    return res


def _run_jobs(fn, jobs: list, workers: int) -> list[dict]:
    if workers <= 1:
        return [fn(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs, chunksize=1))


def _clamp_array(values: np.ndarray) -> np.ndarray:
    low = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES], dtype=np.float64)
    high = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES], dtype=np.float64)
    return np.clip(values, low, high)


def cem_step(
    *,
    candidates: list[np.ndarray],
    scores: list[float],
    mu: np.ndarray,
    sigma: np.ndarray,
    min_sigma: np.ndarray,
    elite_count: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Update Gaussian diagonal search distribution from top-k elites."""
    order = np.argsort(scores)[::-1][: max(1, min(elite_count, len(candidates)))]
    elites = np.stack([candidates[i] for i in order], axis=0)
    elite_mean = elites.mean(axis=0)
    elite_std = elites.std(axis=0, ddof=0) if len(elites) > 1 else sigma
    new_mu = _clamp_array((1.0 - smoothing) * mu + smoothing * elite_mean)
    new_sigma = np.maximum(min_sigma, (1.0 - smoothing) * sigma + smoothing * elite_std)
    return new_mu, new_sigma


def _save_run_payload(path: Path, params: RoreaMLParams, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": params.to_dict(), **extra}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=tuple(INIT_PRESETS), default="anchor")
    parser.add_argument("--init-file", type=Path, default=None)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument("--std-scale", type=float, default=1.0)
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--train-seed-base", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-seed-base", type=int, default=2000)
    parser.add_argument("--max-decisions", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("output/rorea_cem"))
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = [args.train_seed_base + i for i in range(args.train_episodes)]
    val_seeds = [args.eval_seed_base + i for i in range(args.eval_episodes)]

    print(
        f"Verifying anchor equivalence (max_decisions={args.max_decisions})...",
        flush=True,
    )
    verify_anchor_equivalence(
        [train_seeds[0]], max_decisions=min(args.max_decisions, 1500)
    )
    print("Anchor verification passed (0.00 drift).", flush=True)

    print(
        f"Evaluating rorea_global baseline on {len(train_seeds)} train seeds "
        f"using {args.workers} workers...",
        flush=True,
    )
    t_base = time.perf_counter()
    base_results = _run_jobs(
        _play_base_job,
        [(s, args.max_decisions) for s in train_seeds],
        args.workers,
    )
    base_by_seed = {r["seed"]: float(r["deliveries"]) for r in base_results}
    base_scores = [base_by_seed[s] for s in train_seeds]
    print(
        f"Train baseline mean={np.mean(base_scores):.2f} (sd={np.std(base_scores, ddof=1) if len(base_scores) > 1 else 0:.2f}) "
        f"in {time.perf_counter() - t_base:.1f}s",
        flush=True,
    )

    init_params = (
        RoreaMLParams.load(args.init_file)
        if args.init_file is not None
        else INIT_PRESETS[args.init]
    )
    mu = init_params.to_array()
    init_sigma = np.array(
        [INIT_STDS[name] * args.std_scale for name in PARAM_NAMES], dtype=np.float64
    )
    sigma = init_sigma.copy()
    min_sigma = init_sigma * 0.12

    rng = np.random.default_rng(args.seed)
    best_train_score = float("-inf")
    best_train_params = init_params
    best_train_stats: dict = {}
    history: list[dict] = []

    for gen in range(args.generations):
        t_gen = time.perf_counter()
        candidates = [_clamp_array(mu.copy())]
        for _ in range(args.pop_size - 1):
            sample = _clamp_array(mu + rng.normal(0.0, 1.0, size=len(mu)) * sigma)
            candidates.append(sample)

        jobs = [
            (cand_idx, seed, candidates[cand_idx], args.max_decisions)
            for cand_idx in range(len(candidates))
            for seed in train_seeds
        ]
        results = _run_jobs(_play_ml_job, jobs, args.workers)
        per_cand: dict[int, dict[int, float]] = {i: {} for i in range(len(candidates))}
        for res in results:
            per_cand[res["cand_idx"]][res["seed"]] = float(res["deliveries"])

        cand_stats: list[dict] = []
        cand_objectives: list[float] = []
        for idx in range(len(candidates)):
            scores = [per_cand[idx][s] for s in train_seeds]
            st = paired_stats(scores, base_scores)
            cand_stats.append(st)
            # Primary fitness: mean paired delivery improvement, with tiny win-rate tie-break.
            win_rate = st["won"] / max(1, len(train_seeds))
            cand_objectives.append(st["vs_base"] + 1e-3 * win_rate)

        gen_best_idx = int(np.argmax(cand_objectives))
        gen_best_stats = cand_stats[gen_best_idx]
        gen_best_params = RoreaMLParams.from_array(candidates[gen_best_idx])
        if cand_objectives[gen_best_idx] > best_train_score:
            best_train_score = cand_objectives[gen_best_idx]
            best_train_params = gen_best_params
            best_train_stats = gen_best_stats
            _save_run_payload(
                args.out_dir / "best_train.json",
                best_train_params,
                {"generation": gen, "train_stats": best_train_stats},
            )

        mu, sigma = cem_step(
            candidates=candidates,
            scores=cand_objectives,
            mu=mu,
            sigma=sigma,
            min_sigma=min_sigma,
            elite_count=args.elite_count,
            smoothing=args.smoothing,
        )
        elite_order = np.argsort(cand_objectives)[::-1][: args.elite_count]
        elite_mean_gap = float(np.mean([cand_stats[i]["vs_base"] for i in elite_order]))
        gen_time = time.perf_counter() - t_gen
        rec = {
            "generation": gen,
            "elapsed_s": round(gen_time, 1),
            "best_stats": gen_best_stats,
            "elite_mean_gap": round(elite_mean_gap, 2),
            "center_stats": cand_stats[0],
            "mu": RoreaMLParams.from_array(mu).to_dict(),
        }
        history.append(rec)
        print(
            f"[gen {gen:02d}] {gen_time:6.1f}s | center {cand_stats[0]['vs_base']:+6.2f} "
            f"| best {gen_best_stats['mean']:6.2f} ({gen_best_stats['vs_base']:+6.2f} "
            f"+/-{gen_best_stats['ci95']:5.2f}, {gen_best_stats['won']}/{gen_best_stats['lost']}/{gen_best_stats['tied']}) "
            f"| elite_avg {elite_mean_gap:+6.2f}",
            flush=True,
        )
        (args.out_dir / "history.json").write_text(
            json.dumps(
                {
                    "train_seeds": train_seeds,
                    "base_mean": float(np.mean(base_scores)),
                    "generations": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    final_mu_params = RoreaMLParams.from_array(mu)
    _save_run_payload(
        args.out_dir / "final_mu.json",
        final_mu_params,
        {"train_best_stats": best_train_stats},
    )

    if args.skip_val:
        return 0

    # Strict out-of-sample checkpoint selection on disjoint Validation seeds.
    print(
        f"\nSelecting best checkpoint on {len(val_seeds)} Val seeds ({val_seeds[0]}..{val_seeds[-1]})...",
        flush=True,
    )
    val_base = _run_jobs(
        _play_base_job,
        [(s, args.max_decisions) for s in val_seeds],
        args.workers,
    )
    val_base_map = {r["seed"]: float(r["deliveries"]) for r in val_base}
    val_base_scores = [val_base_map[s] for s in val_seeds]

    shortlist: list[tuple[str, RoreaMLParams]] = [
        ("init", init_params),
        ("best_train", best_train_params),
        ("final_mu", final_mu_params),
    ]
    val_jobs = [
        (idx, seed, params.to_array(), args.max_decisions)
        for idx, (_, params) in enumerate(shortlist)
        for seed in val_seeds
    ]
    val_res = _run_jobs(_play_ml_job, val_jobs, args.workers)
    val_map: dict[int, dict[int, float]] = {i: {} for i in range(len(shortlist))}
    for res in val_res:
        val_map[res["cand_idx"]][res["seed"]] = float(res["deliveries"])

    val_summary = []
    best_val_idx = 0
    best_val_gap = float("-inf")
    for idx, (label, params) in enumerate(shortlist):
        scores = [val_map[idx][s] for s in val_seeds]
        st = paired_stats(scores, val_base_scores)
        val_summary.append({"label": label, "stats": st, "params": params.to_dict()})
        print(
            f"  val[{label:<10}] mean={st['mean']:7.2f} ({st['vs_base']:+7.2f} +/-{st['ci95']:6.2f}, "
            f"MDE80=+/-{st['mde80']:6.2f}, W/L/T={st['won']}/{st['lost']}/{st['tied']})",
            flush=True,
        )
        if st["vs_base"] > best_val_gap:
            best_val_gap = st["vs_base"]
            best_val_idx = idx

    win_label, win_params = shortlist[best_val_idx]
    _save_run_payload(
        args.out_dir / "best_val.json",
        win_params,
        {
            "selected_from": win_label,
            "val_summary": val_summary,
            "val_seeds": val_seeds,
        },
    )
    print(
        f"Selected {win_label!r} on Val -> {args.out_dir / 'best_val.json'}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
