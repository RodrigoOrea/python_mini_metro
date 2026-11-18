"""Collect counterfactual CRN dataset for Phase 2 contextual residual ML policy."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import rl.contextual_ml as ctx  # noqa: E402
from rl.objective_ml import BEST_PHASE1_PARAMS, PARAM_NAMES, ParametricPolicyParams  # noqa: E402
from rl.policy_parametric_cem import _get_ml_state, choose  # noqa: E402
from rl.replan_ml import MLPlannerState  # noqa: E402
from rl.semantic_env import SemanticMetroEnv  # noqa: E402
from save_game import serialize_game  # noqa: E402
from save_load import deserialize_game  # noqa: E402

DEFAULT_TRAIN_EPISODES, DEFAULT_VAL_EPISODES, DEFAULT_TRAIN_SEED_BASE = 24, 8, 1000
DEFAULT_SAMPLE_EVERY, DEFAULT_HORIZON, DEFAULT_K_PERTURB = 1200, 3000, 8
DEFAULT_FUTURES, DEFAULT_MIN_REMAINING = 1, 200
VAL_FORBIDDEN = frozenset(range(2000, 2020))
TEST_FORBIDDEN = frozenset(range(9000, 9030))
_ML_ATTRS = (
    "last_rebuild_decision last_replan_topology last_rebalance_decision "
    "last_crisis_check_decision pending_rebalance_target_id pending_rebalance_until"
).split()


def build_split_seeds(
    train_seed_base: int, train_episodes: int, val_episodes: int
) -> tuple[list[int], list[int]]:
    """Build disjoint train and dev-val seed lists strictly inside the Train pool."""
    if train_episodes < 1 or val_episodes < 0:
        raise ValueError(f"Invalid split: train={train_episodes}, val={val_episodes}")
    tr = [train_seed_base + i for i in range(train_episodes)]
    va = [train_seed_base + train_episodes + i for i in range(val_episodes)]
    used = set(tr) | set(va)
    if (used & VAL_FORBIDDEN) or (used & TEST_FORBIDDEN):
        raise ValueError(f"Split hygiene violation on seeds {sorted(used)}")
    return tr, va


def _sample_perturbations(rng: np.random.Generator, k: int, std: float) -> np.ndarray:
    if k < 1:
        raise ValueError(f"k_perturb must be >= 1, got {k}")
    dirs: list[np.ndarray] = []
    for _ in range(k // 2):
        u = np.clip(rng.normal(0.0, std, size=ctx.PARAM_DIM), -1.0, 1.0)
        dirs.extend((u, -u))
    if k % 2 == 1:
        dirs.append(np.clip(rng.normal(0.0, std, size=ctx.PARAM_DIM), -1.0, 1.0))
    return np.stack(dirs, axis=0).astype(np.float64)


def _rollout_horizon(
    env: SemanticMetroEnv,
    var: dict,
    dec: int,
    st: MLPlannerState,
    params: ParametricPolicyParams,
    horizon: int,
) -> float:
    env._mediator = deserialize_game(var)
    env._decision, env._last_deliveries = dec, env._mediator.deliveries
    env._line_born, env._mask_cache = {}, None
    new_st = MLPlannerState(mediator_id=id(env._mediator), last_decision=dec)
    for attr in _ML_ATTRS:
        setattr(new_st, attr, getattr(st, attr))
    env._parametric_cem_state = new_st
    total = 0.0
    for _ in range(horizon):
        _, reward, term, trunc, _ = env.step(choose(env, params))
        total += float(reward)
        if term or trunc:
            break
    return total


def compute_counterfactual_target(
    d_base: float, d_perturbs: np.ndarray, eff_u: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """Compute bounded `[-1, 1]^15` softmax target, SNR weight, max_adv, std(D)."""
    adv = d_perturbs - d_base
    d_std = float(np.std(np.concatenate(([d_base], d_perturbs)), ddof=0))
    max_adv = float(np.max(adv))
    if max_adv <= 0.0:
        return np.zeros(ctx.PARAM_DIM, dtype=np.float64), 0.25, max_adv, d_std
    temp = max(d_std, 1.0)
    pool_adv = np.concatenate(([0.0], adv))
    valid = np.concatenate(([True], adv > 0.0))
    logits = np.where(valid, pool_adv / temp, -np.inf)
    exp_l = np.where(valid, np.exp(logits - np.max(logits)), 0.0)
    probs = exp_l / np.sum(exp_l)
    pool_u = np.vstack((np.zeros((1, ctx.PARAM_DIM)), eff_u))
    target = np.clip(probs @ pool_u, -1.0, 1.0).astype(np.float64)
    return target, float(np.clip(0.25 + max_adv / temp, 0.25, 3.0)), max_adv, d_std


def collect_episode(
    seed: int,
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    horizon: int = DEFAULT_HORIZON,
    k_perturb: int = DEFAULT_K_PERTURB,
    futures: int = DEFAULT_FUTURES,
    max_decisions: int = 200_000,
    perturb_std: float = 0.55,
) -> dict:
    """Roll one episode of `BEST_PHASE1_PARAMS` and sample counterfactual regimes."""
    t0 = time.perf_counter()
    env = SemanticMetroEnv(max_decisions=max_decisions)
    r_env = SemanticMetroEnv(max_decisions=max_decisions + horizon + 16)
    rng = np.random.default_rng(seed * 10007 + 17)
    base, scale = (
        BEST_PHASE1_PARAMS.to_array(),
        np.maximum(ctx.DEFAULT_DELTA_SCALES, 1e-12),
    )
    samples: list[dict] = []
    delivered = dec = last_sample = 0
    terminated = False
    try:
        env.reset(seed=seed)
        while dec < max_decisions:
            m, st = env._mediator, _get_ml_state(env)
            due = (dec - last_sample) >= sample_every and not m.is_game_over
            if (
                due
                and len(m.stations) >= 4
                and len(m.paths) >= 1
                and st.active_plan is None
            ):
                feats, doc = ctx.extract_context_features(m), serialize_game(m)
                dirs = _sample_perturbations(rng, k_perturb, perturb_std)
                cands = [
                    ParametricPolicyParams.from_array(
                        base + ctx.DEFAULT_DELTA_SCALES * u
                    )
                    for u in dirs
                ]
                eff_u = np.clip(
                    (np.array([p.to_array() for p in cands]) - base) / scale, -1, 1
                )
                fkeys = rng.integers(0, 2**31 - 1, size=max(1, futures)).tolist()
                d_b, d_k = 0.0, np.zeros(k_perturb, dtype=np.float64)
                for fk in fkeys:
                    var = copy.deepcopy(doc)
                    var["rng"] = {
                        "python": random.Random(fk).getstate(),
                        "numpy": np.random.default_rng(fk).bit_generator.state,
                    }
                    d_b += _rollout_horizon(
                        r_env, var, dec, st, BEST_PHASE1_PARAMS, horizon
                    )
                    for i, cand in enumerate(cands):
                        d_k[i] += _rollout_horizon(r_env, var, dec, st, cand, horizon)
                d_base = d_b / len(fkeys)
                tgt, wt, m_adv, d_sd = compute_counterfactual_target(
                    d_base, d_k / len(fkeys), eff_u
                )
                r: dict = {
                    "seed": int(seed),
                    "decision": dec,
                    "features": feats,
                    "target": tgt,
                }
                r["weight"], r["deliveries_at_t"], r["d_base"] = (
                    float(wt),
                    int(m.deliveries),
                    float(d_base),
                )
                r["max_adv"], r["d_std"] = float(m_adv), float(d_sd)
                samples.append(r)
                last_sample = dec
            _, reward, term, trunc, _ = env.step(choose(env, BEST_PHASE1_PARAMS))
            delivered += float(reward)
            dec += 1
            if term or trunc:
                terminated = bool(term)
                break
    finally:
        env.close()
        r_env.close()
    tot_d = int(delivered)
    for r in samples:
        r["remaining_decisions"], r["return_to_go"] = (
            dec - r["decision"],
            float(tot_d - r["deliveries_at_t"]),
        )
    ep: dict = {"seed": int(seed), "deliveries": tot_d, "decisions": dec}
    ep["terminated"], ep["samples"], ep["elapsed"] = (
        terminated,
        samples,
        time.perf_counter() - t0,
    )
    return ep


def _collect_job(job: tuple[int, dict]) -> dict:
    return collect_episode(job[0], **job[1])


def _clean_rows(
    episodes: list[dict], min_remaining: int
) -> tuple[list[dict], int, int]:
    kept, n_nf, n_nt = [], 0, 0
    skeys = ("weight", "return_to_go", "d_base", "max_adv", "d_std")
    for ep in episodes:
        for r in ep["samples"]:
            fin = np.all(np.isfinite(r["features"])) and np.all(
                np.isfinite(r["target"])
            )
            if not (fin and all(np.isfinite(r[k]) for k in skeys)):
                n_nf += 1
            elif r["remaining_decisions"] < min_remaining:
                n_nt += 1
            else:
                kept.append(r)
    return kept, n_nf, n_nt


def _pack_split(rows: list[dict]) -> dict[str, np.ndarray]:
    x_shape, y_shape = (0, ctx.CONTEXT_DIM), (0, ctx.PARAM_DIM)
    out = {
        "X": np.stack([r["features"] for r in rows]).astype(np.float64)
        if rows
        else np.zeros(x_shape),
        "Y": np.stack([r["target"] for r in rows]).astype(np.float64)
        if rows
        else np.zeros(y_shape),
    }
    for (
        kv
    ) in "W:weight RTG:return_to_go D_base:d_base A_max:max_adv D_std:d_std".split():
        k, s = kv.split(":")
        out[k] = np.array([r[s] for r in rows], dtype=np.float64)
    for kv in ("seeds:seed", "decisions:decision"):
        k, s = kv.split(":")
        out[k] = np.array([r[s] for r in rows], dtype=np.int64)
    return out


def _run_collection(jobs: list[tuple[int, dict]], workers: int) -> list[dict]:
    results: list[dict] = []
    pctx = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()
    with pctx as pool:
        it = (
            pool.map(_collect_job, jobs, chunksize=1)
            if pool
            else map(_collect_job, jobs)
        )
        for ep in it:
            results.append(ep)
            print(
                f"  seed {ep['seed']}: deliv={ep['deliveries']:4d} dec={ep['decisions']:5d} n={len(ep['samples']):2d} ({ep['elapsed']:.1f}s)",
                flush=True,
            )
    return results


def _write_outputs(
    args: argparse.Namespace,
    tr_seeds: list[int],
    va_seeds: list[int],
    results: list[dict],
    elapsed: float,
) -> None:
    tr_set = set(tr_seeds)
    tr_rows, nf_tr, nt_tr = _clean_rows(
        [e for e in results if e["seed"] in tr_set], args.min_remaining
    )
    va_rows, nf_va, nt_va = _clean_rows(
        [e for e in results if e["seed"] not in tr_set], args.min_remaining
    )
    if not tr_rows:
        raise RuntimeError("No clean training samples remained after filtering")
    tr, va = _pack_split(tr_rows), _pack_split(va_rows)
    x_mean, raw_std = tr["X"].mean(axis=0), tr["X"].std(axis=0, ddof=0)
    x_std = np.where(raw_std > 1e-6, raw_std, 1.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    a = {f"{k}_train": v for k, v in tr.items()} | {
        f"{k}_val": v for k, v in va.items()
    }
    a["x_mean"], a["x_std"], a["delta_scale"] = (
        x_mean,
        x_std,
        ctx.DEFAULT_DELTA_SCALES.copy(),
    )
    a["feature_names"], a["param_names"] = (
        np.array(ctx.CONTEXT_FEATURE_NAMES),
        np.array(PARAM_NAMES),
    )
    a["train_seeds"], a["val_seeds"] = (
        np.array(tr_seeds, np.int64),
        np.array(va_seeds, np.int64),
    )
    np.savez_compressed(args.out_dir / "dataset.npz", **a)

    n_tr, n_va = int(tr["X"].shape[0]), int(va["X"].shape[0])
    s = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    s["train_seeds"], s["val_seeds"], s["min_remaining_decisions"] = (
        tr_seeds,
        va_seeds,
        args.min_remaining,
    )
    s["raw_samples"], s["n_train"], s["n_val"] = (
        sum(len(e["samples"]) for e in results),
        n_tr,
        n_va,
    )
    s["dropped_nonfinite"], s["dropped_near_terminal"] = nf_tr + nf_va, nt_tr + nt_va
    s["dropped_total"] = nf_tr + nf_va + nt_tr + nt_va
    s["train_mean_weight"], s["train_mean_rtg"] = (
        float(tr["W"].mean()),
        float(tr["RTG"].mean()),
    )
    for pfx, sp, cnt in (("train", tr, n_tr), ("val", va, n_va)):
        s[f"{pfx}_positive_advantage_rate"] = (
            float(np.mean(sp["A_max"] > 0.0)) if cnt else 0.0
        )
        s[f"{pfx}_mean_max_advantage"] = float(np.mean(sp["A_max"])) if cnt else 0.0
    x_tr = tr["X"]
    s["feature_stats"] = {
        n: {
            "mean": float(x_mean[i]),
            "std": float(x_tr[:, i].std()),
            "min": float(x_tr[:, i].min()),
            "max": float(x_tr[:, i].max()),
        }
        for i, n in enumerate(ctx.CONTEXT_FEATURE_NAMES)
    }
    s["x_mean"], s["x_std"], s["elapsed_seconds"] = (
        x_mean.tolist(),
        x_std.tolist(),
        round(elapsed, 2),
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(s, indent=2), encoding="utf-8"
    )
    print(
        f"Saved dataset.npz (train={n_tr}, val={n_va}, dropped={s['dropped_total']}, pos_adv={s['train_positive_advantage_rate']:.2%}) -> {args.out_dir}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seed-base", type=int, default=DEFAULT_TRAIN_SEED_BASE)
    p.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    p.add_argument("--val-episodes", type=int, default=DEFAULT_VAL_EPISODES)
    p.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--k-perturb", type=int, default=DEFAULT_K_PERTURB)
    p.add_argument("--futures", type=int, default=DEFAULT_FUTURES)
    p.add_argument("--min-remaining", type=int, default=DEFAULT_MIN_REMAINING)
    p.add_argument("--max-decisions", type=int, default=200_000)
    p.add_argument("--perturb-std", type=float, default=0.55)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--out-dir", type=Path, default=Path("output/phase2_dataset"))
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    tr_seeds, va_seeds = build_split_seeds(
        args.train_seed_base, args.train_episodes, args.val_episodes
    )
    cfg = {
        k: getattr(args, k)
        for k in "sample_every horizon k_perturb futures max_decisions perturb_std".split()
    }
    print(
        f"Collecting Phase 2 CRN dataset: train={len(tr_seeds)} val={len(va_seeds)} horizon={args.horizon} k={args.k_perturb} futures={args.futures} workers={args.workers}",
        flush=True,
    )
    results = _run_collection([(s, cfg) for s in tr_seeds + va_seeds], args.workers)
    _write_outputs(args, tr_seeds, va_seeds, results, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
