"""Clipped PPO fine-tuning for Phase 2 contextual residual ML policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from train_contextual_il import TorchContextualActorCritic  # noqa: E402
from train_parametric_cem import paired_stats  # noqa: E402

import rl.contextual_ml as ctx  # noqa: E402
from rl.heuristic_global_replan import play as play_global_replan  # noqa: E402
from rl.objective_ml import ParametricPolicyParams  # noqa: E402
from rl.policy_parametric_cem import choose as choose_parametric_cem  # noqa: E402
from rl.policy_parametric_cem import play as play_parametric_cem  # noqa: E402
from rl.semantic_env import SemanticMetroEnv  # noqa: E402

DEFAULT_EVAL_EPISODES = 20
VAL_SEEDS = frozenset(range(2000, 2020))
TEST_FORBIDDEN = frozenset(range(9000, 9030))
_LOG_2PI = math.log(2.0 * math.pi)
_FLAGS = (
    "updates:i:20 episodes-per-update:i:12 train-seed-base:i:1000 "
    "train-pool-size:i:24 eval-seed-base:i:2000 eval-every:i:5 update-interval:i:100 "
    "max-decisions:i:200000 ppo-epochs:i:4 batch-size:i:64 seed:i:42 lr:f:3e-4 "
    "gamma:f:1.0 gae-lambda:f:0.98 clip-eps:f:0.15 init-std:f:0.25 final-std:f:0.05 "
    "value-coef:f:0.5 anchor-coef:f:0.02 phase1-anchor-coef:f:0.005 grad-clip:f:0.5 "
    "target-kl:f:0.05"
).split()


def _validate_rl_args(args: argparse.Namespace) -> int:
    """Validate RL hyperparameters upfront and resolve `hidden_dim`."""
    for name in (
        "updates",
        "episodes_per_update",
        "ppo_epochs",
        "batch_size",
        "max_decisions",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(args, name)}")
    if args.init_std <= 0.0 or args.final_std <= 0.0:
        raise ValueError(
            f"init_std and final_std must be > 0, got {args.init_std}, {args.final_std}"
        )
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {args.gamma}")
    if args.il_ckpt is not None:
        if args.hidden_dim is not None:
            raise ValueError(
                "--hidden-dim cannot be passed with --il-ckpt (inherited from checkpoint)"
            )
        return 0
    hidden = 32 if args.hidden_dim is None else int(args.hidden_dim)
    if hidden < 1:
        raise ValueError(f"hidden_dim must be >= 1, got {hidden}")
    return hidden


def _t64(arr: Any, dev: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float64, device=dev)


def build_rl_splits(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    """Construct disjoint Train and Val seed lists while guarding Test."""
    if args.train_pool_size < 1:
        raise ValueError(f"train_pool_size must be >= 1, got {args.train_pool_size}")
    tr = [args.train_seed_base + i for i in range(args.train_pool_size)]
    if (set(tr) & VAL_SEEDS) or (set(tr) & TEST_FORBIDDEN):
        raise ValueError(f"Train seeds overlap Val/Test pool: {tr[:5]}")
    if args.skip_val:
        return tr, []
    if args.eval_episodes < 1:
        raise ValueError(f"eval_episodes must be >= 1, got {args.eval_episodes}")
    va = [args.eval_seed_base + i for i in range(args.eval_episodes)]
    if (set(va) & TEST_FORBIDDEN) or (set(tr) & set(va)):
        raise ValueError(f"Invalid Val seed pool: {va[:5]}")
    return tr, va


def _collect_rl_episode(job: tuple[Any, ...]) -> dict[str, Any]:
    ep_seed, rng_seed, p_dict, std, interval, max_dec = job
    pol = ctx.ContextualResidualMLP.from_dict(p_dict)
    base_a, scale = pol.base_params.to_array(), pol.delta_scale
    rng, env = np.random.default_rng(rng_seed), SemanticMetroEnv(max_decisions=max_dec)
    xs, acts, lps, segs = [], [], [], []
    cur_seg, total, dec, term_flag = 0.0, 0.0, 0, False
    last_upd: int | None = None
    last_topo: Any = None
    active, last_x = pol.base_params, None
    try:
        env.reset(seed=ep_seed)
        while dec < max_dec:
            m = env._mediator
            ps = tuple((p.id, len(p.stations)) for p in m.paths)
            topo = (len(m.stations), len(m.paths), ps)
            if last_upd is None or topo != last_topo or (dec - last_upd) >= interval:
                if xs:
                    segs.append(cur_seg)
                cur_seg, x = 0.0, ctx.extract_context_features(m)
                mu = pol.forward_raw(x)
                if std > 0.0:
                    raw = mu + std * rng.normal(0.0, 1.0, size=ctx.PARAM_DIM)
                    z = (raw - mu) / std
                    lp = -0.5 * float(np.sum(z * z + 2.0 * math.log(std) + _LOG_2PI))
                else:
                    raw, lp = mu.copy(), 0.0
                nd = np.tanh(raw)
                active = pol.base_params
                if np.any(nd):
                    active = ParametricPolicyParams.from_array(base_a + scale * nd)
                xs.append(x)
                acts.append(raw)
                lps.append(lp)
                last_upd, last_topo = dec, topo
            _, reward, term, trunc, _ = env.step(choose_parametric_cem(env, active))
            r_f = float(reward)
            cur_seg += r_f
            total += r_f
            dec += 1
            if term or trunc:
                term_flag = bool(term)
                break
        if xs:
            segs.append(cur_seg)
        if not term_flag:
            last_x = ctx.extract_context_features(env._mediator)
    finally:
        env.close()
    empty_x, empty_a = np.zeros((0, len(pol.x_mean))), np.zeros((0, ctx.PARAM_DIM))
    return {
        "seed": int(ep_seed),
        "deliveries": int(total),
        "terminated": term_flag,
        "last_x": last_x,
        "X": np.stack(xs) if xs else empty_x,
        "A": np.stack(acts) if acts else empty_a,
        "LOGP": np.array(lps, dtype=np.float64),
        "R": np.array(segs, dtype=np.float64),
    }


def _eval_worker(job: tuple[Any, ...]) -> tuple[int, float]:
    kind, seed, p_dict, max_dec, interval = job
    if kind == "global_replan":
        return seed, float(
            play_global_replan(seed, max_decisions=max_dec)["deliveries"]
        )
    if kind == "parametric_cem":
        res = play_parametric_cem(seed, ctx.BEST_PHASE1_PARAMS, max_decisions=max_dec)
        return seed, float(res["deliveries"])
    pol = ctx.ContextualResidualMLP.from_dict(p_dict)
    res = ctx.play_contextual(
        seed, pol, max_decisions=max_dec, update_interval=interval
    )
    return seed, float(res["deliveries"])


def _run_pool(fn: Any, jobs: list[Any], workers: int) -> list[Any]:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs, chunksize=1))


def _eval_seeds(
    kind: str, p_dict: Any, seeds: list[int], args: argparse.Namespace
) -> list[float]:
    jobs = [(kind, s, p_dict, args.max_decisions, args.update_interval) for s in seeds]
    res = dict(_run_pool(_eval_worker, jobs, args.workers))
    return [res[s] for s in seeds]


def _val_baselines(
    out_dir: Path, seeds: list[int], args: argparse.Namespace
) -> tuple[list[float], list[float]]:
    cache, max_d = out_dir / "val_baselines.json", args.max_decisions
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("val_seeds") == seeds and c.get("max_decisions") == max_d:
            return list(map(float, c["parametric_cem"])), list(
                map(float, c["global_replan"])
            )
    ml = _eval_seeds("parametric_cem", None, seeds, args)
    gl = _eval_seeds("global_replan", None, seeds, args)
    doc = {
        "val_seeds": seeds,
        "max_decisions": max_d,
        "parametric_cem": ml,
        "global_replan": gl,
    }
    cache.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return ml, gl


def _compute_gae_batch(
    episodes: list[dict[str, Any]],
    model: TorchContextualActorCritic,
    dev: torch.device,
    args: argparse.Namespace,
    init_rtg: bool,
) -> tuple[torch.Tensor, ...]:
    if init_rtg:
        rtgs = [float(sum(ep["R"][i:])) for ep in episodes for i in range(len(ep["R"]))]
        if rtgs:
            arr = np.asarray(rtgs, dtype=np.float64)
            model.rtg_mean.fill_(float(arr.mean()))
            model.rtg_std.fill_(max(1.0, float(arr.std(ddof=0))))
    buckets: list[list[np.ndarray]] = [[] for _ in range(5)]
    model.eval()
    with torch.no_grad():
        for ep in episodes:
            if ep["X"].shape[0] == 0:
                continue
            vals = model.forward_value(_t64(ep["X"], dev)).cpu().numpy()
            nv_end = 0.0
            if not ep["terminated"] and ep["last_x"] is not None:
                lx = _t64(ep["last_x"][None], dev)
                nv_end = float(model.forward_value(lx).item())
            ep_adv, gae = np.zeros(len(vals), dtype=np.float64), 0.0
            for k in range(len(vals) - 1, -1, -1):
                nv = nv_end if k == len(vals) - 1 else float(vals[k + 1])
                delta = float(ep["R"][k]) + args.gamma * nv - float(vals[k])
                gae = delta + args.gamma * args.gae_lambda * gae
                ep_adv[k] = gae
            row = (ep["X"], ep["A"], ep["LOGP"], ep_adv, ep_adv + vals)
            for dst, src in zip(buckets, row):
                dst.append(src)
    if not buckets[0]:
        raise RuntimeError("Collected zero contextual decisions in rollout")
    return tuple(_t64(np.concatenate(b, axis=0), dev) for b in buckets)


def _ppo_update(
    model: TorchContextualActorCritic,
    ref_model: TorchContextualActorCritic,
    opt: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, ...],
    std: float,
    args: argparse.Namespace,
    gen: torch.Generator,
) -> dict[str, float]:
    xb_a, ab_a, lp_a, adv_a, ret_a = batch
    n, dev = xb_a.shape[0], xb_a.device
    adv_s = adv_a.std(unbiased=False).clamp_min(1e-8)
    adv_n = (adv_a - adv_a.mean()) / adv_s if n > 1 else adv_a
    model.train()
    keys = ("p_loss", "v_loss", "a_loss", "kl", "clip")
    acc, steps = {k: 0.0 for k in keys}, 0
    for _ in range(args.ppo_epochs):
        perm = torch.randperm(n, generator=gen).to(dev)
        ep_kl, ep_n = 0.0, 0
        for s in range(0, n, args.batch_size):
            i = perm[s : s + args.batch_size]
            xb, ab, old_lp, adv_b, ret_b = xb_a[i], ab_a[i], lp_a[i], adv_n[i], ret_a[i]
            raw_mu, v_pred = model(xb)
            log_r = (
                torch.distributions.Normal(raw_mu, std).log_prob(ab).sum(-1) - old_lp
            )
            ratio = torch.exp(log_r)
            clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
            p_loss = -torch.min(ratio * adv_b, clipped * adv_b).mean()
            v_loss = 0.5 * F.mse_loss(v_pred, (ret_b - model.rtg_mean) / model.rtg_std)
            pred_d = torch.tanh(raw_mu)
            with torch.no_grad():
                ref_d = ref_model.predict_normalized_delta(xb)
            ac, pc = args.anchor_coef, args.phase1_anchor_coef
            a_loss = ac * (pred_d - ref_d).pow(2).mean() + pc * pred_d.pow(2).mean()
            loss = p_loss + args.value_coef * v_loss + a_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            with torch.no_grad():
                kl = float(((ratio - 1.0) - log_r).mean())
                clip = float(((ratio - 1.0).abs() > args.clip_eps).double().mean())
            ep_kl, ep_n, steps = ep_kl + kl, ep_n + 1, steps + 1
            vals = (p_loss.item(), v_loss.item(), a_loss.item(), kl, clip)
            for k, val in zip(keys, vals):
                acc[k] += val
        if args.target_kl > 0.0 and ep_n and (ep_kl / ep_n) > args.target_kl:
            break
    return {k: v / max(1, steps) for k, v in acc.items()}


def _eval_and_log(
    tag: str,
    model: TorchContextualActorCritic,
    base: ParametricPolicyParams,
    seeds: list[int],
    baselines: tuple[list[float], list[float]],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    scores = _eval_seeds("ctx", model.to_contextual_mlp(base).to_dict(), seeds, args)
    vm, vg = paired_stats(scores, baselines[0]), paired_stats(scores, baselines[1])
    obj = vm["mean"] + 1e-3 * (vm["won"] / max(1, len(seeds)))
    m_s = f"{vm['vs_base']:+6.2f}+/-{vm['ci95']:5.2f} ({vm['won']}/{vm['lost']}/{vm['tied']})"
    g_s = f"{vg['vs_base']:+6.2f}+/-{vg['ci95']:5.2f} ({vg['won']}/{vg['lost']}/{vg['tied']})"
    print(
        f"  [val {tag:<5}] mean={vm['mean']:7.2f} | vs_ml={m_s} | vs_gl={g_s}",
        flush=True,
    )
    return obj, {"tag": tag, "vs_parametric_cem": vm, "vs_global_replan": vg}


def train_contextual_rl(args: argparse.Namespace) -> dict[str, Any]:
    """Run clipped PPO residual fine-tuning with strict Train vs Val separation."""
    hidden_dim = _validate_rl_args(args)
    train_seeds, val_seeds = build_rl_splits(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev, il_extra = torch.device(args.device), {}
    if args.il_ckpt is not None:
        load_ckpt = TorchContextualActorCritic.load_checkpoint
        model, base_p, il_extra = load_ckpt(args.il_ckpt, device=dev)
    else:
        model = TorchContextualActorCritic(hidden_dim=hidden_dim).to(dev)
        base_p = ctx.BEST_PHASE1_PARAMS
    print(
        f"Loaded AC ({args.il_ckpt or 'zero_init'}, hidden={model.hidden_dim})",
        flush=True,
    )

    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def save_pair(stem: str, extra: dict[str, Any]) -> None:
        model.to_contextual_mlp(base_p).save(args.out_dir / f"{stem}.json")
        model.save_checkpoint(
            args.out_dir / f"{stem}.pt", base_params=base_p, extra=extra
        )

    val_hist, train_hist, best_obj, baselines = [], [], float("-inf"), ([], [])
    if not args.skip_val:
        baselines = _val_baselines(args.out_dir, val_seeds, args)
        best_obj, r0 = _eval_and_log("init", model, base_p, val_seeds, baselines, args)
        val_hist.append(r0)
        save_pair("best_val", r0)

    t0, interval = time.perf_counter(), max(1, int(args.update_interval))
    n_ep, pool_n, max_d = args.episodes_per_update, len(train_seeds), args.max_decisions
    for upd in range(args.updates):
        frac = upd / max(1, args.updates - 1)
        std = float(args.init_std + (args.final_std - args.init_std) * frac)
        ep_s = [train_seeds[(upd * n_ep + i) % pool_n] for i in range(n_ep)]
        p_dict = model.to_contextual_mlp(base_p).to_dict()
        r_s = rng.integers(0, 2**31 - 1, size=n_ep).tolist()
        jobs = [(s, rs, p_dict, std, interval, max_d) for s, rs in zip(ep_s, r_s)]
        episodes = _run_pool(_collect_rl_episode, jobs, args.workers)
        init_rtg = args.il_ckpt is None and upd == 0
        batch = _compute_gae_batch(episodes, model, dev, args, init_rtg)
        pm = _ppo_update(model, ref_model, opt, batch, std, args, gen)
        m_deliv = float(np.mean([e["deliveries"] for e in episodes]))
        row = {"update": upd, "std": round(std, 4), "train_mean_deliv": m_deliv}
        row.update({"n_steps": int(batch[0].shape[0]), **pm})
        train_hist.append(row)
        print(
            f"[upd {upd:02d}] std={std:.3f} n={row['n_steps']:4d} deliv={m_deliv:6.1f}"
            f" p={pm['p_loss']:+.4f} v={pm['v_loss']:.4f} kl={pm['kl']:.4f}"
            f" clip={pm['clip']:.3f}",
            flush=True,
        )
        due = (upd + 1) % max(1, args.eval_every) == 0 or upd == args.updates - 1
        if not args.skip_val and due:
            tag = f"u{upd:02d}"
            obj, rec = _eval_and_log(tag, model, base_p, val_seeds, baselines, args)
            val_hist.append(rec)
            if obj > best_obj:
                best_obj = obj
                save_pair("best_val", rec)

    save_pair("final_model", {"updates": args.updates})
    summary = {
        "il_ckpt": str(args.il_ckpt) if args.il_ckpt else None,
        "il_extra": il_extra,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "train_history": train_hist,
        "val_history": val_hist,
    }
    (args.out_dir / "rl_history.json").write_text(
        json.dumps(summary, indent=2), "utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--il-ckpt", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("output/phase2_rl"))
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip-val", action="store_true")
    for spec in _FLAGS:
        k, kind, v = spec.split(":")
        tp = int if kind == "i" else float
        p.add_argument(f"--{k}", type=tp, default=tp(v))
    return p


def main(argv: list[str] | None = None) -> int:
    train_contextual_rl(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
