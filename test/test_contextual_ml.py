"""Contract tests for Phase 2 contextual ML policy and counterfactual CRN collector."""

from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../scripts")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from collect_phase2_dataset import (  # noqa: E402
    DEFAULT_TRAIN_EPISODES,
    DEFAULT_VAL_EPISODES,
    _clean_rows,
    build_split_seeds,
    collect_episode,
)
from collect_phase2_dataset import main as collect_main  # noqa: E402

from rl.contextual_ml import (  # noqa: E402
    BEST_PHASE1_PARAMS,
    CONTEXT_DIM,
    CONTEXT_FEATURE_NAMES,
    DEFAULT_DELTA_SCALES,
    PARAM_DIM,
    ContextualResidualMLP,
    choose_contextual,
    extract_context_features,
    play_contextual,
)
from rl.policy_parametric_cem import choose as choose_parametric_cem  # noqa: E402
from rl.policy_parametric_cem import play as play_parametric_cem  # noqa: E402
from rl.semantic_env import SemanticMetroEnv  # noqa: E402


class TestContextualMLContract(unittest.TestCase):
    def test_context_features_contract_and_non_degeneracy(self):
        self.assertEqual((CONTEXT_DIM, len(CONTEXT_FEATURE_NAMES)), (16, 16))
        self.assertEqual(DEFAULT_DELTA_SCALES.shape, (PARAM_DIM,))
        self.assertTrue(np.all(DEFAULT_DELTA_SCALES > 0.0))
        env = SemanticMetroEnv(max_decisions=800)
        snaps: list[np.ndarray] = []
        try:
            env.reset(seed=1000)
            snaps.append(extract_context_features(env._mediator))
            for step_idx in range(800):
                _, _, term, trunc, _ = env.step(
                    choose_parametric_cem(env, BEST_PHASE1_PARAMS)
                )
                if step_idx in (15, 300, 550, 799) or term or trunc:
                    snaps.append(extract_context_features(env._mediator))
                if term or trunc:
                    break
        finally:
            env.close()
        mat = np.stack(snaps, axis=0)
        self.assertEqual((mat.shape[1], mat.dtype), (16, np.float64))
        self.assertTrue(np.all(np.isfinite(mat)))
        self.assertGreaterEqual(int(np.sum(mat.std(axis=0) > 1e-6)), 8)

    def test_zero_init_matches_best_phase1_bit_for_bit_on_seed_9000(self):
        policy = ContextualResidualMLP.zero_init()
        self.assertTrue(np.array_equal(policy.W2, np.zeros_like(policy.W2)))
        self.assertTrue(np.array_equal(policy.b2, np.zeros_like(policy.b2)))
        env_ref = SemanticMetroEnv(max_decisions=1500)
        env_ctx = SemanticMetroEnv(max_decisions=1500)
        try:
            env_ref.reset(seed=9000)
            env_ctx.reset(seed=9000)
            deliv_ref = deliv_ctx = 0.0
            for step_idx in range(1500):
                x = extract_context_features(env_ctx._mediator)
                self.assertTrue(np.array_equal(policy.predict_delta(x), np.zeros(15)))
                self.assertEqual(policy.predict_params(x), BEST_PHASE1_PARAMS)
                a_ref = choose_parametric_cem(env_ref, BEST_PHASE1_PARAMS)
                a_ctx = choose_contextual(env_ctx, policy, update_interval=50)
                self.assertEqual(a_ctx, a_ref, f"Diverged at step {step_idx}")
                _, r_ref, t_ref, tr_ref, _ = env_ref.step(a_ref)
                _, r_ctx, t_ctx, tr_ctx, _ = env_ctx.step(a_ctx)
                deliv_ref += float(r_ref)
                deliv_ctx += float(r_ctx)
                self.assertEqual((t_ctx, tr_ctx), (t_ref, tr_ref))
                if t_ref or tr_ref:
                    break
            self.assertEqual(deliv_ctx, deliv_ref)
        finally:
            env_ref.close()
            env_ctx.close()
        ref_ep = play_parametric_cem(1000, BEST_PHASE1_PARAMS, max_decisions=400)
        self.assertEqual(play_contextual(1000, policy, max_decisions=400), ref_ep)
        self.assertEqual(play_contextual(1000, None, max_decisions=400), ref_ep)

    def test_json_save_load_roundtrip_and_nonzero_legality(self):
        rng = np.random.default_rng(123)
        x_std = np.linspace(0.1, 2.0, CONTEXT_DIM)
        x_std[8] = 0.0
        policy = ContextualResidualMLP.zero_init(
            hidden_dim=16,
            seed=7,
            x_mean=rng.uniform(0.0, 1.0, CONTEXT_DIM),
            x_std=x_std,
        )
        policy.W2 = rng.normal(0.0, 0.25, size=policy.W2.shape)
        policy.b2 = rng.normal(0.0, 0.1, size=policy.b2.shape)
        x_one = rng.uniform(0.0, 1.5, size=CONTEXT_DIM)
        x_batch = rng.uniform(0.0, 1.5, size=(4, CONTEXT_DIM))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ctx_policy.json"
            policy.save(path)
            loaded = ContextualResidualMLP.load(path)
        self.assertEqual(
            (loaded.base_params, loaded.hidden_dim),
            (policy.base_params, policy.hidden_dim),
        )
        for k in ("x_mean", "x_std", "delta_scale", "W1", "b1", "W2", "b2"):
            self.assertTrue(np.array_equal(getattr(loaded, k), getattr(policy, k)))
        self.assertTrue(
            np.array_equal(loaded.forward_raw(x_batch), policy.forward_raw(x_batch))
        )
        self.assertTrue(
            np.array_equal(loaded.predict_delta(x_one), policy.predict_delta(x_one))
        )
        self.assertEqual(loaded.predict_params(x_one), policy.predict_params(x_one))

        env = SemanticMetroEnv(max_decisions=500)
        try:
            env.reset(seed=1000)
            for _ in range(500):
                mask = env.action_masks()
                act = choose_contextual(env, loaded, update_interval=50)
                self.assertTrue(bool(mask[act]))
                _, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    break
            env.reset(seed=1001)
            self.assertTrue(
                bool(
                    env.action_masks()[
                        choose_contextual(env, loaded, update_interval=50)
                    ]
                )
            )
        finally:
            env.close()

    def test_collect_phase2_dataset_hygiene_and_cli_smoke(self):
        self.assertEqual((DEFAULT_TRAIN_EPISODES, DEFAULT_VAL_EPISODES), (24, 8))
        self.assertEqual(
            build_split_seeds(1000, 24, 8),
            (list(range(1000, 1024)), list(range(1024, 1032))),
        )
        for bad_base in (2000, 9000):
            with self.assertRaises(ValueError):
                build_split_seeds(bad_base, 4, 2)
        ep = collect_episode(
            1000,
            sample_every=300,
            horizon=400,
            k_perturb=3,
            futures=1,
            max_decisions=1200,
        )
        self.assertGreaterEqual(len(ep["samples"]), 1)
        r = {
            "target": np.zeros(PARAM_DIM),
            "weight": 1.0,
            "return_to_go": 10.0,
            "d_base": 10.0,
            "max_adv": 0.0,
            "d_std": 0.0,
        }
        bad = {
            "samples": [
                {
                    **r,
                    "features": np.full(CONTEXT_DIM, np.nan),
                    "remaining_decisions": 500,
                },
                {**r, "features": np.zeros(CONTEXT_DIM), "remaining_decisions": 150},
            ]
        }
        self.assertEqual(_clean_rows([bad], min_remaining=200)[1:], (1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "p2_data"
            argv = shlex.split(
                "--train-seed-base 1000 --train-episodes 1 --val-episodes 1 --horizon 400 --sample-every 300 --k-perturb 3 --max-decisions 1200 --workers 1"
            ) + ["--out-dir", str(out)]
            self.assertEqual(collect_main(argv), 0)
            data = np.load(out / "dataset.npz")
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(data["X_train"].shape[0], 1)
            self.assertGreaterEqual(data["X_val"].shape[0], 1)
            self.assertEqual(
                (data["X_train"].shape[1], data["Y_train"].shape[1]),
                (CONTEXT_DIM, PARAM_DIM),
            )
            self.assertTrue(np.all(np.abs(data["Y_train"]) <= 1.0 + 1e-12))
            self.assertTrue(
                np.all(data["W_train"] > 0.0) and np.all(data["RTG_train"] >= 0.0)
            )
            self.assertTrue(
                np.all(np.isfinite(data["x_mean"])) and np.all(data["x_std"] > 0.0)
            )
            self.assertTrue(np.allclose(data["x_mean"], data["X_train"].mean(axis=0)))
            self.assertEqual(
                (summary["n_train"], summary["n_val"]),
                (int(data["X_train"].shape[0]), int(data["X_val"].shape[0])),
            )
            self.assertEqual(len(summary["feature_stats"]), CONTEXT_DIM)


if __name__ == "__main__":
    unittest.main()
