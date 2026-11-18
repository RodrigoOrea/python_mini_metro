"""Contract and smoke tests for Phase 2 PyTorch IL & PPO RL pipeline."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../scripts")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rl.contextual_ml import (  # noqa: E402
    BEST_PHASE1_PARAMS,
    CONTEXT_DIM,
    DEFAULT_DELTA_SCALES,
    PARAM_DIM,
    ContextualResidualMLP,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _make_mini_npz(path: Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(0.0, 2.0, size=(64, CONTEXT_DIM))
    x_va = rng.uniform(0.0, 2.0, size=(24, CONTEXT_DIM))
    x_mean = x_tr.mean(axis=0)
    raw_std = x_tr.std(axis=0, ddof=0)
    x_std = np.where(raw_std > 1e-6, raw_std, 1.0)
    proj = rng.normal(0.0, 0.3, size=(CONTEXT_DIM, PARAM_DIM))
    y_tr = 0.4 * np.tanh(((x_tr - x_mean) / x_std) @ proj)
    y_va = 0.4 * np.tanh(((x_va - x_mean) / x_std) @ proj)
    w_tr = rng.uniform(0.5, 2.0, size=len(x_tr))
    w_va = rng.uniform(0.5, 2.0, size=len(x_va))
    rtg_tr = 400.0 + 80.0 * ((x_tr[:, 0] - x_mean[0]) / x_std[0])
    rtg_va = 400.0 + 80.0 * ((x_va[:, 0] - x_mean[0]) / x_std[0])
    np.savez_compressed(
        path,
        X_train=x_tr,
        Y_train=y_tr,
        W_train=w_tr,
        RTG_train=rtg_tr,
        X_val=x_va,
        Y_val=y_va,
        W_val=w_va,
        RTG_val=rtg_va,
        x_mean=x_mean,
        x_std=x_std,
        delta_scale=DEFAULT_DELTA_SCALES.copy(),
    )
    return path


class TestPhase2CLIContract(unittest.TestCase):
    def test_rl_script_eval_episodes_default_meets_protocol_floor(self) -> None:
        if str(REPO_ROOT / "test") not in sys.path:
            sys.path.append(str(REPO_ROOT / "test"))
        from test_evaluation_protocol import MEASURED_FLOOR, _declared_episode_defaults

        declared = _declared_episode_defaults(REPO_ROOT / "scripts")
        self.assertIn("train_contextual_rl.py --eval-episodes", declared)
        default_val = declared["train_contextual_rl.py --eval-episodes"]
        self.assertIsInstance(default_val, int)
        self.assertGreaterEqual(int(default_val), max(20, MEASURED_FLOOR))

    def test_phase2_scripts_parse_cleanly_without_syntax_errors(self) -> None:
        for name in ("train_contextual_il.py", "train_contextual_rl.py"):
            src = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
            ast.parse(src, filename=name)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional (RL extras)")
class TestPhase2PyTorchPipeline(unittest.TestCase):
    def test_pytorch_to_contextual_mlp_json_parity_under_1e_6(self) -> None:
        import torch
        from train_contextual_il import TorchContextualActorCritic

        rng = np.random.default_rng(123)
        x_mean = rng.uniform(0.0, 2.0, size=CONTEXT_DIM)
        x_std = rng.uniform(0.05, 2.0, size=CONTEXT_DIM)
        x_std[5] = 0.0  # Verify safe_std matches ContextualResidualMLP when std=0.

        model0 = TorchContextualActorCritic(hidden_dim=32, x_mean=x_mean, x_std=x_std)
        x_in = rng.uniform(-1.0, 4.0, size=(32, CONTEXT_DIM))
        with torch.no_grad():
            raw0 = model0.forward_raw(
                torch.as_tensor(x_in, dtype=torch.float32)
            ).numpy()
        self.assertTrue(np.all(raw0 == 0.0))
        mlp0 = model0.to_contextual_mlp(BEST_PHASE1_PARAMS)
        self.assertEqual(mlp0.predict_params(x_in[0]), BEST_PHASE1_PARAMS)

        with torch.no_grad():
            model0.actor_fc2.weight.normal_(0.0, 0.3)
            model0.actor_fc2.bias.normal_(0.0, 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "exported.json"
            model0.to_contextual_mlp(BEST_PHASE1_PARAMS).save(jpath)
            loaded = ContextualResidualMLP.load(jpath)
        with torch.no_grad():
            xt = torch.as_tensor(x_in, dtype=torch.float32)
            pt_raw = model0.forward_raw(xt).numpy()
            pt_delta = model0.predict_normalized_delta(xt).numpy()
        self.assertLess(float(np.max(np.abs(pt_raw - loaded.forward_raw(x_in)))), 1e-6)
        self.assertLess(
            float(np.max(np.abs(pt_delta - loaded.predict_normalized_delta(x_in)))),
            1e-6,
        )

    def test_il_2_epoch_and_rl_1_update_smoke_pipeline(self) -> None:
        from train_contextual_il import main as il_main
        from train_contextual_rl import (
            DEFAULT_EVAL_EPISODES,
            build_parser,
            build_rl_splits,
        )
        from train_contextual_rl import main as rl_main

        self.assertEqual(
            build_parser().parse_args([]).eval_episodes, DEFAULT_EVAL_EPISODES
        )
        with self.assertRaises(ValueError):
            build_rl_splits(build_parser().parse_args(["--train-seed-base", "2000"]))
        with self.assertRaises(ValueError):
            build_rl_splits(build_parser().parse_args(["--eval-seed-base", "9000"]))
        with self.assertRaises(ValueError):
            rl_main(["--final-std", "0.0"])
        with self.assertRaises(ValueError):
            rl_main(["--gamma", "0.0"])
        with self.assertRaises(ValueError):
            rl_main(["--updates", "0"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = _make_mini_npz(root / "mini.npz")
            il_dir = root / "il_out"
            rc_il = il_main(
                [
                    "--dataset",
                    str(npz_path),
                    "--out-dir",
                    str(il_dir),
                    "--hidden-dim",
                    "16",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "16",
                    "--lr",
                    "1e-2",
                ]
            )
            self.assertEqual(rc_il, 0)
            il_pt, il_json = il_dir / "best_il.pt", il_dir / "model.json"
            self.assertTrue(il_pt.exists() and il_json.exists())
            with self.assertRaises(ValueError):
                rl_main(["--il-ckpt", str(il_pt), "--hidden-dim", "64"])
            il_hist = json.loads(
                (il_dir / "il_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(il_hist["history"][0]["val"]["pred_delta_rms"], 0.0)
            self.assertLess(
                il_hist["history"][-1]["val"]["loss"],
                il_hist["history"][0]["val"]["loss"],
            )
            il_mlp = ContextualResidualMLP.load(il_json)
            self.assertEqual(il_mlp.hidden_dim, 16)

            rl_dir = root / "rl_out"
            rc_rl = rl_main(
                [
                    "--il-ckpt",
                    str(il_pt),
                    "--out-dir",
                    str(rl_dir),
                    "--updates",
                    "1",
                    "--episodes-per-update",
                    "2",
                    "--train-pool-size",
                    "4",
                    "--max-decisions",
                    "250",
                    "--ppo-epochs",
                    "2",
                    "--batch-size",
                    "4",
                    "--workers",
                    "1",
                    "--skip-val",
                ]
            )
            self.assertEqual(rc_rl, 0)
            final_json = rl_dir / "final_model.json"
            self.assertTrue(
                final_json.exists() and (rl_dir / "final_model.pt").exists()
            )
            rl_hist = json.loads(
                (rl_dir / "rl_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(rl_hist["train_history"]), 1)
            self.assertGreater(rl_hist["train_history"][0]["n_steps"], 0)
            rl_mlp = ContextualResidualMLP.load(final_json)
            p = rl_mlp.predict_params(np.ones(CONTEXT_DIM))
            self.assertEqual(p, p.clamped())


if __name__ == "__main__":
    unittest.main()
