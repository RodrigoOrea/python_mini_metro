"""Contract tests for Phase 1 paired CEM optimizer (`scripts/train_rorea_cem.py`)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../scripts")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from train_rorea_cem import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    cem_step,
    paired_stats,
    verify_anchor_equivalence,
)
from train_rorea_cem import (
    main as train_main,
)

from rl.policy_rorea_ml import ANCHOR_PARAMS, RoreaMLParams  # noqa: E402


class TestTrainRoreaCEM(unittest.TestCase):
    def test_eval_episodes_default_respects_protocol_floor(self):
        self.assertGreaterEqual(DEFAULT_EVAL_EPISODES, 20)

    def test_paired_stats_calculates_exact_ci_and_mde(self):
        base = [100.0, 200.0, 300.0, 400.0]
        cand = [120.0, 220.0, 300.0, 440.0]
        stats = paired_stats(cand, base)
        self.assertAlmostEqual(stats["mean"], 270.0)
        self.assertAlmostEqual(stats["vs_base"], 20.0)
        self.assertEqual(stats["won"], 3)
        self.assertEqual(stats["lost"], 0)
        self.assertEqual(stats["tied"], 1)
        self.assertGreater(stats["mde80"], stats["ci95"])

    def test_verify_anchor_equivalence_passes_on_real_env(self):
        verify_anchor_equivalence(seeds=[42], max_decisions=400)

    def test_cem_step_moves_mean_toward_elites_within_bounds(self):
        mu = ANCHOR_PARAMS.to_array()
        sigma = RoreaMLParams(alpha_dist=0.2, w_fleet_wait=1.0).to_array() * 0.0 + 0.5
        c1 = mu.copy()
        c2 = mu.copy()
        c2[11] = 6.0  # w_fleet_wait
        c3 = mu.copy()
        c3[11] = 8.0
        new_mu, new_sigma = cem_step(
            candidates=[c1, c2, c3],
            scores=[10.0, 40.0, 50.0],
            mu=mu,
            sigma=sigma,
            min_sigma=sigma * 0.1,
            elite_count=2,
            smoothing=0.5,
        )
        self.assertGreater(new_mu[11], mu[11])
        restored = RoreaMLParams.from_array(new_mu)
        self.assertEqual(restored, restored.clamped())

    def test_end_to_end_smoke_writes_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "cem_out"
            rc = train_main(
                [
                    "--generations",
                    "1",
                    "--pop-size",
                    "3",
                    "--elite-count",
                    "2",
                    "--train-episodes",
                    "2",
                    "--train-seed-base",
                    "42",
                    "--max-decisions",
                    "250",
                    "--workers",
                    "1",
                    "--skip-val",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(rc, 0)
            best_path = out_dir / "best_train.json"
            hist_path = out_dir / "history.json"
            self.assertTrue(best_path.exists())
            self.assertTrue(hist_path.exists())
            loaded = RoreaMLParams.load(best_path)
            self.assertEqual(loaded, loaded.clamped())
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            self.assertEqual(len(hist["generations"]), 1)


if __name__ == "__main__":
    unittest.main()
