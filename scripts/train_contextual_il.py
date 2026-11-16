"""PyTorch supervised pretraining (IL + ValueHead) for Phase 2 contextual residual policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rl.contextual_ml import (  # noqa: E402
    BEST_PHASE1_PARAMS,
    CONTEXT_DIM,
    DEFAULT_DELTA_SCALES,
    PARAM_DIM,
    ContextualRoreaMLP,
)
from rl.objective_ml import RoreaMLParams  # noqa: E402

_NPZ_KEYS = (
    "X_train Y_train W_train RTG_train X_val Y_val W_val RTG_val x_mean x_std delta_scale"
).split()


class TorchContextualActorCritic(nn.Module):
    """Exact PyTorch mirror of `ContextualRoreaMLP` plus a ReLU `ValueHead`."""

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        x_mean: np.ndarray | torch.Tensor | None = None,
        x_std: np.ndarray | torch.Tensor | None = None,
        delta_scale: np.ndarray | torch.Tensor | None = None,
        rtg_mean: float = 0.0,
        rtg_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")

        bufs = (
            (
                "x_mean",
                np.zeros(CONTEXT_DIM) if x_mean is None else x_mean,
                (CONTEXT_DIM,),
            ),
            ("x_std", np.ones(CONTEXT_DIM) if x_std is None else x_std, (CONTEXT_DIM,)),
            (
                "delta_scale",
                DEFAULT_DELTA_SCALES.copy() if delta_scale is None else delta_scale,
                (PARAM_DIM,),
            ),
            ("rtg_mean", float(rtg_mean), ()),
            ("rtg_std", max(1.0, float(rtg_std)), ()),
        )
        for name, val, shp in bufs:
            self.register_buffer(
                name, torch.as_tensor(val, dtype=torch.float64).reshape(shp)
            )

        self.actor_fc1 = nn.Linear(CONTEXT_DIM, self.hidden_dim, dtype=torch.float64)
        self.actor_fc2 = nn.Linear(self.hidden_dim, PARAM_DIM, dtype=torch.float64)
        self.value_fc1 = nn.Linear(CONTEXT_DIM, self.hidden_dim, dtype=torch.float64)
        self.value_fc2 = nn.Linear(self.hidden_dim, 1, dtype=torch.float64)

        nn.init.xavier_uniform_(self.actor_fc1.weight)
        nn.init.kaiming_uniform_(self.value_fc1.weight, nonlinearity="relu")
        for layer in (self.actor_fc1, self.actor_fc2, self.value_fc1, self.value_fc2):
            nn.init.zeros_(layer.bias)
            if layer in (self.actor_fc2, self.value_fc2):
                nn.init.zeros_(layer.weight)

    def normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize and outlier-clip features matching `ContextualRoreaMLP`."""
        x64 = x.to(dtype=torch.float64, device=self.x_mean.device)
        safe = torch.where(self.x_std > 1e-6, self.x_std, 1.0)
        return torch.clamp((x64 - self.x_mean) / safe, -10.0, 10.0)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unbounded actor logits before `tanh` and `delta_scale`."""
        return self.actor_fc2(torch.tanh(self.actor_fc1(self.normalize_features(x))))

    def predict_normalized_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute bounded residual coordinates in `[-1, 1]^15`."""
        return torch.tanh(self.forward_raw(x))

    def forward_value_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Predict standardized return-to-go."""
        return self.value_fc2(
            torch.relu(self.value_fc1(self.normalize_features(x)))
        ).squeeze(-1)

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Predict unnormalized return-to-go in deliveries."""
        return self.forward_value_norm(x) * self.rtg_std + self.rtg_mean

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(raw_actor_logits, normalized_value)`."""
        return self.forward_raw(x), self.forward_value_norm(x)

    def to_contextual_mlp(
        self, base_params: RoreaMLParams = BEST_PHASE1_PARAMS
    ) -> ContextualRoreaMLP:
        """Export exact `float64` weights and normalization stats to `ContextualRoreaMLP`."""
        arrs = {
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "delta_scale": self.delta_scale,
            "W1": self.actor_fc1.weight.T,
            "b1": self.actor_fc1.bias,
            "W2": self.actor_fc2.weight.T,
            "b2": self.actor_fc2.bias,
        }
        return ContextualRoreaMLP(
            base_params=base_params,
            hidden_dim=self.hidden_dim,
            **{k: v.detach().cpu().numpy().copy() for k, v in arrs.items()},
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        base_params: RoreaMLParams = BEST_PHASE1_PARAMS,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Persist actor+critic weights and metadata in a `weights_only`-safe checkpoint."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "torch_contextual_ac_v1",
                "hidden_dim": int(self.hidden_dim),
                "base_params": base_params.to_dict(),
                "state_dict": {
                    k: v.detach().cpu() for k, v in self.state_dict().items()
                },
                "extra": extra or {},
            },
            out,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple[TorchContextualActorCritic, RoreaMLParams, dict[str, Any]]:
        """Load actor+critic checkpoint onto `device`."""
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        model = cls(hidden_dim=int(payload["hidden_dim"])).to(device)
        model.load_state_dict(payload["state_dict"])
        base = RoreaMLParams.from_dict(payload["base_params"])
        return model, base, dict(payload.get("extra", {}))


def load_dataset_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate Phase 2 `.npz` dataset collected on Train."""
    with np.load(Path(path), allow_pickle=False) as data:
        missing = [k for k in _NPZ_KEYS if k not in data]
        if missing:
            raise KeyError(f"Dataset {path} missing required keys: {missing}")
        out = {k: np.asarray(data[k], dtype=np.float64) for k in _NPZ_KEYS}
    if out["X_train"].shape[0] < 1:
        raise ValueError("X_train must contain at least 1 sample")
    return out


def compute_il_loss(
    model: TorchContextualActorCritic,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted SmoothL1 + normalized Value MSE + L2 anchor and metrics."""
    x, y, w, rtg = batch
    raw_out, v_pred = model(x)
    pred = torch.tanh(raw_out)
    per = F.smooth_l1_loss(pred, y, reduction="none", beta=args.smooth_l1_beta).mean(-1)
    wc = w.clamp_min(1e-8)
    p_loss = (wc * per).sum() / wc.sum()
    v_loss = F.mse_loss(v_pred, (rtg - model.rtg_mean) / model.rtg_std)
    a_loss = pred.pow(2).mean()
    total = p_loss + args.value_coef * v_loss + args.l2_anchor * a_loss

    with torch.no_grad():
        nz = y.norm(dim=-1) > 1e-8
        cos = (
            float(F.cosine_similarity(pred[nz], y[nz], dim=-1, eps=1e-8).mean())
            if bool(torch.any(nz))
            else 0.0
        )
        rms = float(torch.sqrt(pred.pow(2).mean()))
    return total, {
        "loss": float(total.detach()),
        "policy_loss": float(p_loss.detach()),
        "value_loss": float(v_loss.detach()),
        "cos_sim": cos,
        "pred_delta_rms": rms,
    }


def _emit_epoch(
    epoch: int, tr: dict[str, float], va: dict[str, float], best: bool
) -> None:
    mark = "*" if (best and epoch > 0) else " "
    print(
        f"[epoch {epoch:02d}]{mark} train_loss={tr['loss']:.5f} val_loss={va['loss']:.5f} "
        f"val_cos_sim={va['cos_sim']:+.4f} pred_delta_rms={va['pred_delta_rms']:.5f}",
        flush=True,
    )


def train_contextual_il(args: argparse.Namespace) -> dict[str, Any]:
    """Run supervised IL pretraining and export `best_il.pt` + `model.json`."""
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1:
        raise ValueError("epochs, batch_size, and hidden_dim must all be >= 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ds = load_dataset_npz(args.dataset)
    model = TorchContextualActorCritic(
        hidden_dim=args.hidden_dim,
        x_mean=ds["x_mean"],
        x_std=ds["x_std"],
        delta_scale=ds["delta_scale"],
        rtg_mean=float(ds["RTG_train"].mean()),
        rtg_std=max(1.0, float(ds["RTG_train"].std(ddof=0))),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ts = {
        k: torch.as_tensor(ds[k], dtype=torch.float64, device=device)
        for k in _NPZ_KEYS[:8]
    }
    tr_b = (ts["X_train"], ts["Y_train"], ts["W_train"], ts["RTG_train"])
    va_b = (
        (ts["X_val"], ts["Y_val"], ts["W_val"], ts["RTG_val"])
        if ds["X_val"].shape[0] > 0
        else tr_b
    )

    @torch.no_grad()
    def eval_split(batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        model.eval()
        return compute_il_loss(model, batch, args)[1]

    def save_best(epoch: int, val_loss: float) -> None:
        model.save_checkpoint(
            best_pt, extra={"best_epoch": epoch, "best_val_loss": val_loss}
        )
        model.to_contextual_mlp(BEST_PHASE1_PARAMS).save(best_json)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_pt, best_json = args.out_dir / "best_il.pt", args.out_dir / "model.json"
    init_tr, init_va = eval_split(tr_b), eval_split(va_b)
    best_val_loss, best_epoch = init_va["loss"], 0
    save_best(0, best_val_loss)
    _emit_epoch(0, init_tr, init_va, True)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    n_tr, t0 = tr_b[0].shape[0], time.perf_counter()
    history: list[dict[str, Any]] = [
        {"epoch": 0, "train": init_tr, "val": init_va, "best": True}
    ]

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=gen).to(device)
        for s in range(0, n_tr, args.batch_size):
            idx = perm[s : s + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_il_loss(model, tuple(v[idx] for v in tr_b), args)
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        tr_m, va_m = eval_split(tr_b), eval_split(va_b)
        improved = va_m["loss"] < best_val_loss
        if improved:
            best_val_loss, best_epoch = va_m["loss"], epoch
            save_best(epoch, best_val_loss)
        history.append({"epoch": epoch, "train": tr_m, "val": va_m, "best": improved})
        _emit_epoch(epoch, tr_m, va_m, improved)

    summary = {
        "dataset": str(args.dataset),
        "n_train": n_tr,
        "n_val": int(ds["X_val"].shape[0]),
        "hidden_dim": int(args.hidden_dim),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "history": history,
    }
    (args.out_dir / "il_history.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"IL complete: best_epoch={best_epoch} best_val_loss={best_val_loss:.5f} -> {best_pt}, {best_json}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    d = "output/phase2_dataset/dataset.npz"
    p.add_argument("--dataset", type=Path, default=Path(d))
    p.add_argument("--out-dir", type=Path, default=Path("output/phase2_il"))
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--l2-anchor", type=float, default=1e-2)
    p.add_argument("--smooth-l1-beta", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p


def main(argv: list[str] | None = None) -> int:
    train_contextual_il(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
