"""Phase 2 contextual state-dependent residual ML policy over `RoreaMLParams`."""

from __future__ import annotations

import json
import math
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rl.objective_ml import (
    BEST_PHASE1_PARAMS,
    PARAM_NAMES,
    LiveDemandSnapshot,
    RoreaMLParams,
    extract_live_demand,
)
from rl.policy_rorea_ml import choose as choose_rorea_ml

CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    ("stations_norm", "lines_norm", "mean_stops_norm", "max_stops_norm")
    + ("unserved_frac", "line_credits_norm", "avail_loco_norm", "avail_car_norm")
    + ("total_metros_norm", "total_carriages_norm", "max_wait_norm", "p90_wait_sq")
    + ("mean_station_fill", "max_line_load", "line_load_std", "cross_line_imbalance")
)
CONTEXT_DIM = len(CONTEXT_FEATURE_NAMES)
PARAM_DIM = len(PARAM_NAMES)
_FEATURE_UPPER = np.array(
    [2.0, 2.0, 3.0, 3.0, 1.0, 4.0, 2.0, 2.0] + [2.0, 2.0, 2.0, 2.0, 3.0, 5.0, 3.0, 4.0],
    dtype=np.float64,
)

# Order matches `PARAM_NAMES` in `rl.objective_ml`.
_SCALE_VALUES = (0.60, 0.75, 4.50, 0.90, 0.60, 1.20, 3.00, 0.90) + (
    0.06,
    0.08,
    45.0,
    2.50,
    1.50,
    2.00,
    1.50,
)
DEFAULT_DELTA_SCALES = np.array(_SCALE_VALUES, dtype=np.float64)
DEFAULT_DELTA_SCALES.setflags(write=False)
DEFAULT_DELTA_SCALE_MAP: dict[str, float] = dict(
    zip(PARAM_NAMES, _SCALE_VALUES, strict=True)
)
_ARRAY_FIELDS = ("x_mean", "x_std", "delta_scale", "W1", "b1", "W2", "b2")

__all__ = [
    "BEST_PHASE1_PARAMS",
    "CONTEXT_DIM",
    "CONTEXT_FEATURE_NAMES",
    "DEFAULT_DELTA_SCALES",
    "DEFAULT_DELTA_SCALE_MAP",
    "PARAM_DIM",
    "ContextualRoreaMLP",
    "choose_contextual",
    "extract_context_features",
    "play_contextual",
]


def extract_context_features(
    mediator: Any, demand: LiveDemandSnapshot | None = None
) -> np.ndarray:
    """Extract 16 normalized, finite telemetry features from a live Mediator."""
    snap = extract_live_demand(mediator) if demand is None else demand
    stations, paths = mediator.stations, mediator.paths
    n_st, n_pa = len(stations), len(paths)
    st_cap = max(1.0, float(getattr(mediator, "num_stations", 20)))
    pa_cap = max(1.0, float(getattr(mediator, "num_paths", 4)))
    stops = [len(p.stations) for p in paths]
    served = {st.id for p in paths for st in p.stations}
    unserved = (
        1.0 - sum(1 for st in stations if st.id in served) / n_st if n_st else 0.0
    )
    n_metros = float(
        getattr(mediator, "num_metros", len(getattr(mediator, "metros", ())))
    )
    n_cars = float(
        getattr(mediator, "num_carriages", getattr(mediator, "assigned_carriages", 0))
    )
    av_loco = float(getattr(mediator, "available_locomotives", 0.0)) / max(
        1.0, n_metros
    )
    av_car = float(getattr(mediator, "available_carriages", 0.0)) / max(1.0, n_cars)
    p90_wsq = (
        float(np.percentile(snap.station_wait_sq, 90)) if snap.station_wait_sq else 0.0
    )
    mean_fill = float(np.mean(snap.station_pax_norm)) if snap.station_pax_norm else 0.0
    loads = np.asarray(snap.line_load, dtype=np.float64)
    max_load = float(loads.max()) if loads.size else 0.0
    load_std = float(loads.std()) if loads.size >= 2 else 0.0
    imbalance = (
        float((loads.max() - loads.min()) / (loads.mean() + 0.25))
        if loads.size >= 2
        else 0.0
    )
    raw = np.array(
        [
            n_st / st_cap,
            n_pa / pa_cap,
            float(np.mean(stops)) / 8.0 if stops else 0.0,
            max(stops) / 10.0 if stops else 0.0,
            unserved,
            float(getattr(mediator, "line_credits", 0.0)) / 200.0,
            av_loco,
            av_car,
            n_metros / 8.0,
            n_cars / 8.0,
            snap.max_station_wait_norm,
            p90_wsq,
            mean_fill,
            max_load,
            load_std,
            imbalance,
        ],
        dtype=np.float64,
    )
    cleaned = np.nan_to_num(raw, nan=0.0, posinf=5.0, neginf=0.0)
    return np.clip(cleaned, 0.0, _FEATURE_UPPER)


@dataclass
class ContextualRoreaMLP:
    """Two-layer tanh residual MLP mapping context features to `RoreaMLParams`."""

    base_params: RoreaMLParams
    hidden_dim: int
    x_mean: np.ndarray
    x_std: np.ndarray
    delta_scale: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def __post_init__(self) -> None:
        self.hidden_dim = int(self.hidden_dim)
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        shapes = {
            "x_mean": (CONTEXT_DIM,),
            "x_std": (CONTEXT_DIM,),
            "delta_scale": (PARAM_DIM,),
            "W1": (CONTEXT_DIM, self.hidden_dim),
            "b1": (self.hidden_dim,),
            "W2": (self.hidden_dim, PARAM_DIM),
            "b2": (PARAM_DIM,),
        }
        for name in _ARRAY_FIELDS:
            arr = np.asarray(getattr(self, name), dtype=np.float64).reshape(
                shapes[name]
            )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Non-finite values in ContextualRoreaMLP.{name}")
            setattr(self, name, arr)

    @classmethod
    def zero_init(
        cls,
        base_params: RoreaMLParams = BEST_PHASE1_PARAMS,
        hidden_dim: int = 32,
        seed: int = 0,
        *,
        x_mean: np.ndarray | None = None,
        x_std: np.ndarray | None = None,
        delta_scale: np.ndarray | None = None,
    ) -> ContextualRoreaMLP:
        """Initialize with deterministic Xavier W1 and exact zero readout W2, b2."""
        rng = np.random.default_rng(seed)
        lim = math.sqrt(6.0 / (CONTEXT_DIM + hidden_dim))
        return cls(
            base_params=base_params,
            hidden_dim=hidden_dim,
            x_mean=np.zeros(CONTEXT_DIM) if x_mean is None else x_mean,
            x_std=np.ones(CONTEXT_DIM) if x_std is None else x_std,
            delta_scale=DEFAULT_DELTA_SCALES.copy()
            if delta_scale is None
            else delta_scale,
            W1=rng.uniform(-lim, lim, size=(CONTEXT_DIM, hidden_dim)),
            b1=np.zeros(hidden_dim),
            W2=np.zeros((hidden_dim, PARAM_DIM)),
            b2=np.zeros(PARAM_DIM),
        )

    def normalize_features(self, x: np.ndarray) -> np.ndarray:
        """Standardize and outlier-clip raw feature vector(s) of shape `(..., 16)`."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim not in (1, 2) or arr.shape[-1] != CONTEXT_DIM:
            raise ValueError(f"Expected (..., {CONTEXT_DIM}), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Context feature input contains NaN or Inf")
        safe_std = np.where(self.x_std > 1e-6, self.x_std, 1.0)
        return np.clip((arr - self.x_mean) / safe_std, -10.0, 10.0)

    def forward_raw(self, x: np.ndarray) -> np.ndarray:
        """Compute raw unbounded MLP output logits before `tanh` and `delta_scale`."""
        x_norm = self.normalize_features(x)
        return np.tanh(x_norm @ self.W1 + self.b1) @ self.W2 + self.b2

    def predict_normalized_delta(self, x: np.ndarray) -> np.ndarray:
        """Compute bounded residual coordinates in `[-1, 1]^15`."""
        return np.tanh(self.forward_raw(x))

    def predict_delta(self, x: np.ndarray) -> np.ndarray:
        """Compute scaled parameter-space residual `delta_scale * tanh(raw_out)`."""
        return self.delta_scale * np.tanh(self.forward_raw(x))

    def predict_params(self, x: np.ndarray) -> RoreaMLParams:
        """Predict clamped `RoreaMLParams` for one 1D context vector `x`."""
        delta = self.predict_delta(x)
        if delta.ndim != 1 or delta.shape[0] != PARAM_DIM:
            raise ValueError(f"predict_params expects 1D vector of len {CONTEXT_DIM}")
        # Preserve bit-for-bit equality with base_params when readout is exact zero.
        if not np.any(delta):
            return self.base_params
        return RoreaMLParams.from_array(self.base_params.to_array() + delta)

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy weights, normalization stats, and metadata to JSON dict."""
        data: dict[str, Any] = {
            "format": "contextual_rorea_mlp_v1",
            "context_dim": CONTEXT_DIM,
            "context_features": list(CONTEXT_FEATURE_NAMES),
            "param_names": list(PARAM_NAMES),
            "hidden_dim": self.hidden_dim,
            "base_params": self.base_params.to_dict(),
        }
        for name in _ARRAY_FIELDS:
            data[name] = getattr(self, name).tolist()
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ContextualRoreaMLP:
        """Deserialize and validate a `ContextualRoreaMLP` from a JSON dict."""
        if "context_dim" in raw and int(raw["context_dim"]) != CONTEXT_DIM:
            raise ValueError(f"Expected context_dim={CONTEXT_DIM}")
        if (
            "context_features" in raw
            and tuple(raw["context_features"]) != CONTEXT_FEATURE_NAMES
        ):
            raise ValueError("Mismatched context_features specification")
        if "param_names" in raw and tuple(raw["param_names"]) != PARAM_NAMES:
            raise ValueError("Mismatched param_names specification")
        return cls(
            base_params=RoreaMLParams.from_dict(raw["base_params"]),
            hidden_dim=int(raw["hidden_dim"]),
            **{name: raw[name] for name in _ARRAY_FIELDS},
        )

    def save(self, path: str | Path) -> None:
        """Write policy JSON to `path`."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ContextualRoreaMLP:
        """Load policy JSON from `path`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class _ContextCache:
    mediator_ref: weakref.ref[Any]
    policy_ref: weakref.ref[ContextualRoreaMLP]
    last_decision: int
    last_update_decision: int
    topo_key: tuple[int, int, tuple[tuple[str, int], ...]]
    params: RoreaMLParams


def choose_contextual(
    env: Any, policy: ContextualRoreaMLP | None = None, *, update_interval: int = 100
) -> int:
    """Select an action using state-conditioned `policy` cached across decisions."""
    if policy is None:
        return choose_rorea_ml(env, BEST_PHASE1_PARAMS)
    mediator = env._mediator
    decision = int(getattr(env, "_decision", 0))
    topo = (
        len(mediator.stations),
        len(mediator.paths),
        tuple((p.id, len(p.stations)) for p in mediator.paths),
    )
    cache: _ContextCache | None = getattr(env, "_rorea_ctx_cache", None)
    stale = (
        cache is None
        or cache.mediator_ref() is not mediator
        or cache.policy_ref() is not policy
        or decision < cache.last_decision
        or topo != cache.topo_key
        or (decision - cache.last_update_decision) >= max(1, int(update_interval))
    )
    if stale:
        active_params = policy.predict_params(extract_context_features(mediator))
        cache = _ContextCache(
            weakref.ref(mediator),
            weakref.ref(policy),
            decision,
            decision,
            topo,
            active_params,
        )
        env._rorea_ctx_cache = cache
    else:
        cache.last_decision = decision
        active_params = cache.params
    return choose_rorea_ml(env, active_params)


def play_contextual(
    seed: int,
    policy: ContextualRoreaMLP | None = None,
    max_decisions: int = 200_000,
    *,
    update_interval: int = 100,
) -> dict[str, int]:
    """Run one full `SemanticMetroEnv` episode with `choose_contextual`."""
    from rl.semantic_env import SemanticMetroEnv

    env = SemanticMetroEnv(max_decisions=max_decisions)
    env.reset(seed=seed)
    delivered, decisions, acted = 0.0, 0, 0
    try:
        while decisions < max_decisions:
            action = choose_contextual(env, policy, update_interval=update_interval)
            acted += int(action != 0)
            _, reward, term, trunc, _ = env.step(action)
            delivered += float(reward)
            decisions += 1
            if term or trunc:
                break
        m = env._mediator
        return {
            "seed": seed,
            "deliveries": int(delivered),
            "decisions": decisions,
            "actions_taken": acted,
            "lines": len(m.paths),
            "stations": len(m.stations),
        }
    finally:
        env.close()
