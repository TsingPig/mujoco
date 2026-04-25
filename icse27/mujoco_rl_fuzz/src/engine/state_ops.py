"""State perturbation helpers + state stats.

Used by both the main process (when constructing perturbation requests for the
worker) and the worker itself (statistics post-rollout).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Special string values transported via JSON to denote NaN/Inf
SPECIAL = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}


def encode_value(v: float) -> Any:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "+inf" if v > 0 else "-inf"
    return float(v)


def decode_value(v: Any) -> float:
    if isinstance(v, str) and v in SPECIAL:
        return SPECIAL[v]
    return float(v)


def state_stats(qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray, ncon: int) -> dict[str, float]:
    def _max_abs(a: np.ndarray) -> float:
        if a.size == 0:
            return 0.0
        # Use nanmax so NaN-injected arrays still report finite max if other entries exist.
        finite = np.where(np.isfinite(a), np.abs(a), 0.0)
        return float(finite.max()) if finite.size else 0.0

    return {
        "max_abs_qpos": _max_abs(qpos),
        "max_abs_qvel": _max_abs(qvel),
        "max_abs_ctrl": _max_abs(ctrl),
        "has_nan": bool(np.isnan(qpos).any() or np.isnan(qvel).any() or np.isnan(ctrl).any()),
        "has_inf": bool(np.isinf(qpos).any() or np.isinf(qvel).any() or np.isinf(ctrl).any()),
        "ncon": int(ncon),
    }
