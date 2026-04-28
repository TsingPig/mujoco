"""Shared state featurizer: produces both `state_vec` (RL) and `state_text` (LLM)
from the same `RawState`. Single source of truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..mutations.registry import MUTATOR_IDS

# Default warning vocabulary; should match cfg.warnings_track for full coverage
DEFAULT_WARN_VOCAB: list[str] = [
    "INERTIA", "CONTACTFULL", "CNSTRFULL", "VGEOMFULL",
    "BADQPOS", "BADQVEL", "BADQACC", "BADCTRL",
]


@dataclass
class RawState:
    model_shape: tuple = (0, 0, 0, 0, 0)   # nq,nv,nu,nbody,ngeom
    last_compile_ok: bool = True
    last_runtime_ok: bool = True
    warning_counts: dict[str, int] = field(default_factory=dict)
    max_abs_qpos: float = 0.0
    max_abs_qvel: float = 0.0
    max_abs_ctrl: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    last_reward: float = 0.0
    last_novelty: float = 0.0
    mutation_history_ids: list[int] = field(default_factory=list)
    # Fix 4: identity of the *current* seed XML being mutated.
    # Without this the policy can't tell which of N seeds it is acting on.
    seed_idx: int = 0
    n_seeds: int = 1


def _safe_log1p(x: float) -> float:
    if not math.isfinite(x):
        return 10.0
    return float(math.log1p(abs(x)))


def featurize(s: RawState, history_k: int,
              warn_vocab: Optional[list[str]] = None,
              n_mut: Optional[int] = None,
              max_seeds: int = 32) -> np.ndarray:
    warn_vocab = warn_vocab or DEFAULT_WARN_VOCAB
    n_mut = n_mut or len(MUTATOR_IDS)

    parts: list[float] = []
    # 5 model dims (log1p)
    for v in s.model_shape:
        parts.append(_safe_log1p(v))
    # 3 max_abs (log1p)
    parts.append(_safe_log1p(s.max_abs_qpos))
    parts.append(_safe_log1p(s.max_abs_qvel))
    parts.append(_safe_log1p(s.max_abs_ctrl))
    # 4 bool flags
    parts.append(1.0 if s.last_compile_ok else 0.0)
    parts.append(1.0 if s.last_runtime_ok else 0.0)
    parts.append(1.0 if s.has_nan else 0.0)
    parts.append(1.0 if s.has_inf else 0.0)
    # 2 reward / novelty
    parts.append(float(s.last_reward))
    parts.append(float(s.last_novelty))
    numeric = parts  # 14

    # warning multi-hot count (log1p of count)
    warn_vec = [_safe_log1p(s.warning_counts.get(w, 0)) for w in warn_vocab]

    # mutation history: average one-hot of last K
    hist_vec = [0.0] * n_mut
    if s.mutation_history_ids:
        recent = s.mutation_history_ids[-history_k:]
        for mid in recent:
            if 0 <= mid < n_mut:
                hist_vec[mid] += 1.0
        denom = max(1, len(recent))
        hist_vec = [v / denom for v in hist_vec]

    # Fix 4: one-hot of current seed (truncated to max_seeds slots)
    seed_vec = [0.0] * max_seeds
    if 0 <= s.seed_idx < max_seeds:
        seed_vec[s.seed_idx] = 1.0

    return np.asarray(numeric + warn_vec + hist_vec + seed_vec, dtype=np.float32)


def state_text(s: RawState, warn_vocab: Optional[list[str]] = None) -> str:
    """Compact human-readable serialisation, used as LLM prompt slot."""
    warn_vocab = warn_vocab or DEFAULT_WARN_VOCAB
    nq, nv, nu, nb, ng = (list(s.model_shape) + [0] * 5)[:5]
    warn_str = ", ".join(f"{w}={s.warning_counts.get(w, 0)}" for w in warn_vocab
                         if s.warning_counts.get(w, 0) > 0) or "none"
    return (
        f"model: nq={nq} nv={nv} nu={nu} nbody={nb} ngeom={ng}\n"
        f"warnings: {warn_str}\n"
        f"max_abs: qpos={s.max_abs_qpos:.3g} qvel={s.max_abs_qvel:.3g} "
        f"ctrl={s.max_abs_ctrl:.3g}\n"
        f"flags: compile_ok={s.last_compile_ok} runtime_ok={s.last_runtime_ok} "
        f"has_nan={s.has_nan} has_inf={s.has_inf}\n"
        f"recent_mutators: {s.mutation_history_ids[-8:]}\n"
        f"last_reward: {s.last_reward:.3f}  novelty: {s.last_novelty:.3f}"
    )


def state_dim(history_k: int, warn_vocab: Optional[list[str]] = None,
              n_mut: Optional[int] = None, max_seeds: int = 32) -> int:
    warn_vocab = warn_vocab or DEFAULT_WARN_VOCAB
    n_mut = n_mut or len(MUTATOR_IDS)
    return 14 + len(warn_vocab) + n_mut + max_seeds


def raw_state_from_result(result, history: list[int], last_reward: float,
                          last_novelty: float,
                          seed_idx: int = 0, n_seeds: int = 1) -> RawState:
    return RawState(
        model_shape=tuple(result.model_shape) if result.model_shape else (0, 0, 0, 0, 0),
        last_compile_ok=bool(result.compile_ok),
        last_runtime_ok=bool(result.runtime_ok),
        warning_counts={w.wtype: w.count for w in result.warnings},
        max_abs_qpos=float(result.state_stats.get("max_abs_qpos", 0.0) or 0.0),
        max_abs_qvel=float(result.state_stats.get("max_abs_qvel", 0.0) or 0.0),
        max_abs_ctrl=float(result.state_stats.get("max_abs_ctrl", 0.0) or 0.0),
        has_nan=bool(result.state_stats.get("has_nan", False)),
        has_inf=bool(result.state_stats.get("has_inf", False)),
        last_reward=float(last_reward),
        last_novelty=float(last_novelty),
        mutation_history_ids=list(history),
        seed_idx=int(seed_idx),
        n_seeds=int(n_seeds),
    )
