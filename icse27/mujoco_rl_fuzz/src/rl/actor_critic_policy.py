"""Adapts the ActorCritic network behind the unified `Policy` protocol.

The policy keeps an internal rollout buffer of `rollout_per_update` recent
transitions; when full, it triggers a single algorithm.update() call.

Important: `param_seed` is sampled from `param_seed_table` indexed by the
policy's `param_idx` head — this is how RL/LLM share a finite discrete action
space without having to learn each mutator's idiosyncratic param schema.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..mutations.registry import MUTATOR_IDS
from .algorithms import make_algorithm
from .features import featurize, state_dim
from .networks import _HAS_TORCH, require_torch
from .policy_api import Action, Transition

if _HAS_TORCH:
    import torch


# A fixed table of param seeds. param_idx in [0, len(table)) -> int seed
PARAM_SEED_TABLE = [
    13, 29, 47, 73, 101, 137, 173, 211,
    251, 293, 337, 383, 431, 479, 521, 577,
]


class ActorCriticPolicy:
    """Either A2C or PPO under the hood, selected by `algorithm`."""

    def __init__(self, algorithm: str = "ppo", history_k: int = 8,
                 n_rollout_buckets: int = 5, hidden_dim: int = 128,
                 lr: float = 3e-4, gamma: float = 0.99,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 rollout_per_update: int = 64, device: str = "cpu",
                 seed: int = 0):
        require_torch()
        self.name = f"actor_critic_{algorithm}"
        self.history_k = history_k
        self.n_rollout_buckets = n_rollout_buckets
        self.n_param = len(PARAM_SEED_TABLE)
        self.n_mut = len(MUTATOR_IDS)
        self.d_in = state_dim(history_k=history_k)
        self.algo = make_algorithm(
            algorithm,
            d_in=self.d_in, n_mut=self.n_mut,
            n_param_buckets=self.n_param,
            n_rollout_buckets=n_rollout_buckets,
            d_hidden=hidden_dim,
            lr=lr, gamma=gamma,
            entropy_coef=entropy_coef, value_coef=value_coef,
            device=device,
        )
        self.rollout_per_update = rollout_per_update
        self.device = device
        self._buf: list[dict] = []
        self._last_act: dict | None = None
        self._last_state: np.ndarray | None = None
        self._last_mask: list[bool] | None = None
        torch.manual_seed(seed)

    def select(self, state_vec, state_text: str,
               action_mask: Optional[list[bool]] = None) -> Action:
        if state_vec is None:
            state_vec = np.zeros(self.d_in, dtype=np.float32)
        if action_mask is None:
            action_mask = [True] * self.n_mut
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.tensor([action_mask], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            out = self.algo.net.act(x, mask=m)
        mut_idx = int(out["mut_idx"].item())
        param_idx = int(out["param_idx"].item())
        roll_idx = int(out["roll_idx"].item())
        log_prob = float(out["log_prob"].item())
        value = float(out["value"].item())
        self._last_act = {"mut_idx": mut_idx, "param_idx": param_idx,
                          "roll_idx": roll_idx, "log_prob": log_prob,
                          "value": value}
        self._last_state = state_vec
        self._last_mask = list(action_mask)
        return Action(
            mutator_idx=mut_idx,
            param_seed=PARAM_SEED_TABLE[param_idx % self.n_param],
            rollout_bucket_idx=roll_idx,
        )

    def update(self, batch: list[Transition]) -> dict[str, float]:
        if not batch or self._last_act is None or self._last_state is None:
            return {}
        # The runner feeds one transition per step; we accumulate.
        for tr in batch:
            self._buf.append({
                "state_vec": self._last_state,
                "mask": self._last_mask or [True] * self.n_mut,
                "mut_idx": self._last_act["mut_idx"],
                "param_idx": self._last_act["param_idx"],
                "roll_idx": self._last_act["roll_idx"],
                "log_prob": self._last_act["log_prob"],
                "value": self._last_act["value"],
                "reward": float(tr.reward),
            })
        if len(self._buf) >= self.rollout_per_update:
            metrics = self.algo.update(self._buf)
            self._buf.clear()
            return metrics
        return {}
