"""REINFORCE (Williams 1992): Monte-Carlo policy gradient with baseline.

Actor-only -- the critic head's output is reused only as a leave-one-out
mean baseline for variance reduction (no value loss).

    L = - sum_t  log_pi(a_t|s_t) * (R_t - b)
    where R_t = sum_{k=t..T} gamma^{k-t} * r_k
          b   = mean(R)  (constant baseline; standard variance-reduction)

Useful as the *simplest possible* policy-gradient ablation.
"""
from __future__ import annotations

import numpy as np

from ..networks import _HAS_TORCH, ActorCritic, require_torch

if _HAS_TORCH:
    import torch
    from torch import optim


class ReinforceAlgorithm:
    name = "reinforce"

    def __init__(self, d_in: int, n_mut: int, n_param_buckets: int = 16,
                 n_rollout_buckets: int = 5, d_hidden: int = 128,
                 lr: float = 3e-4, gamma: float = 0.99,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 use_baseline: bool = True, device: str = "cpu",
                 **_unused):
        # NOTE: REINFORCE has no value loss; we silently accept value_coef etc.
        # so the same caller kwargs as A2C/PPO work without branching.
        require_torch()
        self.net = ActorCritic(d_in, n_mut, n_param_buckets,
                               n_rollout_buckets, d_hidden).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_baseline = use_baseline
        self.device = device

    def update(self, batch: list[dict]) -> dict[str, float]:
        if not batch:
            return {}
        x = torch.tensor(np.stack([b["state_vec"] for b in batch]),
                         dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([b["mask"] for b in batch]),
                             dtype=torch.bool, device=self.device)
        mut = torch.tensor([b["mut_idx"] for b in batch], dtype=torch.long, device=self.device)
        par = torch.tensor([b["param_idx"] for b in batch], dtype=torch.long, device=self.device)
        rol = torch.tensor([b["roll_idx"] for b in batch], dtype=torch.long, device=self.device)
        rewards = [float(b["reward"]) for b in batch]
        # Discounted MC return
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if self.use_baseline:
            returns_t = returns_t - returns_t.mean()

        log_prob, entropy, _ = self.net.evaluate(x, masks, mut, par, rol)
        loss = -(log_prob * returns_t).mean() - self.entropy_coef * entropy.mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.opt.step()
        return {
            "loss": float(loss.item()),
            "entropy": float(entropy.mean().item()),
            "return_mean": float(np.mean(rewards)),
        }
