"""A2C (synchronous N-step advantage actor-critic).

InternalTransition: dict with keys
  state_vec (np), mask (bool list), mut_idx, param_idx, roll_idx,
  log_prob (float), value (float), reward (float)
"""
from __future__ import annotations

import numpy as np

from ..networks import _HAS_TORCH, ActorCritic, require_torch

if _HAS_TORCH:
    import torch
    from torch import optim


class A2CAlgorithm:
    def __init__(self, d_in: int, n_mut: int, n_param_buckets: int = 16,
                 n_rollout_buckets: int = 5, d_hidden: int = 128,
                 lr: float = 3e-4, gamma: float = 0.99,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5, device: str = "cpu"):
        require_torch()
        self.net = ActorCritic(d_in, n_mut, n_param_buckets,
                               n_rollout_buckets, d_hidden).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
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

        # N-step return (no bootstrap; assume episode-terminal at batch end)
        returns: list[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        log_prob, entropy, value = self.net.evaluate(x, masks, mut, par, rol)
        adv = (ret_t - value).detach()
        # standardise advantages
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(log_prob * adv).mean()
        value_loss = ((ret_t - value) ** 2).mean()
        ent_loss = -entropy.mean()
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * ent_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.mean().item()),
            "mean_return": float(ret_t.mean().item()),
        }
