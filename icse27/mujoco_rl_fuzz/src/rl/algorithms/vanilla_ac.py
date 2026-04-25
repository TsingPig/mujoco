"""Vanilla 1-step Actor-Critic (a.k.a. "GzFuzz-style").

This is the simplest form of policy-gradient + critic baseline:
    advantage_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    L_actor  = -log_pi(a_t|s_t) * advantage_t.detach()
    L_critic =  0.5 * advantage_t^2
    L_total  =  L_actor + value_coef * L_critic - entropy_coef * H(pi)

Compared with A2C/PPO this version:
  * uses 1-step TD (no n-step / no GAE)
  * does NOT standardise advantages
  * does ONE gradient step per buffer flush (no minibatch / no epochs)
This matches GzFuzz's published actor-critic baseline.
"""
from __future__ import annotations

import numpy as np

from ..networks import _HAS_TORCH, ActorCritic, require_torch

if _HAS_TORCH:
    import torch
    from torch import optim


class VanillaACAlgorithm:
    name = "vanilla_ac"

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
        if len(batch) < 2:
            return {}
        x = torch.tensor(np.stack([b["state_vec"] for b in batch]),
                         dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([b["mask"] for b in batch]),
                             dtype=torch.bool, device=self.device)
        mut = torch.tensor([b["mut_idx"] for b in batch], dtype=torch.long, device=self.device)
        par = torch.tensor([b["param_idx"] for b in batch], dtype=torch.long, device=self.device)
        rol = torch.tensor([b["roll_idx"] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([float(b["reward"]) for b in batch],
                               dtype=torch.float32, device=self.device)

        log_prob, entropy, value = self.net.evaluate(x, masks, mut, par, rol)
        # 1-step TD target: V(s_{t+1}) bootstrap; last step uses 0
        with torch.no_grad():
            v_next = torch.cat([value[1:], value.new_zeros(1)])
            td_target = rewards + self.gamma * v_next
            advantage = td_target - value
        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = 0.5 * (advantage ** 2).mean()
        entropy_bonus = entropy.mean()
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.opt.step()
        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_bonus.item()),
        }
