"""PPO + GAE (clipped surrogate). Recommended main RL algorithm.

Same InternalTransition format as A2C. Stores `log_prob` from the *behaviour*
policy at action-time, and uses it as the importance-sampling reference.
"""
from __future__ import annotations

import numpy as np

from ..networks import _HAS_TORCH, ActorCritic, require_torch

if _HAS_TORCH:
    import torch
    from torch import optim


class PPOAlgorithm:
    def __init__(self, d_in: int, n_mut: int, n_seed: int = 16,
                 n_param_buckets: int = 16, n_rollout_buckets: int = 5, d_hidden: int = 128,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, epochs: int = 4, minibatch: int = 32,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5, device: str = "cpu"):
        require_torch()
        self.net = ActorCritic(d_in, n_mut, n_seed, n_param_buckets,
                               n_rollout_buckets, d_hidden).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch = minibatch
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def _gae(self, rewards: list[float], values: list[float]) -> tuple[list[float], list[float]]:
        # bootstrap with 0 (assume terminal at end of batch)
        adv: list[float] = [0.0] * len(rewards)
        gae = 0.0
        next_v = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
            next_v = values[t]
        returns = [a + v for a, v in zip(adv, values)]
        return adv, returns

    def update(self, batch: list[dict]) -> dict[str, float]:
        if not batch:
            return {}
        x = torch.tensor(np.stack([b["state_vec"] for b in batch]),
                         dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([b["mask"] for b in batch]),
                             dtype=torch.bool, device=self.device)
        see = torch.tensor([b["seed_idx"] for b in batch], dtype=torch.long, device=self.device)
        mut = torch.tensor([b["mut_idx"] for b in batch], dtype=torch.long, device=self.device)
        par = torch.tensor([b["param_idx"] for b in batch], dtype=torch.long, device=self.device)
        rol = torch.tensor([b["roll_idx"] for b in batch], dtype=torch.long, device=self.device)
        old_log = torch.tensor([float(b["log_prob"]) for b in batch],
                               dtype=torch.float32, device=self.device)

        rewards = [float(b["reward"]) for b in batch]
        values = [float(b["value"]) for b in batch]
        adv_l, ret_l = self._gae(rewards, values)
        adv = torch.tensor(adv_l, dtype=torch.float32, device=self.device)
        ret = torch.tensor(ret_l, dtype=torch.float32, device=self.device)
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = x.shape[0]
        idx_all = np.arange(N)
        last_metrics = {}
        for _ in range(self.epochs):
            np.random.shuffle(idx_all)
            for s in range(0, N, self.minibatch):
                mb = idx_all[s:s + self.minibatch]
                if len(mb) == 0:
                    continue
                mb_t = torch.tensor(mb, dtype=torch.long, device=self.device)
                log_prob, entropy, value = self.net.evaluate(
                    x[mb_t], masks[mb_t], see[mb_t], mut[mb_t], par[mb_t], rol[mb_t])
                ratio = torch.exp(log_prob - old_log[mb_t])
                surr1 = ratio * adv[mb_t]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb_t]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((ret[mb_t] - value) ** 2).mean()
                ent_loss = -entropy.mean()
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * ent_loss
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                last_metrics = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.mean().item()),
                    "mean_return": float(ret.mean().item()),
                    "approx_kl": float((old_log[mb_t] - log_prob).mean().item()),
                }
        return last_metrics
