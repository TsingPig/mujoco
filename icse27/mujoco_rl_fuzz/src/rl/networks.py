"""Shared trunk + hierarchical (mutator, param_seed_bucket) heads + value head.

Notes:
- We discretise `param_seed` into K small buckets (e.g. 16) so the head can be
  a finite categorical. Each bucket maps to a fixed seed via `param_seed_table`.
- All in PyTorch; degrades to a friendly error if torch is unavailable.
"""
from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
    _HAS_TORCH = True
except Exception:                          # pragma: no cover
    torch = None                            # type: ignore
    nn = None                               # type: ignore
    _HAS_TORCH = False


def require_torch():
    if not _HAS_TORCH:
        raise RuntimeError("torch not installed; install with `pip install torch`.")


if _HAS_TORCH:

    class SharedTrunk(nn.Module):
        def __init__(self, d_in: int, d_hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_hidden), nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)


    class ActorCritic(nn.Module):
        """Two-stage policy: p(mutator) and p(param_bucket | mutator) + V(s)."""

        def __init__(self, d_in: int, n_mut: int, n_param_buckets: int = 16,
                     n_rollout_buckets: int = 5, d_hidden: int = 128):
            super().__init__()
            self.n_mut = n_mut
            self.n_param = n_param_buckets
            self.n_roll = n_rollout_buckets
            self.trunk = SharedTrunk(d_in, d_hidden)
            self.head_mut = nn.Linear(d_hidden, n_mut)
            self.head_param = nn.Linear(d_hidden + n_mut, n_param_buckets)
            self.head_roll = nn.Linear(d_hidden + n_mut, n_rollout_buckets)
            self.value = nn.Linear(d_hidden, 1)

        def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None):
            h = self.trunk(x)
            mut_logits = self.head_mut(h)
            if mask is not None:
                mut_logits = mut_logits.masked_fill(~mask, float("-inf"))
            v = self.value(h).squeeze(-1)
            return h, mut_logits, v

        def act(self, x, mask=None):
            """Sample a hierarchical action. Returns dict with idxs and log_probs."""
            h, mut_logits, v = self.forward(x, mask=mask)
            dist_m = Categorical(logits=mut_logits)
            mut_idx = dist_m.sample()
            mut_oh = torch.nn.functional.one_hot(mut_idx, self.n_mut).float()
            cond = torch.cat([h, mut_oh], dim=-1)
            param_logits = self.head_param(cond)
            roll_logits = self.head_roll(cond)
            dist_p = Categorical(logits=param_logits)
            dist_r = Categorical(logits=roll_logits)
            p_idx = dist_p.sample()
            r_idx = dist_r.sample()
            log_prob = dist_m.log_prob(mut_idx) + dist_p.log_prob(p_idx) + dist_r.log_prob(r_idx)
            entropy = dist_m.entropy() + dist_p.entropy() + dist_r.entropy()
            return {
                "mut_idx": mut_idx, "param_idx": p_idx, "roll_idx": r_idx,
                "log_prob": log_prob, "entropy": entropy, "value": v,
            }

        def evaluate(self, x, mask, mut_idx, p_idx, r_idx):
            h, mut_logits, v = self.forward(x, mask=mask)
            dist_m = Categorical(logits=mut_logits)
            mut_oh = torch.nn.functional.one_hot(mut_idx, self.n_mut).float()
            cond = torch.cat([h, mut_oh], dim=-1)
            dist_p = Categorical(logits=self.head_param(cond))
            dist_r = Categorical(logits=self.head_roll(cond))
            log_prob = dist_m.log_prob(mut_idx) + dist_p.log_prob(p_idx) + dist_r.log_prob(r_idx)
            entropy = dist_m.entropy() + dist_p.entropy() + dist_r.entropy()
            return log_prob, entropy, v
