"""Unified Policy protocol. random / bandit / actor-critic / LLM all implement this.

Action = (mutator_idx: int, param_seed: int, rollout_steps: int, seed_idx: int = 0)
- mutator_idx selects which BaseMutator to invoke
- param_seed is fed to the mutator's internal RNG (decouples policy from each
  mutator's idiosyncratic param schema; keeps action space uniform)
- rollout_steps is bucketised at the runner level
- seed_idx is the policy's choice of which seed to mutate (0..n_seeds-1); random ignores it

This indirection lets RL/LLM share the SAME 10-way × integer-bucket × seed-count
discrete action space without learning per-mutator parameter schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class Action:
    mutator_idx: int
    param_seed: int = 0
    rollout_bucket_idx: int = 2          # default = index in cfg.rollout.steps_buckets
    seed_idx: int = 0                     # default = policy chooses which seed (0..n_seeds-1)


@dataclass
class Transition:
    state_vec: Any                       # np.ndarray
    state_text: str
    action: Action
    reward: float
    reward_components: dict[str, float] = field(default_factory=dict)
    next_state_vec: Optional[Any] = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    name: str

    def select(self, state_vec, state_text: str,
               action_mask: Optional[Any] = None) -> Action: ...

    def update(self, batch: list[Transition]) -> dict[str, float]: ...
