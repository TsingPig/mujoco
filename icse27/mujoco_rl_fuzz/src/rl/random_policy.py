from __future__ import annotations
import random
from typing import Optional

from .policy_api import Action, Policy, Transition
from ..mutations.registry import MUTATOR_IDS


class RandomPolicy:
    name = "random"

    def __init__(self, n_rollout_buckets: int, seed: int = 0):
        self._rng = random.Random(seed)
        self._n_mut = len(MUTATOR_IDS)
        self._n_rb = max(1, n_rollout_buckets)

    def select(self, state_vec, state_text: str,
               action_mask: Optional[list[bool]] = None) -> Action:
        if action_mask:
            choices = [i for i, m in enumerate(action_mask) if m]
            mut = self._rng.choice(choices) if choices else self._rng.randrange(self._n_mut)
        else:
            mut = self._rng.randrange(self._n_mut)
        return Action(
            mutator_idx=mut,
            param_seed=self._rng.randrange(2**31),
            rollout_bucket_idx=self._rng.randrange(self._n_rb),
        )

    def update(self, batch: list[Transition]) -> dict[str, float]:
        return {}
