"""Pluggable RL algorithms.

Available:
  - "reinforce" : REINFORCE with mean baseline (simplest policy gradient)
  - "vanilla_ac": 1-step TD actor-critic (GzFuzz-style baseline)
  - "a2c"       : synchronous N-step advantage actor-critic
  - "ppo"       : PPO + GAE, clipped surrogate (RECOMMENDED MAIN)
  - "impala"    : placeholder (V-trace, NotImplementedError; reserved for v0.3)
  - "a3c"       : not implemented intentionally (Windows-hostile)

Each algorithm exposes:
    .update(transitions: list[InternalTransition]) -> dict[str, float]
"""
from .a2c import A2CAlgorithm
from .ppo import PPOAlgorithm
from .reinforce import ReinforceAlgorithm
from .vanilla_ac import VanillaACAlgorithm

__all__ = ["A2CAlgorithm", "PPOAlgorithm", "ReinforceAlgorithm",
           "VanillaACAlgorithm", "make_algorithm"]


def make_algorithm(name: str, **kwargs):
    n = name.lower()
    if n == "reinforce":
        return ReinforceAlgorithm(**kwargs)
    if n in ("vanilla_ac", "ac"):
        return VanillaACAlgorithm(**kwargs)
    if n == "a2c":
        return A2CAlgorithm(**kwargs)
    if n == "ppo":
        return PPOAlgorithm(**kwargs)
    if n == "impala":
        raise NotImplementedError(
            "IMPALA (V-trace) is reserved for v0.3. Use 'ppo' as the advanced default.")
    if n == "a3c":
        raise NotImplementedError(
            "A3C is intentionally not supported (Windows + multiprocessing fork issues "
            "+ resource contention with fuzzing subprocesses). Use 'ppo' or 'a2c'.")
    raise ValueError(f"unknown algorithm: {name}")
