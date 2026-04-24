"""Config loader: YAML -> FuzzConfig dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class RunCfg:
    mode: str = "random"
    budget: int = 1000
    timeout_sec: float = 10.0
    seed: int = 0
    out_dir: str = "outputs"
    log_dir: str = "logs"


@dataclass
class SeedsCfg:
    dir: str = "seeds"
    pattern: str = "*.xml"


@dataclass
class RolloutCfg:
    steps_buckets: list[int] = field(default_factory=lambda: [1, 5, 10, 50, 100])
    default_steps: int = 50


@dataclass
class RLCfg:
    policy: str = "actor_critic"      # random | bandit | actor_critic
    hidden_dim: int = 128
    history_k: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    rollout_per_update: int = 64


@dataclass
class LLMCfg:
    enabled: bool = False
    model_name: str = "Qwen/Qwen2.5-1.5B"
    max_new_tokens: int = 64


@dataclass
class FuzzConfig:
    run: RunCfg = field(default_factory=RunCfg)
    seeds: SeedsCfg = field(default_factory=SeedsCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    mutators: list[str] = field(default_factory=list)
    buckets: dict[str, Any] = field(default_factory=dict)
    warnings_track: list[str] = field(default_factory=list)
    reward: dict[str, float] = field(default_factory=dict)
    rl: RLCfg = field(default_factory=RLCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)


def _merge(dc_cls, raw: dict | None):
    if not raw:
        return dc_cls()
    return dc_cls(**{k: v for k, v in raw.items() if k in dc_cls.__dataclass_fields__})


def load_config(path: str) -> FuzzConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = FuzzConfig(
        run=_merge(RunCfg, raw.get("run")),
        seeds=_merge(SeedsCfg, raw.get("seeds")),
        rollout=_merge(RolloutCfg, raw.get("rollout")),
        mutators=list(raw.get("mutators") or []),
        buckets=dict(raw.get("buckets") or {}),
        warnings_track=list(raw.get("warnings_track") or []),
        reward=dict(raw.get("reward") or {}),
        rl=_merge(RLCfg, raw.get("rl")),
        llm=_merge(LLMCfg, raw.get("llm")),
    )
    return cfg
