"""Param bucket utilities + bucket-key sampling.

Buckets are loaded from the FuzzConfig (`cfg.buckets`). All mutators consume
*bucket keys* (strings or indices) so that RL/LLM action spaces stay discrete.
"""
from __future__ import annotations

from typing import Any


class Buckets:
    def __init__(self, raw: dict[str, Any]):
        self._raw = raw or {}

    def keys(self, name: str) -> list[str]:
        v = self._raw.get(name)
        if v is None:
            return []
        if isinstance(v, dict):
            return list(v.keys())
        if isinstance(v, list):
            return [str(i) for i in range(len(v))]
        return []

    def value(self, name: str, key: str) -> Any:
        v = self._raw.get(name)
        if isinstance(v, dict):
            return v.get(key)
        if isinstance(v, list):
            try:
                return v[int(key)]
            except (ValueError, IndexError):
                return None
        return None

    def sample_key(self, name: str, rng) -> str:
        ks = self.keys(name)
        if not ks:
            return ""
        return rng.choice(ks)
