"""Oracle base + verdict.

Oracles are pure functions of (ExecutionResult, raw_worker_dict).
They never run MuJoCo. The runner aggregates their verdicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..result import ExecutionResult


@dataclass
class OracleVerdict:
    name: str
    triggered: bool
    severity: int = 0           # 0..10
    tags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class BaseOracle:
    name: str = "abstract"

    def evaluate(self, result: ExecutionResult, raw: dict) -> OracleVerdict:
        raise NotImplementedError
