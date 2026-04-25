from __future__ import annotations
import math
from .base import BaseOracle, OracleVerdict
from ..result import ExecutionResult


class ConsistencyOracle(BaseOracle):
    """Same-process double-run determinism check (worker computed `consistency_diff`)."""
    name = "consistency"
    threshold: float = 1e-9

    def evaluate(self, result: ExecutionResult, raw: dict) -> OracleVerdict:
        diff = result.consistency_diff
        if diff is None:
            return OracleVerdict(self.name, triggered=False)
        if math.isnan(diff):
            return OracleVerdict(self.name, triggered=True, severity=5,
                                 tags=["consistency_nan"], details={"diff": diff})
        if diff > self.threshold:
            return OracleVerdict(self.name, triggered=True, severity=5,
                                 tags=["nondeterministic"], details={"diff": diff})
        return OracleVerdict(self.name, triggered=False, details={"diff": diff})
