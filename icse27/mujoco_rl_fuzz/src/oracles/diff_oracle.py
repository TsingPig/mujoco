"""Differential oracle: same XML run under different solvers/integrators.

Implementation is in the runner (it must spawn extra subprocesses); this file
only provides verdict aggregation given diff values from the runner.
"""
from __future__ import annotations
from .base import BaseOracle, OracleVerdict
from ..result import ExecutionResult


class SolverDiffOracle(BaseOracle):
    name = "solver_diff"
    threshold: float = 1e-3

    def evaluate(self, result: ExecutionResult, raw: dict) -> OracleVerdict:
        diff = (raw or {}).get("solver_diff")
        if diff is None:
            return OracleVerdict(self.name, triggered=False)
        if not isinstance(diff, dict):
            return OracleVerdict(self.name, triggered=False)
        max_d = float(diff.get("max_qpos_diff", 0.0))
        ref = diff.get("reference_solver")
        cmp = diff.get("compared_solver")
        triggered = max_d > self.threshold
        return OracleVerdict(self.name, triggered=triggered,
                             severity=6 if triggered else 0,
                             tags=[f"solver_diff:{ref}->{cmp}"] if triggered else [],
                             details=diff)
