"""Finite-state oracle: flags non-finite values in qpos/qvel/qacc."""
from __future__ import annotations

from ..runner.types import OracleReport


class FiniteStateOracle:
    name = "finite_state"

    def evaluate(self, trace, *, replay_meta=None) -> OracleReport:
        if trace is None:
            return OracleReport(name=self.name, severity=0.0, failed=False,
                                notes={"reason": "no_trace"})
        if not trace.finite:
            return OracleReport(name=self.name, severity=10.0, failed=True,
                                notes={"reason": "non_finite_state"})
        return OracleReport(name=self.name, severity=0.0, failed=False)
