"""Instability oracle: flags qvel/qacc magnitude blowups."""
from __future__ import annotations

from ..runner.types import OracleReport


class InstabilityOracle:
    name = "instability"

    # Empirical thresholds; tuned conservatively. Refined later via calibration.
    QVEL_MAX_THRESHOLD = 1e3
    QACC_MAX_THRESHOLD = 1e5

    def evaluate(self, trace, *, replay_meta=None) -> OracleReport:
        if trace is None:
            return OracleReport(name=self.name)
        v = trace.qvel_norm_max
        a = trace.qacc_norm_max
        sev = 0.0
        notes = {}
        if v > self.QVEL_MAX_THRESHOLD:
            sev += min(5.0, v / self.QVEL_MAX_THRESHOLD)
            notes["qvel_max"] = v
        if a > self.QACC_MAX_THRESHOLD:
            sev += min(5.0, a / self.QACC_MAX_THRESHOLD)
            notes["qacc_max"] = a
        return OracleReport(name=self.name, severity=sev, failed=sev > 0.5, notes=notes)
