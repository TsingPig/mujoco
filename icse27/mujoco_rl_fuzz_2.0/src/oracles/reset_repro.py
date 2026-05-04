"""Reset-reproducibility oracle (placeholder summary-level check).

A full implementation requires re-executing the same trajectory protocol
multiple times and comparing summaries. This first-stage oracle reads a
pre-computed reproducibility hint from replay_meta if present.
"""
from __future__ import annotations

from ..runner.types import OracleReport


class ResetReproOracle:
    name = "reset_repro"

    def evaluate(self, trace, *, replay_meta=None) -> OracleReport:
        meta = replay_meta or {}
        delta = meta.get("reset_repro_delta")
        if delta is None:
            return OracleReport(name=self.name, severity=0.0, failed=False,
                                notes={"status": "not_evaluated"})
        try:
            d = float(delta)
        except Exception:
            return OracleReport(name=self.name, severity=0.0, failed=False)
        if d > 1e-3:
            return OracleReport(name=self.name, severity=min(5.0, d * 10),
                                failed=True, notes={"delta": d})
        return OracleReport(name=self.name, severity=0.0, failed=False,
                            notes={"delta": d})
