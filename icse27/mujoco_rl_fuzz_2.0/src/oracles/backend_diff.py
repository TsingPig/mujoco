"""Backend differential oracle (interface only in v2.0-β).

The interface accepts replay_meta with `backend_diff_residual` to support
future MJX-vs-classic comparisons. By default it reports unsupported.
"""
from __future__ import annotations

from ..runner.types import OracleReport


class BackendDiffOracle:
    name = "backend_diff"

    RESIDUAL_THRESHOLD = 1e-3

    def evaluate(self, trace, *, replay_meta=None) -> OracleReport:
        meta = replay_meta or {}
        residual = meta.get("backend_diff_residual")
        if residual is None:
            return OracleReport(name=self.name, severity=0.0, failed=False,
                                notes={"status": "unsupported_in_classic_only_mode"})
        try:
            r = float(residual)
        except Exception:
            return OracleReport(name=self.name, severity=0.0, failed=False,
                                notes={"status": "invalid_residual"})
        if r > self.RESIDUAL_THRESHOLD:
            return OracleReport(name=self.name, severity=min(5.0, r / self.RESIDUAL_THRESHOLD),
                                failed=True, notes={"residual": r})
        return OracleReport(name=self.name, severity=0.0, failed=False,
                            notes={"residual": r})
