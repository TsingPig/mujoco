from __future__ import annotations
from .base import BaseOracle, OracleVerdict
from ..result import ExecutionResult


class RuntimeOracle(BaseOracle):
    """Crashes / timeouts / NaN-Inf / runtime exceptions / known warnings."""
    name = "runtime"

    def evaluate(self, result: ExecutionResult, raw: dict) -> OracleVerdict:
        tags: list[str] = []
        sev = 0

        if result.timeout:
            tags.append("timeout"); sev = max(sev, 7)
        if result.returncode != 0:
            tags.append(f"returncode:{result.returncode}"); sev = max(sev, 9)
        if not result.compile_ok:
            return OracleVerdict(self.name, triggered=False)  # delegated to compile_oracle
        if not result.runtime_ok and result.exception_type:
            tags.append(f"runtime_etype:{result.exception_type}"); sev = max(sev, 8)
        if result.state_stats.get("has_nan"):
            tags.append("has_nan"); sev = max(sev, 6)
        if result.state_stats.get("has_inf"):
            tags.append("has_inf"); sev = max(sev, 6)
        for w in result.warnings:
            tags.append(f"warn:{w.wtype}"); sev = max(sev, 4)

        return OracleVerdict(self.name, triggered=bool(tags), severity=sev, tags=tags)
