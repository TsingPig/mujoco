from __future__ import annotations
from .base import BaseOracle, OracleVerdict
from ..result import ExecutionResult


class CompileOracle(BaseOracle):
    name = "compile"

    def evaluate(self, result: ExecutionResult, raw: dict) -> OracleVerdict:
        if result.compile_ok:
            return OracleVerdict(self.name, triggered=False, severity=0,
                                 tags=["compile_ok"])
        return OracleVerdict(self.name, triggered=True, severity=3,
                             tags=["compile_fail",
                                   f"etype:{result.exception_type}"],
                             details={"traceback": result.traceback_summary})
