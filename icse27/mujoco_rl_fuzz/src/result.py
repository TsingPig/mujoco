"""ExecutionResult + WarningRecord dataclasses (guide Appendix A)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WarningRecord:
    wtype: str
    count: int
    message: str = ""


@dataclass
class ExecutionResult:
    tc_id: str
    compile_ok: bool = False
    runtime_ok: bool = False
    warnings: list[WarningRecord] = field(default_factory=list)
    exception_type: Optional[str] = None
    traceback_summary: Optional[str] = None
    returncode: int = 0
    timeout: bool = False
    steps_done: int = 0
    state_stats: dict[str, float] = field(default_factory=dict)
    consistency_diff: Optional[float] = None
    model_shape: tuple[int, ...] = ()


def result_to_dict(r: ExecutionResult) -> dict[str, Any]:
    return {
        "tc_id": r.tc_id,
        "compile_ok": r.compile_ok,
        "runtime_ok": r.runtime_ok,
        "warnings": [w.__dict__ for w in r.warnings],
        "exception_type": r.exception_type,
        "traceback_summary": r.traceback_summary,
        "returncode": r.returncode,
        "timeout": r.timeout,
        "steps_done": r.steps_done,
        "state_stats": r.state_stats,
        "consistency_diff": r.consistency_diff,
        "model_shape": list(r.model_shape),
    }
