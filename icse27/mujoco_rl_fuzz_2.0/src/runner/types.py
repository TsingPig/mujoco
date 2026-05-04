"""Runner request/result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..trajectory.recorder import TraceSummary


@dataclass
class OracleReport:
    name: str
    severity: float = 0.0          # 0.0 = pass, >0 = anomalous
    failed: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRequest:
    seed: Any                       # an ActorSeed | SyntheticSceneSeed | OpenEnvSeed | TrajectorySeed
    protocol: Optional[Any] = None  # TrajectoryProtocol; if None, defaults from seed
    oracles: Optional[List[str]] = None  # subset of registered oracles


@dataclass
class RunResult:
    ok: bool
    seed_id: str = ""
    layer: str = ""
    trace: Optional[TraceSummary] = None
    oracle_reports: List[OracleReport] = field(default_factory=list)
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> float:
        return sum(r.severity for r in self.oracle_reports)

    @property
    def any_failed(self) -> bool:
        return any(r.failed for r in self.oracle_reports)
