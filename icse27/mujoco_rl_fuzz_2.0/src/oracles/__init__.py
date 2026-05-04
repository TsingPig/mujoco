"""Static, dynamic, and differential oracles.

Each oracle takes a TraceSummary and (optionally) extra context and
returns an OracleReport. Oracles never raise.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any

from .finite_state import FiniteStateOracle
from .instability import InstabilityOracle
from .contact_force import ContactForceOracle
from .reset_repro import ResetReproOracle
from .sensor_reward import SensorRewardOracle
from .backend_diff import BackendDiffOracle


_REGISTRY = {
    "finite_state": FiniteStateOracle(),
    "instability": InstabilityOracle(),
    "contact_force": ContactForceOracle(),
    "reset_repro": ResetReproOracle(),
    "sensor_reward": SensorRewardOracle(),
    "backend_diff": BackendDiffOracle(),
}


def list_oracles() -> List[str]:
    return list(_REGISTRY.keys())


def run_oracles(trace, *, oracle_filter: Optional[List[str]] = None,
                replay_meta: Optional[Dict[str, Any]] = None):
    out = []
    for name, oracle in _REGISTRY.items():
        if oracle_filter is not None and name not in oracle_filter:
            continue
        try:
            rep = oracle.evaluate(trace, replay_meta=replay_meta or {})
        except Exception as exc:  # pragma: no cover - defensive
            from ..runner.types import OracleReport
            rep = OracleReport(name=name, severity=0.0, failed=False,
                               notes={"oracle_exception": str(exc)})
        out.append(rep)
    return out


__all__ = [
    "list_oracles", "run_oracles",
    "FiniteStateOracle", "InstabilityOracle", "ContactForceOracle",
    "ResetReproOracle", "SensorRewardOracle", "BackendDiffOracle",
]
