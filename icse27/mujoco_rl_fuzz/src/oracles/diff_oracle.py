"""Differential solver oracle.

Re-runs the same XML/state under different (solver, integrator) combinations
in the worker, then compares final qpos. Disagreement above `threshold` on a
finite, well-conditioned trajectory is the strongest non-crash bug signal we
have: at least one solver got the dynamics wrong.

The actual differential rollouts happen inside the worker
(`subprocess_worker.run_one`). This oracle just reads the resulting payload.
"""
from __future__ import annotations

import math
from typing import Any

from .base import BaseOracle, OracleVerdict
from ..result import ExecutionResult


class SolverDiffOracle(BaseOracle):
    name = "solver_diff"
    # Threshold on same-integrator solver disagreement. With iterations=200
    # and tol=1e-10 all three convex solvers should converge to within
    # ~1e-6. Anything above 1e-4 is suspicious; 1e-3 is almost certainly a
    # bug in one of the solvers.
    threshold: float = 1e-3

    def evaluate(self, result: ExecutionResult, raw: dict) -> OracleVerdict:
        sd: Any = getattr(result, "solver_diff", None) or (raw or {}).get("solver_diff")
        if not sd or not isinstance(sd, dict):
            return OracleVerdict(self.name, triggered=False)
        if "error" in sd:
            return OracleVerdict(self.name, triggered=False,
                                 details={"error": sd["error"]})
        per = sd.get("per_combo_diff_vs_ref") or {}
        solver_only = sd.get("solver_only_max_diff")
        if solver_only is None:
            return OracleVerdict(self.name, triggered=False, details={"per": per})
        try:
            solver_only = float(solver_only)
        except Exception:
            return OracleVerdict(self.name, triggered=False)
        # Same-integrator solver triplet must agree at converged tolerance.
        if not math.isfinite(solver_only):
            return OracleVerdict(self.name, triggered=False,
                                 details={"solver_only_max_diff": solver_only})
        if solver_only <= self.threshold:
            return OracleVerdict(self.name, triggered=False,
                                 details={"solver_only_max_diff": solver_only})
        # Real signal: flag.
        tags = ["real_bug_candidate", "solver_disagreement"]
        return OracleVerdict(
            self.name,
            triggered=True,
            severity=9,
            tags=tags,
            details={
                "solver_only_max_diff": solver_only,
                "per_combo": per,
            },
        )
