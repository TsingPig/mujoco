"""Real-bug oracle: NaN/Inf produced spontaneously by MuJoCo.

Definition of "real bug" used here:
  - The MJCF compiled successfully (no fuzzer-internal invalid input).
  - The fuzzer did NOT inject NaN/Inf into qpos/qvel/ctrl.
  - The rollout still produced NaN/Inf in qpos/qvel, OR raised BADQPOS/BADQVEL/
    BADQACC, OR aborted before the requested step count was reached.

If all three hold, MuJoCo turned a finite, well-formed system into garbage on
its own — this is the kind of finding worth reporting upstream. We mark
severity 9 (just below crash) and tag the testcase as `real_bug_candidate`.
"""
from __future__ import annotations

from .base import BaseOracle, OracleVerdict
from ..result import ExecutionResult


_BAD_WARNS = {"BADQPOS", "BADQVEL", "BADQACC"}


class SpontaneousNanOracle(BaseOracle):
    name = "spontaneous_nan"

    def evaluate(self, result: ExecutionResult, raw: dict) -> OracleVerdict:
        # Only meaningful when we actually got past compile.
        if not result.compile_ok:
            return OracleVerdict(self.name, triggered=False)

        injected = bool((raw or {}).get("injected_nan_inf", False))
        if injected:
            # Fuzzer fed NaN/Inf in; resulting NaN is expected propagation.
            return OracleVerdict(self.name, triggered=False, tags=["injected_nan_skipped"])

        tags: list[str] = []
        sev = 0

        if result.state_stats.get("has_nan"):
            tags.append("spontaneous_nan"); sev = max(sev, 9)
        if result.state_stats.get("has_inf"):
            tags.append("spontaneous_inf"); sev = max(sev, 9)

        for w in result.warnings:
            if w.wtype in _BAD_WARNS:
                tags.append(f"spontaneous_warn:{w.wtype}")
                sev = max(sev, 8)

        # Early termination of rollout while compile_ok and no injection
        # usually means an internal MuJoCo numerical failure.
        requested = int((raw or {}).get("rollout_steps_requested", 0))
        if requested > 0 and result.steps_done < requested and not result.runtime_ok \
                and result.exception_type:
            tags.append(f"early_abort:{result.exception_type}")
            sev = max(sev, 7)

        return OracleVerdict(
            self.name,
            triggered=bool(tags),
            severity=sev,
            tags=(["real_bug_candidate"] + tags) if tags else [],
        )
