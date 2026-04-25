"""Reward decomposition (guide §14)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..result import ExecutionResult


@dataclass
class RewardBreakdown:
    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)


def compute_reward(result: ExecutionResult, is_new_signature: bool,
                   sig_kind: str, weights: dict[str, float]) -> RewardBreakdown:
    """All weights come from cfg.reward (default in configs/default.yaml)."""
    w = weights
    comp: dict[str, float] = {}

    if result.returncode != 0 or result.timeout:
        comp["crash"] = w.get("crash", 10.0)
    if result.exception_type and result.exception_type.startswith("Fatal"):
        comp["fatal_error"] = w.get("fatal_error", 8.0)
    if result.timeout:
        comp["timeout"] = w.get("timeout", 5.0)

    # warning bonuses
    if result.warnings:
        if is_new_signature and sig_kind in ("warning_only", "runtime"):
            comp["new_warning"] = w.get("new_warning", 4.0)
        else:
            comp["known_warning"] = w.get("known_warning", 2.0)

    if result.consistency_diff is not None and result.consistency_diff > 1e-9:
        comp["inconsistency"] = w.get("inconsistency", 3.0)

    if is_new_signature and sig_kind == "ok":
        comp["novel_state_bucket"] = w.get("novel_state_bucket", 1.0)

    if not result.compile_ok and not result.warnings and result.returncode == 0:
        comp["trivial_compile_fail"] = w.get("trivial_compile_fail", -1.0)

    if not is_new_signature and sig_kind in ("warning_only", "compile", "runtime"):
        comp["duplicate_signature"] = w.get("duplicate_signature", -2.0)

    total = float(sum(comp.values()))
    return RewardBreakdown(total=total, components=comp)
