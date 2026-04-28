"""Mutator base protocol.

A mutator is a high-semantic edit on an MJCF tree, parameterised only by
an `intensity_mode` (a fixed string from `IntensityTable`). The mutator
itself owns the mapping `(intensity_mode, rng) -> concrete attribute values`
by querying the table; callers never pass raw numbers.

Contract:

    if mut.applicable(model):
        snap = model.snapshot()
        result = mut.apply(model, intensity_mode, rng)
        if not result.ok:
            model.restore(snap)
        else:
            outcome = model.compile()
            if not outcome.ok:
                model.restore(snap)
                result.ok = False
                result.reason = "post_compile_fail:" + (outcome.error_msg or "")
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from ..mjcf.spec_loader import SafeModel


@dataclass
class ApplyResult:
    ok: bool
    intensity_mode: str
    reason: str = ""
    runtime_directive: Optional[Dict] = None  # for SET_QPOS/QVEL/CTRL etc.
    params_used: Dict = field(default_factory=dict)


class Mutator(Protocol):
    id: str
    intensity_modes: List[str]
    # Modes that are intentionally invalid-but-parseable; CI gate skips them.
    invalid_parseable_modes: List[str]
    # If True the mutator never writes XML, only emits a runtime directive.
    runtime_only: bool

    def applicable(self, model: SafeModel) -> bool: ...
    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult: ...


# Sentinel used by some mutators when they do not accept intensities.
NO_INTENSITY = "_default"
