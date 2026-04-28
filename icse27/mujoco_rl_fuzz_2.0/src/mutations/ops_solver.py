"""Solver/option mutators."""
from __future__ import annotations

import random
from typing import List

from ..mjcf.invariants import option_element
from ..mjcf.spec_loader import SafeModel
from . import intensity as I
from .base import ApplyResult


class MutateTimestep:
    id = "MUTATE_TIMESTEP"
    intensity_modes = list(I.TIMESTEP_MODES.keys())
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return True

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        opt = option_element(model.root)
        ts = I.TIMESTEP_MODES[intensity_mode]
        opt.set("timestep", f"{ts:.6g}")
        return ApplyResult(True, intensity_mode, params_used={"timestep": ts})


class MutateIntegrator:
    id = "MUTATE_INTEGRATOR"
    intensity_modes = list(I.INTEGRATOR_MODES)
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return True

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        opt = option_element(model.root)
        opt.set("integrator", intensity_mode)
        return ApplyResult(True, intensity_mode, params_used={"integrator": intensity_mode})


class MutateSolverIter:
    id = "MUTATE_SOLVER_ITER"
    intensity_modes = list(I.SOLVER_ITER_MODES.keys())
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return True

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        opt = option_element(model.root)
        n = I.SOLVER_ITER_MODES[intensity_mode]
        opt.set("iterations", str(n))
        return ApplyResult(True, intensity_mode, params_used={"iterations": n})
