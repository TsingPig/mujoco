"""Mutator registry — single source of truth for canonical action ordering."""
from __future__ import annotations

from typing import Dict, List

from .base import Mutator
from .ops_actuator import (
    ActuatorAdd, ActuatorDelete, ActuatorMutateRange, EqualityAdd, TendonAdd,
)
from .ops_geom import (
    MutateContactMargin, MutateFriction, MutateGeomShape, MutateGeomSize,
    MutateInertial, MutateSolrefSolimp,
)
from .ops_joint import MutateDampingFriction, MutateJointLimit, MutateJointType
from .ops_solver import MutateIntegrator, MutateSolverIter, MutateTimestep
from .ops_state import SetCtrlRuntime, SetQposRuntime, SetQvelRuntime
from .ops_struct import StructDuplicateSubtree, StructGrowLink, StructShrink


_ALL: List[Mutator] = [
    StructGrowLink(),
    StructShrink(),
    StructDuplicateSubtree(),
    MutateGeomShape(),
    MutateGeomSize(),
    MutateInertial(),
    MutateJointType(),
    MutateJointLimit(),
    MutateDampingFriction(),
    MutateFriction(),
    MutateSolrefSolimp(),
    MutateContactMargin(),
    ActuatorAdd(),
    ActuatorMutateRange(),
    ActuatorDelete(),
    EqualityAdd(),
    TendonAdd(),
    MutateTimestep(),
    MutateIntegrator(),
    MutateSolverIter(),
    SetQposRuntime(),
    SetQvelRuntime(),
    SetCtrlRuntime(),
]


MUTATORS: Dict[str, Mutator] = {m.id: m for m in _ALL}
MUTATOR_IDS: List[str] = [m.id for m in _ALL]


def get(mut_id: str) -> Mutator:
    return MUTATORS[mut_id]


def all_mutators() -> List[Mutator]:
    return list(_ALL)
