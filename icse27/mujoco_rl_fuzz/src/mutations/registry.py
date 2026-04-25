"""Mutator registry: single source of truth mapping id <-> instance <-> int.

Used by:
- runner: lookup by id
- RL: discrete action_id <-> mutator string (via `MUTATOR_IDS`)
- LLM: prompt enumerates `MUTATOR_IDS`
"""
from __future__ import annotations

from .base import BaseMutator
from .xml_mutators import (
    StructGrowMutator, StructShrinkMutator, StructRewireMutator,
    GeomPerturbMutator, JointPerturbMutator, InertialPerturbMutator,
    ContactPerturbMutator, ActuatorEditMutator, SolverToggleMutator,
    StatePerturbMutator,
)


_REGISTERED: list[BaseMutator] = [
    StructGrowMutator(),
    StructShrinkMutator(),
    StructRewireMutator(),
    GeomPerturbMutator(),
    JointPerturbMutator(),
    InertialPerturbMutator(),
    ContactPerturbMutator(),
    ActuatorEditMutator(),
    SolverToggleMutator(),
    StatePerturbMutator(),
]

MUTATORS: dict[str, BaseMutator] = {m.id: m for m in _REGISTERED}
MUTATOR_IDS: list[str] = [m.id for m in _REGISTERED]


def get(mid: str) -> BaseMutator:
    return MUTATORS[mid]


def filter_whitelist(allowed: list[str]) -> list[BaseMutator]:
    if not allowed:
        return list(_REGISTERED)
    return [MUTATORS[m] for m in allowed if m in MUTATORS]


def id_to_index(mid: str) -> int:
    return MUTATOR_IDS.index(mid)


def index_to_id(i: int) -> str:
    return MUTATOR_IDS[i]
