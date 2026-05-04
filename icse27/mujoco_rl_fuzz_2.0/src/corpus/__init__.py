"""Layered seed corpus: schema, manifest, provenance.

Layers:
- L0 ActorSeed: curated single-actor MJCF models.
- L1 SyntheticSceneSeed: actor + arena/object/prop composition.
- L2 OpenEnvSeed: open-source environment registered through an adapter.
- L3 TrajectorySeed: replayable trajectory/mjData protocol.
"""
from __future__ import annotations

from .schema import (
    ActorSeed,
    SyntheticSceneSeed,
    OpenEnvSeed,
    TrajectorySeed,
    LayeredSeed,
    SEED_LAYERS,
    seed_from_dict,
)
from .provenance import Provenance, make_provenance
from .manifest import (
    load_manifest,
    save_manifest,
    append_manifest,
    iter_manifest,
    validate_manifest,
)

__all__ = [
    "ActorSeed",
    "SyntheticSceneSeed",
    "OpenEnvSeed",
    "TrajectorySeed",
    "LayeredSeed",
    "SEED_LAYERS",
    "seed_from_dict",
    "Provenance",
    "make_provenance",
    "load_manifest",
    "save_manifest",
    "append_manifest",
    "iter_manifest",
    "validate_manifest",
]
