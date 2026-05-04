"""Layered seed schema (dataclasses).

Four layers:
    L0 actor          -> ActorSeed
    L1 synthetic_scene-> SyntheticSceneSeed
    L2 open_env       -> OpenEnvSeed
    L3 trajectory     -> TrajectorySeed

All seeds share an `seed_id`, `layer`, `provenance`, optional `tags` list,
and a serializable `to_dict()` / `from_dict()` round trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from .provenance import Provenance


SEED_LAYERS = ("actor", "synthetic_scene", "open_env", "trajectory")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_provenance(x: Any) -> Provenance:
    if isinstance(x, Provenance):
        return x
    if x is None:
        return Provenance()
    return Provenance.from_dict(x)


# -----------------------------------------------------------------------------
# L0 Actor
# -----------------------------------------------------------------------------
@dataclass
class ActorSeed:
    seed_id: str
    layer: str = "actor"
    source_repo: Optional[str] = None
    source_path: Optional[str] = None     # original path inside source repo
    local_path: Optional[str] = None      # workspace-relative path to model dir
    asset_dir: Optional[str] = None       # mujoco asset_dir for compile()
    license_path: Optional[str] = None
    model_xml: Optional[str] = None       # workspace-relative path to MJCF file
    tags: List[str] = field(default_factory=list)
    compile_status: str = "unknown"       # ok | failed | unknown
    model_features: Dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActorSeed":
        d = dict(d)
        d.pop("layer", None)
        prov = _as_provenance(d.pop("provenance", None))
        return cls(layer="actor", provenance=prov, **d)


# -----------------------------------------------------------------------------
# L1 Synthetic Scene
# -----------------------------------------------------------------------------
@dataclass
class SyntheticSceneSeed:
    seed_id: str
    layer: str = "synthetic_scene"
    actor_seed_id: Optional[str] = None
    object_asset_ids: List[str] = field(default_factory=list)
    arena_asset_id: Optional[str] = None
    scene_xml: Optional[str] = None       # workspace-relative path to composed XML
    placement_config: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    template_name: Optional[str] = None
    source: str = "generated"
    tags: List[str] = field(default_factory=list)
    compile_status: str = "unknown"
    model_features: Dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntheticSceneSeed":
        d = dict(d)
        d.pop("layer", None)
        prov = _as_provenance(d.pop("provenance", None))
        return cls(layer="synthetic_scene", provenance=prov, **d)


# -----------------------------------------------------------------------------
# L2 Open-source Environment
# -----------------------------------------------------------------------------
@dataclass
class OpenEnvSeed:
    seed_id: str
    layer: str = "open_env"
    source_repo: Optional[str] = None
    package_name: Optional[str] = None
    env_id: Optional[str] = None
    adapter_name: Optional[str] = None
    constructor_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_access_method: Optional[str] = None  # e.g. "env.unwrapped.model", "env.sim.model", "metadata_only"
    action_space_summary: Dict[str, Any] = field(default_factory=dict)
    observation_space_summary: Dict[str, Any] = field(default_factory=dict)
    reward_available: Optional[bool] = None
    reset_available: Optional[bool] = None
    rollout_available: Optional[bool] = None
    local_source_path: Optional[str] = None
    license_path: Optional[str] = None
    dependency_status: str = "unknown"   # ok | missing | partial | unknown
    runnable_status: str = "unknown"     # runnable | partial | metadata_only | broken | unknown
    smoke_trace: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpenEnvSeed":
        d = dict(d)
        d.pop("layer", None)
        prov = _as_provenance(d.pop("provenance", None))
        return cls(layer="open_env", provenance=prov, **d)


# -----------------------------------------------------------------------------
# L3 Trajectory
# -----------------------------------------------------------------------------
@dataclass
class TrajectorySeed:
    seed_id: str
    layer: str = "trajectory"
    parent_seed_id: Optional[str] = None
    parent_layer: Optional[str] = None  # actor | synthetic_scene | open_env
    initial_state: Dict[str, Any] = field(default_factory=dict)   # qpos/qvel spec
    action_sequence_spec: Dict[str, Any] = field(default_factory=dict)
    reset_protocol: Dict[str, Any] = field(default_factory=dict)
    rollout_profile: Dict[str, Any] = field(default_factory=dict)
    solver_profile: Dict[str, Any] = field(default_factory=dict)
    backend_profile: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 0
    trace_summary: Dict[str, Any] = field(default_factory=dict)
    replay_status: str = "unknown"  # ok | failed | partial | unknown
    tags: List[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectorySeed":
        d = dict(d)
        d.pop("layer", None)
        prov = _as_provenance(d.pop("provenance", None))
        return cls(layer="trajectory", provenance=prov, **d)


LayeredSeed = Union[ActorSeed, SyntheticSceneSeed, OpenEnvSeed, TrajectorySeed]


_LAYER_TO_CLS = {
    "actor": ActorSeed,
    "synthetic_scene": SyntheticSceneSeed,
    "open_env": OpenEnvSeed,
    "trajectory": TrajectorySeed,
}


def seed_from_dict(d: Dict[str, Any]) -> LayeredSeed:
    layer = d.get("layer")
    if layer not in _LAYER_TO_CLS:
        raise ValueError(f"unknown seed layer: {layer!r}")
    return _LAYER_TO_CLS[layer].from_dict(d)
