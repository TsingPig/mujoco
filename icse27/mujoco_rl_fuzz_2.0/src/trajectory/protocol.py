"""Trajectory protocol dataclasses.

A trajectory protocol is a *replayable specification*; we never persist
raw mjData binaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


ACTION_KINDS = (
    "zero_control",
    "random_control",
    "sinusoidal_control",
    "single_actuator_sweep",
    "push_like",
    "grasp_lift_like",
    "reset_replay",
)


@dataclass
class InitialState:
    qpos: Optional[List[float]] = None
    qvel: Optional[List[float]] = None
    use_keyframe: Optional[str] = None  # name of keyframe in MJCF, if any
    perturb_scale: float = 0.0          # additive Gaussian perturbation on qpos

    def to_dict(self): return asdict(self)


@dataclass
class ActionSequence:
    kind: str = "zero_control"          # one of ACTION_KINDS
    horizon: int = 100
    amplitude: float = 0.0
    frequency: float = 1.0
    actuator_index: Optional[int] = None  # for single_actuator_sweep
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self): return asdict(self)


@dataclass
class ResetProtocol:
    n_resets: int = 1
    same_action_each: bool = True

    def to_dict(self): return asdict(self)


@dataclass
class SolverProfile:
    timestep: Optional[float] = None
    integrator: Optional[str] = None     # "EULER", "RK4", "IMPLICIT"
    iterations: Optional[int] = None

    def to_dict(self): return asdict(self)


@dataclass
class BackendProfile:
    backend: str = "classic"             # classic | mjx
    device: str = "cpu"

    def to_dict(self): return asdict(self)


@dataclass
class RolloutProfile:
    record_every: int = 1
    capture_sensors: bool = True
    capture_contacts: bool = True

    def to_dict(self): return asdict(self)


@dataclass
class TrajectoryProtocol:
    initial_state: InitialState = field(default_factory=InitialState)
    action_sequence: ActionSequence = field(default_factory=ActionSequence)
    reset_protocol: ResetProtocol = field(default_factory=ResetProtocol)
    rollout_profile: RolloutProfile = field(default_factory=RolloutProfile)
    solver_profile: SolverProfile = field(default_factory=SolverProfile)
    backend_profile: BackendProfile = field(default_factory=BackendProfile)
    random_seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_state": self.initial_state.to_dict(),
            "action_sequence": self.action_sequence.to_dict(),
            "reset_protocol": self.reset_protocol.to_dict(),
            "rollout_profile": self.rollout_profile.to_dict(),
            "solver_profile": self.solver_profile.to_dict(),
            "backend_profile": self.backend_profile.to_dict(),
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryProtocol":
        d = d or {}
        return cls(
            initial_state=InitialState(**(d.get("initial_state") or {})),
            action_sequence=ActionSequence(**(d.get("action_sequence") or {})),
            reset_protocol=ResetProtocol(**(d.get("reset_protocol") or {})),
            rollout_profile=RolloutProfile(**(d.get("rollout_profile") or {})),
            solver_profile=SolverProfile(**(d.get("solver_profile") or {})),
            backend_profile=BackendProfile(**(d.get("backend_profile") or {})),
            random_seed=int(d.get("random_seed", 0)),
        )
