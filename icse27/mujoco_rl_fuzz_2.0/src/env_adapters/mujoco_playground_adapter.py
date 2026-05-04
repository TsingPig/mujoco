"""MuJoCo Playground (MJX-backed) adapter.

This adapter is intentionally metadata-leaning. The first stage records
env entries and lets users register them; full MJX rollouts are handled by
the backend differential oracle in a later milestone.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import OpenEnvAdapter, _safe_import


_DEFAULT_ENVS: List[Dict[str, Any]] = [
    {"env_id": "playground:CartpoleBalance",        "tags": ["mjx"]},
    {"env_id": "playground:CartpoleSwingup",        "tags": ["mjx"]},
    {"env_id": "playground:HumanoidWalk",           "tags": ["mjx"]},
    {"env_id": "playground:WalkerWalk",             "tags": ["mjx"]},
    {"env_id": "playground:CheetahRun",             "tags": ["mjx"]},
    {"env_id": "playground:HopperHop",              "tags": ["mjx"]},
    {"env_id": "playground:FingerSpin",             "tags": ["mjx"]},
    {"env_id": "playground:ReacherEasy",            "tags": ["mjx"]},
    {"env_id": "playground:LeapCubeReorient",       "tags": ["mjx", "manipulation"]},
    {"env_id": "playground:PandaPickCube",          "tags": ["mjx", "manipulation"]},
    {"env_id": "playground:Go1JoystickFlat",        "tags": ["mjx", "locomotion"]},
    {"env_id": "playground:BarkourJoystick",        "tags": ["mjx", "locomotion"]},
    {"env_id": "playground:H1Walk",                 "tags": ["mjx", "humanoid"]},
    {"env_id": "playground:G1Walk",                 "tags": ["mjx", "humanoid"]},
]


class MujocoPlaygroundAdapter(OpenEnvAdapter):
    name = "mujoco_playground"
    package = "mujoco_playground"

    def list_envs(self, max_envs: int = 50) -> List[Dict[str, Any]]:
        return _DEFAULT_ENVS[:max_envs]

    def make(self, env_id: str, **kwargs):
        mp = _safe_import("mujoco_playground")
        if mp is None:
            raise ImportError("mujoco_playground not installed")
        # Best-effort: many playground envs live under mujoco_playground.registry.
        registry = getattr(mp, "registry", None)
        if registry is None:
            raise NotImplementedError("mujoco_playground registry not found in this version")
        # Strip optional 'playground:' prefix.
        name = env_id.split(":", 1)[1] if ":" in env_id else env_id
        load = getattr(registry, "load", None)
        if not callable(load):
            raise NotImplementedError("mujoco_playground.registry.load not available")
        return load(name, **kwargs)
