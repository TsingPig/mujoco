"""Gymnasium-Robotics adapter."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import OpenEnvAdapter, _safe_import


_DEFAULT_ENVS: List[Dict[str, Any]] = [
    {"env_id": "FetchReach-v3"},
    {"env_id": "FetchPush-v3"},
    {"env_id": "FetchSlide-v3"},
    {"env_id": "FetchPickAndPlace-v3"},
    {"env_id": "HandReach-v2"},
    {"env_id": "HandManipulateBlock-v2"},
    {"env_id": "HandManipulateEgg-v2"},
    {"env_id": "HandManipulatePen-v2"},
    {"env_id": "AdroitHandDoor-v1"},
    {"env_id": "AdroitHandHammer-v1"},
    {"env_id": "AdroitHandPen-v1"},
    {"env_id": "AdroitHandRelocate-v1"},
    # Plain Gymnasium-MuJoCo classics that ship with mujoco bindings.
    {"env_id": "InvertedPendulum-v5"},
    {"env_id": "InvertedDoublePendulum-v5"},
    {"env_id": "Reacher-v5"},
    {"env_id": "Pusher-v5"},
    {"env_id": "HalfCheetah-v5"},
    {"env_id": "Hopper-v5"},
    {"env_id": "Walker2d-v5"},
    {"env_id": "Ant-v5"},
    {"env_id": "Humanoid-v5"},
    {"env_id": "HumanoidStandup-v5"},
    {"env_id": "Swimmer-v5"},
]


class GymRoboticsAdapter(OpenEnvAdapter):
    name = "gym_robotics"
    package = "gymnasium"

    def list_envs(self, max_envs: int = 50) -> List[Dict[str, Any]]:
        return _DEFAULT_ENVS[:max_envs]

    def make(self, env_id: str, **kwargs):
        gym = _safe_import("gymnasium")
        if gym is None:
            raise ImportError("gymnasium not installed")
        # Try to register Gymnasium-Robotics envs if available.
        gymrob = _safe_import("gymnasium_robotics")
        if gymrob is not None:
            try:
                gymrob.register_robotics_envs()
            except Exception:
                pass
        return gym.make(env_id, **kwargs)
