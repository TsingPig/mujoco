"""MyoSuite adapter (biomechanics)."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import OpenEnvAdapter, _safe_import


_DEFAULT_ENVS: List[Dict[str, Any]] = [
    {"env_id": "myoElbowPose1D6MRandom-v0", "tags": ["biomech", "elbow"]},
    {"env_id": "myoFingerReachFixed-v0",    "tags": ["biomech", "finger"]},
    {"env_id": "myoHandPoseFixed-v0",       "tags": ["biomech", "hand"]},
    {"env_id": "myoHandReachFixed-v0",      "tags": ["biomech", "hand"]},
    {"env_id": "myoLegStandRandom-v0",      "tags": ["biomech", "leg"]},
    {"env_id": "myoLegWalk-v0",             "tags": ["biomech", "leg"]},
]


class MyoSuiteAdapter(OpenEnvAdapter):
    name = "myosuite"
    package = "myosuite"

    def list_envs(self, max_envs: int = 50) -> List[Dict[str, Any]]:
        return _DEFAULT_ENVS[:max_envs]

    def make(self, env_id: str, **kwargs):
        ms = _safe_import("myosuite")
        if ms is None:
            raise ImportError("myosuite not installed")
        gym = _safe_import("gymnasium") or _safe_import("gym")
        if gym is None:
            raise ImportError("gymnasium/gym not installed")
        return gym.make(env_id, **kwargs)
