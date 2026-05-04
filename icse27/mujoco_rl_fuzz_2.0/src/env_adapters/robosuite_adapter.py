"""robosuite adapter."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import OpenEnvAdapter, _safe_import, AdapterStatus, STATUS_OK, STATUS_MISSING


# A small curated default robot per env so the adapter works without the
# user having to know robosuite internals.
_DEFAULT_ENVS: List[Dict[str, Any]] = [
    {"env_id": "Lift",          "constructor_kwargs": {"robots": "Panda"}},
    {"env_id": "Stack",         "constructor_kwargs": {"robots": "Panda"}},
    {"env_id": "PickPlace",     "constructor_kwargs": {"robots": "Panda"}},
    {"env_id": "NutAssembly",   "constructor_kwargs": {"robots": "Panda"}},
    {"env_id": "Door",          "constructor_kwargs": {"robots": "Panda"}},
    {"env_id": "Wipe",          "constructor_kwargs": {"robots": "Panda"}},
    {"env_id": "TwoArmLift",    "constructor_kwargs": {"robots": ["Panda", "Panda"]}},
    {"env_id": "TwoArmPegInHole", "constructor_kwargs": {"robots": ["Panda", "Panda"]}},
]


class RobosuiteAdapter(OpenEnvAdapter):
    name = "robosuite"
    package = "robosuite"

    def list_envs(self, max_envs: int = 50) -> List[Dict[str, Any]]:
        return _DEFAULT_ENVS[:max_envs]

    def make(self, env_id: str, **kwargs):
        rs = _safe_import("robosuite")
        if rs is None:
            raise ImportError("robosuite not installed")
        defaults = dict(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            ignore_done=True,
            horizon=200,
        )
        defaults.update(kwargs)
        return rs.make(env_id, **defaults)
