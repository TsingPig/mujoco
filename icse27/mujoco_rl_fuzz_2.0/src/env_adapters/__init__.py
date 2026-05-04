"""Open-source environment adapters (L2).

Adapters wrap heterogeneous environment APIs behind a common interface so
that the runner can reset/step/inspect them without depending on each
package directly. Missing dependencies degrade gracefully into
``metadata_only`` entries instead of raising.
"""
from __future__ import annotations

from .base import (
    OpenEnvAdapter,
    EnvSmokeResult,
    AdapterStatus,
    register_adapter,
    get_adapter,
    list_adapters,
)
from .robosuite_adapter import RobosuiteAdapter
from .gym_robotics_adapter import GymRoboticsAdapter
from .dm_control_adapter import DmControlAdapter
from .mujoco_playground_adapter import MujocoPlaygroundAdapter
from .myosuite_adapter import MyoSuiteAdapter

# Side-effect: register built-in adapters.
register_adapter(RobosuiteAdapter())
register_adapter(GymRoboticsAdapter())
register_adapter(DmControlAdapter())
register_adapter(MujocoPlaygroundAdapter())
register_adapter(MyoSuiteAdapter())

__all__ = [
    "OpenEnvAdapter",
    "EnvSmokeResult",
    "AdapterStatus",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "RobosuiteAdapter",
    "GymRoboticsAdapter",
    "DmControlAdapter",
    "MujocoPlaygroundAdapter",
    "MyoSuiteAdapter",
]
