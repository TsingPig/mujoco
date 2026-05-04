"""dm_control / Composer adapter."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import OpenEnvAdapter, _safe_import


_DEFAULT_TASKS: List[Dict[str, Any]] = [
    {"env_id": "cartpole/balance"},
    {"env_id": "cartpole/swingup"},
    {"env_id": "cheetah/run"},
    {"env_id": "walker/walk"},
    {"env_id": "walker/run"},
    {"env_id": "hopper/stand"},
    {"env_id": "hopper/hop"},
    {"env_id": "humanoid/stand"},
    {"env_id": "humanoid/walk"},
    {"env_id": "humanoid/run"},
    {"env_id": "fish/swim"},
    {"env_id": "swimmer/swimmer6"},
    {"env_id": "reacher/easy"},
    {"env_id": "reacher/hard"},
    {"env_id": "manipulator/bring_ball"},
    {"env_id": "ball_in_cup/catch"},
    {"env_id": "finger/spin"},
    {"env_id": "pendulum/swingup"},
    {"env_id": "acrobot/swingup"},
    {"env_id": "point_mass/easy"},
]


class _DmcWrapper:
    """Minimal common interface around a dm_control Environment."""
    def __init__(self, env):
        self.env = env

    def reset(self):
        ts = self.env.reset()
        return ts.observation

    def step(self, action):
        ts = self.env.step(action)
        return ts.observation, float(ts.reward) if ts.reward is not None else 0.0, ts.last(), False, {}

    def action_spec(self):
        return self.env.action_spec()

    @property
    def action_space(self):  # gym-compat shim
        spec = self.env.action_spec()
        class _AS:
            shape = spec.shape
            def sample(_self):
                import numpy as np
                lo = getattr(spec, "minimum", None)
                hi = getattr(spec, "maximum", None)
                if lo is not None and hi is not None:
                    return np.random.uniform(lo, hi).astype(float)
                return np.zeros(spec.shape, dtype=float)
        return _AS()

    @property
    def physics(self):
        return getattr(self.env, "physics", None)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


class DmControlAdapter(OpenEnvAdapter):
    name = "dm_control"
    package = "dm_control"

    def list_envs(self, max_envs: int = 50) -> List[Dict[str, Any]]:
        return _DEFAULT_TASKS[:max_envs]

    def make(self, env_id: str, **kwargs):
        suite = _safe_import("dm_control.suite")
        if suite is None:
            raise ImportError("dm_control not installed")
        domain, task = env_id.split("/", 1)
        env = suite.load(domain_name=domain, task_name=task, **kwargs)
        return _DmcWrapper(env)
