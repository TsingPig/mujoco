"""Adapter base interface for open-source MuJoCo environments."""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# AdapterStatus values.
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_PARTIAL = "partial"
STATUS_BROKEN = "broken"
STATUS_UNKNOWN = "unknown"


@dataclass
class AdapterStatus:
    dependency: str = STATUS_UNKNOWN  # ok|missing|partial|unknown
    runnable: str = STATUS_UNKNOWN    # runnable|partial|metadata_only|broken|unknown
    notes: Optional[str] = None


@dataclass
class EnvSmokeResult:
    """Result of resetting + stepping an env a few times."""
    ok: bool
    reset_ok: bool = False
    step_ok: bool = False
    n_steps: int = 0
    obs_summary: Dict[str, Any] = field(default_factory=dict)
    reward_summary: Dict[str, Any] = field(default_factory=dict)
    done_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    model_access_method: Optional[str] = None


def _safe_import(modname: str):
    try:
        return importlib.import_module(modname)
    except Exception:
        return None


class OpenEnvAdapter:
    """Base class. Subclasses override the methods below.

    Adapters should never raise to the caller. Failures are reflected via
    `AdapterStatus.dependency = "missing"` or by returning an
    ``EnvSmokeResult(ok=False, error=...)``.
    """
    name: str = "base"
    package: str = ""

    # ---- discovery ----
    def is_available(self) -> AdapterStatus:
        mod = _safe_import(self.package) if self.package else None
        if mod is None:
            return AdapterStatus(dependency=STATUS_MISSING, runnable="metadata_only",
                                 notes=f"package '{self.package}' not importable")
        return AdapterStatus(dependency=STATUS_OK, runnable=STATUS_UNKNOWN)

    def list_envs(self, max_envs: int = 50) -> List[Dict[str, Any]]:
        """Return a list of env entry dicts. Each dict must contain
        ``env_id`` and may contain ``constructor_kwargs``.

        Default: empty list (override in subclasses).
        """
        return []

    # ---- runtime ----
    def make(self, env_id: str, **kwargs):  # pragma: no cover - subclass override
        raise NotImplementedError

    def smoke(self, env_id: str, *, n_steps: int = 5, **kwargs) -> EnvSmokeResult:
        """Reset + step a few times. Never raises."""
        try:
            env = self.make(env_id, **kwargs)
        except Exception as exc:
            return EnvSmokeResult(ok=False, error=f"make_failed: {exc}")
        try:
            try:
                obs = env.reset()
            except Exception as exc:
                return EnvSmokeResult(ok=False, error=f"reset_failed: {exc}")
            reset_ok = True
            steps = 0
            reward_total = 0.0
            done_seen = False
            try:
                action_sample = self._sample_action(env)
            except Exception:
                action_sample = None
            try:
                for i in range(n_steps):
                    if action_sample is None:
                        try:
                            action_sample = self._sample_action(env)
                        except Exception:
                            action_sample = None
                    if action_sample is None:
                        return EnvSmokeResult(ok=False, reset_ok=True,
                                              error="no_action_sampler")
                    out = env.step(action_sample)
                    steps += 1
                    # Try to parse common (obs, reward, done, info) or
                    # (obs, reward, terminated, truncated, info) shapes.
                    if isinstance(out, tuple):
                        if len(out) >= 2:
                            try:
                                reward_total += float(out[1])
                            except Exception:
                                pass
                        if len(out) >= 3:
                            done_seen = done_seen or bool(out[2])
                step_ok = True
            except Exception as exc:
                return EnvSmokeResult(ok=False, reset_ok=reset_ok, n_steps=steps,
                                      error=f"step_failed: {exc}")
            return EnvSmokeResult(
                ok=True, reset_ok=reset_ok, step_ok=step_ok, n_steps=steps,
                reward_summary={"total": reward_total},
                done_summary={"seen_done": done_seen},
                model_access_method=self._detect_model_access(env),
            )
        finally:
            try:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

    # ---- helpers ----
    def _sample_action(self, env) -> Any:
        # Try gymnasium-style action_space.sample().
        sp = getattr(env, "action_space", None)
        if sp is not None and hasattr(sp, "sample"):
            return sp.sample()
        # Try dm_control: env.action_spec().shape -> zeros.
        spec_fn = getattr(env, "action_spec", None)
        if callable(spec_fn):
            spec = spec_fn()
            shape = getattr(spec, "shape", None)
            if shape is not None:
                try:
                    import numpy as np
                    return np.zeros(shape, dtype=float)
                except Exception:
                    return None
        return None

    def _detect_model_access(self, env) -> Optional[str]:
        # Common access paths.
        candidates = [
            "unwrapped.model",
            "unwrapped.sim.model",
            "sim.model",
            "physics.model",
            "model",
        ]
        for c in candidates:
            try:
                obj = env
                for part in c.split("."):
                    obj = getattr(obj, part)
                if obj is not None:
                    return c
            except Exception:
                continue
        return None


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
_REGISTRY: Dict[str, OpenEnvAdapter] = {}


def register_adapter(adapter: OpenEnvAdapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> OpenEnvAdapter:
    return _REGISTRY[name]


def list_adapters() -> List[str]:
    return list(_REGISTRY.keys())
