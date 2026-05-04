"""Trajectory replay over actor / synthetic_scene / open_env seeds.

Replay never raises to the caller. Failures are returned as
ReplayResult(ok=False, error=...).
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mujoco

from .protocol import TrajectoryProtocol, ACTION_KINDS
from .recorder import TraceRecorder, TraceSummary


@dataclass
class ReplayResult:
    ok: bool
    trace: Optional[TraceSummary] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Action-sequence generation
# -----------------------------------------------------------------------------
def _action_at(seq, nu: int, t: float, step_idx: int, rng: random.Random) -> List[float]:
    if nu <= 0:
        return []
    kind = seq.kind
    if kind == "zero_control":
        return [0.0] * nu
    if kind == "random_control":
        amp = max(seq.amplitude, 1e-6) if seq.amplitude > 0 else 0.1
        return [rng.uniform(-amp, amp) for _ in range(nu)]
    if kind == "sinusoidal_control":
        amp = seq.amplitude if seq.amplitude > 0 else 0.5
        f = seq.frequency if seq.frequency > 0 else 1.0
        return [amp * math.sin(2 * math.pi * f * t + i) for i in range(nu)]
    if kind == "single_actuator_sweep":
        idx = (seq.actuator_index if seq.actuator_index is not None else 0) % nu
        amp = seq.amplitude if seq.amplitude > 0 else 0.5
        a = [0.0] * nu
        a[idx] = amp * math.sin(2 * math.pi * (seq.frequency or 1.0) * t)
        return a
    if kind == "push_like":
        amp = seq.amplitude if seq.amplitude > 0 else 0.5
        return [amp if step_idx < seq.horizon // 4 else 0.0 for _ in range(nu)]
    if kind == "grasp_lift_like":
        # crude: first half negative (close), second half positive (lift)
        amp = seq.amplitude if seq.amplitude > 0 else 0.5
        sign = -1.0 if step_idx < seq.horizon // 2 else 1.0
        return [sign * amp] * nu
    if kind == "reset_replay":
        return [0.0] * nu
    return [0.0] * nu


# -----------------------------------------------------------------------------
# Apply solver/backend overrides
# -----------------------------------------------------------------------------
_INTEGRATOR_MAP = {
    "EULER": getattr(mujoco.mjtIntegrator, "mjINT_EULER", 0),
    "RK4":   getattr(mujoco.mjtIntegrator, "mjINT_RK4", 1),
    "IMPLICIT": getattr(mujoco.mjtIntegrator, "mjINT_IMPLICIT", 2),
    "IMPLICITFAST": getattr(mujoco.mjtIntegrator, "mjINT_IMPLICITFAST", 3),
}


def _apply_solver_profile(model, prof) -> None:
    if prof is None:
        return
    if prof.timestep is not None and prof.timestep > 0:
        try:
            model.opt.timestep = float(prof.timestep)
        except Exception:
            pass
    if prof.integrator and prof.integrator in _INTEGRATOR_MAP:
        try:
            model.opt.integrator = _INTEGRATOR_MAP[prof.integrator]
        except Exception:
            pass
    if prof.iterations is not None and prof.iterations > 0:
        try:
            model.opt.iterations = int(prof.iterations)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Initial state
# -----------------------------------------------------------------------------
def _apply_initial_state(model, data, init, rng: random.Random) -> None:
    import numpy as np
    if init is None:
        return
    # keyframe
    if init.use_keyframe:
        try:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, init.use_keyframe)
            if kid >= 0:
                mujoco.mj_resetDataKeyframe(model, data, kid)
                return
        except Exception:
            pass
    # explicit qpos/qvel
    if init.qpos is not None:
        n = min(len(init.qpos), data.qpos.size)
        data.qpos[:n] = np.array(init.qpos[:n], dtype=float)
    if init.qvel is not None:
        n = min(len(init.qvel), data.qvel.size)
        data.qvel[:n] = np.array(init.qvel[:n], dtype=float)
    # perturbation
    if init.perturb_scale and init.perturb_scale > 0:
        try:
            data.qpos[:] += np.array([rng.gauss(0.0, init.perturb_scale) for _ in range(data.qpos.size)])
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Core MJCF replay
# -----------------------------------------------------------------------------
def _replay_xml(xml_text: str, asset_dir: Optional[str], protocol: TrajectoryProtocol) -> ReplayResult:
    try:
        if asset_dir is not None and os.path.isdir(asset_dir):
            old_cwd = os.getcwd()
            try:
                os.chdir(asset_dir)
                model = mujoco.MjModel.from_xml_string(xml_text)
            finally:
                os.chdir(old_cwd)
        else:
            model = mujoco.MjModel.from_xml_string(xml_text)
    except Exception as exc:
        return ReplayResult(ok=False, error=f"compile_failed: {type(exc).__name__}: {exc}")

    _apply_solver_profile(model, protocol.solver_profile)
    data = mujoco.MjData(model)
    rng = random.Random(protocol.random_seed)
    _apply_initial_state(model, data, protocol.initial_state, rng)

    rec = TraceRecorder()
    horizon = max(int(protocol.action_sequence.horizon), 1)
    capture_every = max(int(protocol.rollout_profile.record_every), 1)
    capture_sensors = bool(protocol.rollout_profile.capture_sensors)
    capture_contacts = bool(protocol.rollout_profile.capture_contacts)
    nu = int(model.nu)

    # First-body z proxy for "fall" tracking
    body_z_index = 0 if model.nbody > 1 else None

    try:
        for step_idx in range(horizon):
            t = float(data.time)
            ctrl = _action_at(protocol.action_sequence, nu, t, step_idx, rng)
            if nu > 0:
                for i in range(nu):
                    data.ctrl[i] = ctrl[i]
            try:
                mujoco.mj_step(model, data)
            except Exception as exc:
                rec.warn(f"mj_step_exception:{type(exc).__name__}")
                return ReplayResult(ok=False, trace=rec.finalize(),
                                    error=f"mj_step_failed: {exc}")
            if (step_idx % capture_every) != 0:
                continue
            ncon = int(data.ncon) if capture_contacts else None
            body_z = None
            if body_z_index is not None and body_z_index < model.nbody:
                try:
                    body_z = float(data.xpos[body_z_index, 2])
                except Exception:
                    body_z = None
            sensor = None
            if capture_sensors and model.nsensordata > 0:
                try:
                    sensor = list(data.sensordata)
                except Exception:
                    sensor = None
            rec.step(qpos=list(data.qpos), qvel=list(data.qvel), qacc=list(data.qacc),
                     ncon=ncon, body_z=body_z, sensor=sensor)
    except Exception as exc:  # pragma: no cover - hard-defensive
        return ReplayResult(ok=False, trace=rec.finalize(),
                            error=f"unexpected: {type(exc).__name__}: {exc}")

    return ReplayResult(ok=True, trace=rec.finalize(),
                        meta={"nq": int(model.nq), "nv": int(model.nv), "nu": nu})


# -----------------------------------------------------------------------------
# Public replay entry points
# -----------------------------------------------------------------------------
def replay_actor(actor_xml_path: str, protocol: TrajectoryProtocol) -> ReplayResult:
    if not os.path.isfile(actor_xml_path):
        return ReplayResult(ok=False, error=f"actor_xml_missing: {actor_xml_path}")
    try:
        with open(actor_xml_path, "r", encoding="utf-8") as f:
            xml_text = f.read()
    except Exception as exc:
        return ReplayResult(ok=False, error=f"read_failed: {exc}")
    return _replay_xml(xml_text, os.path.dirname(os.path.abspath(actor_xml_path)), protocol)


def replay_scene(scene_xml_path: str, protocol: TrajectoryProtocol) -> ReplayResult:
    return replay_actor(scene_xml_path, protocol)


def replay_env(env_seed, protocol: TrajectoryProtocol) -> ReplayResult:
    """Replay an L2 OpenEnvSeed via its adapter.

    Replay here is best-effort: we reset, then step ``horizon`` times with
    samples from the adapter's action_space. If the env or the model is
    not accessible, the result is partial.
    """
    from ..env_adapters.base import get_adapter
    name = getattr(env_seed, "adapter_name", None)
    env_id = getattr(env_seed, "env_id", None)
    if not name or not env_id:
        return ReplayResult(ok=False, error="env_seed missing adapter_name/env_id")
    try:
        adapter = get_adapter(name)
    except Exception as exc:
        return ReplayResult(ok=False, error=f"adapter_missing: {exc}")
    res = adapter.smoke(env_id, n_steps=protocol.action_sequence.horizon,
                        **(env_seed.constructor_kwargs or {}))
    if not res.ok:
        return ReplayResult(ok=False, error=res.error or "env_smoke_failed",
                            meta={"reset_ok": res.reset_ok, "n_steps": res.n_steps})
    rec = TraceRecorder()
    # Synthesize a minimal trace from smoke summary.
    rec.step(qpos=[], qvel=[], qacc=[], ncon=0, body_z=None, sensor=None,
             reward=res.reward_summary.get("total"))
    summary = rec.finalize()
    summary.n_steps = res.n_steps
    if res.done_summary.get("seen_done"):
        summary.warnings.append("env_done_during_smoke")
    return ReplayResult(ok=True, trace=summary,
                        meta={"model_access_method": res.model_access_method,
                              "reward_total": res.reward_summary.get("total")})
