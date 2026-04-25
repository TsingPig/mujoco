"""Subprocess worker: compile + rollout + warning capture for ONE testcase.

Invoked as:
    python -m src.engine.subprocess_worker --input <spec.json> --output <result.json>

Spec JSON schema (all keys optional except xml_path):
    {
      "xml_path": "...",
      "rollout_steps": 50,
      "state_perturb": {                # applied AFTER MjData() reset
          "qpos": [v0, v1, ...],
          "qvel": [...],
          "ctrl": [...]                 # values may be the strings "nan"/"+inf"/"-inf"
      },
      "disable_clamp_ctrl": true,       # else BADCTRL is silently clipped
      "consistency_check": true,        # run a 2nd MjData with same setup, diff qpos
      "solver_override": {              # apply to model.opt before running
          "integrator": 0|1|2|3,        # mjtIntegrator
          "solver":     0|1|2,          # mjtSolver
          "iterations": int,
          "tolerance":  float
      }
    }

Output JSON: see `out` dict below. ALWAYS written, even on internal errors.
The worker NEVER raises to the parent; it serialises the error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

# Add project root to sys.path so we can `import src.engine.state_ops`
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

# mujoco import is heavy; do it after sys.path setup so import errors surface cleanly
try:
    import mujoco  # noqa: E402
except Exception as _e:  # pragma: no cover
    sys.stderr.write(f"[worker] mujoco import failed: {_e}\n")
    raise

from src.engine.state_ops import decode_value, state_stats  # noqa: E402


WARN_NAMES = [
    "INERTIA", "CONTACTFULL", "CNSTRFULL", "VGEOMFULL",
    "BADQPOS", "BADQVEL", "BADQACC", "BADCTRL",
]

_WARN_BUF: list[str] = []


def _warn_cb(msg):
    if isinstance(msg, bytes):
        msg = msg.decode("utf-8", errors="replace")
    _WARN_BUF.append(str(msg))


def _empty_result() -> dict[str, Any]:
    return {
        "compile": {"ok": False, "exception_type": None, "traceback": None},
        "runtime": {"ok": False, "exception_type": None, "traceback": None, "steps_done": 0},
        "warnings": [],
        "state_stats": {},
        "consistency": {"ran": False, "diff": None},
        "model_shape": [],
        "solver_used": None,
        "integrator_used": None,
    }


def _apply_perturb(target: np.ndarray, values: list[Any]) -> None:
    if target.size == 0 or not values:
        return
    n = min(target.size, len(values))
    for i in range(n):
        target[i] = decode_value(values[i])


def _collect_warnings(data) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    try:
        for i in range(len(data.warning)):
            cnt = int(data.warning[i].number)
            if cnt > 0:
                name = WARN_NAMES[i] if i < len(WARN_NAMES) else f"WARN_{i}"
                counts[name] = cnt
    except Exception:
        pass

    msgs: dict[str, str] = {k: "" for k in counts}
    for m in _WARN_BUF:
        for k in list(msgs.keys()):
            if not msgs[k] and k in m:
                msgs[k] = m[:200]

    if not counts and _WARN_BUF:
        counts["UNCATEGORIZED"] = len(_WARN_BUF)
        msgs["UNCATEGORIZED"] = _WARN_BUF[0][:200]

    return [{"wtype": k, "count": v, "message": msgs.get(k, "")} for k, v in counts.items()]


def run_one(spec: dict) -> dict:
    out = _empty_result()
    mujoco.set_mju_user_warning(_warn_cb)
    _WARN_BUF.clear()

    xml_path = spec["xml_path"]
    n_steps = int(spec.get("rollout_steps", 50))
    perturb = spec.get("state_perturb") or {}
    consistency = bool(spec.get("consistency_check", False))
    disable_clamp_ctrl = bool(spec.get("disable_clamp_ctrl", False))
    solver_ov = spec.get("solver_override") or {}

    # ---- compile ----
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        out["compile"]["ok"] = True
    except Exception as e:
        out["compile"]["exception_type"] = type(e).__name__
        out["compile"]["traceback"] = "".join(
            traceback.format_exception_only(type(e), e)
        )[:500]
        out["warnings"] = _collect_warnings_no_data()
        return out

    out["model_shape"] = [int(model.nq), int(model.nv), int(model.nu),
                          int(model.nbody), int(model.ngeom)]

    # ---- option overrides ----
    try:
        if "integrator" in solver_ov:
            model.opt.integrator = int(solver_ov["integrator"])
        if "solver" in solver_ov:
            model.opt.solver = int(solver_ov["solver"])
        if "iterations" in solver_ov:
            model.opt.iterations = int(solver_ov["iterations"])
        if "tolerance" in solver_ov:
            model.opt.tolerance = float(solver_ov["tolerance"])
        if disable_clamp_ctrl:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    except Exception:
        pass
    out["integrator_used"] = int(model.opt.integrator)
    out["solver_used"] = int(model.opt.solver)

    # ---- rollout ----
    data = None
    try:
        data = mujoco.MjData(model)
        _apply_perturb(np.asarray(data.qpos), perturb.get("qpos") or [])
        _apply_perturb(np.asarray(data.qvel), perturb.get("qvel") or [])
        _apply_perturb(np.asarray(data.ctrl), perturb.get("ctrl") or [])

        for i in range(n_steps):
            mujoco.mj_step(model, data)
            out["runtime"]["steps_done"] = i + 1
        out["runtime"]["ok"] = True

        out["state_stats"] = state_stats(
            np.asarray(data.qpos),
            np.asarray(data.qvel),
            np.asarray(data.ctrl) if data.ctrl.size else np.zeros(0),
            int(data.ncon),
        )
    except Exception as e:
        out["runtime"]["exception_type"] = type(e).__name__
        out["runtime"]["traceback"] = "".join(
            traceback.format_exception_only(type(e), e)
        )[:500]

    # ---- consistency: 2nd MjData, same perturb, compare qpos ----
    if consistency and out["compile"]["ok"]:
        try:
            data2 = mujoco.MjData(model)
            _apply_perturb(np.asarray(data2.qpos), perturb.get("qpos") or [])
            _apply_perturb(np.asarray(data2.qvel), perturb.get("qvel") or [])
            _apply_perturb(np.asarray(data2.ctrl), perturb.get("ctrl") or [])
            for _ in range(n_steps):
                mujoco.mj_step(model, data2)
            if data is not None:
                a = np.asarray(data.qpos)
                b = np.asarray(data2.qpos)
                if a.size and b.size:
                    d = np.abs(a - b)
                    diff = float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")
                    out["consistency"] = {"ran": True, "diff": diff}
        except Exception:
            out["consistency"] = {"ran": False, "diff": None}

    # ---- warnings ----
    out["warnings"] = _collect_warnings(data) if data is not None else _collect_warnings_no_data()
    return out


def _collect_warnings_no_data() -> list[dict[str, Any]]:
    if not _WARN_BUF:
        return []
    return [{"wtype": "UNCATEGORIZED", "count": len(_WARN_BUF), "message": _WARN_BUF[0][:200]}]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as f:
        spec = json.load(f)

    try:
        out = run_one(spec)
    except BaseException as e:
        # Programmer error in worker itself: still write structured failure
        out = _empty_result()
        out["runtime"]["exception_type"] = "WorkerInternalError:" + type(e).__name__
        out["runtime"]["traceback"] = "".join(
            traceback.format_exception_only(type(e), e)
        )[:500]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
