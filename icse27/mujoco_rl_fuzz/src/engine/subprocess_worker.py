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

    # ---- v8: differential solver oracle ----
    # Re-run same XML + same perturb under DIFFERENT solver/integrator combos
    # and compare final qpos. Large divergence on a well-conditioned system is
    # a strong real-bug signal (one of the solvers got it wrong).
    # We only enable this path when the rollout was deterministic and finite
    # (otherwise diff is meaningless).
    out["solver_diff"] = None
    if (consistency and out["compile"]["ok"] and out["runtime"]["ok"]
            and data is not None
            and not out["state_stats"].get("has_nan")
            and not out["state_stats"].get("has_inf")):
        try:
            ref_qpos = np.array(data.qpos, copy=True)
            results = {}
            results_qpos: dict[str, np.ndarray] = {}
            # (solver_id, integrator_id, label)
            #   mjtSolver: 0=PGS, 1=CG, 2=Newton
            #   mjtIntegrator: 0=Euler, 1=RK4, 2=implicit, 3=implicitfast
            combos = [
                (2, 0, "Newton+Euler"),
                (1, 0, "CG+Euler"),
                (0, 0, "PGS+Euler"),
                (2, 2, "Newton+implicit"),
                (2, 3, "Newton+implicitfast"),
            ]
            for sid, iid, label in combos:
                m2 = mujoco.MjModel.from_xml_path(xml_path)
                m2.opt.solver = sid
                m2.opt.integrator = iid
                # Use generous iterations + tight tolerance so each solver
                # has the best chance to actually converge — disagreement
                # then implicates one solver, not lack of effort.
                m2.opt.iterations = 200
                m2.opt.tolerance = 1e-10
                if disable_clamp_ctrl:
                    m2.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
                d2 = mujoco.MjData(m2)
                _apply_perturb(np.asarray(d2.qpos), perturb.get("qpos") or [])
                _apply_perturb(np.asarray(d2.qvel), perturb.get("qvel") or [])
                _apply_perturb(np.asarray(d2.ctrl), perturb.get("ctrl") or [])
                ok = True
                for _ in range(n_steps):
                    try:
                        mujoco.mj_step(m2, d2)
                    except Exception:
                        ok = False
                        break
                qp = np.asarray(d2.qpos)
                if ok and qp.size == ref_qpos.size and np.isfinite(qp).all():
                    diff = float(np.nanmax(np.abs(qp - ref_qpos)))
                    results_qpos[label] = np.array(qp, copy=True)
                else:
                    diff = float("nan")
                results[label] = diff
            # ---- v8 metric: same-integrator solver disagreement ----
            # Compare the actual final qpos vectors (not their diffs vs ref)
            # across Newton/CG/PGS+Euler. With iterations=200 and tol=1e-10
            # the three convex solvers MUST converge to the same fixed point;
            # any disagreement implicates one solver. Cross-integrator diffs
            # (implicit/implicitfast) are legitimately different and kept
            # for diagnostic context only, not flagged.
            try:
                triplet = ["Newton+Euler", "CG+Euler", "PGS+Euler"]
                qps = {k: results_qpos.get(k) for k in triplet
                       if results_qpos.get(k) is not None}
                solver_only_max = 0.0
                if len(qps) >= 2:
                    keys = list(qps.keys())
                    for i in range(len(keys)):
                        for j in range(i + 1, len(keys)):
                            a, b = qps[keys[i]], qps[keys[j]]
                            if a.size == b.size and np.isfinite(a).all() and np.isfinite(b).all():
                                d = float(np.nanmax(np.abs(a - b)))
                                if d > solver_only_max:
                                    solver_only_max = d
            except Exception:
                solver_only_max = 0.0
            finite_diffs = [v for v in results.values() if np.isfinite(v)]
            max_diff = max(finite_diffs) if finite_diffs else float("nan")
            out["solver_diff"] = {
                "per_combo_diff_vs_ref": results,
                "max_diff": max_diff,
                "solver_only_max_diff": solver_only_max,
                "ref_solver": int(model.opt.solver),
                "ref_integrator": int(model.opt.integrator),
            }
        except Exception as e:
            out["solver_diff"] = {"error": f"{type(e).__name__}: {e}"[:200]}

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
