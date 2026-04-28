"""Runtime state perturbations.

These mutators do NOT modify the XML; instead they emit a `runtime_directive`
dict that the (future) episodic runner will pass into the subprocess worker.

Encoding (kept identical to v1 worker for forward compatibility):

    runtime_directive = {
        "kind": "set_qpos" | "set_qvel" | "set_ctrl",
        "values": [float | "nan" | "+inf" | "-inf", ...],
        "first_n": int,                # apply to first N entries; rest unchanged
        "disable_clamp_ctrl": bool,    # set on `ctrl` + nan_inf to bypass clamp
    }
"""
from __future__ import annotations

import math
import random
from typing import List

from ..mjcf.spec_loader import SafeModel
from . import intensity as I
from .base import ApplyResult


def _nan_inf_token(rng: random.Random) -> str:
    return rng.choice(["nan", "+inf", "-inf"])


def _pick_qpos(rng: random.Random, mode: str) -> list:
    if mode == "zero":
        return [0.0]
    if mode == "small":
        return [rng.uniform(-0.05, 0.05) for _ in range(3)]
    if mode == "large":
        return [rng.uniform(-50.0, 50.0) for _ in range(3)]
    if mode == "near_limit":
        return [rng.choice([-3.13, 3.13, -1e3, 1e3])]
    if mode == "nan_inf":
        return [_nan_inf_token(rng) for _ in range(2)]
    return [0.0]


def _pick_qvel(rng: random.Random, mode: str) -> list:
    if mode == "zero":
        return [0.0]
    if mode == "small":
        return [rng.uniform(-0.5, 0.5) for _ in range(3)]
    if mode == "large":
        return [rng.uniform(-1e3, 1e3) for _ in range(3)]
    if mode == "nan_inf":
        return [_nan_inf_token(rng) for _ in range(2)]
    return [0.0]


def _pick_ctrl(rng: random.Random, mode: str) -> list:
    if mode == "zero":
        return [0.0]
    if mode == "mid":
        return [rng.uniform(-0.5, 0.5) for _ in range(3)]
    if mode == "clip_max":
        return [rng.choice([-1.0, 1.0]) for _ in range(3)]
    if mode == "over_clip":
        return [rng.choice([-1e6, 1e6]) for _ in range(3)]
    if mode == "nan_inf":
        return [_nan_inf_token(rng) for _ in range(2)]
    return [0.0]


class _RuntimeBase:
    runtime_only = True
    invalid_parseable_modes = ["nan_inf"]

    def applicable(self, model: SafeModel) -> bool:  # always applicable
        return True


class SetQposRuntime(_RuntimeBase):
    id = "SET_QPOS_RUNTIME"
    intensity_modes = list(I.QPOS_MODES)

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        vals = _pick_qpos(rng, intensity_mode)
        directive = {"kind": "set_qpos", "values": vals,
                     "first_n": len(vals), "disable_clamp_ctrl": False}
        return ApplyResult(True, intensity_mode, runtime_directive=directive,
                           params_used={"values": vals})


class SetQvelRuntime(_RuntimeBase):
    id = "SET_QVEL_RUNTIME"
    intensity_modes = list(I.QVEL_MODES)

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        vals = _pick_qvel(rng, intensity_mode)
        directive = {"kind": "set_qvel", "values": vals,
                     "first_n": len(vals), "disable_clamp_ctrl": False}
        return ApplyResult(True, intensity_mode, runtime_directive=directive,
                           params_used={"values": vals})


class SetCtrlRuntime(_RuntimeBase):
    id = "SET_CTRL_RUNTIME"
    intensity_modes = list(I.CTRL_MODES)

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        vals = _pick_ctrl(rng, intensity_mode)
        # nan_inf or over_clip on ctrl needs CLAMPCTRL disabled, otherwise
        # MuJoCo silently clamps and BADCTRL never fires.
        disable_clamp = intensity_mode in ("nan_inf", "over_clip")
        directive = {"kind": "set_ctrl", "values": vals,
                     "first_n": len(vals),
                     "disable_clamp_ctrl": disable_clamp}
        return ApplyResult(True, intensity_mode, runtime_directive=directive,
                           params_used={"values": vals, "disable_clamp_ctrl": disable_clamp})
