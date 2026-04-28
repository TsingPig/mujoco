"""Actuator / equality / tendon mutators."""
from __future__ import annotations

import random
from typing import List

from lxml import etree

from ..mjcf.invariants import (
    actuator_root,
    all_bodies,
    all_joints,
    equality_root,
    hingelike_joints,
    tendon_root,
)
from ..mjcf.spec_loader import SafeModel
from . import intensity as I
from .base import ApplyResult


_STD_ACTUATOR_TAGS = ("motor", "position", "velocity", "general", "intvelocity", "damper")


# ----------------------------- ACTUATOR_ADD ----------------------------------
class ActuatorAdd:
    id = "ACTUATOR_ADD"
    intensity_modes = list(I.ACTUATOR_KIND_MODES.keys())
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(hingelike_joints(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        joints = [j for j in hingelike_joints(model.root) if j.get("name")]
        if not joints:
            return ApplyResult(False, intensity_mode, reason="no_named_joints")
        j = rng.choice(joints)
        kind_spec = I.ACTUATOR_KIND_MODES[intensity_mode]
        act = actuator_root(model.root)
        el = etree.SubElement(act, kind_spec["tag"])
        el.set("name", model.fresh_name(f"act_{intensity_mode}"))
        el.set("joint", j.get("name"))
        lo, hi = I.CTRL_RANGE_MODES[kind_spec["ctrlrange"]]
        el.set("ctrlrange", I.fmt_floats(lo, hi))
        el.set("ctrllimited", "true")
        for k, v in kind_spec["extra"].items():
            el.set(k, v)
        return ApplyResult(True, intensity_mode, params_used={"joint": j.get("name")})


# ----------------------------- ACTUATOR_MUTATE_RANGE -------------------------
class ActuatorMutateRange:
    id = "ACTUATOR_MUTATE_RANGE"
    intensity_modes = list(I.CTRL_RANGE_MODES.keys())
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        act = model.root.find("actuator")
        return act is not None and any(c.tag in _STD_ACTUATOR_TAGS for c in act)

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        act = model.root.find("actuator")
        if act is None:
            return ApplyResult(False, intensity_mode, reason="no_actuators")
        eligible = [c for c in act if c.tag in _STD_ACTUATOR_TAGS]
        if not eligible:
            return ApplyResult(False, intensity_mode, reason="no_std_actuators")
        target = rng.choice(eligible)
        lo, hi = I.CTRL_RANGE_MODES[intensity_mode]
        target.set("ctrlrange", I.fmt_floats(lo, hi))
        target.set("ctrllimited", "true")
        return ApplyResult(True, intensity_mode, params_used={"range": (lo, hi)})


# ----------------------------- ACTUATOR_DELETE -------------------------------
class ActuatorDelete:
    id = "ACTUATOR_DELETE"
    intensity_modes = ["_default"]
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        act = model.root.find("actuator")
        return act is not None and any(c.tag in _STD_ACTUATOR_TAGS for c in act)

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        act = model.root.find("actuator")
        if act is None:
            return ApplyResult(False, intensity_mode, reason="no_actuators")
        # Plugin actuators may be required by their plugin instance; skip them.
        eligible = [c for c in act
                    if c.tag in _STD_ACTUATOR_TAGS and c.find("plugin") is None]
        if not eligible:
            return ApplyResult(False, intensity_mode, reason="no_safe_actuators")
        target = rng.choice(eligible)
        act.remove(target)
        return ApplyResult(True, intensity_mode)


# ----------------------------- EQUALITY_ADD ----------------------------------
class EqualityAdd:
    id = "EQUALITY_ADD"
    intensity_modes = list(I.EQUALITY_MODES)
    invalid_parseable_modes = ["over_constrain_pair"]
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len([b for b in all_bodies(model.root) if b.get("name")]) >= 2

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        named = [b for b in all_bodies(model.root) if b.get("name")]
        if len(named) < 2:
            return ApplyResult(False, intensity_mode, reason="need_two_named_bodies")
        b1, b2 = rng.sample(named, 2)
        eq = equality_root(model.root)

        if intensity_mode == "connect_close":
            el = etree.SubElement(eq, "connect")
            el.set("body1", b1.get("name"))
            el.set("body2", b2.get("name"))
            el.set("anchor", "0 0 0")
        elif intensity_mode == "connect_far":
            el = etree.SubElement(eq, "connect")
            el.set("body1", b1.get("name"))
            el.set("body2", b2.get("name"))
            el.set("anchor", I.fmt_floats(rng.uniform(0.5, 2.0), 0.0, 0.0))
        elif intensity_mode == "weld":
            el = etree.SubElement(eq, "weld")
            el.set("body1", b1.get("name"))
            el.set("body2", b2.get("name"))
        elif intensity_mode == "over_constrain_pair":
            for _ in range(3):
                el = etree.SubElement(eq, "connect")
                el.set("name", model.fresh_name("eq"))
                el.set("body1", b1.get("name"))
                el.set("body2", b2.get("name"))
                el.set("anchor", I.fmt_floats(rng.uniform(-0.1, 0.1), 0.0, 0.0))
        else:
            return ApplyResult(False, intensity_mode, reason="unknown_mode")
        if "name" not in el.attrib:
            el.set("name", model.fresh_name(f"eq_{intensity_mode}"))
        return ApplyResult(True, intensity_mode, params_used={"b1": b1.get("name"), "b2": b2.get("name")})


# ----------------------------- TENDON_ADD ------------------------------------
class TendonAdd:
    id = "TENDON_ADD"
    intensity_modes = list(I.TENDON_MODES)
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        named_joints = [j for j in hingelike_joints(model.root) if j.get("name")]
        return len(named_joints) >= 2

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        named_joints = [j for j in hingelike_joints(model.root) if j.get("name")]
        if len(named_joints) < 2:
            return ApplyResult(False, intensity_mode, reason="need_two_named_joints")
        ten = tendon_root(model.root)
        # `fixed` tendon: combination of joints with coefficients.
        el = etree.SubElement(ten, "fixed")
        el.set("name", model.fresh_name(f"ten_{intensity_mode}"))
        if intensity_mode == "fixed_short":
            picks = rng.sample(named_joints, 2)
            coeffs = [1.0, -1.0]
        elif intensity_mode == "fixed_long":
            picks = rng.sample(named_joints, min(len(named_joints), 4))
            coeffs = [1.0] * len(picks)
        elif intensity_mode == "spatial_loop":
            # Long-but-loopy combination of all available joints.
            picks = list(named_joints)
            coeffs = [1.0 if i % 2 == 0 else -1.0 for i in range(len(picks))]
        else:
            return ApplyResult(False, intensity_mode, reason="unknown_mode")
        for j, c in zip(picks, coeffs):
            sub = etree.SubElement(el, "joint")
            sub.set("joint", j.get("name"))
            sub.set("coef", f"{c:.6g}")
        return ApplyResult(True, intensity_mode, params_used={"joints": [j.get("name") for j in picks]})
