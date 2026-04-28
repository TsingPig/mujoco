"""Joint-related mutators."""
from __future__ import annotations

import random
from typing import List

from ..mjcf.invariants import all_joints, hingelike_joints
from ..mjcf.spec_loader import SafeModel
from . import intensity as I
from .base import ApplyResult


# ----------------------------- MUTATE_JOINT_TYPE -----------------------------
class MutateJointType:
    id = "MUTATE_JOINT_TYPE"
    intensity_modes = list(I.JOINT_TYPES)
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        # Skip if no joints, or if the only joints already attach to worldbody
        # children (free joint constraints).
        return len(all_joints(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        joints = all_joints(model.root)
        candidates = [j for j in joints if j.getparent().tag == "body"]
        # Collect names referenced by sensor / equality / actuator / tendon —
        # changing such a joint to ball/free breaks those constraints.
        ref_names = set()
        for tag in ("sensor", "equality", "actuator", "tendon"):
            sec = model.root.find(tag)
            if sec is None:
                continue
            for el in sec.iter():
                for k in ("joint", "joint1", "joint2"):
                    v = el.get(k)
                    if v:
                        ref_names.add(v)
        if intensity_mode in ("ball", "free"):
            candidates = [j for j in candidates if j.get("name") not in ref_names]
        # `free` joints must attach to a body whose parent is <worldbody>.
        if intensity_mode == "free":
            candidates = [
                j for j in candidates
                if j.getparent().getparent() is not None
                and j.getparent().getparent().tag == "worldbody"
            ]
        # `ball` joint cannot be in a body with rotation (quat/axisangle/euler).
        if intensity_mode == "ball":
            rot_attrs = ("quat", "axisangle", "euler", "xyaxes", "zaxis")
            candidates = [
                j for j in candidates
                if not any(j.getparent().get(a) for a in rot_attrs)
            ]
        if not candidates:
            return ApplyResult(False, intensity_mode, reason="no_eligible_joints")
        j = rng.choice(candidates)
        # `range`/`limited`/`axis` may conflict; clean attributes that don't
        # apply to the new type.
        new_type = intensity_mode
        if new_type in ("ball", "free"):
            for attr in ("range", "limited", "axis"):
                if attr in j.attrib:
                    del j.attrib[attr]
        else:
            # Hinge/slide need an axis; ensure one is set.
            if not j.get("axis"):
                j.set("axis", "0 0 1")
        # `free` joint must be the body's only joint and the body must be a
        # direct child of worldbody. Refuse otherwise.
        if new_type == "free":
            body = j.getparent()
            if body.getparent() is not None and body.getparent().tag != "worldbody":
                return ApplyResult(False, intensity_mode, reason="free_needs_worldbody_parent")
            # Drop sibling joints.
            for sib in list(body.findall("joint")):
                if sib is not j:
                    body.remove(sib)
        j.set("type", new_type)
        return ApplyResult(True, intensity_mode, params_used={"type": new_type})


# ----------------------------- MUTATE_JOINT_LIMIT ----------------------------
class MutateJointLimit:
    id = "MUTATE_JOINT_LIMIT"
    intensity_modes = list(I.JOINT_LIMIT_MODES.keys())
    invalid_parseable_modes = ["inverted", "singular_zero_range"]
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(hingelike_joints(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        joints = hingelike_joints(model.root)
        if not joints:
            return ApplyResult(False, intensity_mode, reason="no_hinge_or_slide")
        j = rng.choice(joints)
        lo, hi = I.JOINT_LIMIT_MODES[intensity_mode]
        j.set("limited", "true")
        j.set("range", I.fmt_floats(lo, hi))
        return ApplyResult(True, intensity_mode, params_used={"range": (lo, hi)})


# ----------------------------- MUTATE_DAMPING_FRICTION ----------------------
class MutateDampingFriction:
    id = "MUTATE_DAMPING_FRICTION"
    intensity_modes = list(I.DAMPING_FRICTION_MODES.keys())
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(hingelike_joints(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        joints = hingelike_joints(model.root)
        if not joints:
            return ApplyResult(False, intensity_mode, reason="no_hinge_or_slide")
        j = rng.choice(joints)
        spec = I.DAMPING_FRICTION_MODES[intensity_mode]
        j.set("damping", f"{spec['damping']:.6g}")
        j.set("frictionloss", f"{spec['frictionloss']:.6g}")
        j.set("armature", f"{spec['armature']:.6g}")
        return ApplyResult(True, intensity_mode, params_used=dict(spec))
