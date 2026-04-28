"""Structural mutators: add/remove/duplicate body subtrees.

All mutations are MJCF-valid by construction:
  * every new body has joint + geom + explicit inertial (mass > mjMINVAL)
  * unique names auto-suffixed via SafeModel.fresh_name
  * shrink only removes leaf or fully-contained subtrees so dangling
    references don't appear (no joint/geom referenced from actuator/eq)
"""
from __future__ import annotations

import random
from copy import deepcopy
from typing import List

from lxml import etree

from ..mjcf.invariants import all_bodies, worldbody
from ..mjcf.spec_loader import SafeModel
from . import intensity as I
from .base import ApplyResult


def _make_inertial(mass: float = 1.0, diag: float = 0.01) -> etree._Element:
    el = etree.Element("inertial")
    el.set("pos", "0 0 0")
    el.set("mass", f"{mass:.6g}")
    el.set("diaginertia", I.fmt_floats(diag, diag, diag))
    return el


def _make_geom(name: str, shape: str = "sphere", size: float = 0.05,
               rgba: str = "0.6 0.6 0.9 1") -> etree._Element:
    g = etree.Element("geom")
    g.set("name", name)
    g.set("type", shape)
    if shape == "sphere":
        g.set("size", f"{size:.6g}")
    elif shape in ("capsule", "cylinder"):
        g.set("size", f"{size:.6g} {size*2:.6g}")
    else:  # box / ellipsoid
        g.set("size", I.fmt_floats(size, size, size))
    g.set("rgba", rgba)
    return g


def _make_body(model: SafeModel, name_prefix: str,
               pos: str = "0 0 0.1",
               jtype: str = "hinge",
               shape: str = "sphere",
               size: float = 0.05,
               mass: float = 1.0) -> etree._Element:
    body = etree.Element("body")
    body.set("name", model.fresh_name(name_prefix))
    body.set("pos", pos)
    body.append(_make_inertial(mass=mass, diag=0.01))
    j = etree.SubElement(body, "joint")
    j.set("name", model.fresh_name(f"{name_prefix}_jnt"))
    j.set("type", jtype)
    # Explicitly mark unlimited so any inherited <default><joint limited="true"/>
    # in the seed doesn't force a missing range to (0,0).
    if jtype in ("hinge", "slide"):
        j.set("limited", "false")
    body.append(_make_geom(model.fresh_name(f"{name_prefix}_geom"), shape, size))
    return body


# ----------------------------- STRUCT_GROW_LINK ------------------------------
class StructGrowLink:
    id = "STRUCT_GROW_LINK"
    intensity_modes = list(I.STRUCT_GROW_MODES)
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return worldbody(model.root) is not None

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        wb = worldbody(model.root)
        bodies = all_bodies(model.root)
        parent = rng.choice(bodies) if bodies else wb

        if intensity_mode == "simple_pendulum":
            parent.append(_make_body(model, "grow", jtype="hinge",
                                     shape="sphere", size=0.04, mass=0.5))
        elif intensity_mode == "compound_arm":
            link = _make_body(model, "arm", pos="0.05 0 0",
                              jtype="hinge", shape="capsule", size=0.02, mass=0.2)
            parent.append(link)
            link.append(_make_body(model, "arm_lo", pos="0.05 0 0",
                                   jtype="hinge", shape="capsule", size=0.02, mass=0.2))
        elif intensity_mode == "cluster_balls":
            for _ in range(3):
                parent.append(_make_body(model, "ball",
                                         pos=f"{rng.uniform(-0.05,0.05):.3f} 0 0.1",
                                         jtype="hinge", shape="sphere", size=0.03, mass=0.1))
        # Adding a kinematic link changes nq/nv; any pre-existing <keyframe>
        # with fixed-length qpos arrays is now invalid. Drop the section.
        kf = model.root.find("keyframe")
        if kf is not None:
            kf.getparent().remove(kf)
        return ApplyResult(True, intensity_mode)


# ----------------------------- STRUCT_SHRINK ---------------------------------
class StructShrink:
    id = "STRUCT_SHRINK"
    intensity_modes = list(I.STRUCT_SHRINK_MODES)
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(all_bodies(model.root)) >= 1

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        bodies = all_bodies(model.root)
        if not bodies:
            return ApplyResult(False, intensity_mode, reason="no_bodies")

        # Collect every name referenced from actuator/equality/tendon/sensor/
        # contact/deformable/composite. Includes joint/geom/site/body/flex
        # references so we never delete a body that *contains* something
        # referenced from elsewhere.
        ref_attrs = (
            "body", "body1", "body2",
            "joint", "joint1", "joint2",
            "geom", "geom1", "geom2",
            "site", "site1", "site2",
            "tendon", "tendon1", "tendon2",
            "flex", "objname", "name1", "name2",
        )
        used_names = set()
        for tag in ("actuator", "equality", "tendon", "sensor",
                    "contact", "deformable"):
            sec = model.root.find(tag)
            if sec is None:
                continue
            for el in sec.iter():
                for k in ref_attrs:
                    v = el.get(k)
                    if v:
                        used_names.add(v)

        def body_owns_referenced_child(b: etree._Element) -> bool:
            for el in b.iter():
                if el is b:
                    nm = el.get("name")
                    if nm and nm in used_names:
                        return True
                    continue
                if el.tag in ("joint", "geom", "site", "body", "flex"):
                    nm = el.get("name")
                    if nm and nm in used_names:
                        return True
            return False

        def has_flex_or_composite(b: etree._Element) -> bool:
            # Plugin/elasticity flexes attach to specific bodies; removing such
            # bodies invalidates the flex element.
            for el in b.iter():
                if el.tag in ("flex", "composite", "plugin"):
                    return True
            return False

        def is_leaf(b: etree._Element) -> bool:
            return b.find("body") is None

        candidates = [b for b in bodies
                      if not body_owns_referenced_child(b)
                      and not has_flex_or_composite(b)
                      and (intensity_mode != "leaf_only" or is_leaf(b))]
        if not candidates:
            return ApplyResult(False, intensity_mode, reason="no_safe_target")
        target = rng.choice(candidates)
        target.getparent().remove(target)
        return ApplyResult(True, intensity_mode, params_used={"removed": target.get("name")})


# ----------------------------- STRUCT_DUPLICATE_SUBTREE ----------------------
class StructDuplicateSubtree:
    id = "STRUCT_DUPLICATE_SUBTREE"
    intensity_modes = list(I.STRUCT_DUP_MODES)
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(all_bodies(model.root)) >= 1

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        bodies = all_bodies(model.root)
        if not bodies:
            return ApplyResult(False, intensity_mode, reason="no_bodies")
        n = {"single": 1, "chain_x3": 3, "chain_x10": 10}[intensity_mode]
        src = rng.choice(bodies)
        parent = src.getparent()
        for _ in range(n):
            clone = deepcopy(src)
            # Re-suffix every name attribute we encounter to avoid collisions.
            suffix = "_" + model.fresh_name("dup")
            for el in clone.iter():
                nm = el.get("name")
                if nm:
                    el.set("name", nm + suffix)
            parent.append(clone)
        # Adding bodies changes nq/nv; any pre-existing <keyframe> with fixed
        # qpos arrays is now invalid. Drop the section.
        kf = model.root.find("keyframe")
        if kf is not None:
            kf.getparent().remove(kf)
        return ApplyResult(True, intensity_mode, params_used={"n": n})
