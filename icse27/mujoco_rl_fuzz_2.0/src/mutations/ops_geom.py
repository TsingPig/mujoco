"""Geom & inertial mutators."""
from __future__ import annotations

import random
from typing import List

from lxml import etree

from ..mjcf.invariants import all_bodies, non_plane_geoms
from ..mjcf.spec_loader import SafeModel
from . import intensity as I
from .base import ApplyResult


# ----------------------------- MUTATE_GEOM_SHAPE -----------------------------
class MutateGeomShape:
    id = "MUTATE_GEOM_SHAPE"
    intensity_modes = list(I.GEOM_SHAPES)
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(non_plane_geoms(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        geoms = non_plane_geoms(model.root)
        if not geoms:
            return ApplyResult(False, intensity_mode, reason="no_geoms")
        g = rng.choice(geoms)
        g.set("type", intensity_mode)
        # Recompute size to match the shape's expected dim using the geom's
        # current characteristic length (median of existing size scalars).
        old_size = (g.get("size") or "0.05").split()
        try:
            base = float(old_size[0]) if old_size else 0.05
        except ValueError:
            base = 0.05
        dim = I.GEOM_SIZE_DIM[intensity_mode]
        if intensity_mode == "sphere":
            g.set("size", f"{base:.6g}")
        elif intensity_mode in ("capsule", "cylinder"):
            g.set("size", f"{base:.6g} {base*2:.6g}")
        else:
            g.set("size", I.fmt_floats(base, base, base))
        # Strip mesh attribute if any (incompatible with primitive shapes).
        if "mesh" in g.attrib:
            del g.attrib["mesh"]
        # `fromto` is only valid for capsule/cylinder/box/ellipsoid.
        if intensity_mode == "sphere" and "fromto" in g.attrib:
            del g.attrib["fromto"]
        return ApplyResult(True, intensity_mode, params_used={"dim": dim, "base": base})


# ----------------------------- MUTATE_GEOM_SIZE ------------------------------
class MutateGeomSize:
    id = "MUTATE_GEOM_SIZE"
    intensity_modes = list(I.GEOM_SIZE_MODES.keys())
    # near_zero may push a derived inertia below mjMINVAL when the body has no
    # explicit <inertial>; treat as deliberately-rejected case.
    invalid_parseable_modes: List[str] = ["near_zero"]
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        # Only typed geoms (else we cannot know the size dim; class defaults
        # may make our value count wrong).
        return any(g.get("type") for g in non_plane_geoms(model.root))

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        geoms = [g for g in non_plane_geoms(model.root) if g.get("type")]
        if not geoms:
            return ApplyResult(False, intensity_mode, reason="no_typed_geoms")
        g = rng.choice(geoms)
        gtype = g.get("type", "sphere")
        if gtype not in I.GEOM_SIZE_DIM:
            return ApplyResult(False, intensity_mode, reason=f"unsupported_type:{gtype}")
        dim = I.GEOM_SIZE_DIM[gtype]
        # When `fromto` is set, capsule/cylinder accept only a single radius.
        if g.get("fromto") is not None and gtype in ("capsule", "cylinder"):
            dim = 1
        # Respect the existing size's element count when it's smaller (some
        # MJCF authors set fewer dims and rely on defaults; supplying extra
        # values can re-trigger schema checks).
        existing = (g.get("size") or "").split()
        if existing and len(existing) < dim:
            dim = len(existing)
        sizes = I.pick_size(rng, intensity_mode, dim)
        g.set("size", I.fmt_floats(*sizes))
        return ApplyResult(True, intensity_mode, params_used={"sizes": sizes})


# ----------------------------- MUTATE_INERTIAL -------------------------------
class MutateInertial:
    id = "MUTATE_INERTIAL"
    intensity_modes = list(I.INERTIAL_MODES.keys())
    invalid_parseable_modes = ["negative_principal"]
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(all_bodies(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        bodies = all_bodies(model.root)
        if not bodies:
            return ApplyResult(False, intensity_mode, reason="no_bodies")
        b = rng.choice(bodies)
        spec = I.INERTIAL_MODES[intensity_mode]
        diag = spec["diag_scale"]
        if isinstance(diag, tuple):
            diag_vals = diag
        else:
            diag_vals = (diag, diag, diag)

        # Replace any existing inertial.
        old = b.find("inertial")
        if old is not None:
            b.remove(old)
        el = etree.SubElement(b, "inertial")
        # Inertial must come first by MJCF schema; move to front.
        b.remove(el)
        b.insert(0, el)
        el.set("pos", "0 0 0")
        el.set("mass", f"{float(spec['mass']):.6g}")
        el.set("diaginertia", I.fmt_floats(*diag_vals))
        return ApplyResult(True, intensity_mode, params_used={"mass": spec["mass"], "diag": diag_vals})


# ----------------------------- MUTATE_FRICTION -------------------------------
class MutateFriction:
    id = "MUTATE_FRICTION"
    intensity_modes = list(I.FRICTION_MODES.keys())
    invalid_parseable_modes = ["boundary_neg_eps"]
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(non_plane_geoms(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        geoms = non_plane_geoms(model.root)
        if not geoms:
            return ApplyResult(False, intensity_mode, reason="no_geoms")
        g = rng.choice(geoms)
        triple = I.FRICTION_MODES[intensity_mode]
        g.set("friction", I.fmt_floats(*triple))
        return ApplyResult(True, intensity_mode, params_used={"friction": triple})


# ----------------------------- MUTATE_SOLREF_SOLIMP --------------------------
class MutateSolrefSolimp:
    id = "MUTATE_SOLREF_SOLIMP"
    intensity_modes = list(I.SOLREF_MODES.keys())
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(non_plane_geoms(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        geoms = non_plane_geoms(model.root)
        if not geoms:
            return ApplyResult(False, intensity_mode, reason="no_geoms")
        g = rng.choice(geoms)
        ref = I.SOLREF_MODES[intensity_mode]
        g.set("solref", I.fmt_floats(*ref))
        # Pair solimp from a fixed mapping (default), using the same key when avail.
        imp_key = intensity_mode if intensity_mode in I.SOLIMP_MODES else "default"
        imp = I.SOLIMP_MODES[imp_key]
        g.set("solimp", I.fmt_floats(*imp))
        return ApplyResult(True, intensity_mode,
                           params_used={"solref": ref, "solimp": imp})


# ----------------------------- MUTATE_CONTACT_MARGIN -------------------------
class MutateContactMargin:
    id = "MUTATE_CONTACT_MARGIN"
    intensity_modes = list(I.CONTACT_MARGIN_MODES.keys())
    invalid_parseable_modes: List[str] = []
    runtime_only = False

    def applicable(self, model: SafeModel) -> bool:
        return len(non_plane_geoms(model.root)) > 0

    def apply(self, model: SafeModel, intensity_mode: str, rng: random.Random) -> ApplyResult:
        geoms = non_plane_geoms(model.root)
        if not geoms:
            return ApplyResult(False, intensity_mode, reason="no_geoms")
        g = rng.choice(geoms)
        margin = I.CONTACT_MARGIN_MODES[intensity_mode]
        g.set("margin", f"{margin:.6g}")
        # gap must be <= margin per MuJoCo; use 0.
        g.set("gap", "0")
        return ApplyResult(True, intensity_mode, params_used={"margin": margin})
