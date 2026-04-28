"""Centralised IntensityTable.

Every numeric/categorical knob a mutator can set is enumerated here.
Mutators NEVER inline magic numbers; they call `pick(...)` instead.
This guarantees that `(mutator_id, intensity_mode)` fully determines the
*distribution* of values, even if the rng inside picks one.

Naming convention:
    "<small|medium|large|extreme|near_zero|nan_inf|...>"

Numeric tables are float lists; the mutator picks one with rng.choice.
Categorical tables are string lists.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

# --- Geom shape ---
GEOM_SHAPES = ("sphere", "capsule", "box", "cylinder", "ellipsoid")
# size dim per shape per MuJoCo docs
GEOM_SIZE_DIM = {"sphere": 1, "capsule": 2, "cylinder": 2, "box": 3, "ellipsoid": 3}

GEOM_SIZE_MODES: Dict[str, List[float]] = {
    # Each value is the per-dimension scale picked uniformly with rng.
    "tiny":      [1e-3, 2e-3, 5e-3],
    "small":     [0.005, 0.01, 0.02],
    "medium":    [0.05, 0.1, 0.15],
    "large":     [0.3, 0.5, 0.8],
    "huge":      [1.0, 2.0, 5.0],
    # Whitelisted invalid_parseable: zero radius is rejected at compile,
    # so use eps just above mjMINVAL=1e-15 (compile passes; runtime risky).
    "near_zero": [1e-6, 5e-7, 1e-7],
}

# --- Inertial ---
INERTIAL_MODES: Dict[str, Dict[str, Any]] = {
    "near_zero_mass":    {"mass": 1e-4,  "diag_scale": 1e-4},
    "small_mass":        {"mass": 1e-2,  "diag_scale": 1e-3},
    "default_mass":      {"mass": 1.0,   "diag_scale": 1e-2},
    "huge_mass":         {"mass": 1e3,   "diag_scale": 10.0},
    "near_zero_inertia": {"mass": 1.0,   "diag_scale": 1e-7},
    # diag values must satisfy triangle inequality A+B>=C; pick (0.5, 1.0, 1.4).
    "anisotropic":       {"mass": 1.0,   "diag_scale": (0.5, 1.0, 1.4)},
    # Negative principal moments — mj_loadXML rejects (compile error).
    # Kept here because we want to exercise the rejection path *deliberately*
    # via a runtime test, not the CI gate.
    "negative_principal": {"mass": 1.0,  "diag_scale": (-1.0, 1.0, 1.0)},
}

# --- Joint ---
JOINT_TYPES = ("hinge", "slide", "ball", "free")

JOINT_LIMIT_MODES: Dict[str, Tuple[float, float]] = {
    "narrow":               (-0.05, 0.05),
    "medium":               (-1.0, 1.0),
    "wide":                 (-3.14, 3.14),
    "inverted":             (1.0, -1.0),       # invalid_parseable: lo > hi
    "singular_zero_range":  (0.0, 0.0),        # zero-width range
}

DAMPING_FRICTION_MODES: Dict[str, Dict[str, float]] = {
    "low":     {"damping": 1e-4, "frictionloss": 0.0,    "armature": 0.0},
    "medium":  {"damping": 0.1,  "frictionloss": 0.01,   "armature": 0.01},
    "high":    {"damping": 10.0, "frictionloss": 1.0,    "armature": 0.5},
    "extreme": {"damping": 1e4,  "frictionloss": 100.0,  "armature": 10.0},
}

# --- Contact ---
FRICTION_MODES: Dict[str, Tuple[float, float, float]] = {
    "frictionless":   (0.0, 0.0, 0.0),
    "small":          (0.1, 0.005, 0.0001),
    "medium":         (1.0, 0.005, 0.0001),
    "large":          (10.0, 0.05, 0.001),
    "extreme":        (1e3, 1.0, 0.1),
    # Whitelisted invalid_parseable: tiny negative epsilon (sliding friction).
    "boundary_neg_eps": (-1e-9, 0.005, 0.0001),
}

# (timeconst, dampratio) two-arg form — accepted by mujoco.
SOLREF_MODES: Dict[str, Tuple[float, float]] = {
    "default":         (0.02, 1.0),
    "soft":            (0.5, 1.0),
    "stiff":           (0.001, 1.0),
    "near_singular":   (1e-6, 1.0),
    "extreme_damping": (0.02, 100.0),
}

# Five-element solimp = (dmin, dmax, width, midpoint, power).
SOLIMP_MODES: Dict[str, Tuple[float, float, float, float, float]] = {
    "default":   (0.9, 0.95, 0.001, 0.5, 2.0),
    "soft":      (0.5, 0.7, 0.01, 0.5, 1.0),
    "stiff":     (0.99, 0.999, 0.0001, 0.5, 4.0),
}

CONTACT_MARGIN_MODES: Dict[str, float] = {
    "zero":  0.0,
    "small": 1e-4,
    "large": 0.05,
    "huge":  1.0,
}

# --- Actuator ---
CTRL_RANGE_MODES: Dict[str, Tuple[float, float]] = {
    "tight":   (-0.1, 0.1),
    "normal":  (-1.0, 1.0),
    "wide":    (-100.0, 100.0),
    "extreme": (-1e6, 1e6),
}
ACTUATOR_KIND_MODES: Dict[str, Dict[str, Any]] = {
    "motor_unbounded": {"tag": "motor", "ctrlrange": "extreme", "extra": {}},
    "motor_tight":     {"tag": "motor", "ctrlrange": "tight",   "extra": {}},
    "position":        {"tag": "position", "ctrlrange": "normal", "extra": {"kp": "100"}},
    "velocity":        {"tag": "velocity", "ctrlrange": "normal", "extra": {"kv": "10"}},
}

# --- Equality / tendon ---
EQUALITY_MODES = ("connect_close", "connect_far", "weld", "over_constrain_pair")
TENDON_MODES = ("fixed_short", "fixed_long", "spatial_loop")

# --- Solver / option ---
TIMESTEP_MODES: Dict[str, float] = {
    "tiny":    1e-5,
    "small":   1e-4,
    "default": 0.002,
    "large":   0.05,
    "huge":    0.5,
}
INTEGRATOR_MODES = ("Euler", "RK4", "implicit", "implicitfast")
SOLVER_ITER_MODES: Dict[str, int] = {
    "min_1":    1,
    "low_5":    5,
    "default":  100,
    "high_500": 500,
}
SOLVER_NAME_MODES = ("PGS", "CG", "Newton")

# --- Structural growth / duplication ---
STRUCT_GROW_MODES = ("simple_pendulum", "compound_arm", "cluster_balls")
STRUCT_SHRINK_MODES = ("leaf_only", "random_subtree")
STRUCT_DUP_MODES = ("single", "chain_x3", "chain_x10")

# --- Runtime state perturbations ---
QPOS_MODES = ("zero", "small", "large", "near_limit", "nan_inf")
QVEL_MODES = ("zero", "small", "large", "nan_inf")
CTRL_MODES = ("zero", "mid", "clip_max", "over_clip", "nan_inf")


# Assemble the canonical per-mutator intensity list. Source of truth.
INTENSITY_TABLE: Dict[str, List[str]] = {
    "STRUCT_GROW_LINK":         list(STRUCT_GROW_MODES),
    "STRUCT_SHRINK":            list(STRUCT_SHRINK_MODES),
    "STRUCT_DUPLICATE_SUBTREE": list(STRUCT_DUP_MODES),
    "MUTATE_GEOM_SHAPE":        list(GEOM_SHAPES),
    "MUTATE_GEOM_SIZE":         list(GEOM_SIZE_MODES.keys()),
    "MUTATE_INERTIAL":          list(INERTIAL_MODES.keys()),
    "MUTATE_JOINT_TYPE":        list(JOINT_TYPES),
    "MUTATE_JOINT_LIMIT":       list(JOINT_LIMIT_MODES.keys()),
    "MUTATE_DAMPING_FRICTION":  list(DAMPING_FRICTION_MODES.keys()),
    "MUTATE_FRICTION":          list(FRICTION_MODES.keys()),
    "MUTATE_SOLREF_SOLIMP":     list(SOLREF_MODES.keys()),
    "MUTATE_CONTACT_MARGIN":    list(CONTACT_MARGIN_MODES.keys()),
    "ACTUATOR_ADD":             list(ACTUATOR_KIND_MODES.keys()),
    "ACTUATOR_MUTATE_RANGE":    list(CTRL_RANGE_MODES.keys()),
    "ACTUATOR_DELETE":          ["_default"],
    "EQUALITY_ADD":             list(EQUALITY_MODES),
    "TENDON_ADD":               list(TENDON_MODES),
    "MUTATE_TIMESTEP":          list(TIMESTEP_MODES.keys()),
    "MUTATE_INTEGRATOR":        list(INTEGRATOR_MODES),
    "MUTATE_SOLVER_ITER":       list(SOLVER_ITER_MODES.keys()),
    "SET_QPOS_RUNTIME":         list(QPOS_MODES),
    "SET_QVEL_RUNTIME":         list(QVEL_MODES),
    "SET_CTRL_RUNTIME":         list(CTRL_MODES),
}


def fmt_floats(*xs: float) -> str:
    return " ".join(f"{x:.10g}" for x in xs)


def pick_size(rng: random.Random, mode: str, dim: int) -> List[float]:
    """Pick `dim` size scalars for a geom under the given size mode."""
    pool = GEOM_SIZE_MODES[mode]
    return [rng.choice(pool) for _ in range(dim)]
