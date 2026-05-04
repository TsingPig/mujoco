"""Placement sampling for synthetic scenes.

A placement is a (pos, quat) pair sampled from a small declarative spec.
Kept minimal on purpose; richer sampling can be added later.
"""
from __future__ import annotations

import random
from typing import Dict, Any, List, Tuple


def sample_placement(spec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Sample a placement from a declarative spec.

    Recognised keys:
        pos: [x,y,z] fixed position; or "uniform_xy" + bounds
        quat: fixed quaternion (default identity)
        x_range, y_range, z: bounds for uniform_xy
    """
    out: Dict[str, Any] = {}
    if "pos" in spec and isinstance(spec["pos"], (list, tuple)):
        out["pos"] = [float(v) for v in spec["pos"]]
    elif spec.get("pos") == "uniform_xy":
        x_lo, x_hi = spec.get("x_range", (-0.2, 0.2))
        y_lo, y_hi = spec.get("y_range", (-0.2, 0.2))
        z = float(spec.get("z", 0.05))
        out["pos"] = [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z]
    else:
        out["pos"] = [0.0, 0.0, 0.05]
    out["quat"] = list(spec.get("quat", [1.0, 0.0, 0.0, 0.0]))
    return out
