"""Compose an L1 SyntheticSceneSeed from an actor MJCF and a template.

Strategy
--------
We build a small wrapper MJCF that:
  1. <include file="..."/> the actor model.xml (so we don't need to know
     the actor's internal structure; mesh paths, defaults, etc. are
     resolved through the actor's directory by way of meshdir/texturedir).
  2. Adds a <worldbody> with arena + prop bodies derived from the template.
  3. Optionally moves the included actor by wrapping it in a body offset.

This is intentionally conservative: we never edit the actor file. If the
include or compile fails, the result is marked failed and the caller
should route the seed to quarantine.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lxml import etree

from ..mjcf.spec_loader import safe_compile
from .placements import sample_placement
from .templates import get_template


@dataclass
class ComposeRequest:
    actor_xml_path: str            # absolute path to actor model.xml
    template_name: str
    out_dir: str                   # output directory for scene_xml + provenance
    actor_seed_id: Optional[str] = None
    seed: int = 0
    extra_tags: List[str] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComposeResult:
    ok: bool
    scene_xml_path: Optional[str] = None
    error: Optional[str] = None
    placement_config: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    model_features: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


def _make_arena(arena: Optional[Dict[str, Any]]) -> Optional[etree._Element]:
    if not arena:
        return None
    geom = etree.Element("geom")
    geom.set("name", "arena_floor")
    geom.set("type", str(arena.get("type", "plane")))
    sz = arena.get("size", [2.0, 2.0, 0.1])
    geom.set("size", " ".join(f"{v}" for v in sz))
    if "rgba" in arena:
        geom.set("rgba", " ".join(f"{v}" for v in arena["rgba"]))
    if "friction" in arena:
        geom.set("friction", " ".join(f"{v}" for v in arena["friction"]))
    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    return geom


def _make_prop(prop: Dict[str, Any]) -> etree._Element:
    body = etree.Element("body")
    body.set("name", f"prop_{prop['name']}")
    pos = prop.get("pos", [0, 0, 0])
    body.set("pos", " ".join(f"{v}" for v in pos))
    if "euler" in prop:
        body.set("euler", " ".join(f"{v}" for v in prop["euler"]))
    if not prop.get("static", False) and prop.get("freejoint", False):
        etree.SubElement(body, "freejoint")
    geom = etree.SubElement(body, "geom")
    geom.set("name", f"propgeom_{prop['name']}")
    geom.set("type", str(prop.get("type", "box")))
    sz = prop.get("size", [0.05, 0.05, 0.05])
    geom.set("size", " ".join(f"{v}" for v in sz))
    if "rgba" in prop:
        geom.set("rgba", " ".join(f"{v}" for v in prop["rgba"]))
    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    if not prop.get("static", False) and "mass" in prop:
        # Light inertial fallback so freejoint bodies are well-defined.
        inertial = etree.SubElement(body, "inertial")
        inertial.set("pos", "0 0 0")
        inertial.set("mass", str(prop["mass"]))
        # diagonal inertia matrix; rough sphere-equivalent
        m = float(prop["mass"])
        s = max(sz) if sz else 0.05
        I = max(2.0 / 5.0 * m * s * s, 1e-6)
        inertial.set("diaginertia", f"{I} {I} {I}")
    return body


def compose_scene(req: ComposeRequest) -> ComposeResult:
    actor_path = os.path.abspath(req.actor_xml_path)
    if not os.path.isfile(actor_path):
        return ComposeResult(ok=False, error=f"actor xml not found: {actor_path}")

    try:
        template = get_template(req.template_name)
    except KeyError as exc:
        return ComposeResult(ok=False, error=str(exc))

    rng = random.Random(req.seed)

    # Build wrapper MJCF.
    root = etree.Element("mujoco")
    root.set("model", f"scene_{req.template_name}_{req.seed}")
    # Compiler defaults; actor include may override its own.
    compiler = etree.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    # Include the actor relative to the wrapper file.
    include = etree.SubElement(root, "include")
    actor_rel = os.path.relpath(actor_path, req.out_dir).replace("\\", "/")
    include.set("file", actor_rel)

    worldbody = etree.SubElement(root, "worldbody")
    arena_geom = _make_arena(template.get("arena"))
    if arena_geom is not None:
        worldbody.append(arena_geom)

    placement = sample_placement(template.get("actor_placement", {}), rng)

    for prop in template.get("props", []):
        worldbody.append(_make_prop(prop))

    # Serialize wrapper.
    os.makedirs(req.out_dir, exist_ok=True)
    scene_xml_path = os.path.join(req.out_dir, "scene.xml")
    tree = etree.ElementTree(root)
    tree.write(scene_xml_path, pretty_print=True, encoding="utf-8", xml_declaration=False)

    # Compile.
    with open(scene_xml_path, "r", encoding="utf-8") as f:
        xml_text = f.read()
    outcome = safe_compile(xml_text, asset_dir=req.out_dir)
    if not outcome.ok:
        return ComposeResult(
            ok=False,
            scene_xml_path=scene_xml_path,
            error=f"compile_failed: {outcome.error_type}: {outcome.error_msg}",
            placement_config=placement,
            generation_config={"template": req.template_name, "seed": req.seed},
            tags=list(template.get("tags", [])) + list(req.extra_tags),
        )

    return ComposeResult(
        ok=True,
        scene_xml_path=scene_xml_path,
        placement_config=placement,
        generation_config={"template": req.template_name, "seed": req.seed},
        model_features={
            "nq": outcome.nq, "nv": outcome.nv, "nu": outcome.nu,
            "nbody": outcome.nbody, "ngeom": outcome.ngeom,
            "njnt": outcome.njnt, "neq": outcome.neq, "ntendon": outcome.ntendon,
        },
        tags=list(template.get("tags", [])) + list(req.extra_tags),
    )
