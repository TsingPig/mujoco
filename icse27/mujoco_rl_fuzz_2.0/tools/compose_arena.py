"""Compose a multi-instance arena from a single seed.

Wraps the seed's worldbody bodies inside MuJoCo 3.x `<replicate>` so the same
robot/object subtree is stamped N times at given offsets. The resulting XML is
saved as a *new* curated seed: `seeds/curated/composed_<base>__x<N>/model.xml`.

Why: most upstream toy seeds are single-body (1 humanoid, 1 car). Replicating
them 8/16/32× into a grid creates dense contact + many-DoF scenes that exercise
mutators much harder (collision graph blowup, integrator stress, solver
divergence). Powered by MuJoCo's native `<replicate>` element (no name-mangling
needed; mujoco appends `_i` automatically).

Usage:
    python tools/compose_arena.py --list
    python tools/compose_arena.py mujoco__model_humanoid_humanoid --count 16 --offset 1.5 1.5 0
    python tools/compose_arena.py mujoco__model_car_car --count 25 --grid 5 5 --spacing 1.0
    python tools/compose_arena.py mujoco__model_balloons_balloons --count 8 --rotate-z 45 --keep-actuators

Notes:
- Default DROPS `<actuator>` / `<sensor>` / `<tendon>` / `<equality>` because
  references to per-body joint names become ambiguous after replication. Pass
  `--keep-actuators` to also wrap those sections in matching `<replicate>`s
  (best-effort; may fail to compile for some seeds).
- Floor + lights are added if missing.
- The composed seed is compiled before being written; failures are dumped to
  `.cache/viz/composed_<base>__x<N>__BROKEN.xml`.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = ROOT / "seeds" / "curated"
CACHE_DIR = ROOT / ".cache" / "viz"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    cand = SEEDS_DIR / arg / "model.xml"
    if cand.is_file():
        return cand
    raise SystemExit(f"seed not found: {arg!r}")


def _ensure_floor(worldbody: etree._Element) -> None:
    has_plane = any(g.tag == "geom" and g.get("type") == "plane"
                    for g in worldbody.iter("geom"))
    if not has_plane:
        floor = etree.SubElement(worldbody, "geom", attrib={
            "name": "_arena_floor", "type": "plane",
            "size": "20 20 0.1", "rgba": "0.8 0.85 0.9 1",
        })
        floor.tail = "\n  "
    has_light = any(True for _ in worldbody.iter("light"))
    if not has_light:
        light = etree.SubElement(worldbody, "light", attrib={
            "name": "_arena_light", "pos": "0 0 8", "dir": "0 0 -1",
            "directional": "true",
        })
        light.tail = "\n  "


def compose(seed_xml: Path, count: int, offset: Tuple[float, float, float],
            euler_z: float, keep_actuators: bool,
            grid: Tuple[int, int] | None, spacing: float) -> str:
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(seed_xml), parser)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise SystemExit(f"{seed_xml}: not a <mujoco> root")

    # Insert a model name marker.
    root.set("model", f"arena_{seed_xml.parent.name}_x{count}")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise SystemExit(f"{seed_xml}: no <worldbody>")

    # Strip <keyframe> (qpos length will not match after replication).
    for kf in root.findall("keyframe"):
        root.remove(kf)

    # Optionally drop downstream sections that reference per-body names.
    if not keep_actuators:
        for tag in ("actuator", "sensor", "tendon", "equality", "contact"):
            for el in root.findall(tag):
                root.remove(el)

    # Lift body subtrees out of worldbody; keep lights/world geoms in place.
    bodies: List[etree._Element] = [c for c in list(worldbody)
                                    if c.tag == "body"]
    if not bodies:
        raise SystemExit(f"{seed_xml}: worldbody has no <body> to replicate")
    body_names = {b.get("name") for b in bodies if b.get("name")}
    for b in bodies:
        worldbody.remove(b)

    # Strip target=/objname= references on remaining world-level elements that
    # point at bodies we just replicated (they'd now be ambiguous _i0/_i1/...).
    for el in list(worldbody.iter()):
        for ref_attr in ("target", "objname", "body", "body1", "body2"):
            ref = el.get(ref_attr)
            if ref and ref in body_names:
                el.attrib.pop(ref_attr, None)

    _ensure_floor(worldbody)

    # Build replicate(s). Two modes: linear (`--count` + `--offset`) or grid.
    if grid is not None:
        nx, ny = grid
        # Outer replicate over rows (Y), inner over columns (X).
        outer = etree.SubElement(worldbody, "replicate", attrib={
            "count": str(ny), "offset": f"0 {spacing} 0", "sep": "_row",
        })
        inner = etree.SubElement(outer, "replicate", attrib={
            "count": str(nx), "offset": f"{spacing} 0 0", "sep": "_col",
        })
        for b in bodies:
            inner.append(deepcopy(b))
    else:
        attrs = {"count": str(count),
                 "offset": f"{offset[0]} {offset[1]} {offset[2]}",
                 "sep": "_i"}
        if euler_z:
            attrs["euler"] = f"0 0 {euler_z}"
        rep = etree.SubElement(worldbody, "replicate", attrib=attrs)
        for b in bodies:
            rep.append(deepcopy(b))

    # Pretty-ish serialize.
    etree.indent(tree, space="  ")
    return etree.tostring(tree, xml_declaration=False, pretty_print=True,
                          encoding="unicode")


def _try_compile(xml_text: str, asset_dir: Path) -> Tuple[bool, str]:
    import mujoco
    cwd = os.getcwd()
    try:
        os.chdir(asset_dir)
        mujoco.MjModel.from_xml_string(xml_text)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        os.chdir(cwd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed", nargs="?", help="curated seed name or path/to/model.xml")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--count", type=int, default=8, help="replicate count (linear mode)")
    ap.add_argument("--offset", type=float, nargs=3, default=(1.5, 0.0, 0.0),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--rotate-z", type=float, default=0.0,
                    help="per-instance rotation about world Z (degrees)")
    ap.add_argument("--grid", type=int, nargs=2, default=None, metavar=("NX", "NY"),
                    help="2D grid mode; overrides --count/--offset")
    ap.add_argument("--spacing", type=float, default=1.5, help="grid cell spacing (m)")
    ap.add_argument("--keep-actuators", action="store_true",
                    help="keep actuator/sensor/tendon/equality (often breaks compile)")
    ap.add_argument("--out-name", default=None,
                    help="custom seed dir name; default = composed_<base>__x<N>")
    args = ap.parse_args()

    if args.list or not args.seed:
        seeds = sorted(SEEDS_DIR.glob("*/model.xml"))
        for p in seeds:
            print(p.parent.name)
        return 0

    seed_xml = _resolve(args.seed)
    base = seed_xml.parent.name
    n_total = args.grid[0] * args.grid[1] if args.grid else args.count
    out_name = args.out_name or f"composed_{base}__x{n_total}"
    out_dir = SEEDS_DIR / out_name

    print(f"composing {base} -> {out_name}")
    xml_text = compose(seed_xml,
                       count=args.count, offset=tuple(args.offset),
                       euler_z=args.rotate_z,
                       keep_actuators=args.keep_actuators,
                       grid=tuple(args.grid) if args.grid else None,
                       spacing=args.spacing)

    ok, err = _try_compile(xml_text, seed_xml.parent)
    if not ok:
        broken = CACHE_DIR / f"{out_name}__BROKEN.xml"
        broken.write_text(xml_text, encoding="utf-8")
        print(f"  COMPILE FAIL: {err}")
        print(f"  dumped: {broken}")
        if not args.keep_actuators:
            print("  hint: try a different seed; some seeds rely on assets that ")
            print("        choke when bodies are stamped many times (mesh sharing).")
        if "mj_stackAlloc" in err or "memory" in err.lower():
            print("  hint: hit MuJoCo's per-step stack budget. Add to the seed XML:")
            print('        <size memory="64M"/>  (inside the top-level <mujoco>)')
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.xml").write_text(xml_text, encoding="utf-8")
    # Mirror asset dirs (assets/ meshes/ textures/) so the new seed is self-contained.
    for sub in ("assets", "meshes", "textures"):
        src = seed_xml.parent / sub
        dst = out_dir / sub
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)
    # Sanity recompile from the new path.
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out_dir / "model.xml"))
    print(f"  OK  -> {out_dir / 'model.xml'}")
    print(f"      nq={m.nq}  nv={m.nv}  nbody={m.nbody}  ngeom={m.ngeom}  nu={m.nu}")
    print(f"  visualize: python tools/visualize_seed.py {out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
