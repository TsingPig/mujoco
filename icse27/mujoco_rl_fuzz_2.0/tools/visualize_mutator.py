"""Visualize a mutator's effect: launch viewer for the model BEFORE,
then again AFTER applying the mutator. Mutated XML is saved under .cache/viz/.

Usage:
    python tools/visualize_mutator.py --list
    python tools/visualize_mutator.py MUTATE_GEOM_SIZE                      # synthetic seed, all legal modes
    python tools/visualize_mutator.py MUTATE_GEOM_SIZE --intensity huge
    python tools/visualize_mutator.py STRUCT_GROW_LINK \\
        --seed mujoco__model_humanoid_humanoid --intensity simple_pendulum
    python tools/visualize_mutator.py MUTATE_JOINT_TYPE --no-before         # skip the before viewer

ESC or window close advances. The mutator is applied in-process via the
project's SafeModel wrapper; the resulting XML is compiled with the seed's
asset_dir as cwd so meshes/textures resolve.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mjcf.spec_loader import SafeModel  # noqa: E402
from src.mutations.registry import MUTATORS, MUTATOR_IDS  # noqa: E402

SEEDS_DIR = ROOT / "seeds" / "curated"
CACHE_DIR = ROOT / ".cache" / "viz"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Same minimal seed as tools/smoke_mutators.py.
SYNTH_SEED = """<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>
    <body name="b1" pos="0 0 0.4">
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom name="g1" type="sphere" size="0.05"/>
      <body name="b2" pos="0.1 0 0">
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
        <joint name="j2" type="hinge" axis="1 0 0"/>
        <geom name="g2" type="capsule" size="0.02 0.05"/>
      </body>
    </body>
    <body name="freebox" pos="0.5 0 0.3">
      <inertial pos="0 0 0" mass="0.3" diaginertia="0.002 0.002 0.002"/>
      <joint name="jf" type="hinge" axis="0 1 0"/>
      <geom name="gf" type="box" size="0.04 0.04 0.04"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def _load_seed(arg: str | None) -> tuple[SafeModel, str, Path | None]:
    """Returns (SafeModel, label, asset_dir or None)."""
    if arg is None:
        sm = SafeModel.from_string(SYNTH_SEED)
        return sm, "synthetic", None
    p = Path(arg)
    if not p.is_file():
        cand = SEEDS_DIR / arg / "model.xml"
        if cand.is_file():
            p = cand
        else:
            raise SystemExit(f"seed not found: {arg!r}")
    sm = SafeModel.load(str(p))
    return sm, p.parent.name, p.parent


def _view(label: str, xml_path: Path, asset_dir: Path | None) -> None:
    import mujoco
    import mujoco.viewer
    print(f"\n=== {label} ===  ({xml_path})")
    cwd = os.getcwd()
    try:
        if asset_dir is not None:
            os.chdir(asset_dir)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    finally:
        os.chdir(cwd)
    data = mujoco.MjData(model)
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print("  close the window (or ESC) to continue ...")
    mujoco.viewer.launch(model, data)


def _apply_and_dump(seed_arg: str | None, mut_id: str, mode: str,
                    rng_seed: int) -> tuple[Path, Path | None, dict]:
    """Apply mutator on a fresh load of the seed; dump mutated XML.

    Returns (mutated_xml_path, asset_dir, diag_dict).
    """
    sm, label, asset_dir = _load_seed(seed_arg)
    mut = MUTATORS[mut_id]
    if not mut.applicable(sm):
        raise SystemExit(f"mutator {mut_id} not applicable to seed {label!r}")
    rng = random.Random(rng_seed)
    ar = mut.apply(sm, mode, rng)
    if not ar.ok:
        raise SystemExit(f"{mut_id}|{mode} apply failed: {ar.reason}")
    if mut.runtime_only:
        print(f"  (runtime-only mutator) directive = {ar.runtime_directive}")
    co = sm.compile()
    if not co.ok:
        # Still dump for debugging.
        out = CACHE_DIR / f"{mut_id}__{label}__{mode}__BROKEN.xml"
        out.write_text(sm.serialize(), encoding="utf-8")
        raise SystemExit(f"{mut_id}|{mode} compiled FAIL: {co.error_msg}\n  dumped: {out}")
    out = CACHE_DIR / f"{mut_id}__{label}__{mode}.xml"
    out.write_text(sm.serialize(), encoding="utf-8")
    # Mirror the seed's asset dir contents next to the dump so viewer can resolve meshes.
    if asset_dir is not None:
        for sub in ("assets", "meshes", "textures"):
            src = asset_dir / sub
            dst = CACHE_DIR / sub
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
    return out, (CACHE_DIR if asset_dir is not None else None), {
        "label": label, "params": ar.params_used,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mutator", nargs="?", help="mutator id, e.g. MUTATE_GEOM_SIZE")
    ap.add_argument("--list", action="store_true", help="list mutator ids and exit")
    ap.add_argument("--seed", default=None,
                    help="seed name (folder under seeds/curated) or path/to/model.xml; "
                         "default = synthetic")
    ap.add_argument("--intensity", default=None,
                    help="single intensity mode; default = iterate all legal modes")
    ap.add_argument("--no-before", action="store_true", help="skip the BEFORE viewer")
    ap.add_argument("--rng-seed", type=int, default=0xCAFE)
    args = ap.parse_args()

    if args.list or not args.mutator:
        for mid in MUTATOR_IDS:
            m = MUTATORS[mid]
            tag = " (runtime-only)" if m.runtime_only else ""
            print(f"{mid}{tag}: {list(m.intensity_modes)}")
        return 0

    if args.mutator not in MUTATORS:
        raise SystemExit(f"unknown mutator: {args.mutator!r}")
    mut = MUTATORS[args.mutator]

    if args.intensity:
        modes = [args.intensity]
    else:
        modes = [m for m in mut.intensity_modes
                 if m not in mut.invalid_parseable_modes]

    # Show BEFORE once.
    sm, label, asset_dir = _load_seed(args.seed)
    before_xml = CACHE_DIR / f"BEFORE__{label}.xml"
    before_xml.write_text(sm.serialize(), encoding="utf-8")
    if asset_dir is not None:
        for sub in ("assets", "meshes", "textures"):
            src = asset_dir / sub
            dst = CACHE_DIR / sub
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
    if not args.no_before:
        _view(f"BEFORE  seed={label}",
              before_xml, CACHE_DIR if asset_dir is not None else None)

    # AFTER per intensity mode.
    for i, mode in enumerate(modes, 1):
        print(f"\n[{i}/{len(modes)}] applying {args.mutator}|{mode} ...")
        out, ad, diag = _apply_and_dump(args.seed, args.mutator, mode, args.rng_seed + i)
        print(f"  params_used = {diag['params']}")
        _view(f"AFTER  {args.mutator}|{mode}  seed={label}", out, ad)

    print("\ndone. dumped XMLs are in", CACHE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
