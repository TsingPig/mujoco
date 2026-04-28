"""Visualize a single curated seed in the MuJoCo native viewer.

Usage:
    python tools/visualize_seed.py --list
    python tools/visualize_seed.py mujoco__model_humanoid_humanoid
    python tools/visualize_seed.py mujoco__model_car_car --static
    python tools/visualize_seed.py path/to/model.xml             # also accepted

ESC or window close exits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = ROOT / "seeds" / "curated"


def _list_seeds() -> list[Path]:
    return sorted(SEEDS_DIR.glob("*/model.xml"))


def _resolve(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    cand = SEEDS_DIR / arg / "model.xml"
    if cand.is_file():
        return cand
    raise SystemExit(f"seed not found: {arg!r}\n"
                     f"  try `python {sys.argv[0]} --list`")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed", nargs="?", help="seed name (folder under seeds/curated) or path/to/model.xml")
    ap.add_argument("--list", action="store_true", help="list curated seeds and exit")
    ap.add_argument("--static", action="store_true", help="open viewer without stepping the simulation")
    args = ap.parse_args()

    if args.list or not args.seed:
        seeds = _list_seeds()
        if not seeds:
            print("(no curated seeds; run `python tools/fetch_seeds.py` first)")
            return 0
        for p in seeds:
            print(p.parent.name)
        return 0

    xml_path = _resolve(args.seed)
    print(f"loading: {xml_path}")

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")

    if args.static:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("  static viewer; close window or ESC to exit")
            while viewer.is_running():
                viewer.sync()
    else:
        # Blocking viewer: handles its own loop and stepping.
        mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
