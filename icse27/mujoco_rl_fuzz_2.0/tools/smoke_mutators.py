"""Standalone smoke validator (no pytest dependency).

Mirrors tests/test_intensity_table.py + tests/test_each_mutator_validity.py.
Runs every mutator * every legal intensity mode against a synthetic seed.

Usage:
    python tools/smoke_mutators.py
Exit code 0 ok, 1 if any failures.
"""
from __future__ import annotations

import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mjcf.spec_loader import SafeModel  # noqa: E402
from src.mutations.intensity import INTENSITY_TABLE  # noqa: E402
from src.mutations.registry import MUTATOR_IDS, all_mutators  # noqa: E402

SEED_XML = """<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 0.2">
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


def check_intensity_table() -> list[str]:
    errs: list[str] = []
    for mid in MUTATOR_IDS:
        if mid not in INTENSITY_TABLE:
            errs.append(f"INTENSITY_TABLE missing {mid}")
    for m in all_mutators():
        if not m.intensity_modes:
            errs.append(f"{m.id}: no intensity_modes")
        for mode in m.invalid_parseable_modes:
            if mode not in m.intensity_modes:
                errs.append(f"{m.id}: invalid mode {mode!r} not in modes")
        if list(m.intensity_modes) != list(INTENSITY_TABLE.get(m.id, [])):
            errs.append(
                f"{m.id}: registry modes {list(m.intensity_modes)} != "
                f"table {list(INTENSITY_TABLE.get(m.id, []))}"
            )
    return errs


def check_mutators() -> tuple[int, int, list[str]]:
    rng = random.Random(0xCAFE)
    n_ok = 0
    n_fail = 0
    errs: list[str] = []
    for mut in all_mutators():
        legal_ok = 0
        for mode in mut.intensity_modes:
            if mode in mut.invalid_parseable_modes:
                continue
            try:
                sm = SafeModel.from_string(SEED_XML)
                snap = sm.snapshot()
                if not mut.applicable(sm):
                    continue
                ar = mut.apply(sm, mode, rng)
                if mut.runtime_only:
                    if not ar.ok or ar.runtime_directive is None:
                        n_fail += 1
                        errs.append(
                            f"FAIL {mut.id}|{mode} runtime: ok={ar.ok} reason={ar.reason}"
                        )
                        continue
                else:
                    if not ar.ok:
                        sm.restore(snap)
                        n_fail += 1
                        errs.append(f"FAIL {mut.id}|{mode} apply: {ar.reason}")
                        continue
                    co = sm.compile()
                    if not co.ok:
                        n_fail += 1
                        errs.append(
                            f"FAIL {mut.id}|{mode} compile: {co.error_msg}"
                        )
                        continue
                n_ok += 1
                legal_ok += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                errs.append(
                    f"EXC  {mut.id}|{mode}: {type(e).__name__}: {e}\n"
                    + traceback.format_exc(limit=3)
                )
        if legal_ok == 0:
            errs.append(f"NOTE {mut.id}: zero legal modes succeeded")
    return n_ok, n_fail, errs


def main() -> int:
    print("=" * 60)
    print("[1] IntensityTable invariants")
    table_errs = check_intensity_table()
    if table_errs:
        for e in table_errs:
            print("  ", e)
        print(f"  {len(table_errs)} ERRORS")
    else:
        print("  OK")

    print("=" * 60)
    print("[2] Mutator legal-mode validity (synthetic seed)")
    n_ok, n_fail, errs = check_mutators()
    for e in errs:
        print("  ", e)
    print(f"  {n_ok} OK, {n_fail} FAIL across {len(MUTATOR_IDS)} mutators")
    print("=" * 60)
    return 0 if not table_errs and n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
