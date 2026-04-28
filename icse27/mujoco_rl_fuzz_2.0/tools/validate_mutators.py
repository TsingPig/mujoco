"""CI gate: (seed x mutator x intensity) cartesian validity check.

For every (seed, mutator, intensity_mode) where mutator.applicable(seed) and
intensity_mode is NOT in mutator.invalid_parseable_modes:
    - apply, then run safe_compile()
    - require ok=True (or applicability returned False up front)

Whitelisted invalid_parseable modes are run separately and only checked to
NOT crash the python interpreter. mujoco's compile error is allowed and
expected for them.
"""
from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path
from typing import List

# Ensure project root is importable when run as script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mjcf.spec_loader import SafeModel
from src.mutations.registry import all_mutators


def gather_seeds(curated_dir: Path) -> List[Path]:
    return sorted(curated_dir.glob("**/model.xml"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="seeds/curated")
    ap.add_argument("--seeds-glob", default="",
                    help="optional explicit list (comma-sep) of seed XML paths")
    ap.add_argument("--reps", type=int, default=2,
                    help="rng repeats per (seed, mut, intensity) cell")
    ap.add_argument("--include-invalid", action="store_true",
                    help="also probe whitelisted invalid_parseable modes "
                         "(only checks no python crash, compile may fail)")
    ap.add_argument("--seed-rng", type=int, default=0)
    args = ap.parse_args()

    if args.seeds_glob:
        seeds = [Path(p) for p in args.seeds_glob.split(",") if p]
    else:
        seeds = gather_seeds(Path(args.curated))
    if not seeds:
        print(f"no seeds; populate {args.curated} first via fetch_seeds.py",
              file=sys.stderr)
        return 1

    muts = all_mutators()
    print(f"seeds={len(seeds)}  mutators={len(muts)}  reps={args.reps}")

    fails: List[dict] = []
    invalid_unexpected_compile_pass: List[dict] = []
    rng = random.Random(args.seed_rng)
    n_total = 0
    n_skipped_inapplicable = 0

    for seed_path in seeds:
        try:
            sm = SafeModel.load(str(seed_path))
            base_compile = sm.compile()
        except Exception as exc:  # noqa: BLE001
            fails.append({"seed": str(seed_path),
                          "stage": "load",
                          "err": f"{type(exc).__name__}: {exc}"})
            continue
        if not base_compile.ok:
            fails.append({"seed": str(seed_path),
                          "stage": "base_compile",
                          "err": base_compile.error_msg})
            continue

        for mut in muts:
            for mode in mut.intensity_modes:
                is_invalid_mode = mode in mut.invalid_parseable_modes
                if is_invalid_mode and not args.include_invalid:
                    continue
                for rep in range(args.reps):
                    n_total += 1
                    snap = sm.snapshot()
                    rrng = random.Random(rng.randint(0, 2**31 - 1))
                    if not mut.applicable(sm):
                        n_skipped_inapplicable += 1
                        sm.restore(snap)
                        continue
                    try:
                        ar = mut.apply(sm, mode, rrng)
                    except Exception as exc:  # noqa: BLE001
                        fails.append({"seed": str(seed_path), "mut": mut.id,
                                      "mode": mode, "rep": rep,
                                      "stage": "apply_exception",
                                      "err": f"{type(exc).__name__}: {exc}",
                                      "tb": traceback.format_exc()[:500]})
                        sm.restore(snap)
                        continue
                    if mut.runtime_only:
                        # No XML mutation; expect a runtime_directive.
                        if ar.runtime_directive is None:
                            fails.append({"seed": str(seed_path), "mut": mut.id,
                                          "mode": mode, "stage": "runtime_directive_missing"})
                        sm.restore(snap)
                        continue
                    if not ar.ok:
                        sm.restore(snap)
                        continue
                    co = sm.compile()
                    if is_invalid_mode:
                        if co.ok:
                            invalid_unexpected_compile_pass.append({
                                "seed": str(seed_path), "mut": mut.id, "mode": mode})
                    else:
                        if not co.ok:
                            fails.append({"seed": str(seed_path), "mut": mut.id,
                                          "mode": mode, "rep": rep,
                                          "stage": "post_compile_fail",
                                          "err": (co.error_msg or "")[:200]})
                    sm.restore(snap)

    print(f"\nTotal cells run     : {n_total}")
    print(f"Inapplicable skipped: {n_skipped_inapplicable}")
    print(f"Failures            : {len(fails)}")
    print(f"invalid-mode passes : {len(invalid_unexpected_compile_pass)} (informational)")
    for f in fails[:20]:
        print(f"  FAIL [{f.get('mut','-')}|{f.get('mode','-')}|{f.get('stage')}]"
              f" {f.get('seed','-')[-60:]}\n    {f.get('err','')[:160]}")
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
