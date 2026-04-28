"""Analyze v8 real_bug candidates: separate true positives from injected FPs."""
from __future__ import annotations
import glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    rows = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "outputs/real_bugs/*.json"))):
        d = json.load(open(fp, encoding="utf-8"))
        res = d.get("result") or {}
        sd = res.get("solver_diff") or {}
        so = sd.get("solver_only_max_diff")
        inj_nan = bool(d.get("injected_nan_inf"))
        inj_bad = bool(d.get("injected_bad"))
        action = d.get("action") or {}
        mut = action.get("mutator") if isinstance(action, dict) else None
        seed = d.get("seed_xml") or d.get("seed_path") or d.get("seed") or ""
        stats = res.get("state_stats") or {}
        warns = res.get("warnings") or []
        bad_w = [w.get("wtype") for w in warns
                 if w.get("wtype") in ("BADQPOS", "BADQVEL", "BADQACC")]
        rows.append({
            "file": os.path.basename(fp),
            "solver_only": so,
            "per_combo": sd.get("per_combo_diff_vs_ref"),
            "inj_nan": inj_nan,
            "inj_bad": inj_bad,
            "mutator": mut,
            "seed": os.path.basename(str(seed)) if seed else "",
            "has_nan_or_inf": bool(stats.get("has_nan") or stats.get("has_inf")),
            "bad_warns": bad_w,
        })

    print(f"{len(rows)} total real_bug candidates\n")

    sd_real = [r for r in rows
               if r["solver_only"] is not None
               and r["solver_only"] > 1e-3
               and not r["inj_nan"]
               and not r["inj_bad"]]
    sd_real.sort(key=lambda r: -(r["solver_only"] or 0))
    print(f"=== Solver-disagreement (clean, no injection): {len(sd_real)} ===")
    for r in sd_real:
        m = (r['mutator'] or '<none>')[:30]
        s = (r['seed'] or '<?>')[:30]
        print(f"  diff={r['solver_only']:.4e}  mutator={m:30s}  seed={s:30s}  file={r['file']}")
        per = r["per_combo"] or {}
        print(f"     per_combo={per}")

    spont = [r for r in rows
             if (r["has_nan_or_inf"] or r["bad_warns"])
             and not r["inj_nan"]
             and not r["inj_bad"]
             # Don't double-count solver_diff finds here
             and not (r["solver_only"] is not None and r["solver_only"] > 1e-3)]
    print(f"\n=== Spontaneous NaN/BAD (clean): {len(spont)} ===")
    for r in spont:
        print(f"  nan/inf={r['has_nan_or_inf']}  bad={r['bad_warns']}  mutator={r['mutator']}  seed={r['seed']}  file={r['file']}")

    inj = [r for r in rows if r["inj_nan"] or r["inj_bad"]]
    print(f"\n=== Injected (FP, ignored): {len(inj)} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
