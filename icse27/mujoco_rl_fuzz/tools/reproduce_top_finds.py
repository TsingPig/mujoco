"""Re-run the top solver-disagreement candidates with progressively larger
iteration budgets to distinguish "PGS just needs more iterations" from a
genuine solver bug.

For each candidate XML we:
  1. Load model + apply the same 50-step rollout (no perturbation: the
     persisted record's qpos perturb isn't reproducible without the param
     RNG state, so we treat the find as "this XML topology + zero ctrl"
     which is the worst case for our claim — if it still disagrees, the
     bug is in the solver convergence, not in our state setup).
  2. Re-run with each (solver, integrator) combo at iters in
     [200, 1000, 5000, 20000] and tol=1e-12.
  3. Print same-integrator solver-disagreement and how it shrinks with
     iters. If it stays > 1e-3 at iters=20000 we have a real bug.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_STEPS = 50
TOL = 1e-12
ITER_BUDGETS = [200, 1000, 5000, 20000]
COMBOS = [
    (2, 0, "Newton+Euler"),
    (1, 0, "CG+Euler"),
    (0, 0, "PGS+Euler"),
    (2, 2, "Newton+implicit"),
    (2, 3, "Newton+implicitfast"),
]


def run_combo(xml_path: str, sid: int, iid: int, iters: int) -> np.ndarray | None:
    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
    except Exception:
        return None
    m.opt.solver = sid
    m.opt.integrator = iid
    m.opt.iterations = iters
    m.opt.tolerance = TOL
    d = mujoco.MjData(m)
    try:
        for _ in range(N_STEPS):
            mujoco.mj_step(m, d)
    except Exception:
        return None
    qp = np.array(d.qpos, copy=True)
    if not np.isfinite(qp).all():
        return None
    return qp


def solver_only_diff(qps: dict[str, np.ndarray]) -> float:
    triplet = ["Newton+Euler", "CG+Euler", "PGS+Euler"]
    vs = [qps[k] for k in triplet if qps.get(k) is not None]
    if len(vs) < 2:
        return float("nan")
    m = 0.0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if vs[i].size != vs[j].size:
                return float("nan")
            d = float(np.nanmax(np.abs(vs[i] - vs[j])))
            if d > m:
                m = d
    return m


def reproduce(tc_id: str) -> tuple[str, dict[int, float]]:
    """Returns (xml_path, {iters -> solver_only_max_diff})."""
    out: dict[int, float] = {}
    xml_path = os.path.join(ROOT, "outputs", "tc", f"{tc_id}.xml")
    if not os.path.exists(xml_path):
        print(f"  [SKIP {tc_id}] xml missing")
        return xml_path, out
    print(f"  [{tc_id}]  xml={xml_path}")
    for it in ITER_BUDGETS:
        qps = {}
        for sid, iid, label in COMBOS:
            qps[label] = run_combo(xml_path, sid, iid, it)
        sod = solver_only_diff(qps)
        out[it] = sod
        ref = qps.get("Newton+implicit")
        per = {}
        if ref is not None:
            for k, v in qps.items():
                if v is None:
                    per[k] = None
                elif v.size != ref.size:
                    per[k] = "size_mismatch"
                else:
                    per[k] = float(np.nanmax(np.abs(v - ref)))
        print(f"    iters={it:6d}  solver_only_max_diff={sod:.4e}  vs_implicit={per}")
    return xml_path, out


def main() -> int:
    # Pick top-N candidates by solver_only_max_diff
    rows = []
    for fp in glob.glob(os.path.join(ROOT, "outputs/real_bugs/*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        sd = (d.get("result") or {}).get("solver_diff") or {}
        so = sd.get("solver_only_max_diff")
        if so is not None:
            rows.append((float(so), d["tc_id"]))
    rows.sort(reverse=True)
    top = rows  # all candidates
    print(f"Reproducing all {len(top)} solver-disagreement finds with iters in {ITER_BUDGETS}")
    print()
    summary = []  # (tc_id, orig, diff_at_max_iters, classification)
    for so, tc in top:
        print(f"=== Original solver_only_max_diff = {so:.4e} ===")
        _, diffs = reproduce(tc)
        d_max = diffs.get(ITER_BUDGETS[-1], float("nan"))
        if not (d_max == d_max):  # nan
            cls = "unreproducible"
        elif d_max <= 1e-3:
            cls = "convergence_artifact"
        elif d_max >= so * 0.5:
            cls = "STABLE_DISAGREEMENT"
        else:
            cls = "PARTIAL"
        summary.append((tc, so, d_max, cls))
        print(f"   -> classification: {cls}  (orig={so:.3e}, at_iters={ITER_BUDGETS[-1]}={d_max:.3e})")
        print()

    print("=" * 70)
    print(f"SUMMARY ({len(summary)} candidates):")
    counts: dict[str, int] = {}
    for tc, so, dm, cls in summary:
        counts[cls] = counts.get(cls, 0) + 1
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v}")
    print()
    print("STABLE_DISAGREEMENT cases (real-bug candidates):")
    for tc, so, dm, cls in sorted(summary, key=lambda r: -r[2]):
        if cls == "STABLE_DISAGREEMENT":
            print(f"  {tc}  orig={so:.3e}  iters={ITER_BUDGETS[-1]}_diff={dm:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
