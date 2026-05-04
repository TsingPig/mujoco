"""tools/run_rule_fuzz.py — rule-based baseline.

Heuristic policies based on seed metadata:
  - articulated robots → "single_actuator_sweep" + "grasp_lift_like"
  - locomotion-like (legged actor or "flat" arena) → "sinusoidal_control"
  - cluttered scenes (>= 2 props) → "push_like" + "random_control"
  - others → "random_control"

Usage:
    python tools/run_rule_fuzz.py --budget 30
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.corpus import iter_manifest, ActorSeed, SyntheticSceneSeed, OpenEnvSeed, TrajectorySeed
from src.runner import RunRequest, run_seed
from src.trajectory.protocol import TrajectoryProtocol, ActionSequence
from src.triage.finding import Finding


def _select_kinds(seed) -> list:
    if isinstance(seed, ActorSeed):
        sid = seed.seed_id.lower()
        if any(k in sid for k in ("ant", "humanoid", "cheetah", "walker", "dog", "quadruped")):
            return ["sinusoidal_control", "single_actuator_sweep"]
        if any(k in sid for k in ("arm", "panda", "iiwa", "ur5", "franka", "robot")):
            return ["single_actuator_sweep", "grasp_lift_like"]
        return ["random_control"]
    if isinstance(seed, SyntheticSceneSeed):
        nprops = len(seed.placement_config.get("props", []) if seed.placement_config else [])
        tmpl = (seed.template_name or "").lower()
        if "clutter" in tmpl or nprops >= 2:
            return ["push_like", "random_control"]
        if "ramp" in tmpl or "obstacle" in tmpl:
            return ["sinusoidal_control"]
        return ["random_control"]
    if isinstance(seed, OpenEnvSeed):
        return ["random_control"]
    if isinstance(seed, TrajectorySeed):
        return [seed.action_sequence_spec.get("kind", "random_control")]
    return ["random_control"]


def _load_corpus(include_envs: bool):
    seeds, parents = [], {}
    curated = REPO / "seeds/curated"
    if curated.exists():
        for sub in sorted(curated.iterdir()):
            if sub.is_dir() and (sub / "model.xml").is_file():
                s = ActorSeed(seed_id=sub.name, source_repo=None, source_path=None,
                              local_path=str(sub.relative_to(REPO)).replace("\\", "/"),
                              asset_dir=str(sub.relative_to(REPO)).replace("\\", "/"),
                              model_xml=str((sub / "model.xml").relative_to(REPO)).replace("\\", "/"),
                              tags=[], compile_status="unknown")
                seeds.append(s); parents[s.seed_id] = s
    sm = REPO / "seeds/synthetic_scenes/manifest.jsonl"
    if sm.exists():
        for s in iter_manifest(str(sm)):
            if isinstance(s, SyntheticSceneSeed) and s.compile_status == "ok":
                seeds.append(s); parents[s.seed_id] = s
    em = REPO / "seeds/open_envs/manifest.jsonl"
    if include_envs and em.exists():
        for s in iter_manifest(str(em)):
            if isinstance(s, OpenEnvSeed) and s.runnable_status == "runnable":
                seeds.append(s); parents[s.seed_id] = s
    tm = REPO / "seeds/trajectory_seeds/manifest.jsonl"
    if tm.exists():
        for s in iter_manifest(str(tm)):
            if isinstance(s, TrajectorySeed) and s.replay_status == "ok":
                seeds.append(s)
    return seeds, parents


def _signature(seed_id, names):
    h = hashlib.sha1(seed_id.encode("utf-8"))
    for n in sorted(names):
        h.update(b"|"); h.update(n.encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--include-envs", action="store_true")
    p.add_argument("--out-dir", default="findings/rule")
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()

    seeds, parents = _load_corpus(args.include_envs)
    if not seeds:
        print("[run_rule_fuzz] no seeds available; build corpus first")
        return 1
    out_dir = (REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n_findings = 0
    t0 = time.time()
    for i in range(args.budget):
        s = seeds[rng.randrange(len(seeds))]
        kinds = _select_kinds(s)
        kind = kinds[rng.randrange(len(kinds))]
        proto = TrajectoryProtocol(
            action_sequence=ActionSequence(kind=kind, horizon=args.horizon,
                                           amplitude=rng.uniform(0.3, 0.9),
                                           frequency=rng.uniform(0.5, 2.0)),
            random_seed=rng.randrange(1_000_000),
        )
        res = run_seed(RunRequest(seed=s, protocol=proto),
                       workspace_root=str(REPO),
                       parent_lookup=lambda sid, _p=parents: _p.get(sid))
        failed = [r for r in res.oracle_reports if r.failed]
        if (not res.ok) or failed:
            sig = _signature(res.seed_id, [r.name for r in failed] or ["replay_failed"])
            f = Finding(
                finding_id=f"rule_{i:05d}_{sig}",
                seed_id=res.seed_id, layer=res.layer, signature=sig,
                oracle_signals=[{"name": r.name, "severity": r.severity,
                                 "failed": r.failed, "notes": r.notes}
                                for r in res.oracle_reports],
                trace_summary=(res.trace.to_dict() if res.trace else {}),
                protocol=proto.to_dict(),
                notes={"replay_error": res.error, "rule_kind": kind},
            )
            f.dump(str(out_dir))
            n_findings += 1
            print(f"[FINDING] {f.finding_id} seed={res.seed_id} kind={kind}")
    print(f"[run_rule_fuzz] budget={args.budget} findings={n_findings} elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
