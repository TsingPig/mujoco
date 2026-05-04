"""tools/run_random_fuzz.py — random baseline.

For a budget of N episodes:
1. Pick a seed at random across L0/L1/L3 (L2 only if --include-envs).
2. Pick a random trajectory protocol (random_control, sinusoidal, ...).
3. Run via src.runner.run_seed and dump anomalous findings.

Usage:
    python tools/run_random_fuzz.py --budget 50
    python tools/run_random_fuzz.py --budget 20 --include-envs
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def _load_corpus(include_envs: bool):
    seeds = []
    parents = {}
    # actors from curated dirs
    curated = REPO / "seeds/curated"
    if curated.exists():
        for sub in sorted(curated.iterdir()):
            if sub.is_dir() and (sub / "model.xml").is_file():
                from src.corpus import ActorSeed
                s = ActorSeed(seed_id=sub.name, source_repo=None, source_path=None,
                              local_path=str(sub.relative_to(REPO)).replace("\\", "/"),
                              asset_dir=str(sub.relative_to(REPO)).replace("\\", "/"),
                              model_xml=str((sub / "model.xml").relative_to(REPO)).replace("\\", "/"),
                              tags=[], compile_status="unknown")
                seeds.append(s)
                parents[s.seed_id] = s
    scene_m = REPO / "seeds/synthetic_scenes/manifest.jsonl"
    if scene_m.exists():
        for s in iter_manifest(str(scene_m)):
            if isinstance(s, SyntheticSceneSeed) and s.compile_status == "ok":
                seeds.append(s)
                parents[s.seed_id] = s
    env_m = REPO / "seeds/open_envs/manifest.jsonl"
    if include_envs and env_m.exists():
        for s in iter_manifest(str(env_m)):
            if isinstance(s, OpenEnvSeed) and s.runnable_status == "runnable":
                seeds.append(s)
                parents[s.seed_id] = s
    traj_m = REPO / "seeds/trajectory_seeds/manifest.jsonl"
    if traj_m.exists():
        for s in iter_manifest(str(traj_m)):
            if isinstance(s, TrajectorySeed) and s.replay_status == "ok":
                seeds.append(s)
    return seeds, parents


def _random_protocol(rng: random.Random, horizon: int) -> TrajectoryProtocol:
    kind = rng.choice(["zero_control", "random_control", "sinusoidal_control",
                       "single_actuator_sweep", "push_like"])
    return TrajectoryProtocol(
        action_sequence=ActionSequence(
            kind=kind, horizon=horizon,
            amplitude=rng.uniform(0.1, 0.8),
            frequency=rng.uniform(0.5, 3.0),
            actuator_index=rng.randrange(0, 4),
        ),
        random_seed=rng.randrange(1_000_000),
    )


def _signature(seed_id: str, oracle_names) -> str:
    h = hashlib.sha1()
    h.update(seed_id.encode("utf-8"))
    for n in sorted(oracle_names):
        h.update(b"|"); h.update(n.encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--include-envs", action="store_true")
    p.add_argument("--out-dir", default="findings/random")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    seeds, parents = _load_corpus(args.include_envs)
    if not seeds:
        print("[run_random_fuzz] no seeds available; build corpus first")
        return 1
    out_dir = (REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    n_findings = 0
    n_runs = 0
    t0 = time.time()
    for i in range(args.budget):
        s = seeds[rng.randrange(len(seeds))]
        proto = _random_protocol(rng, args.horizon)
        req = RunRequest(seed=s, protocol=proto)
        res = run_seed(req, workspace_root=str(REPO),
                       parent_lookup=lambda sid, _p=parents: _p.get(sid))
        n_runs += 1
        failed_reports = [r for r in res.oracle_reports if r.failed]
        if (not res.ok) or failed_reports:
            sig = _signature(res.seed_id, [r.name for r in failed_reports] or ["replay_failed"])
            f = Finding(
                finding_id=f"rand_{i:05d}_{sig}",
                seed_id=res.seed_id,
                layer=res.layer,
                signature=sig,
                oracle_signals=[{"name": r.name, "severity": r.severity,
                                 "failed": r.failed, "notes": r.notes} for r in res.oracle_reports],
                trace_summary=(res.trace.to_dict() if res.trace else {}),
                action_history=[],
                protocol=proto.to_dict(),
                notes={"replay_error": res.error},
            )
            f.dump(str(out_dir))
            n_findings += 1
            print(f"[FINDING] {f.finding_id} seed={res.seed_id} sev={res.severity:.2f}")
    print(f"[run_random_fuzz] runs={n_runs} findings={n_findings} elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
