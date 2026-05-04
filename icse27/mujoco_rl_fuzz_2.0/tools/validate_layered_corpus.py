"""tools/validate_layered_corpus.py

Per-layer summary: count, compile/runnable ratios, dependency states.

Usage:
    python tools/validate_layered_corpus.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.corpus import iter_manifest, ActorSeed, SyntheticSceneSeed, OpenEnvSeed, TrajectorySeed


SCENE_MANIFEST = REPO / "seeds/synthetic_scenes/manifest.jsonl"
ENV_MANIFEST = REPO / "seeds/open_envs/manifest.jsonl"
TRAJ_MANIFEST = REPO / "seeds/trajectory_seeds/manifest.jsonl"


def _summarize(name, items, status_attr):
    n = len(items)
    counts = Counter(getattr(s, status_attr, "?") or "?" for s in items)
    print(f"[{name}] count={n} {status_attr}={dict(counts)}")


def _count_actors() -> int:
    curated = REPO / "seeds/curated"
    if not curated.exists():
        return 0
    n = 0
    for sub in curated.iterdir():
        if sub.is_dir() and (sub / "model.xml").is_file():
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exit-on-empty", action="store_true",
                   help="non-zero exit if any manifest is missing")
    args = p.parse_args()

    actors_n = _count_actors()
    print(f"[L0 actors] count={actors_n}")

    scenes = list(iter_manifest(str(SCENE_MANIFEST))) if SCENE_MANIFEST.exists() else []
    _summarize("L1 synthetic_scenes", scenes, "compile_status")

    envs = list(iter_manifest(str(ENV_MANIFEST))) if ENV_MANIFEST.exists() else []
    _summarize("L2 open_envs (dep)", envs, "dependency_status")
    _summarize("L2 open_envs (run)", envs, "runnable_status")

    trajs = list(iter_manifest(str(TRAJ_MANIFEST))) if TRAJ_MANIFEST.exists() else []
    _summarize("L3 trajectories", trajs, "replay_status")

    if args.exit_on_empty:
        if not scenes or not envs or not trajs:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
