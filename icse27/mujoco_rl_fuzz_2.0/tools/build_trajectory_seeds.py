"""tools/build_trajectory_seeds.py

Generate L3 trajectory seeds for L0 actor seeds, L1 synthetic scenes, and
L2 runnable open envs. Each trajectory seed is a *replayable protocol*,
not a serialized mjData binary.

Usage:
    python tools/build_trajectory_seeds.py
    python tools/build_trajectory_seeds.py --per-actor 2 --per-scene 1 --per-env 1

Outputs:
    seeds/trajectory_seeds/manifest.jsonl
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.corpus import (
    TrajectorySeed, ActorSeed, SyntheticSceneSeed, OpenEnvSeed,
    append_manifest, iter_manifest, make_provenance,
)
from src.trajectory.protocol import (
    TrajectoryProtocol, InitialState, ActionSequence,
    ResetProtocol, RolloutProfile, SolverProfile, BackendProfile,
    ACTION_KINDS,
)
from src.trajectory.replay import replay_actor, replay_scene, replay_env


def _proto(kind: str, horizon: int, seed: int) -> TrajectoryProtocol:
    return TrajectoryProtocol(
        initial_state=InitialState(),
        action_sequence=ActionSequence(kind=kind, horizon=horizon, amplitude=0.3, frequency=1.0),
        reset_protocol=ResetProtocol(),
        rollout_profile=RolloutProfile(),
        solver_profile=SolverProfile(),
        backend_profile=BackendProfile(backend="classic"),
        random_seed=seed,
    )


def _from_proto(parent_seed_id: str, parent_layer: str, proto: TrajectoryProtocol,
                replay_status: str, trace_summary: dict, seed_id: str) -> TrajectorySeed:
    return TrajectorySeed(
        seed_id=seed_id,
        parent_seed_id=parent_seed_id,
        parent_layer=parent_layer,
        initial_state=proto.initial_state.to_dict(),
        action_sequence_spec=proto.action_sequence.to_dict(),
        reset_protocol=proto.reset_protocol.to_dict(),
        rollout_profile=proto.rollout_profile.to_dict(),
        solver_profile=proto.solver_profile.to_dict(),
        backend_profile=proto.backend_profile.to_dict(),
        random_seed=proto.random_seed,
        trace_summary=trace_summary,
        replay_status=replay_status,
        tags=[proto.action_sequence.kind, parent_layer],
        provenance=make_provenance(source="generated",
                                   notes=f"trajectory for {parent_layer}:{parent_seed_id}"),
    )


def _list_actors(curated_dir: Path):
    for sub in sorted(curated_dir.iterdir()):
        if sub.is_dir() and (sub / "model.xml").is_file():
            yield sub.name, sub / "model.xml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="seeds/trajectory_seeds")
    p.add_argument("--curated-dir", default="seeds/curated")
    p.add_argument("--scene-manifest", default="seeds/synthetic_scenes/manifest.jsonl")
    p.add_argument("--env-manifest", default="seeds/open_envs/manifest.jsonl")
    p.add_argument("--per-actor", type=int, default=1)
    p.add_argument("--per-scene", type=int, default=1)
    p.add_argument("--per-env", type=int, default=1)
    p.add_argument("--actor-limit", type=int, default=20,
                   help="cap on number of actors used (perf-bound)")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    out_root = (REPO / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    rng = random.Random(args.seed)
    n_total = 0
    n_ok = 0

    # --- L0 actors ---
    actors = list(_list_actors((REPO / args.curated_dir).resolve()))
    actors = actors[:args.actor_limit]
    for actor_id, xml_path in actors:
        for k in range(args.per_actor):
            kind = rng.choice(["zero_control", "random_control", "sinusoidal_control"])
            proto = _proto(kind, args.horizon, rng.randrange(1_000_000))
            r = replay_actor(str(xml_path), proto)
            replay_status = "ok" if r.ok else "failed"
            trace = r.trace.to_dict() if r.trace else {}
            sid = f"traj_actor_{actor_id[:32]}_{k}_{kind}"
            seed = _from_proto(actor_id, "actor", proto, replay_status, trace, sid)
            append_manifest(str(manifest_path), seed)
            n_total += 1
            n_ok += int(r.ok)

    # --- L1 scenes ---
    scenes = list(iter_manifest(str((REPO / args.scene_manifest).resolve())))
    for sc in scenes:
        if not isinstance(sc, SyntheticSceneSeed) or sc.compile_status != "ok":
            continue
        scene_xml = sc.scene_xml
        if scene_xml and not os.path.isabs(scene_xml):
            scene_xml = str((REPO / scene_xml).resolve())
        for k in range(args.per_scene):
            kind = rng.choice(["zero_control", "random_control"])
            proto = _proto(kind, args.horizon, rng.randrange(1_000_000))
            r = replay_scene(scene_xml, proto) if scene_xml else None
            ok = bool(r and r.ok)
            replay_status = "ok" if ok else "failed"
            trace = r.trace.to_dict() if (r and r.trace) else {}
            sid = f"traj_scene_{sc.seed_id}_{k}_{kind}"
            seed = _from_proto(sc.seed_id, "synthetic_scene", proto, replay_status, trace, sid)
            append_manifest(str(manifest_path), seed)
            n_total += 1
            n_ok += int(ok)

    # --- L2 envs ---
    envs = list(iter_manifest(str((REPO / args.env_manifest).resolve())))
    for env_seed in envs:
        if not isinstance(env_seed, OpenEnvSeed):
            continue
        if env_seed.runnable_status not in ("runnable", "partial"):
            continue
        for k in range(args.per_env):
            proto = _proto("random_control", min(args.horizon, 5), rng.randrange(1_000_000))
            r = replay_env(env_seed, proto)
            ok = r.ok
            replay_status = "ok" if ok else "failed"
            trace = r.trace.to_dict() if r.trace else {}
            sid = f"traj_env_{env_seed.seed_id}_{k}"
            seed = _from_proto(env_seed.seed_id, "open_env", proto, replay_status, trace, sid)
            append_manifest(str(manifest_path), seed)
            n_total += 1
            n_ok += int(ok)

    print(f"[build_trajectory_seeds] total={n_total} ok={n_ok} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
