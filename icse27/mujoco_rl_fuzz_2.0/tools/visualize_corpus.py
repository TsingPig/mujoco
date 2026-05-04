"""tools/visualize_corpus.py

List or visualize layered seeds (L0/L1/L2/L3).

Usage:
    python tools/visualize_corpus.py --list
    python tools/visualize_corpus.py --list --layer actor
    python tools/visualize_corpus.py --list --layer synthetic_scene
    python tools/visualize_corpus.py --list --layer open_env
    python tools/visualize_corpus.py --list --layer trajectory
    python tools/visualize_corpus.py --show <seed_id>

For --show on synthetic_scene: launches mujoco viewer if available,
otherwise prints summary. For trajectory: replays and prints summary.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.corpus import iter_manifest, ActorSeed, SyntheticSceneSeed, OpenEnvSeed, TrajectorySeed
from src.viz.corpus_viewer import render_table


SCENE_MANIFEST = REPO / "seeds/synthetic_scenes/manifest.jsonl"
ENV_MANIFEST = REPO / "seeds/open_envs/manifest.jsonl"
TRAJ_MANIFEST = REPO / "seeds/trajectory_seeds/manifest.jsonl"
ACTOR_LEGACY_MANIFEST = REPO / "seeds/curated/MANIFEST.json"


def _load_actor_seeds(actor_manifest_jsonl: Path):
    """If a JSONL actor manifest exists, use it. Otherwise synthesize seeds
    from seeds/curated/<dir>/model.xml so visualization still works.
    """
    if actor_manifest_jsonl.exists():
        for s in iter_manifest(str(actor_manifest_jsonl)):
            yield s
        return
    curated = REPO / "seeds/curated"
    if not curated.exists():
        return
    for sub in sorted(curated.iterdir()):
        if not sub.is_dir():
            continue
        cand = sub / "model.xml"
        if cand.is_file():
            yield ActorSeed(
                seed_id=sub.name, source_repo=None,
                source_path=None,
                local_path=str(sub.relative_to(REPO)).replace("\\", "/"),
                asset_dir=str(sub.relative_to(REPO)).replace("\\", "/"),
                model_xml=str(cand.relative_to(REPO)).replace("\\", "/"),
                tags=[], compile_status="unknown",
            )


def _load_all(layer_filter=None):
    actor_jsonl = REPO / "seeds/curated/manifest.jsonl"
    if layer_filter in (None, "actor"):
        for s in _load_actor_seeds(actor_jsonl):
            yield s
    if layer_filter in (None, "synthetic_scene"):
        if SCENE_MANIFEST.exists():
            for s in iter_manifest(str(SCENE_MANIFEST)):
                yield s
    if layer_filter in (None, "open_env"):
        if ENV_MANIFEST.exists():
            for s in iter_manifest(str(ENV_MANIFEST)):
                yield s
    if layer_filter in (None, "trajectory"):
        if TRAJ_MANIFEST.exists():
            for s in iter_manifest(str(TRAJ_MANIFEST)):
                yield s


def cmd_list(args) -> int:
    seeds = list(_load_all(args.layer))
    print(render_table(seeds))
    print(f"\n[total: {len(seeds)} seeds]")
    return 0


def cmd_show(args) -> int:
    target = args.show
    found = None
    for s in _load_all():
        if s.seed_id == target:
            found = s
            break
    if found is None:
        print(f"[visualize_corpus] seed not found: {target}")
        return 1
    print(f"layer={found.layer} seed_id={found.seed_id}")
    if isinstance(found, SyntheticSceneSeed):
        scene_xml = found.scene_xml
        if scene_xml and not os.path.isabs(scene_xml):
            scene_xml = str((REPO / scene_xml).resolve())
        print(f"scene_xml: {scene_xml}")
        print(f"actor_seed_id: {found.actor_seed_id}")
        print(f"template: {found.template_name}")
        print(f"model_features: {found.model_features}")
        if args.rollout:
            from src.trajectory.protocol import TrajectoryProtocol, ActionSequence
            from src.trajectory.replay import replay_scene
            proto = TrajectoryProtocol(action_sequence=ActionSequence(kind="zero_control", horizon=args.steps))
            r = replay_scene(scene_xml, proto)
            print(f"rollout ok={r.ok} error={r.error}")
            if r.trace:
                print(r.trace.to_dict())
        elif args.viewer:
            try:
                import mujoco
                import mujoco.viewer  # type: ignore
                model = mujoco.MjModel.from_xml_path(scene_xml)
                data = mujoco.MjData(model)
                mujoco.viewer.launch_passive(model, data)
            except Exception as exc:
                print(f"[viewer unavailable: {exc}]")
        return 0
    if isinstance(found, ActorSeed):
        print(f"model_xml: {found.model_xml}")
        print(f"local_path: {found.local_path}")
        if args.rollout:
            from src.trajectory.protocol import TrajectoryProtocol, ActionSequence
            from src.trajectory.replay import replay_actor
            xml = found.model_xml
            if xml and not os.path.isabs(xml):
                xml = str((REPO / xml).resolve())
            proto = TrajectoryProtocol(action_sequence=ActionSequence(kind="zero_control", horizon=args.steps))
            r = replay_actor(xml, proto)
            print(f"rollout ok={r.ok} error={r.error}")
            if r.trace:
                print(r.trace.to_dict())
        return 0
    if isinstance(found, OpenEnvSeed):
        print(f"adapter={found.adapter_name} env_id={found.env_id}")
        print(f"dependency_status={found.dependency_status} runnable={found.runnable_status}")
        print(f"smoke_trace={found.smoke_trace}")
        return 0
    if isinstance(found, TrajectorySeed):
        print(f"parent_layer={found.parent_layer} parent_seed_id={found.parent_seed_id}")
        print(f"action_sequence={found.action_sequence_spec}")
        print(f"replay_status={found.replay_status}")
        print(f"trace_summary={found.trace_summary}")
        return 0
    print("unsupported seed type")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="list seeds in a table")
    p.add_argument("--layer", choices=["actor", "synthetic_scene", "open_env", "trajectory"],
                   default=None)
    p.add_argument("--show", default=None, help="show details for a single seed_id")
    p.add_argument("--viewer", action="store_true", help="launch mujoco viewer for scene/actor")
    p.add_argument("--rollout", action="store_true", help="run a short rollout instead")
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    if args.show:
        return cmd_show(args)
    if args.list or args.layer:
        return cmd_list(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
