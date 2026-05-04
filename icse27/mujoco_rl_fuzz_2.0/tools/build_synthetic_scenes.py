"""tools/build_synthetic_scenes.py

Generate ~50 L1 synthetic composed scenes from existing curated actor
seeds. Each generated scene is safe-compiled. Failures are routed to
seeds/_quarantine/synthetic_scenes/.

Usage:
    python tools/build_synthetic_scenes.py --count 50
    python tools/build_synthetic_scenes.py --count 5 --seed 7

Outputs:
    seeds/synthetic_scenes/<seed_id>/scene.xml
    seeds/synthetic_scenes/<seed_id>/generation_config.json
    seeds/synthetic_scenes/<seed_id>/provenance.json
    seeds/synthetic_scenes/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.corpus import (
    SyntheticSceneSeed, ActorSeed, append_manifest,
    make_provenance,
)
from src.scene import compose_scene, ComposeRequest, list_templates


def _list_curated_actors(curated_dir: Path):
    """Yield (actor_seed_id, model_xml_abs_path) for every curated dir
    containing model.xml.
    """
    for sub in sorted(curated_dir.iterdir()):
        if not sub.is_dir():
            continue
        cand = sub / "model.xml"
        if cand.is_file():
            yield sub.name, cand


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=50,
                   help="number of synthetic scenes to attempt")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--curated-dir", default="seeds/curated",
                   help="directory containing actor seed dirs")
    p.add_argument("--out-root", default="seeds/synthetic_scenes",
                   help="output directory for L1 scenes")
    p.add_argument("--quarantine", default="seeds/_quarantine/synthetic_scenes",
                   help="quarantine for compile-failed scenes")
    p.add_argument("--templates", nargs="*", default=None,
                   help="restrict to these template names")
    args = p.parse_args()

    curated = (REPO / args.curated_dir).resolve()
    out_root = (REPO / args.out_root).resolve()
    quarantine = (REPO / args.quarantine).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    actors = list(_list_curated_actors(curated))
    if not actors:
        print(f"[build_synthetic_scenes] no actor seeds found under {curated}")
        return 1

    templates = args.templates or list_templates()
    rng = random.Random(args.seed)
    n_ok = 0
    n_fail = 0

    for i in range(args.count):
        actor_id, actor_path = actors[rng.randrange(len(actors))]
        tmpl = templates[rng.randrange(len(templates))]
        seed_id = f"scene_{i:04d}_{tmpl}_{actor_id[:32]}"
        out_dir = out_root / seed_id
        out_dir.mkdir(parents=True, exist_ok=True)
        req = ComposeRequest(
            actor_xml_path=str(actor_path),
            template_name=tmpl,
            out_dir=str(out_dir),
            actor_seed_id=actor_id,
            seed=rng.randrange(1_000_000),
        )
        res = compose_scene(req)
        gen_cfg = {"template": tmpl, "actor_seed_id": actor_id,
                   "actor_xml_path": str(actor_path), **(res.generation_config or {})}
        with open(out_dir / "generation_config.json", "w", encoding="utf-8") as f:
            json.dump(gen_cfg, f, indent=2)
        prov = make_provenance(
            source="generated", local_path=str(out_dir.relative_to(REPO)),
            notes=f"composed from actor {actor_id} via template {tmpl}",
        )
        with open(out_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(prov.to_dict(), f, indent=2)

        if not res.ok:
            # Move to quarantine.
            qdir = quarantine / seed_id
            qdir.mkdir(parents=True, exist_ok=True)
            with open(qdir / "compose_error.txt", "w", encoding="utf-8") as f:
                f.write(res.error or "unknown")
            print(f"[FAIL] {seed_id}: {res.error}")
            n_fail += 1
            continue

        seed = SyntheticSceneSeed(
            seed_id=seed_id,
            actor_seed_id=actor_id,
            object_asset_ids=[],
            arena_asset_id=None,
            scene_xml=str(Path(res.scene_xml_path).relative_to(REPO)).replace("\\", "/"),
            placement_config=res.placement_config,
            generation_config=gen_cfg,
            template_name=tmpl,
            tags=res.tags,
            compile_status="ok",
            model_features=res.model_features,
            provenance=prov,
        )
        append_manifest(str(manifest_path), seed)
        n_ok += 1
        print(f"[OK]   {seed_id}")

    print(f"[build_synthetic_scenes] ok={n_ok} fail={n_fail} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
