"""tools/fetch_env_scenes.py

Register open-source environment scenes (L2) via adapters. Missing
dependencies degrade gracefully into ``metadata_only`` entries.

Usage:
    python tools/fetch_env_scenes.py
    python tools/fetch_env_scenes.py --max-per-adapter 20
    python tools/fetch_env_scenes.py --no-smoke

Outputs:
    seeds/open_envs/manifest.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.corpus import OpenEnvSeed, append_manifest, make_provenance
from src.env_adapters import list_adapters, get_adapter
from src.env_adapters.base import STATUS_OK, STATUS_MISSING


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="seeds/open_envs")
    p.add_argument("--max-per-adapter", type=int, default=30)
    p.add_argument("--no-smoke", action="store_true",
                   help="skip reset+step smoke test (faster, less informative)")
    p.add_argument("--smoke-steps", type=int, default=5)
    args = p.parse_args()

    out_root = (REPO / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    n_total = 0
    n_runnable = 0
    n_partial = 0
    n_meta = 0

    for ad_name in list_adapters():
        adapter = get_adapter(ad_name)
        status = adapter.is_available()
        envs = adapter.list_envs(max_envs=args.max_per_adapter)
        for entry in envs:
            n_total += 1
            env_id = entry["env_id"]
            ctor = entry.get("constructor_kwargs", {})
            tags = list(entry.get("tags", []))
            seed_id = f"env_{ad_name}_{env_id.replace('/', '_').replace(':', '_')}"
            dependency_status = status.dependency
            runnable_status = "metadata_only"
            smoke_trace = {}
            model_access = None
            obs_summary = {}
            reward_avail = None
            reset_avail = None
            rollout_avail = None

            if dependency_status == STATUS_OK and not args.no_smoke:
                res = adapter.smoke(env_id, n_steps=args.smoke_steps, **ctor)
                smoke_trace = {
                    "ok": res.ok, "reset_ok": res.reset_ok,
                    "step_ok": res.step_ok, "n_steps": res.n_steps,
                    "error": res.error,
                }
                model_access = res.model_access_method
                reset_avail = res.reset_ok
                reward_avail = bool(res.reward_summary)
                rollout_avail = res.step_ok
                if res.ok:
                    runnable_status = "runnable"
                    n_runnable += 1
                elif res.reset_ok:
                    runnable_status = "partial"
                    n_partial += 1
                else:
                    runnable_status = "broken"
            elif dependency_status == STATUS_OK and args.no_smoke:
                runnable_status = "unknown"
            else:
                n_meta += 1

            seed = OpenEnvSeed(
                seed_id=seed_id,
                source_repo=None,
                package_name=adapter.package,
                env_id=env_id,
                adapter_name=ad_name,
                constructor_kwargs=ctor,
                model_access_method=model_access,
                action_space_summary=obs_summary,
                observation_space_summary={},
                reward_available=reward_avail,
                reset_available=reset_avail,
                rollout_available=rollout_avail,
                local_source_path=None,
                license_path=None,
                dependency_status=dependency_status,
                runnable_status=runnable_status,
                smoke_trace=smoke_trace,
                tags=tags,
                provenance=make_provenance(
                    source=ad_name, package=adapter.package,
                    notes=status.notes,
                ),
            )
            append_manifest(str(manifest_path), seed)
            print(f"[{runnable_status:14s}] {ad_name:18s} {env_id}")

    print(f"[fetch_env_scenes] total={n_total} runnable={n_runnable} "
          f"partial={n_partial} metadata_only={n_meta} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
