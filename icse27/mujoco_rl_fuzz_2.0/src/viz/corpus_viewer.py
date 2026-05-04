"""Tabular listing of layered seeds."""
from __future__ import annotations

from typing import Iterable, List

from ..corpus.schema import LayeredSeed


COLUMNS = ("seed_id", "layer", "source", "env_id", "local_path",
           "license", "compile_status", "runnable_status", "tags")


def _row(seed: LayeredSeed) -> List[str]:
    layer = seed.layer
    source = ""
    env_id = ""
    local_path = getattr(seed, "local_path", "") or ""
    license_path = getattr(seed, "license_path", "") or ""
    compile_status = getattr(seed, "compile_status", "") or ""
    runnable_status = getattr(seed, "runnable_status", "") or ""
    tags = ",".join(getattr(seed, "tags", []) or [])

    if layer == "actor":
        source = getattr(seed, "source_repo", "") or seed.provenance.source
    elif layer == "synthetic_scene":
        source = getattr(seed, "source", "generated")
        local_path = getattr(seed, "scene_xml", "") or local_path
    elif layer == "open_env":
        source = getattr(seed, "source_repo", "") or getattr(seed, "package_name", "") or ""
        env_id = getattr(seed, "env_id", "") or ""
        local_path = getattr(seed, "local_source_path", "") or ""
    elif layer == "trajectory":
        source = f"parent:{getattr(seed, 'parent_seed_id', '')}"
        compile_status = getattr(seed, "replay_status", "") or ""
        runnable_status = compile_status
    return [seed.seed_id, layer, source, env_id, local_path,
            license_path, compile_status, runnable_status, tags]


def render_table(seeds: Iterable[LayeredSeed]) -> str:
    rows = [list(COLUMNS)]
    for s in seeds:
        rows.append(_row(s))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(COLUMNS))]
    out_lines = []
    for r in rows:
        out_lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out_lines)
