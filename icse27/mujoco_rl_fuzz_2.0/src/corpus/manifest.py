"""JSONL manifest reader/writer for layered seeds.

Each manifest file is a newline-delimited JSON file. One line == one seed.
Layers may share a manifest or be split per directory.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Iterable, List, Optional

from .schema import LayeredSeed, seed_from_dict


def iter_manifest(path: str) -> Iterator[LayeredSeed]:
    """Yield seeds from a JSONL manifest. Skips blank/comment lines.

    Lines that fail to parse or fail schema mapping are skipped silently
    (with a printed warning) so partial corruption doesn't kill exploration.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
                yield seed_from_dict(d)
            except Exception as exc:  # pragma: no cover - tolerant by design
                print(f"[manifest] {path}:{line_no} skipped: {exc}")


def load_manifest(path: str) -> List[LayeredSeed]:
    return list(iter_manifest(path))


def save_manifest(path: str, seeds: Iterable[LayeredSeed]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def append_manifest(path: str, seed: LayeredSeed) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(seed.to_dict(), ensure_ascii=False) + "\n")


def validate_manifest(path: str) -> dict:
    """Return a small dict {total, ok, by_layer, errors}."""
    total = 0
    by_layer: dict = {}
    errors = 0
    if not os.path.exists(path):
        return {"total": 0, "ok": 0, "by_layer": {}, "errors": 0, "missing": True}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            total += 1
            try:
                d = json.loads(line)
                seed = seed_from_dict(d)
                by_layer[seed.layer] = by_layer.get(seed.layer, 0) + 1
            except Exception:
                errors += 1
    return {
        "total": total,
        "ok": total - errors,
        "errors": errors,
        "by_layer": by_layer,
        "missing": False,
    }
