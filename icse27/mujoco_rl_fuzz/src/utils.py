"""Generic helpers: hash, json io, rng."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Any

import numpy as np


def stable_id(*parts: Any, length: int = 16) -> str:
    """Hash arbitrary parts to a short hex id (deterministic)."""
    h = hashlib.blake2s(digest_size=length // 2)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
    return h.hexdigest()


def make_rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def make_np_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def now_ts() -> float:
    return time.time()
