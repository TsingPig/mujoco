"""Seed pool: scan a directory for *.xml files."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from .utils import stable_id


@dataclass
class SeedRecord:
    seed_id: str
    xml_path: str
    name: str


class SeedPool:
    def __init__(self, root: str, pattern: str = "*.xml"):
        self.root = root
        self.pattern = pattern
        self._items: list[SeedRecord] = []
        self.refresh()

    def refresh(self) -> None:
        paths = sorted(glob.glob(os.path.join(self.root, self.pattern)))
        self._items = [
            SeedRecord(seed_id=stable_id(p), xml_path=p, name=os.path.basename(p))
            for p in paths
        ]

    def list(self) -> list[SeedRecord]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, rng) -> SeedRecord:
        if not self._items:
            raise RuntimeError(f"No seeds found in {self.root!r}")
        return rng.choice(self._items)

    def add(self, xml_path: str) -> SeedRecord:
        rec = SeedRecord(seed_id=stable_id(xml_path), xml_path=xml_path,
                         name=os.path.basename(xml_path))
        self._items.append(rec)
        return rec
