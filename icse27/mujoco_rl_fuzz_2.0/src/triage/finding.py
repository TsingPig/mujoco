"""Finding record dumped to disk after an interesting RunResult."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Finding:
    finding_id: str
    seed_id: str
    layer: str
    signature: str
    oracle_signals: List[Dict[str, Any]] = field(default_factory=list)
    trace_summary: Dict[str, Any] = field(default_factory=dict)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    protocol: Dict[str, Any] = field(default_factory=dict)
    reproducible: Optional[bool] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, dirpath: str) -> str:
        os.makedirs(dirpath, exist_ok=True)
        path = os.path.join(dirpath, f"{self.finding_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
