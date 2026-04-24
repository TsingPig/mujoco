"""TestCase + MutationRecord dataclasses (guide Appendix A)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MutationRecord:
    mutator_id: str
    params: dict[str, Any]
    success: bool
    reason: Optional[str] = None


@dataclass
class TestCase:
    tc_id: str
    parent_seed_id: str
    mutations: list[MutationRecord] = field(default_factory=list)
    xml_path: str = ""
    rollout_steps: int = 50
    created_at: float = 0.0
