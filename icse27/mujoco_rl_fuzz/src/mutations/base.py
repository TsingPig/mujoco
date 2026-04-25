"""Mutator base class + result + minimal context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from lxml import etree


@dataclass
class MutationApplyResult:
    ok: bool
    new_xml_path: Optional[str] = None
    reason: Optional[str] = None
    runtime_directives: Optional[dict[str, Any]] = None  # forwarded to worker spec


class BaseMutator:
    """Abstract structural OR runtime mutator.

    - `sample_params` MUST be deterministic given (tree, rng).
    - `apply` MUST NOT raise on expected failure; return ok=False with `reason`.
    - For runtime-only mutators (e.g. STATE_PERTURB, SOLVER_TOGGLE) the XML is
      copied unchanged and `runtime_directives` carries the spec for the worker.
    """
    id: str = "abstract"

    def applicable(self, tree: etree._ElementTree) -> bool:
        return True

    def sample_params(self, tree: etree._ElementTree, rng) -> dict[str, Any]:
        raise NotImplementedError

    def apply(self, tree: etree._ElementTree, params: dict, out_xml_path: str) -> MutationApplyResult:
        raise NotImplementedError
