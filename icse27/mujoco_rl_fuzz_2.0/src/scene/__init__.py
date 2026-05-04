"""Synthetic scene composition (L1)."""
from __future__ import annotations

from .composer import compose_scene, ComposeRequest, ComposeResult
from .templates import TEMPLATES, list_templates, get_template
from .placements import sample_placement
from .validate_scene import validate_scene_xml

__all__ = [
    "compose_scene",
    "ComposeRequest",
    "ComposeResult",
    "TEMPLATES",
    "list_templates",
    "get_template",
    "sample_placement",
    "validate_scene_xml",
]
