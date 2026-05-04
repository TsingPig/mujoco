"""Synthetic scene templates.

A template describes how to build an L1 SyntheticSceneSeed from an actor
seed. Each template is a small declarative dict consumed by
`scene.composer.compose_scene`.

Templates intentionally rely only on MJCF primitives (boxes, planes,
spheres, cylinders) so they can compose with any actor without needing
external mesh assets.
"""
from __future__ import annotations

from typing import Dict, Any, List


# Each template is:
#   tags:       list of tags propagated to the SceneSeed
#   arena:      dict describing the floor/ground (optional)
#   props:      list of dicts describing additional bodies
#   placement:  placement spec for the actor relative to the arena
#   description: human-readable summary
TEMPLATES: Dict[str, Dict[str, Any]] = {
    "arm_table_cube": {
        "tags": ["scene", "tabletop", "contact", "arm"],
        "description": "Actor sits next to a static table with a small cube on top.",
        "arena": {"type": "plane", "size": [2.0, 2.0, 0.1], "rgba": [0.7, 0.7, 0.7, 1]},
        "props": [
            {"name": "table", "type": "box", "size": [0.4, 0.4, 0.02],
             "pos": [0.5, 0.0, 0.4], "rgba": [0.6, 0.4, 0.2, 1], "mass": 5.0,
             "static": True},
            {"name": "cube", "type": "box", "size": [0.025, 0.025, 0.025],
             "pos": [0.5, 0.0, 0.45], "rgba": [0.2, 0.6, 0.9, 1], "mass": 0.05,
             "freejoint": True},
        ],
        "actor_placement": {"pos": [0.0, 0.0, 0.0], "quat": [1, 0, 0, 0]},
    },
    "gripper_object_lift": {
        "tags": ["scene", "manipulation", "contact", "gripper"],
        "description": "Gripper actor placed above a small object on a plane.",
        "arena": {"type": "plane", "size": [1.0, 1.0, 0.05]},
        "props": [
            {"name": "object", "type": "box", "size": [0.02, 0.02, 0.02],
             "pos": [0.0, 0.0, 0.04], "rgba": [0.1, 0.7, 0.3, 1], "mass": 0.02,
             "freejoint": True},
        ],
        "actor_placement": {"pos": [0.0, 0.0, 0.2]},
    },
    "quadruped_slope": {
        "tags": ["scene", "locomotion", "slope", "quadruped"],
        "description": "Quadruped actor placed on top of a tilted ramp.",
        "arena": {"type": "plane", "size": [3.0, 3.0, 0.1]},
        "props": [
            {"name": "ramp", "type": "box", "size": [1.0, 0.6, 0.05],
             "pos": [0.5, 0.0, 0.05], "euler": [0.0, -0.25, 0.0],
             "rgba": [0.4, 0.4, 0.4, 1], "mass": 50.0, "static": True},
        ],
        "actor_placement": {"pos": [0.0, 0.0, 0.4]},
    },
    "humanoid_obstacle": {
        "tags": ["scene", "locomotion", "obstacle", "humanoid"],
        "description": "Humanoid actor faces a low obstacle.",
        "arena": {"type": "plane", "size": [4.0, 4.0, 0.1]},
        "props": [
            {"name": "obstacle", "type": "box", "size": [0.1, 0.5, 0.15],
             "pos": [0.8, 0.0, 0.15], "rgba": [0.9, 0.2, 0.2, 1], "mass": 20.0,
             "static": True},
        ],
        "actor_placement": {"pos": [0.0, 0.0, 1.0]},
    },
    "mobile_base_wall": {
        "tags": ["scene", "navigation", "wall", "mobile_base"],
        "description": "Mobile base actor placed in front of a wall.",
        "arena": {"type": "plane", "size": [4.0, 4.0, 0.1]},
        "props": [
            {"name": "wall", "type": "box", "size": [0.05, 1.0, 0.5],
             "pos": [1.0, 0.0, 0.5], "rgba": [0.8, 0.8, 0.4, 1], "mass": 100.0,
             "static": True},
        ],
        "actor_placement": {"pos": [0.0, 0.0, 0.05]},
    },
    "hand_object": {
        "tags": ["scene", "manipulation", "hand", "contact"],
        "description": "Robot hand actor placed near a small graspable object.",
        "arena": {"type": "plane", "size": [1.0, 1.0, 0.05]},
        "props": [
            {"name": "object", "type": "sphere", "size": [0.025],
             "pos": [0.05, 0.0, 0.05], "rgba": [0.9, 0.9, 0.1, 1], "mass": 0.03,
             "freejoint": True},
        ],
        "actor_placement": {"pos": [0.0, 0.0, 0.15]},
    },
    "actor_floor_friction_variants": {
        "tags": ["scene", "friction", "floor"],
        "description": "Actor placed on a plane with perturbed friction.",
        "arena": {"type": "plane", "size": [3.0, 3.0, 0.1], "friction": [0.2, 0.005, 0.0001]},
        "props": [],
        "actor_placement": {"pos": [0.0, 0.0, 0.1]},
    },
    "actor_contact_stress": {
        "tags": ["scene", "contact_stress"],
        "description": "Actor surrounded by 4 small obstacles to maximize contacts.",
        "arena": {"type": "plane", "size": [2.0, 2.0, 0.1]},
        "props": [
            {"name": "p1", "type": "box", "size": [0.05, 0.05, 0.05],
             "pos": [0.3, 0.0, 0.05], "mass": 0.5, "freejoint": True},
            {"name": "p2", "type": "box", "size": [0.05, 0.05, 0.05],
             "pos": [-0.3, 0.0, 0.05], "mass": 0.5, "freejoint": True},
            {"name": "p3", "type": "box", "size": [0.05, 0.05, 0.05],
             "pos": [0.0, 0.3, 0.05], "mass": 0.5, "freejoint": True},
            {"name": "p4", "type": "box", "size": [0.05, 0.05, 0.05],
             "pos": [0.0, -0.3, 0.05], "mass": 0.5, "freejoint": True},
        ],
        "actor_placement": {"pos": [0.0, 0.0, 0.2]},
    },
    "actor_mass_perturbation_context": {
        "tags": ["scene", "mass_context"],
        "description": "Actor placed near a heavy mass that may settle into contact.",
        "arena": {"type": "plane", "size": [2.0, 2.0, 0.1]},
        "props": [
            {"name": "weight", "type": "box", "size": [0.1, 0.1, 0.1],
             "pos": [0.4, 0.0, 0.1], "mass": 5.0, "freejoint": True},
        ],
        "actor_placement": {"pos": [0.0, 0.0, 0.2]},
    },
    "actor_reset_context": {
        "tags": ["scene", "reset"],
        "description": "Bare actor on a plane, used to test reset reproducibility.",
        "arena": {"type": "plane", "size": [2.0, 2.0, 0.1]},
        "props": [],
        "actor_placement": {"pos": [0.0, 0.0, 0.05]},
    },
}


def list_templates() -> List[str]:
    return list(TEMPLATES.keys())


def get_template(name: str) -> Dict[str, Any]:
    if name not in TEMPLATES:
        raise KeyError(f"unknown scene template: {name}")
    return TEMPLATES[name]
