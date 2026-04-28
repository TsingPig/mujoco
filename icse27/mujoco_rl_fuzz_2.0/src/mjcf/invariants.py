"""Cheap structural invariants used by mutators to decide applicability."""
from __future__ import annotations

from typing import List
from lxml import etree


def all_bodies(root: etree._Element) -> List[etree._Element]:
    """All <body> elements (worldbody itself excluded)."""
    return list(root.iter("body"))


def _is_body_child(el: etree._Element) -> bool:
    p = el.getparent()
    return p is not None and p.tag == "body"


def all_joints(root: etree._Element) -> List[etree._Element]:
    """Only kinematic <joint> elements (children of <body>).

    Excludes <joint> tags that appear inside <equality>, <tendon>, <actuator>,
    <default>, etc., which use a different schema (joint1/joint2/coef).
    """
    return [j for j in root.iter("joint") if _is_body_child(j)]


def hingelike_joints(root: etree._Element) -> List[etree._Element]:
    """Body joints whose explicit `type` is hinge or slide.

    Conservative: joints without an explicit type may inherit ball/free from
    a `<default>` class; we exclude them rather than try to resolve the chain.
    """
    return [j for j in all_joints(root) if j.get("type") in ("hinge", "slide")]


def all_geoms(root: etree._Element) -> List[etree._Element]:
    """Only kinematic <geom> elements (children of <body> or <worldbody>).

    Excludes <geom> in <default> blocks and contact <pair>/<exclude> children.
    """
    out = []
    for g in root.iter("geom"):
        p = g.getparent()
        if p is not None and p.tag in ("body", "worldbody"):
            out.append(g)
    return out


def non_plane_geoms(root: etree._Element) -> List[etree._Element]:
    return [g for g in all_geoms(root) if g.get("type") != "plane"]


def actuator_root(root: etree._Element) -> etree._Element:
    el = root.find("actuator")
    if el is None:
        el = etree.SubElement(root, "actuator")
    return el


def equality_root(root: etree._Element) -> etree._Element:
    el = root.find("equality")
    if el is None:
        el = etree.SubElement(root, "equality")
    return el


def tendon_root(root: etree._Element) -> etree._Element:
    el = root.find("tendon")
    if el is None:
        el = etree.SubElement(root, "tendon")
    return el


def option_element(root: etree._Element) -> etree._Element:
    el = root.find("option")
    if el is None:
        # MJCF requires option to come early; insert as first non-compiler child.
        el = etree.Element("option")
        # Place after <compiler> if present, else first.
        compiler_el = root.find("compiler")
        if compiler_el is not None:
            compiler_el.addnext(el)
        else:
            root.insert(0, el)
    return el


def worldbody(root: etree._Element) -> etree._Element:
    wb = root.find("worldbody")
    if wb is None:
        wb = etree.SubElement(root, "worldbody")
    return wb


def named_or_indexed(elements: List[etree._Element], idx: int) -> etree._Element:
    return elements[idx % len(elements)]
