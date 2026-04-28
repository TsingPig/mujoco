"""MJCF loading + safe in-process compile gate.

We use lxml for all structural mutations (proven path from v1).
MjSpec in MuJoCo 3.2.3 lacks reliable access to `option` and other
core sub-structures, so we treat XML text as the single source of truth.

`SafeModel` wraps an lxml tree with deep-copy snapshot/rollback and a
single entry point `safe_compile()` that returns (ok, exception_msg).
"""
from __future__ import annotations

import copy
import io
import os
from dataclasses import dataclass
from typing import Optional

from lxml import etree

import mujoco


def parse_xml_file(path: str) -> etree._ElementTree:
    parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
    tree = etree.parse(path, parser)
    return tree


def parse_xml_string(text: str) -> etree._ElementTree:
    parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
    return etree.ElementTree(etree.fromstring(text.encode("utf-8"), parser))


def serialize(tree: etree._ElementTree) -> str:
    return etree.tostring(tree, pretty_print=True, encoding="unicode")


@dataclass
class CompileOutcome:
    ok: bool
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    nq: int = 0
    nv: int = 0
    nu: int = 0
    nbody: int = 0
    ngeom: int = 0
    njnt: int = 0
    neq: int = 0
    ntendon: int = 0


def safe_compile(xml_text: str, asset_dir: Optional[str] = None) -> CompileOutcome:
    """Compile XML in-process. Never raises.

    asset_dir: if provided, mujoco will resolve meshdir/texturedir
    relative to this directory.
    """
    try:
        if asset_dir is not None:
            old_cwd = os.getcwd()
            try:
                os.chdir(asset_dir)
                model = mujoco.MjModel.from_xml_string(xml_text)
            finally:
                os.chdir(old_cwd)
        else:
            model = mujoco.MjModel.from_xml_string(xml_text)
    except Exception as exc:  # pragma: no cover - mujoco raises many kinds
        return CompileOutcome(
            ok=False,
            error_type=type(exc).__name__,
            error_msg=str(exc)[:300],
        )
    return CompileOutcome(
        ok=True,
        nq=int(model.nq), nv=int(model.nv), nu=int(model.nu),
        nbody=int(model.nbody), ngeom=int(model.ngeom),
        njnt=int(model.njnt), neq=int(model.neq), ntendon=int(model.ntendon),
    )


class SafeModel:
    """Wraps an lxml ElementTree with snapshot/rollback semantics.

    Intended usage by mutators:

        sm = SafeModel.load(path)
        snap = sm.snapshot()
        ok = some_mutator.apply_inplace(sm)
        if not ok or not sm.compile().ok:
            sm.restore(snap)
    """

    def __init__(self, tree: etree._ElementTree, source_path: Optional[str] = None):
        self.tree = tree
        self.source_path = source_path
        # Auto-incremented per-mutation counter for unique name suffixes.
        self._uid = 0

    # --- factory ---
    @classmethod
    def load(cls, path: str) -> "SafeModel":
        return cls(parse_xml_file(path), source_path=path)

    @classmethod
    def from_string(cls, text: str) -> "SafeModel":
        return cls(parse_xml_string(text))

    # --- snapshot / restore ---
    def snapshot(self) -> bytes:
        return etree.tostring(self.tree, encoding="utf-8")

    def restore(self, snap: bytes) -> None:
        self.tree = etree.ElementTree(etree.fromstring(snap))

    # --- access ---
    @property
    def root(self) -> etree._Element:
        return self.tree.getroot()

    def serialize(self) -> str:
        return serialize(self.tree)

    def asset_dir(self) -> Optional[str]:
        if self.source_path:
            return os.path.dirname(os.path.abspath(self.source_path))
        return None

    def compile(self) -> CompileOutcome:
        return safe_compile(self.serialize(), asset_dir=self.asset_dir())

    # --- helpers ---
    def fresh_name(self, prefix: str) -> str:
        self._uid += 1
        return f"{prefix}_mut{self._uid}"

    def get_or_create(self, parent: etree._Element, tag: str) -> etree._Element:
        el = parent.find(tag)
        if el is None:
            el = etree.SubElement(parent, tag)
        return el
