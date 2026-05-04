"""Validate a synthetic scene XML file via safe_compile."""
from __future__ import annotations

import os
from typing import Dict, Any

from ..mjcf.spec_loader import safe_compile


def validate_scene_xml(path: str) -> Dict[str, Any]:
    """Compile a scene.xml file and return a small status dict.

    Never raises; returns {ok: bool, error_type: ?, error_msg: ?, features: {...}}.
    """
    if not os.path.isfile(path):
        return {"ok": False, "error_type": "FileNotFound", "error_msg": path}
    try:
        with open(path, "r", encoding="utf-8") as f:
            xml_text = f.read()
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error_msg": str(exc)}
    asset_dir = os.path.dirname(os.path.abspath(path))
    out = safe_compile(xml_text, asset_dir=asset_dir)
    if not out.ok:
        return {"ok": False, "error_type": out.error_type, "error_msg": out.error_msg}
    return {
        "ok": True,
        "features": {
            "nq": out.nq, "nv": out.nv, "nu": out.nu,
            "nbody": out.nbody, "ngeom": out.ngeom,
            "njnt": out.njnt, "neq": out.neq, "ntendon": out.ntendon,
        },
    }
