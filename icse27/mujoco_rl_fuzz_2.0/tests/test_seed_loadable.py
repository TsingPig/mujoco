"""All curated seeds must load via mujoco.MjModel.from_xml_path.

Skipped if seeds/curated has no model.xml files yet (run fetch_seeds first).
"""
from pathlib import Path

import pytest


def _seeds():
    return sorted(Path("seeds/curated").glob("**/model.xml"))


@pytest.mark.skipif(not _seeds(), reason="run tools/fetch_seeds.py first")
@pytest.mark.parametrize("path", _seeds(), ids=lambda p: p.parent.name)
def test_seed_loads(path):
    import mujoco
    mujoco.MjModel.from_xml_path(str(path))
