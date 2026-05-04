"""tools/fetch_actor_seeds.py — thin wrapper around tools/fetch_seeds.py.

Kept separate so that the v2.0-β layered pipeline has a layer-explicit
entry point. Forwards all argv to ``fetch_seeds.py``.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    target = REPO / "tools" / "fetch_seeds.py"
    if not target.is_file():
        print(f"[fetch_actor_seeds] missing {target}")
        return 1
    sys.argv = [str(target)] + sys.argv[1:]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
