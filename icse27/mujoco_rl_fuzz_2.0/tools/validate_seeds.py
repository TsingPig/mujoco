"""Validate every curated seed compiles via mujoco.MjModel.from_xml_path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="seeds/curated")
    ap.add_argument("--quarantine", default="seeds/_quarantine")
    args = ap.parse_args()

    import mujoco

    curated = Path(args.curated)
    quarantine = Path(args.quarantine)
    quarantine.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(curated.glob("**/model.xml"))
    if not files:
        print(f"no curated seeds at {curated}")
        return 1

    ok: List[str] = []
    bad: List[Dict] = []
    for f in files:
        try:
            mujoco.MjModel.from_xml_path(str(f))
            ok.append(str(f))
        except Exception as exc:  # noqa: BLE001
            bad.append({"path": str(f), "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    print(f"loadable : {len(ok)}/{len(files)}")
    print(f"failed   : {len(bad)}")
    if bad:
        (quarantine / "REASONS.jsonl").write_text(
            "\n".join(json.dumps(b) for b in bad), encoding="utf-8")
        print(f"  -> wrote reasons to {quarantine / 'REASONS.jsonl'}")
        for b in bad[:10]:
            print(f"    {b['path']}\n      {b['error']}")
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
