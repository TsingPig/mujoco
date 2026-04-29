"""Build the seeds 2.0 categorized index.

Reads `seeds2/categories.yaml`, scans `seeds/curated/` and `seeds/_assets/`,
classifies every seed into one or more bug-taxonomy categories (C1..C12) and
writes:

    seeds2/MANIFEST.json               # every seed -> [categories, stats, paths]
    seeds2/<Cn>/index.txt              # one seed name per line
    seeds2/<Cn>/README.md              # category metadata + member count
    seeds2/SUMMARY.md                  # cross-category overview

This DOES NOT duplicate any XML / mesh files; it only emits an index pointing
back into ``seeds/curated/<seed>/model.xml`` (or ``seeds/_assets/<seed>/...``).
This keeps the repo small and lets the existing visualize_seed.py / mutator
pipelines load the same files.

Usage:
    python tools/build_seeds2.py
    python tools/build_seeds2.py --refetch        # also runs fetch_seeds.py for
                                                  # sources listed under
                                                  # extra_fetch.sources first
    python tools/build_seeds2.py --include-assets # also classify seeds/_assets/
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = ROOT / "seeds"
CURATED_DIR = SEEDS_DIR / "curated"
ASSETS_DIR = SEEDS_DIR / "_assets"
SEEDS2_DIR = ROOT / "seeds2"
CATS_YAML = SEEDS2_DIR / "categories.yaml"
CURATED_MANIFEST = CURATED_DIR / "MANIFEST.json"


def _load_curated_stats() -> Dict[str, dict]:
    """Map seed-dir-name -> stats dict from seeds/curated/MANIFEST.json."""
    out: Dict[str, dict] = {}
    if not CURATED_MANIFEST.is_file():
        return out
    try:
        for entry in json.loads(CURATED_MANIFEST.read_text(encoding="utf-8")):
            cp = entry.get("curated_path", "").replace("\\", "/")
            parts = cp.split("/")
            if len(parts) >= 2:
                out[parts[1]] = entry
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to parse curated MANIFEST: {exc}", file=sys.stderr)
    return out


def _scan_seed_dirs(include_assets: bool) -> List[Tuple[str, Path, str]]:
    """Yield (seed_name, xml_path, tier) for every model.xml found.

    tier is "curated" or "asset". seeds/_assets entries may have the xml at
    arbitrary depths, so we walk for any .xml that compiles to a valid MJCF.
    To keep this fast we trust the directory layout: the XML named
    ``model.xml`` is preferred; otherwise the lexicographically first .xml.
    """
    results: List[Tuple[str, Path, str]] = []
    if CURATED_DIR.is_dir():
        for d in sorted(CURATED_DIR.iterdir()):
            if not d.is_dir():
                continue
            xml = d / "model.xml"
            if xml.is_file():
                results.append((d.name, xml, "curated"))
    if include_assets and ASSETS_DIR.is_dir():
        for d in sorted(ASSETS_DIR.iterdir()):
            if not d.is_dir():
                continue
            xml = d / "model.xml"
            if not xml.is_file():
                xmls = sorted(d.rglob("*.xml"))
                if not xmls:
                    continue
                xml = xmls[0]
            results.append((d.name, xml, "asset"))
    return results


def _compile_match(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns or []]


def classify(seed_name: str, source: str, cats: List[dict]) -> List[str]:
    """Return the list of category ids the seed belongs to."""
    out: List[str] = []
    name_l = seed_name.lower()
    for c in cats:
        cid = c["id"]
        if seed_name in (c.get("explicit") or []):
            out.append(cid)
            continue
        match_pats = _compile_match(c.get("name_match", []))
        excl_pats = _compile_match(c.get("name_exclude", []))
        sources_allowed = set(c.get("sources") or [])
        name_hit = any(p.search(name_l) for p in match_pats)
        src_hit = (source in sources_allowed) and bool(c.get("sources"))
        if not (name_hit or (src_hit and not match_pats)):
            continue
        if any(p.search(name_l) for p in excl_pats):
            continue
        out.append(cid)
    return out


def _infer_source(seed_name: str) -> str:
    if "__" not in seed_name:
        return "unknown"
    return seed_name.split("__", 1)[0]


def _refetch(extra_sources: List[str]) -> None:
    if not extra_sources:
        return
    cmd = [sys.executable, str(ROOT / "tools" / "fetch_seeds.py"),
           "--only", ",".join(extra_sources)]
    print(f"[refetch] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refetch", action="store_true",
                    help="invoke fetch_seeds.py for extra_fetch.sources first")
    ap.add_argument("--include-assets", action="store_true",
                    help="also classify seeds/_assets/ directories")
    args = ap.parse_args()

    if not CATS_YAML.is_file():
        print(f"missing {CATS_YAML}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(CATS_YAML.read_text(encoding="utf-8"))
    cats: List[dict] = cfg.get("categories") or []

    if args.refetch:
        _refetch((cfg.get("extra_fetch") or {}).get("sources", []))

    curated_stats = _load_curated_stats()
    seeds = _scan_seed_dirs(include_assets=args.include_assets)
    print(f"[scan] {len(seeds)} seed dirs (curated + {'assets' if args.include_assets else 'no-assets'})")

    by_cat: Dict[str, List[str]] = {c["id"]: [] for c in cats}
    manifest: List[dict] = []
    for seed_name, xml_path, tier in seeds:
        source = _infer_source(seed_name)
        cat_ids = classify(seed_name, source, cats)
        rel_xml = str(xml_path.relative_to(ROOT)).replace("\\", "/")
        stats = curated_stats.get(seed_name, {})
        manifest.append({
            "name": seed_name,
            "tier": tier,
            "source": source,
            "xml_path": rel_xml,
            "categories": cat_ids,
            "nq": stats.get("nq"),
            "nv": stats.get("nv"),
            "nu": stats.get("nu"),
            "nbody": stats.get("nbody"),
            "ngeom": stats.get("ngeom"),
            "njnt": stats.get("njnt"),
            "neq": stats.get("neq"),
            "ntendon": stats.get("ntendon"),
            "commit": stats.get("commit"),
            "src_rel_path": stats.get("src_rel_path"),
        })
        for cid in cat_ids:
            by_cat[cid].append(seed_name)

    SEEDS2_DIR.mkdir(parents=True, exist_ok=True)
    (SEEDS2_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Per-category folders.
    summary_lines = ["# seeds2 — Categorized seed index", "",
                     "Auto-generated by `tools/build_seeds2.py`. Do not edit.",
                     "", "| Category | Title | Members |", "|---|---|---:|"]
    uncovered = [m["name"] for m in manifest if not m["categories"]]
    for c in cats:
        cid = c["id"]
        cdir = SEEDS2_DIR / cid
        cdir.mkdir(parents=True, exist_ok=True)
        members = sorted(set(by_cat[cid]))
        (cdir / "index.txt").write_text("\n".join(members) + "\n", encoding="utf-8")
        readme = [
            f"# {cid} — {c.get('name_zh', '')}",
            "",
            f"> {c.get('description', '')}",
            "",
            f"**Doc anchor:** `{c.get('doc_anchor', '')}`",
            "",
            f"**Members:** {len(members)}",
            "",
            "## Suggested operators",
            "",
            *[f"- `{op}`" for op in (c.get("operators") or [])],
            "",
            "## Suggested oracles",
            "",
            *[f"- `{o}`" for o in (c.get("oracles") or [])],
            "",
            "## Members",
            "",
            *[f"- `{n}`" for n in members],
            "",
        ]
        (cdir / "README.md").write_text("\n".join(readme), encoding="utf-8")
        summary_lines.append(f"| `{cid}` | {c.get('name_zh', '')} | {len(members)} |")

    summary_lines += [
        "",
        f"**Total seeds scanned:** {len(manifest)}",
        f"**Uncategorized:** {len(uncovered)}",
        "",
    ]
    if uncovered:
        summary_lines += ["## Uncategorized seeds", ""]
        summary_lines += [f"- `{n}`" for n in sorted(uncovered)[:200]]
        if len(uncovered) > 200:
            summary_lines.append(f"- ... ({len(uncovered) - 200} more)")
    (SEEDS2_DIR / "SUMMARY.md").write_text("\n".join(summary_lines),
                                            encoding="utf-8")

    print(f"[ok] manifest -> {SEEDS2_DIR / 'MANIFEST.json'}")
    for c in cats:
        cid = c["id"]
        print(f"  {cid:>4}: {len(set(by_cat[cid])):>4} seeds  ({c.get('name_zh', '')})")
    print(f"  uncategorized: {len(uncovered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
