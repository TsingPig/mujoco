"""Fetch upstream MJCF seeds via shallow git clone.

Usage:
    python tools/fetch_seeds.py
    python tools/fetch_seeds.py --only mujoco,dm_control
    python tools/fetch_seeds.py --workdir .cache/repos --out seeds/curated

Each source clones to .cache/repos/<source>/ (depth=1). MJCFs are walked,
each candidate is compiled with mujoco.MjModel.from_xml_path. On success the
file (and its include/<asset> deps) is normalised and copied into
seeds/curated/<source>__<slug>.xml. Per-source caps from configs/default.yaml
limit the curated set; overflow goes to seeds/_pool/. A MANIFEST.json records
{commit, repo, source, rel_path, nq, nv, nbody, has_actuator, has_eq, has_tendon}
for every curated file.

Source repos LICENSE files are also copied to seeds/curated/<source>__LICENSE
when present.

This script is idempotent: re-running fetches updates via `git fetch --depth=1`.
No git submodules are added to the parent repo.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from lxml import etree


# ---------------------------------------------------------------------------
# Source registry. Each source has:
#   url       : git remote
#   ref       : tag / branch / commit
#   patterns  : list of glob patterns relative to repo root (curated XMLs)
#   exclude   : substrings; any match is dropped
#   prefer    : substrings; sorted to front before cap
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    name: str
    url: str
    ref: str
    patterns: Tuple[str, ...]
    exclude: Tuple[str, ...] = ()
    prefer: Tuple[str, ...] = ()


SOURCES: Dict[str, Source] = {
    "mujoco": Source(
        name="mujoco",
        url="https://github.com/google-deepmind/mujoco.git",
        ref="3.2.3",
        patterns=("model/**/*.xml",),
        exclude=("test", "shared",),
    ),
    "menagerie": Source(
        name="menagerie",
        url="https://github.com/google-deepmind/mujoco_menagerie.git",
        ref="main",
        patterns=("*/scene*.xml", "*/*.xml"),
        exclude=("README", ".github"),
        prefer=("scene",),
    ),
    "dm_control": Source(
        name="dm_control",
        url="https://github.com/google-deepmind/dm_control.git",
        ref="main",
        patterns=(
            "dm_control/suite/*.xml",
            "dm_control/locomotion/**/*.xml",
            "dm_control/manipulation/**/*.xml",
        ),
        exclude=("test", "common", "assets/common"),
    ),
    "gym_robotics": Source(
        name="gym_robotics",
        url="https://github.com/Farama-Foundation/Gymnasium-Robotics.git",
        ref="main",
        patterns=("gymnasium_robotics/envs/**/*.xml",),
        exclude=("test", "shared", "assets/shared"),
    ),
    "mujoco_mpc": Source(
        name="mujoco_mpc",
        url="https://github.com/google-deepmind/mujoco_mpc.git",
        ref="main",
        patterns=("mjpc/tasks/**/task*.xml", "mjpc/tasks/**/*.xml"),
        exclude=("common", "test"),
        prefer=("task.xml",),
    ),
    "robosuite": Source(
        name="robosuite",
        url="https://github.com/ARISE-Initiative/robosuite.git",
        ref="master",
        patterns=("robosuite/models/assets/arenas/*.xml",
                  "robosuite/models/assets/scenes/*.xml"),
        exclude=("test",),
    ),
    "mjx": Source(
        name="mjx",
        url="https://github.com/google-deepmind/mujoco.git",
        ref="3.2.3",
        patterns=("mjx/**/test_data/**/*.xml", "mjx/**/data/**/*.xml"),
        exclude=("test_util",),
    ),
}


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
def _run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                          check=check, capture_output=True, text=True)


def shallow_clone_or_fetch(src: Source, dest: Path) -> Optional[str]:
    """Returns commit hash on success, None on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        try:
            _run(["git", "fetch", "--depth", "1", "origin", src.ref], cwd=dest)
            _run(["git", "checkout", "FETCH_HEAD"], cwd=dest)
        except subprocess.CalledProcessError as exc:
            print(f"  WARN fetch failed for {src.name}: {exc.stderr[:200]}", file=sys.stderr)
            return None
    else:
        try:
            _run(["git", "clone", "--depth", "1", "--branch", src.ref,
                  src.url, str(dest)])
        except subprocess.CalledProcessError as exc:
            # Some refs are commits, not branches -> fall back to default + checkout.
            print(f"  branch clone failed, trying default: {exc.stderr[:200]}", file=sys.stderr)
            try:
                _run(["git", "clone", "--depth", "1", src.url, str(dest)])
                _run(["git", "fetch", "--depth", "1", "origin", src.ref], cwd=dest)
                _run(["git", "checkout", "FETCH_HEAD"], cwd=dest)
            except subprocess.CalledProcessError as exc2:
                print(f"  ERROR clone failed for {src.name}: {exc2.stderr[:200]}", file=sys.stderr)
                return None
    try:
        out = _run(["git", "rev-parse", "HEAD"], cwd=dest)
        return out.stdout.strip()[:12]
    except subprocess.CalledProcessError:
        return None


# ---------------------------------------------------------------------------
# MJCF processing
# ---------------------------------------------------------------------------
def _walk_glob(root: Path, pattern: str) -> List[Path]:
    return sorted(root.glob(pattern))


def candidate_xmls(repo_root: Path, src: Source) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for pat in src.patterns:
        for p in _walk_glob(repo_root, pat):
            if not p.is_file():
                continue
            if any(ex in str(p) for ex in src.exclude):
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    # Stable preferred-first ordering.
    if src.prefer:
        out.sort(key=lambda p: 0 if any(pr in str(p) for pr in src.prefer) else 1)
    return out


def _is_top_level_mjcf(path: Path) -> bool:
    """Cheap filter: skip files that are obviously include-fragments
    (no <mujoco> root, or root has only <asset>/<default>/<contact>)."""
    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError:
        return False
    root = tree.getroot()
    if root.tag != "mujoco":
        return False
    children = {c.tag for c in root}
    # A real scene either has a worldbody or is so atomic it still compiles.
    return "worldbody" in children or {"compiler", "option"} & children == {"compiler", "option"}


def try_compile(path: Path) -> Tuple[bool, str]:
    import mujoco  # local import: heavy
    try:
        mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
    return True, ""


def model_meta(path: Path) -> Dict:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    root = etree.parse(str(path)).getroot()
    return {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "nbody": int(m.nbody), "ngeom": int(m.ngeom),
        "njnt": int(m.njnt), "neq": int(m.neq), "ntendon": int(m.ntendon),
        "has_actuator": root.find("actuator") is not None,
        "has_equality": root.find("equality") is not None,
        "has_tendon": root.find("tendon") is not None,
    }


def slug(path: Path, repo_root: Path) -> str:
    rel = path.relative_to(repo_root)
    s = str(rel).replace(os.sep, "_").replace("/", "_")
    if s.endswith(".xml"):
        s = s[:-4]
    s = s.replace(".", "_")
    return s


def copy_assets_for(xml_path: Path, dest_dir: Path) -> None:
    """Copy the XML's directory `assets/` (if any) into dest_dir/<slug>_assets/.

    Many menagerie / robosuite scenes have `meshdir="assets"` or
    `texturedir="assets"` and rely on co-located mesh/texture files.
    We mirror the parent directory's `assets/` folder verbatim so the
    relative paths still resolve when MuJoCo loads the curated copy.
    """
    parent = xml_path.parent
    candidates = ["assets", "meshes", "textures"]
    for sub in candidates:
        src_dir = parent / sub
        if src_dir.is_dir():
            dst = dest_dir / sub
            if not dst.exists():
                shutil.copytree(src_dir, dst, dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fetch_one(src: Source, workdir: Path, out_dir: Path, cap: int,
              pool_dir: Path, manifest: List[Dict]) -> int:
    print(f"\n=== {src.name} ({src.url} @ {src.ref}) ===")
    repo_dir = workdir / src.name
    commit = shallow_clone_or_fetch(src, repo_dir)
    if commit is None:
        print(f"  skipped: clone failure")
        return 0
    print(f"  commit={commit}")

    # Copy LICENSE if present.
    for lic_name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        p = repo_dir / lic_name
        if p.is_file():
            shutil.copyfile(p, out_dir / f"{src.name}__LICENSE")
            break

    cands = candidate_xmls(repo_dir, src)
    print(f"  {len(cands)} XML candidates")

    n_curated = 0
    for path in cands:
        if not _is_top_level_mjcf(path):
            continue
        ok, err = try_compile(path)
        if not ok:
            continue
        target_dir = out_dir if n_curated < cap else pool_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        s = slug(path, repo_dir)
        # Copy XML preserving relative directory shape under a slug folder so
        # local <include> and asset references continue to resolve.
        sub_dir = target_dir / f"{src.name}__{s}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        new_xml = sub_dir / "model.xml"
        shutil.copyfile(path, new_xml)
        copy_assets_for(path, sub_dir)
        # Re-verify the curated copy compiles standalone.
        ok2, err2 = try_compile(new_xml)
        if not ok2:
            shutil.rmtree(sub_dir, ignore_errors=True)
            continue
        try:
            meta = model_meta(new_xml)
        except Exception as exc:  # noqa: BLE001
            meta = {"meta_error": str(exc)[:200]}
        manifest.append({
            "source": src.name,
            "repo": src.url,
            "commit": commit,
            "src_rel_path": str(path.relative_to(repo_dir)),
            "curated_path": str(new_xml.relative_to(out_dir.parent)),
            "in_curated": (target_dir == out_dir),
            **meta,
        })
        if target_dir == out_dir:
            n_curated += 1
    print(f"  -> curated {n_curated} (cap={cap})")
    return n_curated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".cache/repos")
    ap.add_argument("--out", default="seeds/curated")
    ap.add_argument("--pool", default="seeds/_pool")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of source names")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    caps = cfg["seeds"]["per_source_cap"]

    workdir = Path(args.workdir)
    out_dir = Path(args.out)
    pool_dir = Path(args.pool)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    selected = SOURCES.keys() if not args.only else args.only.split(",")
    manifest: List[Dict] = []
    total = 0
    for name in selected:
        if name not in SOURCES:
            print(f"unknown source: {name}", file=sys.stderr)
            continue
        total += fetch_one(SOURCES[name], workdir, out_dir,
                           caps.get(name, 5), pool_dir, manifest)

    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nTOTAL curated: {total}")
    print(f"Manifest: {out_dir / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
