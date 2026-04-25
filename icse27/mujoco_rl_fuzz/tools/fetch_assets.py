"""Asset fetcher: clone curated upstream/downstream MuJoCo asset repos and build
a curated seed pool. Includes a *compile probe* that runs each XML through
mujoco.MjModel.from_xml_path inside a subprocess (5s timeout) and quarantines
the failures.

Run:
    python -m tools.fetch_assets --root . --sources P0           # P0 only
    python -m tools.fetch_assets --root . --sources P0 P1 P2     # full
    python -m tools.fetch_assets --probe                         # re-probe only

Requires `git` on PATH. Network access required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------- source registry ----------
# Each source is documented with PRIORITY in {P0,P1,P2,P3} and an "include"
# glob list. We keep `git clone --depth=1` for speed.

@dataclass
class Source:
    name: str
    priority: str
    git: str
    include_globs: list[str] = field(default_factory=lambda: ["**/*.xml"])
    exclude_globs: list[str] = field(default_factory=list)
    extra_assets: list[str] = field(default_factory=lambda: ["**/*.stl", "**/*.obj",
                                                              "**/*.msh", "**/*.png"])
    note: str = ""


SOURCES: list[Source] = [
    Source("upstream_models", "P0",
           git="",  # in-repo, no clone needed
           note="In-repo MuJoCo `model/` directory; copied locally."),
    Source("menagerie", "P0",
           git="https://github.com/google-deepmind/mujoco_menagerie.git",
           include_globs=["**/*.xml"],
           exclude_globs=["**/test/**", "**/tests/**"],
           note="Industrial robots, ~50 models. High structural complexity."),
    Source("dm_control", "P0",
           git="https://github.com/google-deepmind/dm_control.git",
           include_globs=["dm_control/suite/*.xml",
                          "dm_control/manipulation/**/*.xml",
                          "dm_control/locomotion/**/*.xml"],
           note="Suite/manipulation/locomotion XMLs."),
    Source("gymnasium_robotics", "P1",
           git="https://github.com/Farama-Foundation/Gymnasium-Robotics.git",
           include_globs=["gymnasium_robotics/envs/**/*.xml"],
           note="Fetch / Hand / AntMaze; contact-heavy."),
    Source("robosuite", "P1",
           git="https://github.com/ARISE-Initiative/robosuite.git",
           include_globs=["robosuite/models/assets/**/*.xml"],
           note="Multi-arm manipulation; composite/tendon usage."),
    Source("mujoco_mpc", "P1",
           git="https://github.com/google-deepmind/mujoco_mpc.git",
           include_globs=["mjpc/tasks/**/*.xml"],
           note="Stiff control scenes; solver-stress candidates."),
    Source("isaaclab_mjcf", "P2",
           git="https://github.com/isaac-sim/IsaacLab.git",
           include_globs=["**/*.xml"],
           note="Large scenes (filter heavily)."),
    Source("dial_mpc", "P2",
           git="https://github.com/Improbable-AI/dial-mpc.git",
           include_globs=["**/*.xml"],
           note="Bipedal / hand; tendon/composite."),
    Source("mjctrl", "P2",
           git="https://github.com/kevinzakka/mjctrl.git",
           include_globs=["**/*.xml"],
           note="Academic demos; varied authoring style."),
]


# ---------- helpers ----------

def _glob_many(root: str, patterns: list[str]) -> list[str]:
    import fnmatch
    matched = []
    for dirpath, _, files in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        for f in files:
            rel = (rel_dir + "/" + f).lstrip("./")
            for p in patterns:
                # convert ** to fnmatch-friendly
                norm = p.replace("\\", "/")
                if fnmatch.fnmatch(rel, norm):
                    matched.append(os.path.join(dirpath, f))
                    break
    return matched


def _git_clone(src: Source, cache_dir: str) -> Optional[str]:
    if not src.git:
        return None
    dst = os.path.join(cache_dir, src.name)
    if os.path.exists(dst):
        print(f"[fetch] {src.name}: cache hit -> {dst}")
        return dst
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[fetch] cloning {src.git} -> {dst}")
    rc = subprocess.run(["git", "clone", "--depth=1", src.git, dst]).returncode
    if rc != 0:
        print(f"[fetch] FAILED to clone {src.name}", file=sys.stderr)
        return None
    return dst


def _compile_probe(xml_path: str, project_root: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
    """Run a worker subprocess to compile-only the XML (rollout_steps=0)."""
    spec = {"xml_path": xml_path, "rollout_steps": 0}
    import tempfile, uuid
    nonce = uuid.uuid4().hex[:8]
    tmp = tempfile.gettempdir()
    in_p = os.path.join(tmp, f"probe_in_{nonce}.json")
    out_p = os.path.join(tmp, f"probe_out_{nonce}.json")
    with open(in_p, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    cmd = [sys.executable, "-m", "src.engine.subprocess_worker",
           "--input", in_p, "--output", out_p]
    try:
        proc = subprocess.run(cmd, cwd=project_root, capture_output=True,
                              timeout=timeout_sec, text=True)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if not os.path.exists(out_p):
        return False, f"no_output (rc={proc.returncode})"
    try:
        with open(out_p, "r", encoding="utf-8") as f:
            r = json.load(f)
    except Exception as e:
        return False, f"parse_err:{e}"
    finally:
        for p in (in_p, out_p):
            try: os.remove(p)
            except OSError: pass
    if r.get("compile", {}).get("ok"):
        return True, "ok"
    return False, r.get("compile", {}).get("exception_type") or "unknown"


def _harvest_one(src: Source, project_root: str, cache_dir: str,
                 pool_dir: str) -> list[str]:
    """Returns list of XML paths copied into pool_dir."""
    if src.name == "upstream_models":
        # Copy from project_root/../model (the in-tree MuJoCo model/ dir)
        candidate = os.path.normpath(os.path.join(project_root, "..", "..", "model"))
        if not os.path.isdir(candidate):
            print(f"[harvest] upstream_models: dir not found at {candidate}, skip.")
            return []
        repo_root = candidate
    else:
        repo_root = _git_clone(src, cache_dir)
        if repo_root is None:
            return []

    xmls = _glob_many(repo_root, src.include_globs)
    if src.exclude_globs:
        excluded = set(_glob_many(repo_root, src.exclude_globs))
        xmls = [x for x in xmls if x not in excluded]

    out_paths = []
    src_pool = os.path.join(pool_dir, src.name)
    os.makedirs(src_pool, exist_ok=True)

    # Copy XMLs preserving relative path; also copy support assets in the same dir
    for xml in xmls:
        rel = os.path.relpath(xml, repo_root)
        dst = os.path.join(src_pool, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(xml, dst)
        out_paths.append(dst)

    # Bulk-copy mesh/texture neighbours by directory
    seen_dirs = {os.path.dirname(x) for x in xmls}
    for d in seen_dirs:
        rel_d = os.path.relpath(d, repo_root)
        out_d = os.path.join(src_pool, rel_d)
        for f in os.listdir(d):
            ext = os.path.splitext(f)[1].lower()
            if ext in (".stl", ".obj", ".msh", ".png", ".jpg", ".dae"):
                src_f = os.path.join(d, f)
                dst_f = os.path.join(out_d, f)
                if not os.path.exists(dst_f):
                    try: shutil.copy2(src_f, dst_f)
                    except Exception: pass

    print(f"[harvest] {src.name}: {len(out_paths)} XMLs copied")
    return out_paths


# ---------- main ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root (the mujoco_rl_fuzz dir).")
    ap.add_argument("--sources", nargs="+", default=["P0"],
                    help="Priorities to include: P0/P1/P2/P3, or specific source names.")
    ap.add_argument("--cache", default=".cache/assets")
    ap.add_argument("--pool", default="seeds/_pool")
    ap.add_argument("--curated", default="seeds/curated")
    ap.add_argument("--quarantine", default="seeds/_quarantine")
    ap.add_argument("--probe", action="store_true",
                    help="Skip cloning; only re-probe existing pool.")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    cache_dir = os.path.join(root, args.cache)
    pool_dir = os.path.join(root, args.pool)
    curated_dir = os.path.join(root, args.curated)
    quar_dir = os.path.join(root, args.quarantine)
    for d in (cache_dir, pool_dir, curated_dir, quar_dir):
        os.makedirs(d, exist_ok=True)

    if not args.probe:
        wanted = []
        for s in SOURCES:
            if s.priority in args.sources or s.name in args.sources:
                wanted.append(s)
        print(f"[fetch_assets] selected {len(wanted)} sources")
        for s in wanted:
            try:
                _harvest_one(s, root, cache_dir, pool_dir)
            except Exception as e:
                print(f"[harvest] {s.name} ERROR: {e}", file=sys.stderr)

    # Probe
    if args.no_probe:
        return 0
    manifest = []
    n_ok = n_bad = 0
    t0 = time.time()
    for src_name in sorted(os.listdir(pool_dir)):
        src_pool = os.path.join(pool_dir, src_name)
        if not os.path.isdir(src_pool):
            continue
        for dirpath, _, files in os.walk(src_pool):
            for f in files:
                if not f.endswith(".xml"):
                    continue
                p = os.path.join(dirpath, f)
                ok, reason = _compile_probe(p, root)
                rel = os.path.relpath(p, src_pool)
                target_root = curated_dir if ok else quar_dir
                dst = os.path.join(target_root, src_name, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    try: shutil.copy2(p, dst)
                    except Exception: pass
                manifest.append({"source": src_name, "rel": rel.replace("\\", "/"),
                                 "ok": ok, "reason": reason})
                n_ok += int(ok); n_bad += int(not ok)
                if (n_ok + n_bad) % 25 == 0:
                    print(f"[probe] +{n_ok+n_bad}  ok={n_ok}  bad={n_bad}  "
                          f"({(n_ok+n_bad)/max(time.time()-t0,1e-6):.1f}/s)")

    with open(os.path.join(curated_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.time(), "items": manifest,
                   "n_ok": n_ok, "n_bad": n_bad}, f, indent=2)
    print(f"[probe] DONE  ok={n_ok}  bad={n_bad}  manifest -> {curated_dir}/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
