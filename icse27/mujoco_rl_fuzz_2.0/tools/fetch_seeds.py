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
            "dm_control/locomotion/walkers/assets/*.xml",
            "dm_control/locomotion/walkers/assets/dog_v2/*.xml",
            "dm_control/locomotion/walkers/assets/fruitfly_v2/*.xml",
            "dm_control/locomotion/walkers/assets/rodent_v2/*.xml",
            "dm_control/locomotion/soccer/assets/**/*.xml",
            "dm_control/locomotion/arenas/assets/**/*.xml",
            "dm_control/manipulation/**/*.xml",
            "dm_control/entities/**/*.xml",
        ),
        exclude=("test", "common", "build_", "_defaults", "include"),
        prefer=("scene", "soccer", "humanoid_CMU", "dog", "rodent"),
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
        patterns=(
            "mjpc/tasks/**/task*.xml",
            "mjpc/tasks/**/scene*.xml",
            "mjpc/tasks/**/*.xml",
        ),
        exclude=("common_assets", "common.xml", "test", "include"),
        prefer=("task.xml", "quadruped", "humanoid", "panda", "manipulation"),
    ),
    "robosuite": Source(
        name="robosuite",
        url="https://github.com/ARISE-Initiative/robosuite.git",
        ref="master",
        patterns=(
            "robosuite/models/assets/arenas/*.xml",
            "robosuite/models/assets/scenes/*.xml",
            "robosuite/models/assets/objects/*.xml",
            "robosuite/models/assets/robots/*/robot.xml",
            "robosuite/models/assets/grippers/*.xml",
        ),
        exclude=("test", "shared"),
        prefer=("arena", "scene", "multi", "two_arm"),
    ),
    "mjx": Source(
        name="mjx",
        url="https://github.com/google-deepmind/mujoco.git",
        ref="3.2.3",
        patterns=("mjx/**/test_data/**/*.xml", "mjx/**/data/**/*.xml"),
        exclude=("test_util",),
    ),
    # ---- realistic / long-horizon manipulation ----------------------------
    "robocasa": Source(
        name="robocasa",
        url="https://github.com/robocasa/robocasa.git",
        ref="main",
        patterns=(
            "robocasa/models/assets/scenes/**/*.xml",
            "robocasa/models/assets/fixtures/**/*.xml",
            "robocasa/models/assets/objects/**/*.xml",
            "robocasa/models/assets/arenas/**/*.xml",
        ),
        exclude=("test", "shared", "_defaults", "include", "common"),
        prefer=("kitchen", "scene", "stove", "sink", "microwave", "fridge"),
    ),
    "libero": Source(
        name="libero",
        url="https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        ref="master",
        patterns=(
            "libero/libero/assets/scenes/**/*.xml",
            "libero/libero/assets/articulated_objects/**/*.xml",
            "libero/libero/assets/stable_hope_objects/**/*.xml",
        ),
        exclude=("test", "shared", "common", "include"),
        prefer=("scene", "kitchen", "study", "living"),
    ),
    "mimicgen": Source(
        name="mimicgen",
        url="https://github.com/NVlabs/mimicgen.git",
        ref="main",
        patterns=(
            "mimicgen/models/robosuite/assets/**/*.xml",
            "mimicgen/envs/robosuite/**/*.xml",
        ),
        exclude=("test", "shared", "common", "include"),
        prefer=("scene", "arena", "coffee", "stack", "threading", "square"),
    ),
    "safety_gymnasium": Source(
        name="safety_gymnasium",
        url="https://github.com/PKU-Alignment/safety-gymnasium.git",
        ref="main",
        patterns=(
            "safety_gymnasium/assets/xmls/**/*.xml",
            "safety_gymnasium/tasks/**/assets/**/*.xml",
        ),
        exclude=("test", "shared", "common"),
        prefer=("car", "point", "doggo", "racecar"),
    ),
    "myosuite": Source(
        name="myosuite",
        url="https://github.com/MyoHub/myosuite.git",
        ref="main",
        patterns=(
            "myosuite/simhive/myo_sim/**/*.xml",
            "myosuite/envs/myo/assets/**/*.xml",
        ),
        exclude=("test", "shared", "common", "include", "_defaults"),
        prefer=("hand", "arm", "leg", "finger", "elbow"),
    ),
    # ---- new actor-rich open source registries -----------------------------
    # NOTE: mujoco_playground's locomotion/manipulation XMLs reference
    # external mesh assets that the upstream installs at pip-install time
    # (no .gitmodules). We only pick up the dm_control_suite subset which
    # ships meshes inline and compiles standalone.
    "mujoco_playground": Source(
        name="mujoco_playground",
        url="https://github.com/google-deepmind/mujoco_playground.git",
        ref="main",
        patterns=(
            "mujoco_playground/_src/dm_control_suite/xmls/*.xml",
        ),
        exclude=("common", "materials", "skybox", "visual"),
        prefer=("humanoid", "cheetah", "walker", "hopper", "manipulator"),
    ),
    # NOTE: ROBEL and OpenAI Robogym were evaluated but their MJCF rely on
    # cross-directory <include> graphs (robel-scenes/dependencies*.xml,
    # robogym/assets/xmls/object/*) that do not survive a flat slug-folder
    # copy. Adding them would require a repo-subtree copier; deferred.
}


# ---------------------------------------------------------------------------
# Actor classification: a seed lands in the curated tier only if it has
# actuators, a non-trivial body tree, and is NOT a fixture / scene fragment /
# loose object. Everything else compiles fine but goes to seeds/_assets/ for
# later use as scene props (L1 scenario layer).
# ---------------------------------------------------------------------------
ACTOR_MIN_NU = 1
ACTOR_MIN_NBODY = 4
import re as _re
_NONACTOR_NAME_RE = _re.compile(
    r"(composed_|articulated_objects|stable_hope_objects|shapenet_|fixtures?_|"
    r"scenes?_|arenas?_|backgrounds?_|wall_painting|book_shelf|table_|"
    r"-visual$|_visual$|_vis$|outlets?_|switches?_|counter_|empty_|"
    r"objects_(?:coffee|bread|can|cereal|milk|drawer|serving|mug)|"
    r"test_data_(?:tendon|sensor|pendula|ray|convex|constraints))",
    _re.I,
)
# Demo families that compile but are physics-engine showcases, not robots.
_DEMO_FAMILY_RE = _re.compile(
    r"(replicate|elasticity|sdf_|balloons?|hammock)", _re.I,
)
_ACTOR_NAME_HINT_RE = _re.compile(
    r"(panda|franka|spot|anymal|humanoid|hand|arm|dog|ant|cassie|walker|tiago|"
    r"kinova|sawyer|xarm|iiwa|go2|go1|barkour|h1|g1|z1|a1|fr3|ur5e|ur10e|"
    r"allegro|shadow|leap|fruitfly|fly|rodent|piper|talos|pendulum|cheetah|"
    r"hopper|swimmer|reacher|pusher|drone|crazyflie|toddlerbot|robot|"
    r"tendon_arm|slider_crank|gripper|finger|softfoot|op3|tidybot|jaco|gr1|"
    r"trossen|aero|wonik|flexiv|rizon|kuka|trs_so_arm|ufactory|lite6)",
    _re.I,
)
# nbody/nu above this threshold strongly suggests soft body / replicated demo
# rather than an articulated robot (real robots: humanoid 17/21≈0.8, dog 63/38≈1.7).
_ACTOR_MAX_NBODY_PER_NU = 30


def _xml_has_softbody(xml_path: Path) -> bool:
    """True if the MJCF (or any of its includes) declares <flex>/<flexcomp>/<composite>."""
    soft_tags = ("flex", "flexcomp", "composite")
    seen = {str(xml_path).lower()}
    try:
        stack = [(xml_path, etree.parse(str(xml_path)).getroot())]
    except etree.XMLSyntaxError:
        return False
    while stack:
        path, root = stack.pop()
        for tag in soft_tags:
            for _ in root.iter(tag):
                return True
        for inc in root.iter("include"):
            ref = inc.get("file")
            if not ref or Path(ref).is_absolute():
                continue
            p2 = (path.parent / ref).resolve()
            k = str(p2).lower()
            if k in seen or not p2.is_file():
                continue
            seen.add(k)
            try:
                stack.append((p2, etree.parse(str(p2)).getroot()))
            except etree.XMLSyntaxError:
                continue
    return False


# Body-name pattern indicating manual replication of an actor:
# e.g. "1a_torso", "2a_torso", "3a_torso" or "1_link", "2_link" -> stack-of-clones,
# typical of mjx test_data/humanoid/0N_humanoids.xml benchmark packs.
_REPLICA_PREFIX_RE = _re.compile(r"^(\d+[a-z]?_)(.+)$")


def _xml_has_replica_stack(xml_path: Path) -> bool:
    """True if the model contains the manual-replication body-naming pattern,
    or a `<replicate>` tag at the top level."""
    try:
        root = etree.parse(str(xml_path)).getroot()
    except etree.XMLSyntaxError:
        return False
    if any(True for _ in root.iter("replicate")):
        return True
    suffix_count: Dict[str, int] = {}
    prefixes = set()
    for body in root.iter("body"):
        name = body.get("name", "")
        m = _REPLICA_PREFIX_RE.match(name)
        if m:
            prefixes.add(m.group(1))
            suffix_count[m.group(2)] = suffix_count.get(m.group(2), 0) + 1
    if not suffix_count:
        return False
    # 2+ distinct numeric prefixes AND some body suffix repeats 2+ times
    # => the same skeleton was duplicated under different prefixes.
    return len(prefixes) >= 2 and max(suffix_count.values()) >= 2


def is_actor_seed(slug_name: str, meta: Dict, xml_path: Optional[Path] = None) -> bool:
    """Decide whether a seed belongs in the actor curated tier.

    Hard demotions (regardless of stats):
      * composed_*  ← synthetic
      * NONACTOR_NAME pattern  ← obvious fixture/scene/object
      * DEMO_FAMILY pattern    ← engine showcase (elasticity/replicate/hammock/...)
      * <flex|flexcomp|composite> in XML  ← soft body, unless name strongly says robot
      * <replicate> tag or "Na_<bodyname>" repeated body-name pattern ← stacked clones
      * nbody/nu > 30          ← far too many bodies per actuator for a real robot
    """
    if slug_name.startswith("composed_"):
        return False
    if _NONACTOR_NAME_RE.search(slug_name):
        return False
    if _DEMO_FAMILY_RE.search(slug_name):
        return False
    nu = int(meta.get("nu", 0))
    nb = int(meta.get("nbody", 0))
    nj = int(meta.get("njnt", 0))
    name_hint = bool(_ACTOR_NAME_HINT_RE.search(slug_name))
    # Soft-body / replica-stack / extreme ratio overrides unless name is a clear robot.
    if not name_hint:
        if nu > 0 and nb / nu > _ACTOR_MAX_NBODY_PER_NU:
            return False
        if xml_path is not None and _xml_has_softbody(xml_path):
            return False
    # Replica-stack rule applies even to robot-named seeds (a stack of pandas
    # is still a benchmark, not a single actor we want to fuzz).
    if xml_path is not None and _xml_has_replica_stack(xml_path):
        return False
    if nu >= ACTOR_MIN_NU and nb >= ACTOR_MIN_NBODY:
        return True
    if name_hint and nb >= 6 and nj >= 4:
        return True
    return False


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
        commit = out.stdout.strip()[:12]
    except subprocess.CalledProcessError:
        return None
    # Initialize any submodules (no-op when there are none). MyoSuite, dm_control's
    # locomotion assets, and a few menagerie robots use submodules; without this
    # the .xml files compile to FileNotFound.
    try:
        _run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=dest)
    except subprocess.CalledProcessError as exc:
        print(f"  WARN submodule init failed for {src.name}: {exc.stderr[:200]}", file=sys.stderr)
    return commit


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
    candidates = ("assets", "meshes", "textures", "shared", "common")
    for sub in candidates:
        src_dir = parent / sub
        if src_dir.is_dir():
            dst = dest_dir / sub
            if not dst.exists():
                shutil.copytree(src_dir, dst, dirs_exist_ok=True)


def _copy_includes_recursive(xml_path: Path, original_dir: Path,
                             dest_dir: Path, seen: Optional[set] = None) -> None:
    """Recursively copy every <include file=...> referenced by xml_path.

    Many menagerie scenes / mujoco_playground suite XMLs include sibling XML
    files (``<include file="piper.xml"/>``, ``<include file="./common/visual.xml"/>``).
    A flat copy of just the entry XML loses these. We walk the include graph,
    copy each referenced .xml to ``dest_dir/<rel_path>`` preserving the
    relative directory shape, and recurse into the copied files.
    Already-absolute include paths are left alone.
    """
    if seen is None:
        seen = set()
    try:
        tree = etree.parse(str(xml_path))
    except etree.XMLSyntaxError:
        return
    root = tree.getroot()
    for inc in root.iter("include"):
        ref = inc.get("file")
        if not ref or Path(ref).is_absolute():
            continue
        # Resolve against the ORIGINAL dir, not dest, since dest might be
        # missing the file we're trying to copy.
        src_inc = (original_dir / ref).resolve()
        if not src_inc.is_file():
            continue
        rel = Path(ref)
        dst_inc = (dest_dir / rel).resolve()
        try:
            dst_inc.relative_to(dest_dir.resolve())
        except ValueError:
            # Refuses ../../escape-style paths.
            continue
        key = str(src_inc).lower()
        if key in seen:
            continue
        seen.add(key)
        dst_inc.parent.mkdir(parents=True, exist_ok=True)
        if not dst_inc.exists():
            shutil.copyfile(src_inc, dst_inc)
        # Recurse into the included XML (its includes are relative to its own dir).
        _copy_includes_recursive(dst_inc, src_inc.parent, dst_inc.parent, seen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fetch_one(src: Source, workdir: Path, out_dir: Path, cap: int,
              pool_dir: Path, manifest: List[Dict],
              assets_dir: Optional[Path] = None) -> int:
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
    n_assets = 0
    for path in cands:
        if not _is_top_level_mjcf(path):
            continue
        ok, err = try_compile(path)
        if not ok:
            continue
        s = slug(path, repo_dir)
        slug_name = f"{src.name}__{s}"
        # Compile once on original to get meta -> decide actor vs asset BEFORE copy.
        try:
            meta_pre = model_meta(path)
        except Exception:  # noqa: BLE001
            continue
        actor = is_actor_seed(slug_name, meta_pre, xml_path=path)
        if actor:
            target_dir = out_dir if n_curated < cap else pool_dir
        else:
            target_dir = assets_dir if assets_dir is not None else pool_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        sub_dir = target_dir / slug_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        new_xml = sub_dir / "model.xml"
        shutil.copyfile(path, new_xml)
        copy_assets_for(path, sub_dir)
        _copy_includes_recursive(new_xml, path.parent, sub_dir)
        _absolutize_asset_dirs(new_xml, path.parent)
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
            "is_actor": actor,
            **meta,
        })
        if target_dir == out_dir:
            n_curated += 1
        elif assets_dir is not None and target_dir == assets_dir:
            n_assets += 1
    print(f"  -> curated {n_curated} (cap={cap})  assets {n_assets}")
    return n_curated


def _absolutize_asset_dirs(curated_xml: Path, original_dir: Path) -> None:
    """Inject absolute meshdir/texturedir/assetdir into <compiler>.

    Many upstream XMLs use relative paths like ``file="../textures/x.png"`` or
    ``file="kitchen_background/foo.png"``. After we copy the XML into a
    flattened ``seeds/curated/<name>/model.xml`` location those paths break.
    Rewriting every <texture/mesh file=> attribute is fragile, so instead we
    pin the compiler's *dir attributes to the original repo location. MJCF
    resolves relative file paths relative to those directories (or to the XML
    file when unset), so this fixes the paths in one shot — at the cost of
    coupling curated copies to ``.cache/repos`` staying around.
    """
    abs_dir = str(original_dir.resolve()).replace("\\", "/")
    try:
        tree = etree.parse(str(curated_xml))
    except etree.XMLSyntaxError:
        return
    root = tree.getroot()
    comp = root.find("compiler")
    if comp is None:
        comp = etree.SubElement(root, "compiler")
        root.insert(0, comp)
    # Don't clobber explicit absolute paths the upstream may have set, and
    # preserve relative subfolder mapping (e.g. ``meshdir="assets"`` must
    # become ``<abs_dir>/assets`` not just ``<abs_dir>``, otherwise mesh
    # files in upstream ``robot/assets/*.obj`` won't be found).
    for attr in ("meshdir", "texturedir", "assetdir"):
        v = comp.get(attr)
        if v is None:
            comp.set(attr, abs_dir)
        elif not Path(v).is_absolute():
            joined = (original_dir / v).resolve()
            comp.set(attr, str(joined).replace("\\", "/"))
    tree.write(str(curated_xml), encoding="utf-8", xml_declaration=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".cache/repos")
    ap.add_argument("--out", default="seeds/curated")
    ap.add_argument("--pool", default="seeds/_pool")
    ap.add_argument("--assets", default="seeds/_assets",
                    help="non-actor seeds (scenes/objects/fixtures) go here")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of source names")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    caps = cfg["seeds"]["per_source_cap"]

    workdir = Path(args.workdir)
    out_dir = Path(args.out)
    pool_dir = Path(args.pool)
    assets_dir = Path(args.assets)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    selected = SOURCES.keys() if not args.only else args.only.split(",")
    # Preserve existing manifest entries from sources NOT being refetched.
    existing: List[Dict] = []
    mpath = out_dir / "MANIFEST.json"
    if args.only and mpath.is_file():
        try:
            existing = [e for e in json.loads(mpath.read_text(encoding="utf-8"))
                        if e.get("source") not in set(selected)]
        except Exception:  # noqa: BLE001
            existing = []
    manifest: List[Dict] = list(existing)
    total = 0
    for name in selected:
        if name not in SOURCES:
            print(f"unknown source: {name}", file=sys.stderr)
            continue
        total += fetch_one(SOURCES[name], workdir, out_dir,
                           caps.get(name, 5), pool_dir, manifest,
                           assets_dir=assets_dir)

    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nTOTAL curated this run: {total}  (manifest entries: {len(manifest)})")
    print(f"Manifest: {out_dir / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
