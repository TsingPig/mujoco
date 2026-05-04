"""Local web GUI for launching the visualization CLIs.

Usage:
    python tools/viz_gui.py            # serves http://127.0.0.1:9000/
    python tools/viz_gui.py --port 9000 --no-open

Pure stdlib (http.server) — no Flask / pip needed. Each click POSTs to the
server, which spawns the matching `visualize_seed.py` / `visualize_mutator.py`
subprocess detached so the MuJoCo viewer window pops up locally.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mutations.registry import MUTATORS, MUTATOR_IDS  # noqa: E402

SEEDS_DIR = ROOT / "seeds" / "curated"
TOOLS_DIR = ROOT / "tools"
PYTHON = sys.executable

# Source -> {license SPDX, upstream repo, 中文用途说明}. All curated sources
# are permissive open-source. Composed seeds inherit the parent's license.
SOURCE_INFO: dict[str, dict[str, str]] = {
    "mujoco": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco",
        "desc":    "MuJoCo 官方仓库自带的示例模型（humanoid、car、cards、tendon_arm 等），用来覆盖引擎核心特性（关节、约束、tendon、复合体）的最小可复现场景。",
    },
    "menagerie": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco_menagerie",
        "desc":    "DeepMind 维护的高质量真实机器人模型库（Franka、UR5e、Spot、ANYmal、ALOHA、Shadow Hand 等），对真实硬件的几何/惯量/驱动器参数都做过校准。",
    },
    "dm_control": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/dm_control",
        "desc":    "DeepMind Control Suite——经典 RL benchmark（cartpole、cheetah、walker、quadruped 等），任务难度可控，适合做控制类策略 fuzz。",
    },
    "gym_robotics": {
        "license": "MIT",
        "repo":    "github.com/Farama-Foundation/Gymnasium-Robotics",
        "desc":    "Farama 维护的 Gymnasium-Robotics（含 Fetch、HandManipulate、PointMaze 等），主要是带稀疏/密集奖励的目标到达类操作任务。",
    },
    "mujoco_mpc": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco_mpc",
        "desc":    "DeepMind MJPC 自带的 MPC 任务模型（Cartpole-Swingup、Acrobot、Quadruped、Humanoid-Walk、Hand 等），约束/接触组合较多。",
    },
    "robosuite": {
        "license": "MIT",
        "repo":    "github.com/ARISE-Initiative/robosuite",
        "desc":    "ARISE 的模块化机械臂操作框架（Lift、Stack、PickPlace、NutAssembly、Door 等），单/双臂 + 多种 gripper，是接近真实工作流的桌面操作场景。",
    },
    "mjx": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco",
        "desc":    "MJX（MuJoCo 的 JAX 后端）官方示例模型，专门挑选了能在 GPU 批量并行下稳定的几何/约束子集。",
    },
    "robocasa": {
        "license": "MIT",
        "repo":    "github.com/robocasa/robocasa",
        "desc":    "RoboCasa（基于 robosuite 的大规模厨房模拟器）：100+ 原子任务（开微波炉、倒水、煎、搅拌、放盘子）+ long-horizon composite（“做早餐”串 5–10 步）。最接近真实「机械臂做饭」工作流的开源场景。",
    },
    "libero": {
        "license": "MIT",
        "repo":    "github.com/Lifelong-Robot-Learning/LIBERO",
        "desc":    "LIBERO：130 个长程操作任务（kitchen / study / living 三套场景），专为 lifelong learning 设计，xml 自带干扰物体，比 robosuite 单任务复杂得多。",
    },
    "mimicgen": {
        "license": "MIT",
        "repo":    "github.com/NVlabs/mimicgen",
        "desc":    "NVIDIA MimicGen：Coffee、Stack-Three、Threading、Square、Hammer-Cleanup、NutAssembly 等多步操作场景，含场景 arena 与 articulated 物体。",
    },
    "safety_gymnasium": {
        "license": "Apache-2.0",
        "repo":    "github.com/PKU-Alignment/safety-gymnasium",
        "desc":    "PKU 安全 RL benchmark（Car / Point / Doggo + hazards/pillars/vases），约束求解 + 多机器人 + 障碍物，对接触/约束 mutator 是好压测。",
    },
    "myosuite": {
        "license": "Apache-2.0",
        "repo":    "github.com/MyoHub/myosuite",
        "desc":    "MyoSuite（基于 MuJoCo 的肌肉骨骼仿真）：手 / 臂 / 腿 / 肘等高保真生物力学模型，含大量 tendon、site、equality 约束，对 tendon/equality 类 mutator 覆盖率极高。",
    },
    "mujoco_playground": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco_playground",
        "desc":    "MuJoCo Playground（DeepMind）：dm_control_suite 标准化版本（acrobot、cheetah、walker、humanoid、manipulator 等），与 dm_control 略有差异，便于做后端兼容性 fuzz。",
    },
    "composed": {
        "license": "inherits parent",
        "repo":    "tools/compose_arena.py",
        "desc":    "由 compose_arena.py 用 <replicate> 把若干父种子拼成的多实例 arena，用于增加 body/contact 数压测引擎与 mutator 的边界。",
    },
    "unknown": {
        "license": "?",
        "repo":    "?",
        "desc":    "未识别来源（种子目录前缀不在已知清单内）。",
    },
}

# 字段 → 中文解释（用于表头 / nq 等的悬浮 tooltip）。
FIELD_TOOLTIPS: dict[str, str] = {
    "name":       "种子目录名（curated/<source>__<slug>/model.xml 中的 <source>__<slug>）。",
    "source":     "种子来源仓库（鼠标悬浮在 source pill 上看仓库用途说明）。",
    "license":    "种子继承自上游仓库的开源许可证（Apache-2.0 / MIT 等）。",
    "nq":         "nq：广义坐标（generalized position）维度。即 qpos 向量长度，由所有关节贡献：hinge/slide=1，ball=4（四元数），free=7（位置 3 + 四元数 4）。",
    "nv":         "nv：广义速度（generalized velocity）维度。即 qvel 向量长度：hinge/slide=1，ball=3（角速度），free=6（线速度 3 + 角速度 3）。一般 nv ≤ nq。",
    "nu":         "nu：actuator（驱动器）数量。每个 <actuator> 标签贡献 1 个控制输入 ctrl[i]，决定可控自由度。",
    "nbody":      "nbody：刚体（body）数量，含 worldbody。决定运动学树规模与碰撞对上限。",
    "ngeom":      "ngeom：几何体（geom）数量，决定碰撞检测代价；contype/conaffinity 进一步过滤接触对。",
    "njnt":       "njnt：关节（joint）数量。free/ball/hinge/slide 各计 1 个 njnt，但贡献的 nq/nv 不同。",
    "neq":        "neq：等式约束（equality constraint）数量，如 connect、weld、joint、tendon、distance。会进入约束求解器。",
    "ntendon":    "ntendon：腱（tendon）数量。spatial/fixed 两种，可形成耦合驱动或长度约束。",
    "complexity": "复杂度评分 = nq + nv + 2·nbody + 2·ngeom + 5·nu。仅用于排序，越大越「重」。",
}

_MANIFEST_CACHE: dict | None = None
_SEED_CACHE: list[dict] | None = None


def _load_manifest() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE
    out: dict = {}
    p = SEEDS_DIR / "MANIFEST.json"
    if p.is_file():
        try:
            for entry in json.loads(p.read_text(encoding="utf-8")):
                cp = entry.get("curated_path", "")
                # curated_path is `curated\\<seed>\\model.xml`; key by <seed>.
                parts = cp.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    out[parts[1]] = entry
        except Exception as exc:  # noqa: BLE001
            print(f"[manifest] parse failed: {exc}", file=sys.stderr)
    _MANIFEST_CACHE = out
    return out


def _infer_source(seed_name: str) -> str:
    if seed_name.startswith("composed_"):
        return "composed"
    pre = seed_name.split("__", 1)[0]
    return pre if pre in SOURCE_INFO else "unknown"


def _upstream_url(repo: str, commit, src_rel_path):
    """Best-effort GitHub blob URL for the seed's original XML.

    Returns None when we can't honestly construct one (avoid hallucinated
    links). Falls back to repo root when commit/path are missing but the
    repo string still looks like a GitHub slug.
    """
    if not repo or not repo.startswith("github.com/"):
        return None
    base = f"https://{repo.rstrip('/')}"
    if not commit or not src_rel_path:
        return base
    rel = str(src_rel_path).replace("\\", "/")
    return f"{base}/blob/{commit}/{rel}"


def _stats_via_mujoco(xml_path: Path) -> dict | None:
    """Compute model stats by compiling. Used when MANIFEST.json lacks the seed
    (e.g. composed_ arenas)."""
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        return {
            "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
            "nbody": int(m.nbody), "ngeom": int(m.ngeom),
            "njnt": int(m.njnt), "neq": int(m.neq), "ntendon": int(m.ntendon),
            "has_actuator": bool(m.nu),
            "has_equality": bool(m.neq),
            "has_tendon": bool(m.ntendon),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[stats] {xml_path.name} failed: {exc}", file=sys.stderr)
        return None


def _complexity(stats: dict | None) -> int:
    """Single scalar score for sorting; favors many-DoF, many-contact scenes."""
    if not stats:
        return 0
    return (stats.get("nq", 0)
            + stats.get("nv", 0)
            + 2 * stats.get("nbody", 0)
            + 2 * stats.get("ngeom", 0)
            + 5 * stats.get("nu", 0))


def list_seeds(force: bool = False) -> list[dict]:
    global _SEED_CACHE
    if _SEED_CACHE is not None and not force:
        return _SEED_CACHE
    if not SEEDS_DIR.is_dir():
        _SEED_CACHE = []
        return _SEED_CACHE
    manifest = _load_manifest()
    out: list[dict] = []
    for xml in sorted(SEEDS_DIR.glob("*/model.xml")):
        name = xml.parent.name
        source = _infer_source(name)
        stats = manifest.get(name)
        if stats is None:
            stats = _stats_via_mujoco(xml) or {}
        license_id, repo = SOURCE_INFO.get(source, SOURCE_INFO["unknown"])["license"], \
                            SOURCE_INFO.get(source, SOURCE_INFO["unknown"])["repo"]
        # Pull upstream-traceability fields straight from MANIFEST when available;
        # these are what makes the per-seed "open in repo" link non-hallucinated.
        man_entry = stats if isinstance(stats, dict) else {}
        commit = man_entry.get("commit")
        src_rel_path = man_entry.get("src_rel_path")
        out.append({
            "name": name,
            "source": source,
            "license": license_id,
            "repo": repo,
            "commit": commit,
            "src_rel_path": src_rel_path,
            "upstream_url": _upstream_url(repo, commit, src_rel_path),
            "stats": {k: stats.get(k) for k in
                      ("nq", "nv", "nu", "nbody", "ngeom",
                       "njnt", "neq", "ntendon")},
            "complexity": _complexity(stats),
            "is_composed": name.startswith("composed_"),
        })
    out.sort(key=lambda s: -s["complexity"])  # default: hardest first
    _SEED_CACHE = out
    return out


def mutator_catalog() -> list[dict]:
    out = []
    for mid in MUTATOR_IDS:
        m = MUTATORS[mid]
        legal = [x for x in m.intensity_modes if x not in m.invalid_parseable_modes]
        invalid = list(m.invalid_parseable_modes)
        out.append({
            "id": mid,
            "runtime_only": bool(getattr(m, "runtime_only", False)),
            "legal": legal,
            "invalid": invalid,
        })
    return out


_CORPUS_CACHE: dict | None = None
_FINDINGS_CACHE: list | None = None


def _load_corpus_layers(force: bool = False) -> dict:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None and not force:
        return _CORPUS_CACHE
    try:
        from src.corpus import iter_manifest, SyntheticSceneSeed, OpenEnvSeed, TrajectorySeed
    except Exception:
        _CORPUS_CACHE = {"synthetic_scenes": [], "open_envs": [], "trajectory_seeds": []}
        return _CORPUS_CACHE
    result: dict = {"synthetic_scenes": [], "open_envs": [], "trajectory_seeds": []}
    sm = ROOT / "seeds/synthetic_scenes/manifest.jsonl"
    if sm.exists():
        for s in iter_manifest(str(sm)):
            if isinstance(s, SyntheticSceneSeed):
                mf = s.model_features or {}
                result["synthetic_scenes"].append({
                    "seed_id": s.seed_id,
                    "actor_seed_id": s.actor_seed_id or "",
                    "template_name": s.template_name or "",
                    "compile_status": s.compile_status or "",
                    "scene_xml": (s.scene_xml or "").replace("\\", "/"),
                    "nq": mf.get("nq", "?"), "nv": mf.get("nv", "?"),
                    "nbody": mf.get("nbody", "?"), "ngeom": mf.get("ngeom", "?"),
                    "nu": mf.get("nu", "?"),
                    "tags": s.tags or [],
                })
    em = ROOT / "seeds/open_envs/manifest.jsonl"
    if em.exists():
        for s in iter_manifest(str(em)):
            if isinstance(s, OpenEnvSeed):
                result["open_envs"].append({
                    "seed_id": s.seed_id,
                    "adapter_name": s.adapter_name or "",
                    "env_id": s.env_id or "",
                    "dependency_status": s.dependency_status or "",
                    "runnable_status": s.runnable_status or "",
                    "package_name": s.package_name or "",
                    "tags": s.tags or [],
                })
    tm = ROOT / "seeds/trajectory_seeds/manifest.jsonl"
    if tm.exists():
        for s in iter_manifest(str(tm)):
            if isinstance(s, TrajectorySeed):
                result["trajectory_seeds"].append({
                    "seed_id": s.seed_id,
                    "parent_seed_id": s.parent_seed_id or "",
                    "parent_layer": s.parent_layer or "",
                    "replay_status": s.replay_status or "",
                    "action_kind": (s.action_sequence_spec or {}).get("kind", "?"),
                    "horizon": (s.action_sequence_spec or {}).get("horizon", 0),
                    "tags": s.tags or [],
                })
    _CORPUS_CACHE = result
    return result


def _load_findings(force: bool = False) -> list:
    global _FINDINGS_CACHE
    if _FINDINGS_CACHE is not None and not force:
        return _FINDINGS_CACHE
    findings_root = ROOT / "findings"
    out: list = []
    if findings_root.exists():
        for fpath in sorted(findings_root.rglob("*.json")):
            try:
                d = json.loads(fpath.read_text(encoding="utf-8"))
                sev = sum(o.get("severity", 0) for o in d.get("oracle_signals", []))
                out.append({
                    "finding_id": d.get("finding_id", fpath.stem),
                    "seed_id": d.get("seed_id", ""),
                    "layer": d.get("layer", ""),
                    "signature": d.get("signature", ""),
                    "severity": round(sev, 2),
                    "oracle_signals": d.get("oracle_signals", []),
                    "notes": d.get("notes", {}),
                    "_path": str(fpath.relative_to(ROOT)).replace("\\", "/"),
                })
            except Exception:
                continue
    out.sort(key=lambda x: -x["severity"])
    _FINDINGS_CACHE = out
    return out


def spawn(cmd: list[str]) -> int:
    """Spawn a detached subprocess so the GUI request returns immediately."""
    print(f"[spawn] {' '.join(cmd)}")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        creationflags=creationflags,
        stdout=None, stderr=None,
    )
    return proc.pid


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>mujoco_rl_fuzz_2.0 · 可视化入口</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; padding: 24px; max-width: 1400px; }
  h1 { margin: 0 0 4px 0; }
  .sub { color: #888; margin-bottom: 18px; font-size: 13px; }
  .card { border: 1px solid #8884; border-radius: 10px; padding: 14px 16px;
          margin-bottom: 18px; background: #ffffff08; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 4px 0; }
  select, button, input[type=text] { font-size: 13px; padding: 5px 9px; border-radius: 6px;
                   border: 1px solid #8884; background: #ffffff10; color: inherit; }
  button { cursor: pointer; }
  button.primary { background: #2563eb; color: white; border-color: #2563eb; }
  button.primary:hover { background: #1d4ed8; }
  button.ghost { background: transparent; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .mut { border: 1px solid #8883; border-radius: 8px; padding: 8px 12px; }
  .mut h3 { margin: 0 0 4px 0; font-size: 13px; font-family: ui-monospace, Consolas, monospace; }
  .mut .modes { font-size: 11px; color: #888; margin-bottom: 4px; word-break: break-all; }
  #log { font-family: ui-monospace, Consolas, monospace; font-size: 12px;
         background: #0001; padding: 10px; border-radius: 8px; max-height: 200px;
         overflow: auto; white-space: pre-wrap; }
  .pill { display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 999px;
          margin-left: 4px; line-height: 1.5; }
  .pill.src    { background: #2563eb22; color: #2563eb; }
  .pill.lic    { background: #16a34a22; color: #16a34a; }
  .pill.warn   { background: #f59e0b22; color: #b45309; }
  .pill.composed { background: #8b5cf622; color: #8b5cf6; }
  .chip { display: inline-flex; align-items: center; gap: 4px; font-size: 11px;
          padding: 3px 9px; border-radius: 999px; cursor: pointer; user-select: none;
          border: 1px solid #8884; background: #ffffff10; margin: 2px 4px 2px 0; }
  .chip:hover { background: #2563eb18; border-color: #2563eb55; }
  .chip.on { background: #2563eb; color: white; border-color: #2563eb; }
  .chip .n { opacity: 0.7; font-variant-numeric: tabular-nums; }
  details > summary { cursor: pointer; font-weight: 600; font-size: 16px;
                      list-style: none; padding: 4px 0; margin: 0; }
  details > summary::before { content: '▸ '; display: inline-block; transition: transform .15s; }
  details[open] > summary::before { content: '▾ '; }
  .card > details > summary { border-radius: 6px; padding: 4px 6px; margin: -4px -6px; }
  .card > details > summary:hover { background: #8881; }
  .card > details[open] > summary { margin-bottom: 10px; }
  body.compact table.seeds td, body.compact table.seeds th { padding: 2px 6px; font-size: 12px; }
  .iconbtn { padding: 2px 6px; font-size: 11px; }
  mark { background: #fde047; color: inherit; padding: 0 1px; border-radius: 2px; }
  table.seeds { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.seeds th { text-align: left; font-weight: 500; color: #888;
                   border-bottom: 1px solid #8883; padding: 6px 6px; cursor: pointer;
                   user-select: none; }
  table.seeds th.active { color: inherit; }
  table.seeds td { padding: 6px 6px; border-bottom: 1px solid #8881; vertical-align: middle; }
  table.seeds tr:hover td { background: #8881; }
  .seed-name { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
  .num { text-align: right; font-variant-numeric: tabular-nums;
         font-family: ui-monospace, Consolas, monospace; }
  .num.bigC { font-weight: 600; color: #2563eb; }
  .toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
  .legend { font-size: 11px; color: #888; margin-top: 6px; }
</style>
</head>
<body>

<h1>mujoco_rl_fuzz_2.0 · 可视化入口</h1>
<div class="sub">点击按钮即在本机启动 <code>mujoco.viewer</code> 窗口。所有种子均为开源（Apache-2.0 / MIT）。鼠标悬浮表头/字段可看中文说明。</div>

<div class="card">
  <details id="src-details">
    <summary>⓪ 数据源说明（每个 source 是什么仓库）— 点击展开</summary>
    <table class="seeds" id="src-table" style="margin-top:10px">
      <thead>
        <tr>
          <th style="width:130px">source</th>
          <th style="width:100px">license</th>
          <th>upstream repo</th>
          <th>用途说明</th>
        </tr>
      </thead>
      <tbody id="src-tbody"></tbody>
    </table>
  </details>
</div>

<div class="card">
  <details open id="sec-seeds">
  <summary>① 种子（seed）<span class="pill src" id="seed-count"></span></summary>
  <div class="toolbar">
    <input type="text" id="seed-filter" placeholder="过滤名字 / 源..." style="min-width:200px"/>
    <label>排序：</label>
    <select id="seed-sort">
      <option value="complexity" selected>复杂度（默认，降序）</option>
      <option value="name">名字（A→Z）</option>
      <option value="nq">DOF / nq</option>
      <option value="nbody">body 数</option>
      <option value="ngeom">geom 数</option>
      <option value="nu">actuator 数</option>
      <option value="source">源</option>
    </select>
    <label><input type="checkbox" id="only-composed"/> 只看合成场景</label>
    <label><input type="checkbox" id="compact-mode"/> 紧凑模式</label>
    <button class="ghost" id="refresh-seeds">⟳ 刷新</button>
  </div>
  <div id="src-chips" style="margin-bottom:8px"></div>
  <table class="seeds">
    <thead>
      <tr>
        <th data-sort="name" title="__TT_name__">name</th>
        <th title="__TT_source__">source / license</th>
        <th class="num" data-sort="nq" title="__TT_nq__">nq</th>
        <th class="num" data-sort="nv" title="__TT_nv__">nv</th>
        <th class="num" data-sort="nbody" title="__TT_nbody__">nbody</th>
        <th class="num" data-sort="ngeom" title="__TT_ngeom__">ngeom</th>
        <th class="num" data-sort="nu" title="__TT_nu__">nu</th>
        <th class="num" data-sort="complexity" title="__TT_complexity__">复杂度</th>
        <th>动作</th>
      </tr>
    </thead>
    <tbody id="seeds"></tbody>
  </table>
  <div class="legend">
    复杂度 = nq + nv + 2·nbody + 2·ngeom + 5·nu。
    紫色「composed」徽章表示由 <code>tools/compose_arena.py</code> 用
    <code>&lt;replicate&gt;</code> 合成的多实例 arena（许可证继承父种子）。
  </div>
  </details>
</div>

<div class="card">
  <details open id="sec-mutators">
  <summary>② Mutator 前后对比</summary>
  <div class="toolbar">
    <label>种子：</label>
    <input type="text" id="mut-seed" list="mut-seed-list" placeholder="输入名字搜索 / 留空 = synthetic 合成种子" style="min-width:380px"/>
    <datalist id="mut-seed-list"></datalist>
    <button class="ghost" id="mut-seed-clear" title="清空 → 用 synthetic 合成种子">✕</button>
    <label><input type="checkbox" id="mut-no-before"/> 跳过 BEFORE</label>
    <input type="text" id="mut-filter" placeholder="过滤 mutator..." style="margin-left:auto;min-width:200px"/>
  </div>
  <div class="grid" id="muts"></div>
  </details>
</div>

<div class="card">
  <details open id="sec-corpus">
  <summary>③ 分层语料库 <small style="font-weight:400;color:#888">(L1 合成场景 / L2 开源环境 / L3 轨迹种子)</small></summary>
  <div class="toolbar">
    <button class="ghost corpus-tab-btn" data-tab="l1">L1 合成场景 <span class="pill src" id="l1-count"></span></button>
    <button class="ghost corpus-tab-btn" data-tab="l2">L2 开源环境 <span class="pill src" id="l2-count"></span></button>
    <button class="ghost corpus-tab-btn" data-tab="l3">L3 轨迹种子 <span class="pill src" id="l3-count"></span></button>
    <button class="ghost" id="refresh-corpus" title="重新读取 manifests">⟳ 刷新</button>
    <span style="margin-left:auto;display:flex;gap:6px">
      <button class="ghost" id="btn-validate" title="运行 validate_layered_corpus.py（弹出新终端）">📋 验证语料库</button>
      <button class="ghost" id="btn-rand-fuzz" title="run_random_fuzz.py --budget 20">🎲 随机 Fuzz×20</button>
      <button class="ghost" id="btn-rule-fuzz" title="run_rule_fuzz.py --budget 20">📐 规则 Fuzz×20</button>
    </span>
  </div>

  <div class="corpus-tab-pane" data-tab="l1">
    <div class="toolbar"><input type="text" id="l1-filter" placeholder="过滤 seed_id / template..." style="min-width:260px"/></div>
    <table class="seeds"><thead><tr>
      <th>seed_id</th><th>template</th><th>actor</th><th>compile</th>
      <th class="num" title="广义坐标维度">nq</th><th class="num">nbody</th><th class="num">nu</th>
      <th>动作</th>
    </tr></thead><tbody id="l1-tbody"></tbody></table>
  </div>

  <div class="corpus-tab-pane" data-tab="l2" style="display:none">
    <div class="toolbar"><input type="text" id="l2-filter" placeholder="过滤 env_id / adapter..." style="min-width:260px"/></div>
    <table class="seeds"><thead><tr>
      <th>env_id</th><th>adapter</th><th>依赖</th><th>可运行</th><th>tags</th>
    </tr></thead><tbody id="l2-tbody"></tbody></table>
  </div>

  <div class="corpus-tab-pane" data-tab="l3" style="display:none">
    <div class="toolbar"><input type="text" id="l3-filter" placeholder="过滤 seed_id / parent..." style="min-width:260px"/></div>
    <table class="seeds"><thead><tr>
      <th>seed_id</th><th>parent</th><th>layer</th><th>replay</th><th>action_kind</th><th class="num">horizon</th>
    </tr></thead><tbody id="l3-tbody"></tbody></table>
  </div>
  </details>
</div>

<div class="card">
  <details open id="sec-findings">
  <summary>④ Findings <span class="pill src" id="findings-count"></span></summary>
  <div class="toolbar">
    <input type="text" id="findings-filter" placeholder="过滤 finding_id / seed_id / layer..." style="min-width:280px"/>
    <button class="ghost" id="refresh-findings" style="margin-left:auto">⟳ 刷新</button>
  </div>
  <table class="seeds"><thead><tr>
    <th>finding_id</th><th>seed_id</th><th>layer</th><th>severity</th>
    <th>failed oracles</th><th>signature</th><th>path</th>
  </tr></thead><tbody id="findings-tbody"></tbody></table>
  <div class="legend">findings/ 目录下所有 .json 文件自动扫描，按 severity 降序排列。</div>
  </details>
</div>

<div class="card">
  <h2 style="margin-top:0">日志</h2>
  <div id="log">(等待操作)</div>
</div>

<script>
let SEEDS = __SEEDS__;
const MUTS = __MUTS__;
const SOURCE_INFO = __SOURCE_INFO__;
let CORPUS = __CORPUS__;
let FINDINGS = __FINDINGS__;

const $log = document.getElementById("log");
function log(msg) {
  const t = new Date().toLocaleTimeString();
  $log.textContent += `\\n[${t}] ${msg}`;
  $log.scrollTop = $log.scrollHeight;
}

async function launch(payload) {
  log("launching " + JSON.stringify(payload));
  try {
    const r = await fetch("/launch", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify(payload),
    });
    const j = await r.json();
    if (j.ok) log("→ pid=" + j.pid + "  cmd: " + j.cmd.join(" "));
    else log("✗ error: " + j.error);
  } catch (e) {
    log("✗ network error: " + e);
  }
}

// ------- seed table -------
const $tbody = document.getElementById("seeds");
const $count = document.getElementById("seed-count");
const $sort  = document.getElementById("seed-sort");
const $filter = document.getElementById("seed-filter");
const $onlyComposed = document.getElementById("only-composed");
const $compact = document.getElementById("compact-mode");
const $mutSeed = document.getElementById("mut-seed");
const $srcChips = document.getElementById("src-chips");

// Persisted UI state (localStorage so 9000 端口刷新不丢配置).
const LS_KEY = "mjfuzz_viz_state_v2";  // bumped: composedOnly default reset
const UI = Object.assign(
  {sort:"complexity", filter:"", composedOnly:false, compact:false, srcSel:[]},
  (() => { try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
           catch { return {}; } })()
);
function saveUI() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(UI)); } catch {}
}
$sort.value = UI.sort;
$filter.value = UI.filter;
$onlyComposed.checked = UI.composedOnly;
$compact.checked = UI.compact;
if (UI.compact) document.body.classList.add("compact");

function escHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
}
function highlight(text, q) {
  const t = escHtml(text);
  if (!q) return t;
  const i = t.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return t;
  return t.slice(0, i) + "<mark>" + t.slice(i, i+q.length) + "</mark>" + t.slice(i+q.length);
}

function getSortVal(s, key) {
  if (key === "name" || key === "source") return s[key];
  if (key === "complexity") return s.complexity;
  return (s.stats && s.stats[key]) || 0;
}

function renderSrcChips() {
  // 统计每个 source 的 seed 数量 → 渲染 chip。点击切换显隐。
  const counts = {};
  for (const s of SEEDS) counts[s.source] = (counts[s.source] || 0) + 1;
  const sources = Object.keys(counts).sort();
  $srcChips.innerHTML = "";
  // “全部” chip
  const all = document.createElement("span");
  all.className = "chip" + (UI.srcSel.length === 0 ? " on" : "");
  all.innerHTML = `\u2605 \u5168\u90e8 <span class="n">${SEEDS.length}</span>`;
  all.onclick = () => { UI.srcSel = []; saveUI(); renderSrcChips(); renderSeeds(); };
  $srcChips.appendChild(all);
  for (const src of sources) {
    const chip = document.createElement("span");
    const on = UI.srcSel.includes(src);
    chip.className = "chip" + (on ? " on" : "");
    const desc = (SOURCE_INFO[src] && SOURCE_INFO[src].desc) || src;
    chip.title = desc;
    chip.innerHTML = `${escHtml(src)} <span class="n">${counts[src]}</span>`;
    chip.onclick = () => {
      if (on) UI.srcSel = UI.srcSel.filter(x => x !== src);
      else UI.srcSel = [...UI.srcSel, src];
      saveUI(); renderSrcChips(); renderSeeds();
    };
    $srcChips.appendChild(chip);
  }
}

function renderSeeds() {
  const sortKey = $sort.value;
  const filt = $filter.value.trim().toLowerCase();
  const composedOnly = $onlyComposed.checked;
  let view = SEEDS.slice();
  if (UI.srcSel.length) view = view.filter(s => UI.srcSel.includes(s.source));
  if (filt) view = view.filter(s =>
      s.name.toLowerCase().includes(filt) ||
      s.source.toLowerCase().includes(filt));
  if (composedOnly) view = view.filter(s => s.is_composed);
  view.sort((a, b) => {
    const av = getSortVal(a, sortKey), bv = getSortVal(b, sortKey);
    if (typeof av === "string") return av.localeCompare(bv);
    return bv - av;  // numeric: descending
  });
  $count.textContent = view.length + " / " + SEEDS.length;
  $tbody.innerHTML = "";
  for (const s of view) {
    const tr = document.createElement("tr");
    const st = s.stats || {};
    const composedTag = s.is_composed ? '<span class="pill composed">composed</span>' : '';
    const srcDesc = (SOURCE_INFO[s.source] && SOURCE_INFO[s.source].desc) || s.source;
    const xmlPath = `seeds/curated/${s.name}/model.xml`;
    // Multi-line hover summary (newline-separated; browsers render literal \\n in title=...).
    const hoverLines = [
      `\u540d\u79f0\uff1a${s.name}`,
      `\u6765\u6e90\uff1a${s.source}\u3000\u8bb8\u53ef\uff1a${s.license}`,
      s.commit ? `\u4e0a\u6e38 commit\uff1a${s.commit}` : null,
      s.src_rel_path ? `\u4e0a\u6e38\u8def\u5f84\uff1a${s.src_rel_path}` : null,
      `\u590d\u6742\u5ea6\uff1a${s.complexity}\u3000nq=${st.nq ?? '?'} nv=${st.nv ?? '?'} nbody=${st.nbody ?? '?'} nu=${st.nu ?? '?'}`,
      `\u672c\u5730\u8def\u5f84\uff1a${xmlPath}`,
      s.upstream_url ? '' : '\u26a0 \u672a\u77e5\u4e0a\u6e38 URL\uff08\u53ef\u80fd\u4e3a composed/synthetic\uff09',
    ].filter(Boolean).join('\\n');
    const linkBtn = s.upstream_url
      ? `<button class="ghost iconbtn" data-act="link" title="\u5728 GitHub \u6253\u5f00\u4e0a\u6e38 XML\uff1a${escHtml(s.upstream_url)}">&#127760;</button>`
      : `<button class="ghost iconbtn" data-act="link" title="\u672a\u77e5\u4e0a\u6e38\u94fe\u63a5" disabled>&#127760;</button>`;
    tr.innerHTML = `
      <td title="${escHtml(hoverLines)}"><span class="seed-name">${highlight(s.name, filt)}</span> ${composedTag}</td>
      <td><span class="pill src" title="${escHtml(srcDesc)}">${escHtml(s.source)}</span><span class="pill lic" title="\u8bb8\u53ef\u8bc1\uff1a${escHtml(s.license)}">${escHtml(s.license)}</span></td>
      <td class="num" title="nq=${st.nq ?? "?"}">${st.nq ?? "?"}</td>
      <td class="num" title="nv=${st.nv ?? "?"}">${st.nv ?? "?"}</td>
      <td class="num" title="nbody=${st.nbody ?? "?"}">${st.nbody ?? "?"}</td>
      <td class="num" title="ngeom=${st.ngeom ?? "?"}">${st.ngeom ?? "?"}</td>
      <td class="num" title="nu=${st.nu ?? "?"}">${st.nu ?? "?"}</td>
      <td class="num bigC" title="\u590d\u6742\u5ea6\u8bc4\u5206\uff08\u8d8a\u5927\u8d8a\u91cd\uff09">${s.complexity}</td>
      <td>
        <button class="primary iconbtn" data-act="run" title="\u5b9e\u65f6\u8dd1\uff08mujoco.viewer.launch\uff09">\u25b6</button>
        <button class="ghost iconbtn" data-act="static" title="\u9759\u6001\u67e5\u770b\uff08\u4e0d\u6b65\u8fdb\uff09">\u25fb</button>
        <button class="ghost iconbtn" data-act="copy" title="\u590d\u5236 XML \u8def\u5f84\uff1a${escHtml(xmlPath)}">\u29c9</button>
        <button class="ghost iconbtn" data-act="mut" title="\u5728\u2461 mutator \u533a\u9009\u4e2d\u6b64\u79cd\u5b50">M</button>
        ${linkBtn}
        <button class="ghost iconbtn" data-act="del" title="\u7acb\u5373\u5220\u9664\u5e76\u79fb\u5165 seeds/_recycle/\uff08\u53ef\u624b\u52a8\u6062\u590d\uff09">&#128465;</button>
      </td>
    `;
    tr.querySelector('[data-act="run"]').onclick    = () => launch({tool:"seed", seed:s.name, static:false});
    tr.querySelector('[data-act="static"]').onclick = () => launch({tool:"seed", seed:s.name, static:true});
    tr.querySelector('[data-act="copy"]').onclick   = async () => {
      try { await navigator.clipboard.writeText(xmlPath); log("\u5df2\u590d\u5236: " + xmlPath); }
      catch { log("\u590d\u5236\u5931\u8d25\uff08\u8bf7\u624b\u52a8\u9009\u4e2d\uff09: " + xmlPath); }
    };
    tr.querySelector('[data-act="mut"]').onclick    = () => {
      $mutSeed.value = s.name; $mutSeed.scrollIntoView({behavior:"smooth", block:"center"});
      $mutSeed.focus(); log("mutator \u4e8c\u533a\u79cd\u5b50 \u2192 " + s.name);
    };
    if (s.upstream_url) {
      tr.querySelector('[data-act="link"]').onclick = () => {
        window.open(s.upstream_url, '_blank', 'noopener');
        log('[link] \u6253\u5f00\u4e0a\u6e38\uff1a' + s.upstream_url);
      };
    }
    tr.querySelector('[data-act="del"]').onclick = async () => {
      try {
        const r = await fetch('/api/delete?name=' + encodeURIComponent(s.name), {method:'POST'});
        const j = await r.json();
        if (j.ok) { log('[del] \u5df2\u56de\u6536\uff1a' + s.name + ' \u2192 ' + j.moved_to); await reloadSeeds(); }
        else     { log('\u5220\u9664\u5931\u8d25\uff1a' + (j.error || 'unknown')); }
      } catch (e) { log('\u5220\u9664\u8bf7\u6c42\u5f02\u5e38\uff1a' + e); }
    };
    $tbody.appendChild(tr);
  }
}

function rebuildMutSeedSelect() {
  const list = document.getElementById("mut-seed-list");
  list.innerHTML = "";
  // 按 source 分组排序，同源内按复杂度降序，方便手动浏览。
  const sorted = SEEDS.slice().sort((a, b) => {
    if (a.source !== b.source) return a.source.localeCompare(b.source);
    return b.complexity - a.complexity;
  });
  for (const s of sorted) {
    const o = document.createElement("option");
    o.value = s.name;
    o.label = `[${s.source}] c=${s.complexity}  nq=${s.stats.nq ?? "?"} nbody=${s.stats.nbody ?? "?"}`;
    list.appendChild(o);
  }
}

// ------- source info table -------
function renderSources() {
  const used = new Set(SEEDS.map(s => s.source));
  const $b = document.getElementById("src-tbody");
  $b.innerHTML = "";
  // 优先显示当前 seed 库里实际出现的 source，再展示其它已知 source（淡显）。
  const order = Object.keys(SOURCE_INFO);
  order.sort((a, b) => {
    const au = used.has(a) ? 0 : 1, bu = used.has(b) ? 0 : 1;
    if (au !== bu) return au - bu;
    return a.localeCompare(b);
  });
  for (const key of order) {
    const info = SOURCE_INFO[key];
    const inUse = used.has(key);
    const tr = document.createElement("tr");
    if (!inUse) tr.style.opacity = "0.5";
    const repoCell = (info.repo && info.repo.startsWith("github.com/"))
      ? `<a href="https://${info.repo}" target="_blank" rel="noopener">${info.repo}</a>`
      : info.repo;
    tr.innerHTML = `
      <td><span class="pill src">${key}</span>${inUse ? "" : ' <span class="pill warn" title="当前 seed 库中没有该来源的种子">未使用</span>'}</td>
      <td><span class="pill lic">${info.license}</span></td>
      <td style="font-family: ui-monospace, Consolas, monospace; font-size: 12px;">${repoCell}</td>
      <td>${info.desc}</td>
    `;
    $b.appendChild(tr);
  }
}

$sort.onchange = () => { UI.sort = $sort.value; saveUI(); renderSeeds(); };
$filter.oninput = () => { UI.filter = $filter.value; saveUI(); renderSeeds(); };
$onlyComposed.onchange = () => { UI.composedOnly = $onlyComposed.checked; saveUI(); renderSeeds(); };
$compact.onchange = () => {
  UI.compact = $compact.checked; saveUI();
  document.body.classList.toggle("compact", UI.compact);
};
document.querySelectorAll("table.seeds th[data-sort]").forEach(th => {
  th.onclick = () => { $sort.value = th.dataset.sort; UI.sort = $sort.value; saveUI(); renderSeeds(); };
});
document.getElementById("refresh-seeds").onclick = () => reloadSeeds();
async function reloadSeeds() {
  const r = await fetch("/api/state?refresh=1"); const j = await r.json();
  SEEDS = j.seeds; renderSrcChips(); renderSeeds(); rebuildMutSeedSelect(); renderSources();
  log("seeds reloaded ("+SEEDS.length+")");
}
document.getElementById("mut-seed-clear").onclick = () => {
  document.getElementById("mut-seed").value = "";
};

// ------- mutators -------
const $muts = document.getElementById("muts");
const $mutFilter = document.getElementById("mut-filter");

function renderMuts() {
  const filt = $mutFilter.value.trim().toLowerCase();
  $muts.innerHTML = "";
  MUTS.forEach(m => {
    if (filt && !m.id.toLowerCase().includes(filt)) return;
    const card = document.createElement("div");
    card.className = "mut";
    const tag = m.runtime_only ? '<span class="pill warn">runtime-only</span>' : '';
    const opts = ['<option value="">(全部合法强度)</option>',
                  ...m.legal.map(x => `<option value="${x}">${x}</option>`)].join("");
    card.innerHTML = `
      <h3>${m.id} ${tag}</h3>
      <div class="modes">legal: ${m.legal.join(", ") || "(none)"}${
        m.invalid.length ? "<br/>invalid_parseable: " + m.invalid.join(", ") : ""}</div>
      <div class="row">
        <select>${opts}</select>
        <button class="primary">▶ Before/After</button>
      </div>
    `;
    const sel = card.querySelector("select");
    card.querySelector("button").onclick = () => launch({
      tool: "mutator",
      mutator: m.id,
      seed: $mutSeed.value || null,
      intensity: sel.value || null,
      no_before: document.getElementById("mut-no-before").checked,
    });
    $muts.appendChild(card);
  });
}
$mutFilter.oninput = renderMuts;

renderSrcChips();
renderSeeds();
rebuildMutSeedSelect();
renderMuts();
renderSources();

// ─────────────────────────────────────────────
// ③ 分层语料库 (L1 / L2 / L3)
// ─────────────────────────────────────────────
function statusColor(v) {
  if (!v) return "#888";
  if (v === "ok" || v === "runnable") return "#16a34a";
  if (v === "partial" || v === "unknown") return "#d97706";
  if (v === "missing" || v === "failed" || v === "broken") return "#dc2626";
  return "#888";
}
function statusPill(v) {
  const c = statusColor(v);
  return `<span style="display:inline-block;padding:1px 7px;border-radius:999px;font-size:10px;background:${c}22;color:${c}">${escHtml(v||"?")}</span>`;
}

const $tabBtns = document.querySelectorAll(".corpus-tab-btn");
const $tabPanes = document.querySelectorAll(".corpus-tab-pane");

function switchTab(name) {
  $tabBtns.forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  $tabPanes.forEach(p => p.style.display = p.dataset.tab === name ? "" : "none");
}
$tabBtns.forEach(b => b.onclick = () => switchTab(b.dataset.tab));

function renderL1() {
  const tbody = document.getElementById("l1-tbody");
  const filt = document.getElementById("l1-filter").value.trim().toLowerCase();
  const items = CORPUS.synthetic_scenes.filter(s =>
    !filt || s.seed_id.toLowerCase().includes(filt) || s.template_name.toLowerCase().includes(filt));
  tbody.innerHTML = "";
  for (const s of items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="seed-name">${highlight(s.seed_id, filt)}</td>
      <td>${escHtml(s.template_name)}</td>
      <td class="seed-name" style="font-size:11px">${escHtml(s.actor_seed_id)}</td>
      <td>${statusPill(s.compile_status)}</td>
      <td class="num">${s.nq}</td><td class="num">${s.nbody}</td><td class="num">${s.nu}</td>
      <td>
        <button class="primary iconbtn" title="visualize_corpus --show">▶</button>
        <button class="ghost iconbtn" title="--rollout">↻</button>
      </td>`;
    tr.querySelectorAll("button")[0].onclick = () => launch({tool:"corpus_show", seed_id:s.seed_id, rollout:false});
    tr.querySelectorAll("button")[1].onclick = () => launch({tool:"corpus_show", seed_id:s.seed_id, rollout:true});
    tbody.appendChild(tr);
  }
  document.getElementById("l1-count").textContent = items.length + " / " + CORPUS.synthetic_scenes.length;
}

function renderL2() {
  const tbody = document.getElementById("l2-tbody");
  const filt = document.getElementById("l2-filter").value.trim().toLowerCase();
  const items = CORPUS.open_envs.filter(s =>
    !filt || s.env_id.toLowerCase().includes(filt) || s.adapter_name.toLowerCase().includes(filt));
  tbody.innerHTML = "";
  for (const s of items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="seed-name" style="font-size:11px">${highlight(s.env_id, filt)}</td>
      <td><span class="pill src">${escHtml(s.adapter_name)}</span></td>
      <td>${statusPill(s.dependency_status)}</td>
      <td>${statusPill(s.runnable_status)}</td>
      <td style="font-size:11px;color:#888">${escHtml((s.tags||[]).join(", "))}</td>`;
    tbody.appendChild(tr);
  }
  document.getElementById("l2-count").textContent = items.length + " / " + CORPUS.open_envs.length;
}

function renderL3() {
  const tbody = document.getElementById("l3-tbody");
  const filt = document.getElementById("l3-filter").value.trim().toLowerCase();
  const items = CORPUS.trajectory_seeds.filter(s =>
    !filt || s.seed_id.toLowerCase().includes(filt) || s.parent_seed_id.toLowerCase().includes(filt));
  tbody.innerHTML = "";
  for (const s of items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="seed-name" style="font-size:11px">${highlight(s.seed_id, filt)}</td>
      <td style="font-size:11px;color:#888">${escHtml(s.parent_seed_id)}</td>
      <td><span class="pill src" style="font-size:10px">${escHtml(s.parent_layer)}</span></td>
      <td>${statusPill(s.replay_status)}</td>
      <td>${escHtml(s.action_kind)}</td>
      <td class="num">${s.horizon}</td>`;
    tbody.appendChild(tr);
  }
  document.getElementById("l3-count").textContent = items.length + " / " + CORPUS.trajectory_seeds.length;
}

document.getElementById("l1-filter").oninput = renderL1;
document.getElementById("l2-filter").oninput = renderL2;
document.getElementById("l3-filter").oninput = renderL3;

document.getElementById("refresh-corpus").onclick = async () => {
  const r = await fetch("/api/corpus?refresh=1"); const j = await r.json();
  CORPUS = j.corpus;
  renderL1(); renderL2(); renderL3();
  log("corpus reloaded  L1=" + CORPUS.synthetic_scenes.length +
      " L2=" + CORPUS.open_envs.length + " L3=" + CORPUS.trajectory_seeds.length);
};

document.getElementById("btn-validate").onclick = () =>
  launch({tool:"run_tool", script:"validate_layered_corpus.py"});
document.getElementById("btn-rand-fuzz").onclick = () =>
  launch({tool:"run_tool", script:"run_random_fuzz.py", args:["--budget","20"]});
document.getElementById("btn-rule-fuzz").onclick = () =>
  launch({tool:"run_tool", script:"run_rule_fuzz.py", args:["--budget","20"]});

switchTab("l1");
renderL1(); renderL2(); renderL3();

// ─────────────────────────────────────────────
// ④ Findings
// ─────────────────────────────────────────────
function renderFindings() {
  const tbody = document.getElementById("findings-tbody");
  const filt = document.getElementById("findings-filter").value.trim().toLowerCase();
  const items = FINDINGS.filter(f =>
    !filt || f.finding_id.toLowerCase().includes(filt) ||
    f.seed_id.toLowerCase().includes(filt) || f.layer.toLowerCase().includes(filt) ||
    f.signature.toLowerCase().includes(filt));
  document.getElementById("findings-count").textContent = items.length + " / " + FINDINGS.length;
  tbody.innerHTML = "";
  for (const f of items) {
    const tr = document.createElement("tr");
    const oracles = (f.oracle_signals || []).filter(o => o.failed).map(o => o.name).join(", ");
    const sevBar = f.severity > 0
      ? `<div style="display:inline-block;width:${Math.min(60, f.severity*10)}px;height:8px;background:#dc2626;border-radius:4px;vertical-align:middle;margin-right:4px"></div>`
      : "";
    tr.innerHTML = `
      <td class="seed-name" style="font-size:11px">${highlight(f.finding_id, filt)}</td>
      <td style="font-size:11px">${escHtml(f.seed_id)}</td>
      <td><span class="pill src">${escHtml(f.layer)}</span></td>
      <td>${sevBar}<span class="num">${f.severity.toFixed(1)}</span></td>
      <td style="font-size:11px;color:#dc2626">${escHtml(oracles||"(replay_failed)")}</td>
      <td style="font-size:10px;color:#888">${escHtml(f.signature)}</td>
      <td style="font-size:10px;color:#888">${escHtml(f._path)}</td>`;
    tbody.appendChild(tr);
  }
}

document.getElementById("findings-filter").oninput = renderFindings;
document.getElementById("refresh-findings").onclick = async () => {
  const r = await fetch("/api/findings?refresh=1"); const j = await r.json();
  FINDINGS = j.findings;
  renderFindings();
  log("findings reloaded (" + FINDINGS.length + ")");
};

renderFindings();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet default access log
        return

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            seeds = list_seeds()
            html = (INDEX_HTML
                    .replace("__SEEDS__", json.dumps(seeds))
                    .replace("__MUTS__", json.dumps(mutator_catalog()))
                    .replace("__SOURCE_INFO__", json.dumps(SOURCE_INFO, ensure_ascii=False))
                    .replace("__CORPUS__", json.dumps(_load_corpus_layers(), ensure_ascii=False))
                    .replace("__FINDINGS__", json.dumps(_load_findings(), ensure_ascii=False)))
            for k, v in FIELD_TOOLTIPS.items():
                html = html.replace(f"__TT_{k}__", v.replace('"', "&quot;"))
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/state":
            return self._send_json(200, {
                "seeds": list_seeds(force=("refresh" in qs)),
                "mutators": mutator_catalog(),
            })
        if path == "/api/corpus":
            return self._send_json(200, {
                "corpus": _load_corpus_layers(force=("refresh" in qs)),
            })
        if path == "/api/findings":
            global _FINDINGS_CACHE
            _FINDINGS_CACHE = None
            return self._send_json(200, {
                "findings": _load_findings(force=("refresh" in qs)),
            })
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/api/delete":
            return self._handle_delete(parsed)
        if path != "/launch":
            self.send_error(404); return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send_json(400, {"ok": False, "error": f"bad json: {e}"})

        tool = payload.get("tool")
        try:
            if tool == "seed":
                seed = payload.get("seed")
                if not seed:
                    raise ValueError("seed required")
                cmd = [PYTHON, str(TOOLS_DIR / "visualize_seed.py"), seed]
                if payload.get("static"):
                    cmd.append("--static")
            elif tool == "mutator":
                mid = payload.get("mutator")
                if not mid:
                    raise ValueError("mutator required")
                cmd = [PYTHON, str(TOOLS_DIR / "visualize_mutator.py"), mid]
                if payload.get("seed"):
                    cmd += ["--seed", payload["seed"]]
                if payload.get("intensity"):
                    cmd += ["--intensity", payload["intensity"]]
                if payload.get("no_before"):
                    cmd.append("--no-before")
            elif tool == "corpus_show":
                seed_id = payload.get("seed_id")
                if not seed_id:
                    raise ValueError("seed_id required")
                cmd = [PYTHON, str(TOOLS_DIR / "visualize_corpus.py"),
                       "--show", seed_id]
                if payload.get("rollout"):
                    cmd.append("--rollout")
                elif payload.get("viewer"):
                    cmd.append("--viewer")
            elif tool == "run_tool":
                script = payload.get("script")
                if not script:
                    raise ValueError("script required")
                # Security: only allow scripts that live in tools/ directory.
                target = (TOOLS_DIR / script).resolve()
                if not str(target).startswith(str(TOOLS_DIR.resolve())):
                    raise ValueError(f"script must be inside tools/: {script!r}")
                cmd = [PYTHON, str(target)] + list(payload.get("args", []))
            else:
                raise ValueError(f"unknown tool: {tool!r}")
            pid = spawn(cmd)
            return self._send_json(200, {"ok": True, "pid": pid, "cmd": cmd})
        except Exception as e:
            return self._send_json(400, {"ok": False, "error": str(e)})

    # ---- /api/delete : move seed dir into seeds/_recycle/ and patch MANIFEST ----
    def _handle_delete(self, parsed) -> None:
        global _SEED_CACHE, _MANIFEST_CACHE
        qs = urllib.parse.parse_qs(parsed.query)
        name = (qs.get("name") or [""])[0]
        # Path-traversal guard: seed names are flat directory names with no separators.
        if not name or any(c in name for c in ("/", "\\")) or name.startswith(".") or name in ("_recycle", "_assets", "_pool", "_quarantine"):
            return self._send_json(400, {"ok": False, "error": f"bad seed name: {name!r}"})
        src_dir = SEEDS_DIR / name
        if not src_dir.is_dir():
            return self._send_json(404, {"ok": False, "error": f"not found: seeds/curated/{name}"})
        recycle_root = SEEDS_DIR.parent / "_recycle"
        recycle_root.mkdir(parents=True, exist_ok=True)
        dst_dir = recycle_root / name
        if dst_dir.exists():
            # Append a numeric suffix to keep prior recycled copies intact.
            i = 2
            while (recycle_root / f"{name}__{i}").exists():
                i += 1
            dst_dir = recycle_root / f"{name}__{i}"
        try:
            shutil.move(str(src_dir), str(dst_dir))
        except Exception as e:
            return self._send_json(500, {"ok": False, "error": f"move failed: {e}"})
        # Patch MANIFEST.json so future loads don't keep claiming it's curated.
        try:
            mp = SEEDS_DIR / "MANIFEST.json"
            if mp.is_file():
                data = json.loads(mp.read_text(encoding="utf-8"))
                changed = False
                for entry in data:
                    cp = entry.get("curated_path", "").replace("\\", "/")
                    if cp == f"curated/{name}/model.xml":
                        entry["in_curated"] = False
                        entry["recycled"] = True
                        entry["curated_path"] = f"_recycle/{dst_dir.name}/model.xml"
                        changed = True
                        break
                if changed:
                    mp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[delete] manifest patch failed: {e}", file=sys.stderr)
        _SEED_CACHE = None
        _MANIFEST_CACHE = None
        return self._send_json(200, {"ok": True, "moved_to": f"seeds/_recycle/{dst_dir.name}"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"serving on {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
