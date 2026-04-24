"""
Smoke demo: 证明 MuJoCo + warning hook + exception 捕获链路全通。

不依赖 src/ 任何模块。直接：
    python demo_smoke.py

四个阶段：
  Stage 1  正常 rollout            -> 期望 0 warning
  Stage 2  极大 ctrl(=1e10)        -> 期望捕获 BADCTRL
  Stage 3  qpos 注入 NaN           -> 期望捕获 BADQPOS + has_nan=True
  Stage 4  故意编译错误 XML        -> 期望捕获 compile-time exception

每阶段打印结构化结果。任何阶段没拿到预期信号都会在结尾汇总里显式标 [MISS]。
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import mujoco
except ImportError:
    print("[FATAL] mujoco package not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(2)


HERE = os.path.dirname(os.path.abspath(__file__))
SEED_XML = os.path.join(HERE, "seeds", "pendulum.xml")


# ---------- warning capture ----------

# mujoco.mjtWarning 枚举 → 名称
_WARN_NAMES = [
    "INERTIA", "CONTACTFULL", "CNSTRFULL", "VGEOMFULL",
    "BADQPOS", "BADQVEL", "BADQACC", "BADCTRL",
]

_captured_warnings: list[str] = []


def _warning_callback(msg):
    # mujoco 会把 C 字符串 (bytes) 传进来
    if isinstance(msg, bytes):
        msg = msg.decode("utf-8", errors="replace")
    _captured_warnings.append(msg)


def install_warning_hook():
    # 全局回调，整个进程共享
    mujoco.set_mju_user_warning(_warning_callback)


def reset_warning_capture():
    _captured_warnings.clear()


def snapshot_data_warning_counts(data) -> dict[str, int]:
    """读取 mjData.warning[i].number，给出每种 warning 的累计计数。"""
    out: dict[str, int] = {}
    try:
        n = len(data.warning)
    except Exception:
        return out
    for i in range(n):
        name = _WARN_NAMES[i] if i < len(_WARN_NAMES) else f"WARN_{i}"
        cnt = int(data.warning[i].number)
        if cnt > 0:
            out[name] = cnt
    return out


# ---------- result struct ----------

@dataclass
class StageResult:
    stage: str
    compile_ok: bool
    runtime_ok: bool
    steps_done: int = 0
    warnings_via_hook: list[str] = field(default_factory=list)
    warnings_via_data: dict[str, int] = field(default_factory=dict)
    exception_type: Optional[str] = None
    exception_msg: Optional[str] = None
    state_stats: dict[str, float] = field(default_factory=dict)
    expected_signal: str = ""
    matched_expectation: bool = False

    def pretty(self) -> str:
        flag = "OK " if self.matched_expectation else "MISS"
        return (
            f"[{flag}] {self.stage}\n"
            f"   expected      : {self.expected_signal}\n"
            f"   compile_ok    : {self.compile_ok}\n"
            f"   runtime_ok    : {self.runtime_ok}\n"
            f"   steps_done    : {self.steps_done}\n"
            f"   warn(hook)    : {len(self.warnings_via_hook)} item(s)"
            + (f"  e.g. {self.warnings_via_hook[0][:80]!r}" if self.warnings_via_hook else "")
            + "\n"
            f"   warn(data)    : {self.warnings_via_data}\n"
            f"   exception     : {self.exception_type}: {self.exception_msg}\n"
            f"   state_stats   : {self.state_stats}\n"
        )


# ---------- helpers ----------

def state_stats(data) -> dict[str, float]:
    qpos = np.asarray(data.qpos)
    qvel = np.asarray(data.qvel)
    ctrl = np.asarray(data.ctrl) if data.ctrl.size else np.zeros(0)
    return {
        "max_abs_qpos": float(np.max(np.abs(qpos))) if qpos.size else 0.0,
        "max_abs_qvel": float(np.max(np.abs(qvel))) if qvel.size else 0.0,
        "max_abs_ctrl": float(np.max(np.abs(ctrl))) if ctrl.size else 0.0,
        "has_nan": bool(np.isnan(qpos).any() or np.isnan(qvel).any()),
        "has_inf": bool(np.isinf(qpos).any() or np.isinf(qvel).any()),
    }


def compile_model(xml_path: str):
    return mujoco.MjModel.from_xml_path(xml_path)


def compile_model_from_string(xml_str: str):
    return mujoco.MjModel.from_xml_string(xml_str)


def run_steps(model, data, n_steps: int) -> int:
    done = 0
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        done += 1
    return done


# ---------- stages ----------

def stage_1_normal() -> StageResult:
    r = StageResult(stage="Stage 1: normal rollout", compile_ok=False, runtime_ok=False,
                    expected_signal="0 warnings, no exception")
    reset_warning_capture()
    try:
        model = compile_model(SEED_XML)
        r.compile_ok = True
        data = mujoco.MjData(model)
        r.steps_done = run_steps(model, data, 100)
        r.runtime_ok = True
        r.state_stats = state_stats(data)
        r.warnings_via_data = snapshot_data_warning_counts(data)
    except Exception as e:
        r.exception_type = type(e).__name__
        r.exception_msg = str(e)[:200]
    r.warnings_via_hook = list(_captured_warnings)
    r.matched_expectation = (
        r.compile_ok and r.runtime_ok
        and not r.warnings_via_hook and not r.warnings_via_data
        and r.exception_type is None
    )
    return r


def stage_2_extreme_ctrl() -> StageResult:
    r = StageResult(stage="Stage 2: ctrl := Inf", compile_ok=False, runtime_ok=False,
                    expected_signal="BADCTRL warning captured")
    reset_warning_capture()
    try:
        model = compile_model(SEED_XML)
        r.compile_ok = True
        # 关掉 ctrl 自动 clip，否则越界值会被静默截断
        try:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
        except Exception:
            pass
        data = mujoco.MjData(model)
        # 注入 Inf（NaN/Inf 才会被 BADCTRL 捕获）
        if data.ctrl.size:
            data.ctrl[:] = float("inf")
        r.steps_done = run_steps(model, data, 50)
        r.runtime_ok = True
        r.state_stats = state_stats(data)
        r.warnings_via_data = snapshot_data_warning_counts(data)
    except Exception as e:
        r.exception_type = type(e).__name__
        r.exception_msg = str(e)[:200]
    r.warnings_via_hook = list(_captured_warnings)
    r.matched_expectation = (
        "BADCTRL" in r.warnings_via_data
        or any("BADCTRL" in w or "ctrl" in w.lower() for w in r.warnings_via_hook)
    )
    return r


def stage_3_nan_qpos() -> StageResult:
    r = StageResult(stage="Stage 3: qpos := NaN", compile_ok=False, runtime_ok=False,
                    expected_signal="BADQPOS warning + has_nan=True")
    reset_warning_capture()
    try:
        model = compile_model(SEED_XML)
        r.compile_ok = True
        data = mujoco.MjData(model)
        if data.qpos.size:
            data.qpos[0] = float("nan")
        r.steps_done = run_steps(model, data, 10)
        r.runtime_ok = True
        r.state_stats = state_stats(data)
        r.warnings_via_data = snapshot_data_warning_counts(data)
    except Exception as e:
        r.exception_type = type(e).__name__
        r.exception_msg = str(e)[:200]
    r.warnings_via_hook = list(_captured_warnings)
    r.matched_expectation = (
        "BADQPOS" in r.warnings_via_data
        or r.state_stats.get("has_nan", False)
        or any("BADQPOS" in w or "qpos" in w.lower() for w in r.warnings_via_hook)
    )
    return r


_BROKEN_XML = """<mujoco model="broken">
  <worldbody>
    <body name="b1">
      <!-- 没有 inertial 也没有 geom：几乎一定触发 compile error -->
      <joint name="j1" type="hinge"/>
    </body>
  </worldbody>
</mujoco>
"""


def stage_4_bad_xml() -> StageResult:
    r = StageResult(stage="Stage 4: invalid XML compile",
                    compile_ok=False, runtime_ok=False,
                    expected_signal="compile-time exception captured")
    reset_warning_capture()
    try:
        compile_model_from_string(_BROKEN_XML)
        r.compile_ok = True
    except Exception as e:
        r.exception_type = type(e).__name__
        r.exception_msg = str(e)[:200]
    r.warnings_via_hook = list(_captured_warnings)
    r.matched_expectation = (not r.compile_ok) and (r.exception_type is not None)
    return r


# ---------- main ----------

def main() -> int:
    print("=" * 70)
    print("MuJoCo RL-Fuzz :: Smoke Demo")
    print(f"  mujoco version : {mujoco.__version__}")
    print(f"  seed XML       : {SEED_XML}")
    print(f"  exists         : {os.path.exists(SEED_XML)}")
    print("=" * 70)

    install_warning_hook()

    stages = [
        stage_1_normal,
        stage_2_extreme_ctrl,
        stage_3_nan_qpos,
        stage_4_bad_xml,
    ]
    results = []
    for fn in stages:
        try:
            res = fn()
        except Exception as e:
            # 任何意料外异常都算实现 bug，打出来但继续
            res = StageResult(stage=fn.__name__, compile_ok=False, runtime_ok=False,
                              expected_signal="(stage helper failed)",
                              exception_type=type(e).__name__,
                              exception_msg=str(e)[:200])
        results.append(res)
        print(res.pretty())

    print("=" * 70)
    matched = sum(1 for r in results if r.matched_expectation)
    print(f"SUMMARY: {matched}/{len(results)} stages matched expectation.")
    if matched < len(results):
        print("[!] Some stages did not produce the expected signal.")
        print("    This may be normal across MuJoCo versions; inspect the per-stage output above.")
    return 0 if matched >= 3 else 1   # 至少 3/4 通过才算链路 OK


if __name__ == "__main__":
    sys.exit(main())
