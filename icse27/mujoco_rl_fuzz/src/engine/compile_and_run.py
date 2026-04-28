"""Main-process side: dispatch a worker subprocess for one TestCase.

Returns a parsed `ExecutionResult` plus the raw worker dict (for oracles).
The subprocess is killed on timeout; on any abnormal exit we still produce
a structured ExecutionResult.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from ..result import ExecutionResult, WarningRecord


@dataclass
class RunSpec:
    xml_path: str
    rollout_steps: int = 50
    state_perturb: Optional[dict] = None
    disable_clamp_ctrl: bool = False
    consistency_check: bool = False
    solver_override: Optional[dict] = None


def _spec_to_dict(s: RunSpec) -> dict:
    d: dict = {"xml_path": s.xml_path, "rollout_steps": int(s.rollout_steps)}
    if s.state_perturb:        d["state_perturb"] = s.state_perturb
    if s.disable_clamp_ctrl:    d["disable_clamp_ctrl"] = True
    if s.consistency_check:     d["consistency_check"] = True
    if s.solver_override:       d["solver_override"] = s.solver_override
    return d


def run_in_subprocess(
    tc_id: str,
    spec: RunSpec,
    timeout_sec: float = 10.0,
    workdir: Optional[str] = None,
) -> tuple[ExecutionResult, dict]:
    """Spawn worker, parse result. Returns (ExecutionResult, raw_dict).

    raw_dict carries the full worker JSON (for downstream oracles).
    On timeout / non-zero exit / missing output, fields are filled accordingly.
    """
    workdir = workdir or tempfile.gettempdir()
    os.makedirs(workdir, exist_ok=True)
    nonce = uuid.uuid4().hex[:8]
    in_path = os.path.join(workdir, f"spec_{tc_id}_{nonce}.json")
    out_path = os.path.join(workdir, f"result_{tc_id}_{nonce}.json")

    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(_spec_to_dict(spec), f)

    cmd = [sys.executable, "-m", "src.engine.subprocess_worker",
           "--input", in_path, "--output", out_path]

    timeout = False
    returncode = 0
    stderr_tail = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_sec, text=True)
        returncode = proc.returncode
        if proc.stderr:
            stderr_tail = proc.stderr[-500:]
    except subprocess.TimeoutExpired as te:
        timeout = True
        returncode = -1
        stderr_tail = (te.stderr or b"")[-500:].decode("utf-8", errors="replace") if te.stderr else ""

    raw: dict[str, Any] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}

    # Normalise into ExecutionResult
    res = ExecutionResult(tc_id=tc_id)
    res.returncode = returncode
    res.timeout = timeout

    if not raw:
        res.compile_ok = False
        res.runtime_ok = False
        res.exception_type = "WorkerNoOutput" if not timeout else "WorkerTimeout"
        res.traceback_summary = stderr_tail
    else:
        res.compile_ok = bool(raw.get("compile", {}).get("ok", False))
        res.runtime_ok = bool(raw.get("runtime", {}).get("ok", False))
        res.steps_done = int(raw.get("runtime", {}).get("steps_done", 0))
        res.state_stats = dict(raw.get("state_stats") or {})
        res.warnings = [WarningRecord(**w) for w in (raw.get("warnings") or [])]
        # Prefer compile-time exception when present, fall back to runtime
        ce = raw.get("compile", {}).get("exception_type")
        re = raw.get("runtime", {}).get("exception_type")
        if ce:
            res.exception_type = ce
            res.traceback_summary = raw.get("compile", {}).get("traceback")
        elif re:
            res.exception_type = re
            res.traceback_summary = raw.get("runtime", {}).get("traceback")
        cons = raw.get("consistency") or {}
        res.consistency_diff = cons.get("diff") if cons.get("ran") else None
        res.solver_diff = raw.get("solver_diff")
        res.model_shape = tuple(raw.get("model_shape") or ())

    # Cleanup spec/result files (keep on failure for forensics)
    for p in (in_path, out_path):
        if os.path.exists(p) and res.runtime_ok and not res.warnings and not res.exception_type:
            try:
                os.remove(p)
            except OSError:
                pass

    raw.setdefault("returncode", returncode)
    raw.setdefault("timeout", timeout)
    return res, raw
