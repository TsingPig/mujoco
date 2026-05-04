"""Trace recorder + summary statistics for trajectory replays."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TraceSummary:
    n_steps: int = 0
    qpos_norm_min: float = 0.0
    qpos_norm_max: float = 0.0
    qpos_norm_final: float = 0.0
    qvel_norm_min: float = 0.0
    qvel_norm_max: float = 0.0
    qvel_norm_final: float = 0.0
    qacc_norm_min: float = 0.0
    qacc_norm_max: float = 0.0
    qacc_norm_final: float = 0.0
    contact_count_min: int = 0
    contact_count_max: int = 0
    contact_count_final: int = 0
    body_z_min: Optional[float] = None
    body_z_max: Optional[float] = None
    body_z_final: Optional[float] = None
    sensor_norm_max: Optional[float] = None
    sensor_norm_final: Optional[float] = None
    reward_total: Optional[float] = None
    finite: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _norm(v) -> float:
    try:
        s = 0.0
        for x in v:
            xx = float(x)
            s += xx * xx
        return math.sqrt(s)
    except Exception:
        return 0.0


def _all_finite(*arrs) -> bool:
    for arr in arrs:
        if arr is None:
            continue
        try:
            for x in arr:
                if not math.isfinite(float(x)):
                    return False
        except Exception:
            continue
    return True


class TraceRecorder:
    """Streams per-step samples and produces a TraceSummary at the end."""

    def __init__(self):
        self._n = 0
        self._first = True
        self._summary = TraceSummary()

    def step(self, *,
             qpos=None, qvel=None, qacc=None,
             ncon: Optional[int] = None,
             body_z: Optional[float] = None,
             sensor=None,
             reward: Optional[float] = None) -> None:
        self._n += 1
        s = self._summary
        s.n_steps = self._n

        qn = _norm(qpos) if qpos is not None else 0.0
        vn = _norm(qvel) if qvel is not None else 0.0
        an = _norm(qacc) if qacc is not None else 0.0
        if self._first:
            s.qpos_norm_min = qn; s.qpos_norm_max = qn
            s.qvel_norm_min = vn; s.qvel_norm_max = vn
            s.qacc_norm_min = an; s.qacc_norm_max = an
        else:
            s.qpos_norm_min = min(s.qpos_norm_min, qn); s.qpos_norm_max = max(s.qpos_norm_max, qn)
            s.qvel_norm_min = min(s.qvel_norm_min, vn); s.qvel_norm_max = max(s.qvel_norm_max, vn)
            s.qacc_norm_min = min(s.qacc_norm_min, an); s.qacc_norm_max = max(s.qacc_norm_max, an)
        s.qpos_norm_final = qn; s.qvel_norm_final = vn; s.qacc_norm_final = an

        if ncon is not None:
            if self._first:
                s.contact_count_min = ncon; s.contact_count_max = ncon
            else:
                s.contact_count_min = min(s.contact_count_min, ncon)
                s.contact_count_max = max(s.contact_count_max, ncon)
            s.contact_count_final = ncon

        if body_z is not None:
            if s.body_z_min is None or body_z < s.body_z_min:
                s.body_z_min = body_z
            if s.body_z_max is None or body_z > s.body_z_max:
                s.body_z_max = body_z
            s.body_z_final = body_z

        if sensor is not None:
            sn = _norm(sensor)
            s.sensor_norm_max = sn if s.sensor_norm_max is None else max(s.sensor_norm_max, sn)
            s.sensor_norm_final = sn

        if reward is not None:
            try:
                rv = float(reward)
                s.reward_total = rv if s.reward_total is None else (s.reward_total + rv)
            except Exception:
                pass

        if not _all_finite(qpos, qvel, qacc):
            s.finite = False

        self._first = False

    def warn(self, msg: str) -> None:
        self._summary.warnings.append(msg)

    def finalize(self) -> TraceSummary:
        return self._summary
