"""Sensor / observation / reward consistency oracle.

First-stage check: flag non-finite or excessively large sensor norms.
"""
from __future__ import annotations

from ..runner.types import OracleReport


class SensorRewardOracle:
    name = "sensor_reward"

    SENSOR_NORM_THRESHOLD = 1e6

    def evaluate(self, trace, *, replay_meta=None) -> OracleReport:
        if trace is None:
            return OracleReport(name=self.name)
        notes = {}
        sev = 0.0
        sn = trace.sensor_norm_max
        if sn is not None:
            notes["sensor_norm_max"] = sn
            if sn > self.SENSOR_NORM_THRESHOLD:
                sev += min(5.0, sn / self.SENSOR_NORM_THRESHOLD)
        # reward NaN check
        rt = trace.reward_total
        if rt is not None:
            notes["reward_total"] = rt
            try:
                import math
                if not math.isfinite(rt):
                    sev += 5.0
                    notes["reward_nonfinite"] = True
            except Exception:
                pass
        return OracleReport(name=self.name, severity=sev, failed=sev > 0.5, notes=notes)
