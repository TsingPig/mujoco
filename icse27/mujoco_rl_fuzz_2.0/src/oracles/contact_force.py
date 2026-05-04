"""Contact-force / contact-count anomaly oracle.

This first-stage oracle uses the trace summary's contact_count as a proxy
for "many contacts". A real contact-force magnitude oracle requires
sampling efc_force during replay; that hook will be added together with
the backend differential oracle.
"""
from __future__ import annotations

from ..runner.types import OracleReport


class ContactForceOracle:
    name = "contact_force"

    CONTACT_COUNT_THRESHOLD = 200

    def evaluate(self, trace, *, replay_meta=None) -> OracleReport:
        if trace is None:
            return OracleReport(name=self.name)
        c = trace.contact_count_max
        if c > self.CONTACT_COUNT_THRESHOLD:
            sev = min(3.0, c / self.CONTACT_COUNT_THRESHOLD)
            return OracleReport(name=self.name, severity=sev, failed=sev > 1.5,
                                notes={"contact_count_max": c})
        return OracleReport(name=self.name, severity=0.0, failed=False,
                            notes={"contact_count_max": c})
