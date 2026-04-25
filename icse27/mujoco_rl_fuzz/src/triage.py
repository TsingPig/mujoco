"""Issue signature + dedup counter."""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .result import ExecutionResult


@dataclass(frozen=True)
class IssueSignature:
    sig_hash: str
    failure_kind: str             # "compile" | "runtime" | "warning_only" | "inconsistency" | "ok"
    exception_type: Optional[str]
    warning_types: tuple[str, ...]
    returncode: int
    model_shape: tuple[int, ...]


def classify(result: ExecutionResult) -> str:
    if result.returncode != 0 or result.timeout:
        return "crash"
    if not result.compile_ok:
        return "compile"
    if not result.runtime_ok:
        return "runtime"
    if result.warnings:
        return "warning_only"
    if result.consistency_diff is not None and result.consistency_diff > 1e-9:
        return "inconsistency"
    return "ok"


def signature_of(result: ExecutionResult) -> IssueSignature:
    wtypes = tuple(sorted(w.wtype for w in result.warnings))
    kind = classify(result)
    payload = repr((kind, result.exception_type, wtypes,
                    result.returncode, tuple(result.model_shape)))
    h = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()
    return IssueSignature(h, kind, result.exception_type, wtypes,
                          result.returncode, tuple(result.model_shape))


class Triage:
    def __init__(self):
        self._counts: Counter[str] = Counter()
        self._sigs: dict[str, IssueSignature] = {}
        self._first_seen_step: dict[str, int] = {}

    def update(self, result: ExecutionResult, step: int) -> tuple[IssueSignature, bool]:
        sig = signature_of(result)
        is_new = sig.sig_hash not in self._sigs
        self._counts[sig.sig_hash] += 1
        if is_new:
            self._sigs[sig.sig_hash] = sig
            self._first_seen_step[sig.sig_hash] = step
        return sig, is_new

    @property
    def n_unique(self) -> int:
        return len(self._sigs)

    @property
    def n_raw(self) -> int:
        return sum(self._counts.values())

    def summary(self) -> dict:
        per_kind: Counter[str] = Counter(s.failure_kind for s in self._sigs.values())
        return {
            "n_raw": self.n_raw,
            "n_unique": self.n_unique,
            "by_kind_unique": dict(per_kind),
            "top_signatures": [
                {"sig": h, "count": c, "kind": self._sigs[h].failure_kind,
                 "etype": self._sigs[h].exception_type,
                 "warnings": list(self._sigs[h].warning_types)}
                for h, c in self._counts.most_common(20) if h in self._sigs
            ],
        }
