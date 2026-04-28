"""Issue signature + dedup counter.

v6 fix: signature now includes a NORMALIZED error-message fingerprint
(`exception_msg_norm`) and EXCLUDES `model_shape` from the hash. This prevents
the previous spurious-novelty bug where the same root-cause failure
("size N must be positive") got 10+ different signatures merely because
STRUCT_GROW changed (nq, nv, nbody). `model_shape` is still kept on the
record for diagnostics, just not part of the identity hash.
"""
from __future__ import annotations

import hashlib
import re
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
    model_shape: tuple[int, ...]   # diagnostic only, NOT in hash
    exception_msg_norm: str = ""   # normalized first ~80 chars of traceback_summary


# Patterns to strip volatile parts (numbers, hex addrs, file paths, line nums)
_NORM_PATTERNS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\b\d+\.\d+(?:[eE][+\-]?\d+)?\b"), "NUM"),  # floats
    (re.compile(r"\b\d+\b"), "N"),                            # ints
    (re.compile(r"line\s+N"), "line"),
    (re.compile(r"[A-Za-z]:\\[^\s'\"]+"), "PATH"),            # windows paths
    (re.compile(r"/[^\s'\"]+"), "PATH"),                       # unix paths
    (re.compile(r"\s+"), " "),
]


def _normalize_msg(msg: Optional[str], max_len: int = 80) -> str:
    """Strip numbers/addrs/paths so 'size 1 ...' and 'size 7 ...' map to same key."""
    if not msg:
        return ""
    s = msg.strip()
    for pat, repl in _NORM_PATTERNS:
        s = pat.sub(repl, s)
    return s[:max_len].strip()


# Exception types that indicate the *fuzzer itself* produced a structurally
# invalid MJCF tree (or a Python-level mutator skip). These are NOT MuJoCo bugs;
# they are fuzzer-internal noise and are excluded from `n_unique`/`top_signatures`.
_INVALID_ETYPES = {
    "MutationSkip",
    "ValueError",      # almost always raised by mujoco's XML parser/validator
    "FileNotFoundError",
    "XMLSyntaxError",
    "FatalError",      # mujoco compile-time fatal (still our bad input usually)
}


def classify(result: ExecutionResult) -> str:
    if result.returncode != 0 or result.timeout:
        return "crash"
    if not result.compile_ok:
        # Compile-time failure ALWAYS routes to `invalid`. Real MuJoCo compile-
        # time bugs would manifest as crashes / asserts / segfaults, which take
        # the `crash` branch above.
        return "invalid"
    if not result.runtime_ok:
        # Runtime exception with successful compile -> potentially a real bug.
        # If exception type is in the invalid set (e.g. ValueError from
        # mj_step on garbage state caused by our perturbation), still
        # surface as runtime so user can decide.
        return "runtime"
    if result.warnings:
        return "warning_only"
    if result.consistency_diff is not None and result.consistency_diff > 1e-9:
        return "inconsistency"
    return "ok"


def is_real_finding_kind(kind: str) -> bool:
    """True iff this kind is a candidate MuJoCo bug (not fuzzer noise)."""
    return kind in ("crash", "runtime", "warning_only", "inconsistency", "ok")


def signature_of(result: ExecutionResult) -> IssueSignature:
    wtypes = tuple(sorted(w.wtype for w in result.warnings))
    kind = classify(result)
    msg_norm = _normalize_msg(result.traceback_summary)
    # Hash payload INTENTIONALLY excludes model_shape; includes msg_norm.
    payload = repr((kind, result.exception_type, msg_norm, wtypes, result.returncode))
    h = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()
    return IssueSignature(
        sig_hash=h,
        failure_kind=kind,
        exception_type=result.exception_type,
        warning_types=wtypes,
        returncode=result.returncode,
        model_shape=tuple(result.model_shape),
        exception_msg_norm=msg_norm,
    )


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

    @property
    def n_unique_real(self) -> int:
        """Distinct signatures that are NOT fuzzer-internal noise."""
        return sum(1 for s in self._sigs.values()
                   if is_real_finding_kind(s.failure_kind))

    @property
    def n_unique_invalid(self) -> int:
        return sum(1 for s in self._sigs.values()
                   if not is_real_finding_kind(s.failure_kind))

    def summary(self) -> dict:
        per_kind: Counter[str] = Counter(s.failure_kind for s in self._sigs.values())
        return {
            "n_raw": self.n_raw,
            "n_unique": self.n_unique,                     # all signatures (legacy)
            "n_unique_real": self.n_unique_real,           # excludes `invalid`
            "n_unique_invalid": self.n_unique_invalid,
            "by_kind_unique": dict(per_kind),
            "top_signatures": [
                {"sig": h, "count": c, "kind": self._sigs[h].failure_kind,
                 "etype": self._sigs[h].exception_type,
                 "warnings": list(self._sigs[h].warning_types),
                 "msg": self._sigs[h].exception_msg_norm}
                for h, c in self._counts.most_common(20) if h in self._sigs
            ],
            "top_signatures_real": [
                {"sig": h, "count": c, "kind": self._sigs[h].failure_kind,
                 "etype": self._sigs[h].exception_type,
                 "warnings": list(self._sigs[h].warning_types),
                 "msg": self._sigs[h].exception_msg_norm}
                for h, c in self._counts.most_common(50)
                if h in self._sigs and is_real_finding_kind(self._sigs[h].failure_kind)
            ][:20],
        }
