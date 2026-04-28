"""Issue signature: stable hash for dedup. Hardened copy of v1 triage.

Only the hashing utilities are kept here; the v1 ExecutionResult / Triage
classes are intentionally left out because v2.0-α has no runner yet.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional, Tuple


_NORM_PATTERNS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\b\d+\.\d+(?:[eE][+\-]?\d+)?\b"), "NUM"),
    (re.compile(r"\b\d+\b"), "N"),
    (re.compile(r"line\s+N"), "line"),
    (re.compile(r"[A-Za-z]:\\[^\s'\"]+"), "PATH"),
    (re.compile(r"/[^\s'\"]+"), "PATH"),
    (re.compile(r"\s+"), " "),
]


def normalize_msg(msg: Optional[str], max_len: int = 80) -> str:
    if not msg:
        return ""
    s = msg.strip()
    for pat, repl in _NORM_PATTERNS:
        s = pat.sub(repl, s)
    return s[:max_len].strip()


@dataclass(frozen=True)
class IssueSignature:
    sig_hash: str
    failure_kind: str
    exception_type: Optional[str]
    warning_types: Tuple[str, ...]
    returncode: int
    exception_msg_norm: str = ""


def signature(failure_kind: str,
              exception_type: Optional[str],
              warning_types: Tuple[str, ...],
              returncode: int,
              exception_msg: Optional[str] = None) -> IssueSignature:
    msg_norm = normalize_msg(exception_msg)
    payload = repr((failure_kind, exception_type, msg_norm,
                    tuple(sorted(warning_types)), returncode))
    h = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()
    return IssueSignature(
        sig_hash=h,
        failure_kind=failure_kind,
        exception_type=exception_type,
        warning_types=tuple(sorted(warning_types)),
        returncode=returncode,
        exception_msg_norm=msg_norm,
    )
