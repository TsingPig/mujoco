"""Provenance metadata for layered seeds.

Every seed in any layer must record where it came from so that downstream
RL-guided exploration can reason about source distribution, license, and
reproducibility.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class Provenance:
    """Provenance for any layered seed.

    Fields are intentionally optional so the same dataclass works for L0
    curated assets, L1 generated scenes, L2 environment registrations, and
    L3 trajectory protocols.
    """
    source: str = "unknown"           # high-level source bucket, e.g. "menagerie", "robosuite", "generated"
    repo: Optional[str] = None        # upstream repo URL, if any
    commit: Optional[str] = None      # commit hash, if known
    package: Optional[str] = None     # python package, e.g. "robosuite", "gymnasium_robotics"
    package_version: Optional[str] = None
    src_rel_path: Optional[str] = None  # original path inside source repo
    local_path: Optional[str] = None    # path inside this workspace
    license_path: Optional[str] = None  # workspace-relative path to LICENSE
    collected_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Provenance":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_provenance(
    source: str,
    *,
    repo: Optional[str] = None,
    commit: Optional[str] = None,
    package: Optional[str] = None,
    package_version: Optional[str] = None,
    src_rel_path: Optional[str] = None,
    local_path: Optional[str] = None,
    license_path: Optional[str] = None,
    notes: Optional[str] = None,
    **extra: Any,
) -> Provenance:
    return Provenance(
        source=source,
        repo=repo,
        commit=commit,
        package=package,
        package_version=package_version,
        src_rel_path=src_rel_path,
        local_path=local_path,
        license_path=license_path,
        notes=notes,
        extra=dict(extra),
    )
