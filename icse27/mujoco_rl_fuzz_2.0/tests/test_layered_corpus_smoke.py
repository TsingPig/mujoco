"""Smoke tests for the v2.0-β layered corpus pipeline.

These tests deliberately avoid asserting on absolute counts; they verify
that *given existing curated actor seeds*, each pipeline stage:
  - imports without error
  - produces a non-empty manifest (or degrades gracefully)
  - is reachable end-to-end via the runner

Slow tests are gated via SKIP_HEAVY env var.
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKIP_HEAVY = os.environ.get("SKIP_HEAVY") == "1"


def _has_curated_actor() -> bool:
    curated = REPO / "seeds" / "curated"
    if not curated.exists():
        return False
    for sub in curated.iterdir():
        if sub.is_dir() and (sub / "model.xml").is_file():
            return True
    return False


def test_corpus_module_imports():
    from src.corpus import (
        ActorSeed, SyntheticSceneSeed, OpenEnvSeed, TrajectorySeed,
        seed_from_dict, append_manifest, iter_manifest, make_provenance,
    )
    assert ActorSeed is not None


def test_scene_module_imports_and_lists_templates():
    from src.scene import list_templates
    tmpls = list_templates()
    assert isinstance(tmpls, list) and len(tmpls) >= 1


def test_env_adapters_register_and_degrade_gracefully():
    from src.env_adapters import list_adapters, get_adapter
    names = list_adapters()
    assert len(names) >= 1
    for n in names:
        ad = get_adapter(n)
        st = ad.is_available()
        assert st.dependency in {"ok", "missing", "partial", "unknown"}


def test_oracles_registry_and_run_on_empty_trace():
    from src.oracles import list_oracles, run_oracles
    names = list_oracles()
    assert len(names) >= 6
    reports = run_oracles(None, oracle_filter=None, replay_meta={})
    assert len(reports) == len(names)
    for r in reports:
        assert hasattr(r, "name") and hasattr(r, "severity")


@pytest.mark.skipif(not _has_curated_actor(), reason="no curated actor seeds")
def test_runner_runs_short_actor_rollout():
    from src.corpus import ActorSeed
    from src.runner import RunRequest, run_seed
    from src.trajectory.protocol import TrajectoryProtocol, ActionSequence
    curated = REPO / "seeds" / "curated"
    actor = next(s for s in curated.iterdir() if s.is_dir() and (s / "model.xml").is_file())
    seed = ActorSeed(seed_id=actor.name, source_repo=None, source_path=None,
                     local_path=str(actor.relative_to(REPO)).replace("\\", "/"),
                     asset_dir=str(actor.relative_to(REPO)).replace("\\", "/"),
                     model_xml=str((actor / "model.xml").relative_to(REPO)).replace("\\", "/"),
                     tags=[], compile_status="unknown")
    req = RunRequest(seed=seed,
                     protocol=TrajectoryProtocol(action_sequence=ActionSequence(kind="zero_control", horizon=5)))
    res = run_seed(req, workspace_root=str(REPO))
    # Either fully ok, or a clean failure record. Never raise.
    assert res.seed_id == seed.seed_id
    assert isinstance(res.oracle_reports, list)


@pytest.mark.skipif(SKIP_HEAVY or not _has_curated_actor(),
                    reason="heavy: builds 3 synthetic scenes")
def test_build_synthetic_scenes_produces_at_least_one(tmp_path):
    out_root = tmp_path / "synthetic_scenes"
    quarantine = tmp_path / "quarantine"
    cmd = [sys.executable, str(REPO / "tools" / "build_synthetic_scenes.py"),
           "--count", "3",
           "--out-root", str(out_root),
           "--quarantine", str(quarantine)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr
    manifest = out_root / "manifest.jsonl"
    # 0 lines is acceptable only if all 3 happened to fail — flag instead.
    if manifest.exists():
        lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) >= 1, f"no scene compiled successfully; stderr:\n{proc.stderr}"


def test_visualize_corpus_list_runs(tmp_path):
    cmd = [sys.executable, str(REPO / "tools" / "visualize_corpus.py"), "--list"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr


def test_validate_layered_corpus_runs():
    cmd = [sys.executable, str(REPO / "tools" / "validate_layered_corpus.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr


def test_fetch_env_scenes_no_smoke_does_not_crash(tmp_path):
    """Smoke is skipped to avoid network/heavy deps; we just validate the
    script runs and writes (possibly empty) manifest."""
    out_root = tmp_path / "open_envs"
    cmd = [sys.executable, str(REPO / "tools" / "fetch_env_scenes.py"),
           "--out-root", str(out_root), "--no-smoke",
           "--max-per-adapter", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr
