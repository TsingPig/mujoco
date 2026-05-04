"""Episode-level execution: load a layered seed, replay its protocol,
evaluate oracles, return a RunResult.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from ..corpus.schema import (
    ActorSeed, SyntheticSceneSeed, OpenEnvSeed, TrajectorySeed, LayeredSeed,
)
from ..trajectory.protocol import TrajectoryProtocol, ActionSequence
from ..trajectory.replay import replay_actor, replay_scene, replay_env, ReplayResult

from .types import RunRequest, RunResult, OracleReport


def _default_protocol() -> TrajectoryProtocol:
    return TrajectoryProtocol(
        action_sequence=ActionSequence(kind="zero_control", horizon=50),
    )


def _resolve_actor_xml(seed: ActorSeed, workspace_root: Optional[str]) -> Optional[str]:
    """Best-effort: return absolute path to actor model.xml."""
    if seed.model_xml:
        if os.path.isabs(seed.model_xml) and os.path.isfile(seed.model_xml):
            return seed.model_xml
        if workspace_root:
            cand = os.path.join(workspace_root, seed.model_xml)
            if os.path.isfile(cand):
                return cand
        if os.path.isfile(seed.model_xml):
            return os.path.abspath(seed.model_xml)
    if seed.local_path:
        cand = os.path.join(seed.local_path, "model.xml")
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def run_seed(req: RunRequest, *, workspace_root: Optional[str] = None,
             parent_lookup=None) -> RunResult:
    """Execute a layered seed and return a RunResult.

    parent_lookup: optional callable(seed_id) -> LayeredSeed used by L3.
    """
    from ..oracles import run_oracles  # late import to avoid cycle

    seed = req.seed
    protocol = req.protocol or _default_protocol()
    if isinstance(seed, ActorSeed):
        xml_path = _resolve_actor_xml(seed, workspace_root)
        if not xml_path:
            return RunResult(ok=False, seed_id=seed.seed_id, layer="actor",
                             error="actor_xml_unresolved")
        rep = replay_actor(xml_path, protocol)
        return _wrap(rep, seed.seed_id, "actor", req.oracles)
    if isinstance(seed, SyntheticSceneSeed):
        xml_path = seed.scene_xml
        if xml_path and not os.path.isabs(xml_path) and workspace_root:
            xml_path = os.path.join(workspace_root, xml_path)
        if not xml_path or not os.path.isfile(xml_path):
            return RunResult(ok=False, seed_id=seed.seed_id, layer="synthetic_scene",
                             error=f"scene_xml_missing: {seed.scene_xml}")
        rep = replay_scene(xml_path, protocol)
        return _wrap(rep, seed.seed_id, "synthetic_scene", req.oracles)
    if isinstance(seed, OpenEnvSeed):
        rep = replay_env(seed, protocol)
        return _wrap(rep, seed.seed_id, "open_env", req.oracles)
    if isinstance(seed, TrajectorySeed):
        # Look up parent and run with its protocol (override with seed-specific
        # solver/backend/initial_state if present).
        if parent_lookup is None:
            return RunResult(ok=False, seed_id=seed.seed_id, layer="trajectory",
                             error="parent_lookup_required_for_trajectory_seed")
        try:
            parent = parent_lookup(seed.parent_seed_id)
        except Exception as exc:
            return RunResult(ok=False, seed_id=seed.seed_id, layer="trajectory",
                             error=f"parent_lookup_failed: {exc}")
        if parent is None:
            return RunResult(ok=False, seed_id=seed.seed_id, layer="trajectory",
                             error=f"parent_not_found: {seed.parent_seed_id}")
        proto = TrajectoryProtocol.from_dict({
            "initial_state": seed.initial_state,
            "action_sequence": seed.action_sequence_spec,
            "reset_protocol": seed.reset_protocol,
            "rollout_profile": seed.rollout_profile,
            "solver_profile": seed.solver_profile,
            "backend_profile": seed.backend_profile,
            "random_seed": seed.random_seed,
        })
        return run_seed(RunRequest(seed=parent, protocol=proto, oracles=req.oracles),
                        workspace_root=workspace_root, parent_lookup=parent_lookup)
    return RunResult(ok=False, error=f"unsupported_seed_type: {type(seed).__name__}")


def _wrap(rep: ReplayResult, seed_id: str, layer: str, oracle_filter):
    from ..oracles import run_oracles
    if not rep.ok:
        return RunResult(ok=False, seed_id=seed_id, layer=layer,
                         trace=rep.trace, error=rep.error, meta=rep.meta)
    reports = run_oracles(rep.trace, oracle_filter=oracle_filter, replay_meta=rep.meta)
    return RunResult(ok=True, seed_id=seed_id, layer=layer,
                     trace=rep.trace, oracle_reports=reports, meta=rep.meta)
