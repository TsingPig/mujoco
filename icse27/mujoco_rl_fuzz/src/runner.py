"""Main fuzzing loop. Policy-agnostic.

For each step:
  1. sample seed
  2. policy.select(state) -> Action
  3. mutate XML / build runtime directives
  4. dispatch to subprocess worker
  5. parse result, run oracles, triage, compute reward
  6. log; persist anomalous testcases; feed Transition to policy.update buffer
"""
from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import asdict
from typing import Optional

from lxml import etree

from .config import FuzzConfig
from .engine.compile_and_run import RunSpec, run_in_subprocess
from .mutations.registry import MUTATORS, MUTATOR_IDS, filter_whitelist
from .oracles.compile_oracle import CompileOracle
from .oracles.consistency_oracle import ConsistencyOracle
from .oracles.runtime_oracle import RuntimeOracle
from .result import ExecutionResult, result_to_dict
from .rl.features import RawState, raw_state_from_result
from .rl.policy_api import Action, Policy, Transition
from .rl.reward import compute_reward
from .seed_pool import SeedPool
from .testcase import MutationRecord, TestCase
from .triage import Triage, signature_of
from .utils import append_jsonl, now_ts, stable_id, write_json


class FuzzRunner:
    def __init__(self, cfg: FuzzConfig, policy: Policy, project_root: str):
        self.cfg = cfg
        self.policy = policy
        self.root = project_root
        self.pool = SeedPool(os.path.join(project_root, cfg.seeds.dir),
                             cfg.seeds.pattern)
        self.allowed = filter_whitelist(cfg.mutators)
        self.allowed_ids = {m.id for m in self.allowed}
        self.triage = Triage()
        self.history: list[int] = []
        self.last_reward = 0.0
        self.last_novelty = 0.0
        self.rng = random.Random(cfg.run.seed)

        self.out_dir = os.path.join(project_root, cfg.run.out_dir)
        self.log_dir = os.path.join(project_root, cfg.run.log_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        for sub in ("crashes", "warnings", "interesting", "minimized", "tc"):
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

        self.oracles = [CompileOracle(), RuntimeOracle(), ConsistencyOracle()]
        self.steps_buckets = list(cfg.rollout.steps_buckets) or [50]
        self.run_log = os.path.join(self.log_dir, f"run_{policy.name}.jsonl")

    # --------------- one step ---------------

    def _build_action_mask(self, tree) -> list[bool]:
        mask = [False] * len(MUTATOR_IDS)
        for m in self.allowed:
            try:
                mask[MUTATOR_IDS.index(m.id)] = bool(m.applicable(tree))
            except Exception:
                mask[MUTATOR_IDS.index(m.id)] = False
        return mask

    def _state_for_policy(self) -> RawState:
        return RawState(
            model_shape=(0, 0, 0, 0, 0),
            last_compile_ok=True,
            last_runtime_ok=True,
            warning_counts={},
            last_reward=self.last_reward,
            last_novelty=self.last_novelty,
            mutation_history_ids=list(self.history),
        )

    def step(self, step_idx: int) -> ExecutionResult:
        seed = self.pool.sample(self.rng)
        try:
            tree = etree.parse(seed.xml_path)
        except Exception as e:
            # Seed itself broken; record and skip
            er = ExecutionResult(tc_id="seed_parse_fail",
                                 exception_type=type(e).__name__,
                                 traceback_summary=str(e)[:200])
            return er

        mask = self._build_action_mask(tree)
        state_vec = None
        state_text = ""
        # Late-import featurize to avoid pulling numpy if not needed for random
        try:
            from .rl.features import featurize, state_text as state_text_fn
            raw = self._state_for_policy()
            state_vec = featurize(raw, history_k=self.cfg.rl.history_k)
            state_text = state_text_fn(raw)
        except Exception:
            pass

        action: Action = self.policy.select(state_vec, state_text, action_mask=mask)
        action.mutator_idx = max(0, min(action.mutator_idx, len(MUTATOR_IDS) - 1))
        action.rollout_bucket_idx = max(0, min(action.rollout_bucket_idx, len(self.steps_buckets) - 1))

        mid = MUTATOR_IDS[action.mutator_idx]
        if mid not in self.allowed_ids:
            # Policy picked a disabled mutator; degrade to first allowed
            mid = self.allowed[0].id if self.allowed else MUTATOR_IDS[0]
            action.mutator_idx = MUTATOR_IDS.index(mid)
        mutator = MUTATORS[mid]

        # Mutate
        m_rng = random.Random(action.param_seed)
        try:
            params = mutator.sample_params(tree, m_rng)
        except Exception as e:
            params = {"_error": str(e)[:200]}

        tc_id = stable_id(seed.seed_id, mid, action.param_seed, step_idx)
        out_xml = os.path.join(self.out_dir, "tc", f"{tc_id}.xml")
        try:
            mres = mutator.apply(tree, params, out_xml)
        except Exception as e:
            mres = type("MR", (), {"ok": False, "new_xml_path": None,
                                   "reason": f"apply_raise:{type(e).__name__}",
                                   "runtime_directives": None})()

        mrec = MutationRecord(mutator_id=mid, params=params,
                              success=bool(mres.ok), reason=getattr(mres, "reason", None))
        rollout_steps = self.steps_buckets[action.rollout_bucket_idx]

        if not mres.ok:
            # Don't run worker; classify as "mutation_skip"
            er = ExecutionResult(tc_id=tc_id, compile_ok=False, runtime_ok=False,
                                 exception_type="MutationSkip",
                                 traceback_summary=mrec.reason or "")
            self._after_step(step_idx, seed.seed_id, tc_id, [mrec], rollout_steps,
                             er, raw={}, action=action,
                             state_vec=state_vec, state_text=state_text)
            return er

        # Build worker spec
        directives = mres.runtime_directives or {}
        spec = RunSpec(
            xml_path=mres.new_xml_path,
            rollout_steps=rollout_steps,
            state_perturb=directives.get("state_perturb"),
            disable_clamp_ctrl=directives.get("disable_clamp_ctrl", False),
            consistency_check=True,
            solver_override=directives.get("solver_override"),
        )

        result, raw = run_in_subprocess(tc_id, spec,
                                        timeout_sec=self.cfg.run.timeout_sec,
                                        workdir=os.path.join(self.out_dir, "tc"))
        self._after_step(step_idx, seed.seed_id, tc_id, [mrec], rollout_steps,
                         result, raw=raw, action=action,
                         state_vec=state_vec, state_text=state_text)
        return result

    # --------------- side effects: triage / persist / reward / policy update ---------------

    def _after_step(self, step_idx, seed_id, tc_id, mrecs, rollout_steps,
                    result: ExecutionResult, raw: dict, action: Action,
                    state_vec, state_text):
        sig, is_new = self.triage.update(result, step_idx)
        rb = compute_reward(result, is_new_signature=is_new,
                            sig_kind=sig.failure_kind, weights=self.cfg.reward)
        self.last_reward = rb.total
        self.last_novelty = 1.0 if is_new else 0.0
        self.history.append(action.mutator_idx)
        if len(self.history) > 64:
            self.history = self.history[-64:]

        verdicts = [o.evaluate(result, raw) for o in self.oracles]

        log_entry = {
            "step": step_idx,
            "tc_id": tc_id,
            "seed_id": seed_id,
            "mutations": [asdict(m) for m in mrecs],
            "rollout_steps": rollout_steps,
            "result": result_to_dict(result),
            "signature": sig.sig_hash,
            "sig_kind": sig.failure_kind,
            "is_new": is_new,
            "reward": rb.total,
            "reward_components": rb.components,
            "verdicts": [{"name": v.name, "triggered": v.triggered,
                          "severity": v.severity, "tags": v.tags} for v in verdicts],
            "policy": self.policy.name,
            "action": asdict(action),
            "state_text": state_text,
            "ts": now_ts(),
        }
        append_jsonl(self.run_log, log_entry)

        # Persist anomalous testcases
        kind = sig.failure_kind
        bucket = None
        if kind == "crash" or result.timeout:
            bucket = "crashes"
        elif kind in ("warning_only", "runtime"):
            bucket = "warnings"
        elif kind == "inconsistency":
            bucket = "interesting"
        elif kind == "compile":
            # Only persist non-trivial compile fails (new signature)
            if is_new:
                bucket = "interesting"
        if bucket and is_new:
            write_json(os.path.join(self.out_dir, bucket, f"{tc_id}.json"), log_entry)

        # Feed transition to policy
        try:
            tr = Transition(state_vec=state_vec, state_text=state_text,
                            action=action, reward=rb.total,
                            reward_components=rb.components, done=False)
            self.policy.update([tr])
        except Exception:
            pass

    # --------------- main loop ---------------

    def loop(self, budget: int) -> dict:
        t0 = time.time()
        for i in range(budget):
            try:
                self.step(i)
            except KeyboardInterrupt:
                break
            except Exception as e:
                append_jsonl(self.run_log,
                             {"step": i, "runner_error": f"{type(e).__name__}: {e}"})
            if (i + 1) % 50 == 0:
                self._print_progress(i + 1, time.time() - t0)
        elapsed = time.time() - t0
        summary = self.triage.summary()
        summary["elapsed_sec"] = elapsed
        summary["budget"] = budget
        summary["policy"] = self.policy.name
        write_json(os.path.join(self.log_dir, f"summary_{self.policy.name}.json"),
                   summary)
        return summary

    def _print_progress(self, i: int, elapsed: float) -> None:
        s = self.triage.summary()
        rate = i / max(elapsed, 1e-6)
        kinds = s.get("by_kind_unique", {})
        print(f"[{self.policy.name}] step={i}  raw={s['n_raw']}  unique={s['n_unique']}  "
              f"by_kind={kinds}  rate={rate:.2f}/s")
