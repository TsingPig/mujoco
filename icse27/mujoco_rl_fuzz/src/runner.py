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
from .oracles.diff_oracle import SolverDiffOracle
from .oracles.runtime_oracle import RuntimeOracle
from .oracles.spontaneous_nan_oracle import SpontaneousNanOracle
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
        # Last execution result (for non-stale state featurisation - Fix 1)
        self.last_result: Optional[ExecutionResult] = None
        # Count-based curiosity (Fix 2): visits per (mutator_id, param_seed)
        self.action_visits: dict[tuple[int, int], int] = {}
        self.rng = random.Random(cfg.run.seed)

        self.out_dir = os.path.join(project_root, cfg.run.out_dir)
        self.log_dir = os.path.join(project_root, cfg.run.log_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        for sub in ("crashes", "warnings", "interesting", "minimized", "tc", "real_bugs"):
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

        self.oracles = [CompileOracle(), RuntimeOracle(), ConsistencyOracle(),
                        SpontaneousNanOracle(), SolverDiffOracle()]
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

    def _state_for_policy(self, seed_idx: int = 0) -> RawState:
        # Fix 1: feed the *previous* execution result back to the policy so
        # state_vec actually changes step-to-step. Falls back to blank state
        # for step 0.
        n_seeds = len(self.pool)
        if self.last_result is not None:
            return raw_state_from_result(
                self.last_result, self.history,
                last_reward=self.last_reward,
                last_novelty=self.last_novelty,
                seed_idx=seed_idx, n_seeds=n_seeds,
            )
        return RawState(
            model_shape=(0, 0, 0, 0, 0),
            last_compile_ok=True,
            last_runtime_ok=True,
            warning_counts={},
            last_reward=self.last_reward,
            last_novelty=self.last_novelty,
            mutation_history_ids=list(self.history),
            seed_idx=seed_idx,
            n_seeds=n_seeds,
        )

    def step(self, step_idx: int) -> ExecutionResult:
        pool_list = self.pool.list()
        n_seeds = len(pool_list)
        if n_seeds == 0:
            raise RuntimeError("No seeds in pool!")
        
        # Get placeholder state (policy will condition on seed later)
        state_vec = None
        state_text = ""
        try:
            from .rl.features import featurize, state_text as state_text_fn
            raw_state = self._state_for_policy(seed_idx=0)
            state_vec = featurize(raw_state, history_k=self.cfg.rl.history_k)
            state_text = state_text_fn(raw_state)
        except Exception:
            pass
        
        # Get mask from first seed (will be re-clamped later)
        mask = None
        try:
            tree = etree.parse(pool_list[0].xml_path)
            mask = self._build_action_mask(tree)
        except Exception:
            pass
        
        # Policy picks action
        action: Action = self.policy.select(state_vec, state_text, action_mask=mask)
        
        # Clamp indices
        action.mutator_idx = max(0, min(action.mutator_idx, len(MUTATOR_IDS) - 1))
        action.rollout_bucket_idx = max(0, min(action.rollout_bucket_idx, len(self.steps_buckets) - 1))
        action.seed_idx = max(0, min(action.seed_idx, n_seeds - 1))
        
        # Get the seed policy picked
        seed = pool_list[action.seed_idx]
        seed_idx = action.seed_idx
        
        # Parse the actual seed
        try:
            tree = etree.parse(seed.xml_path)
        except Exception as e:
            er = ExecutionResult(tc_id="seed_parse_fail",
                                 exception_type=type(e).__name__,
                                 traceback_summary=str(e)[:200])
            return er

        # Re-compute mask for the actual seed
        mask = self._build_action_mask(tree)
        
        # Update state representation for actual seed
        try:
            raw = self._state_for_policy(seed_idx=seed_idx)
            state_vec = featurize(raw, history_k=self.cfg.rl.history_k)
            state_text = state_text_fn(raw)
        except Exception:
            pass

        mid = MUTATOR_IDS[action.mutator_idx]
        if mid not in self.allowed_ids:
            # Policy picked a disabled mutator; degrade to first allowed
            mid = self.allowed[0].id if self.allowed else MUTATOR_IDS[0]
            action.mutator_idx = MUTATOR_IDS.index(mid)
        mutator = MUTATORS[mid]

        tc_id = stable_id(seed.seed_id, mid, action.param_seed, step_idx)
        out_xml = os.path.join(self.out_dir, "tc", f"{tc_id}.xml")

        # v7: GZFuzz-style validity gate.
        # Try the picked (mutator, param_seed). If the mutator declines or
        # the resulting XML fails to compile in this main process, RESAMPLE
        # params up to N times (cheap: no subprocess). This collapses the
        # giant pile of trivial "invalid input" signatures the report was
        # dominated by, and forces the worker to spend its budget on
        # behaviours MuJoCo will actually execute.
        max_validity_retries = int(getattr(self.cfg.run, "validity_retries", 3))
        mres = None
        params: dict = {}
        validity_attempts = 0
        for attempt in range(max_validity_retries + 1):
            validity_attempts = attempt + 1
            m_rng = random.Random(action.param_seed + attempt * 7919)
            try:
                params = mutator.sample_params(tree, m_rng)
            except Exception as e:
                params = {"_error": str(e)[:200]}
                mres = type("MR", (), {"ok": False, "new_xml_path": None,
                                       "reason": f"sample_raise:{type(e).__name__}",
                                       "runtime_directives": None})()
                continue
            try:
                mres = mutator.apply(tree, params, out_xml)
            except Exception as e:
                mres = type("MR", (), {"ok": False, "new_xml_path": None,
                                       "reason": f"apply_raise:{type(e).__name__}",
                                       "runtime_directives": None})()
                continue
            if not mres.ok:
                continue
            # Pre-compile validity check in the main process.
            if mres.new_xml_path and os.path.exists(mres.new_xml_path):
                try:
                    import mujoco as _mj  # local import: cheap, already loaded
                    _mj.MjModel.from_xml_path(mres.new_xml_path)
                    break  # valid -> stop retrying
                except Exception as e:
                    mres = type("MR", (), {"ok": False,
                                           "new_xml_path": mres.new_xml_path,
                                           "reason": f"precompile:{type(e).__name__}",
                                           "runtime_directives": None})()
                    continue
            else:
                # Mutator returned ok but no XML (state-only mutators)
                break

        mrec = MutationRecord(mutator_id=mid, params=params,
                              success=bool(mres and mres.ok),
                              reason=getattr(mres, "reason", None))
        rollout_steps = self.steps_buckets[action.rollout_bucket_idx]

        if not (mres and mres.ok):
            # All retries exhausted; classify as fuzzer-internal "invalid".
            er = ExecutionResult(tc_id=tc_id, compile_ok=False, runtime_ok=False,
                                 exception_type="MutationSkip",
                                 traceback_summary=(mrec.reason or "") +
                                                   f" [retries={validity_attempts}]")
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
        # v8: tag whether the *fuzzer* injected anything non-trivial into the
        # state. Anything beyond `small` scale (1e-2) can plausibly produce
        # MuJoCo warnings as expected behaviour; we should NOT count those as
        # "spontaneous" bugs. This expands the previous `nan/inf`-only filter
        # which generated 10/11 false positives in the v7 bench.
        injected_bad = False
        sp = directives.get("state_perturb") or {}
        for vals in sp.values():
            for v in (vals or []):
                if isinstance(v, str) and v.lower() in ("nan", "+inf", "-inf", "inf"):
                    injected_bad = True
                    break
                # Numeric perturbation strong enough to plausibly cause
                # MuJoCo to legitimately warn -> not a real bug signal.
                try:
                    fv = float(v)
                    if abs(fv) > 1e1:
                        injected_bad = True
                        break
                except (TypeError, ValueError):
                    pass
            if injected_bad:
                break
        raw["injected_nan_inf"] = injected_bad
        raw["rollout_steps_requested"] = int(rollout_steps)
        # v8: solver_override with too-few iterations legitimately produces
        # BADQACC ("solver did not converge"). Tag so SpontaneousNanOracle
        # can ignore. We treat <=2 iterations as deliberately-broken.
        ov = directives.get("solver_override") or {}
        if isinstance(ov.get("iterations"), int) and ov["iterations"] <= 2:
            raw["injected_nan_inf"] = True
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
        # Fix 2: count-based curiosity bonus on (mutator, param_seed).
        # Reward = c / sqrt(n_visits)  -> first visit ~+c, decays slowly.
        key = (action.mutator_idx, int(action.param_seed))
        n_visits = self.action_visits.get(key, 0) + 1
        self.action_visits[key] = n_visits
        c_curiosity = float(self.cfg.reward.get("curiosity_coef", 0.0))
        if c_curiosity > 0.0:
            bonus = c_curiosity / (n_visits ** 0.5)
            rb.components["curiosity"] = bonus
            rb.total += bonus
        self.last_reward = rb.total
        self.last_novelty = 1.0 if is_new else 0.0
        self.last_result = result  # Fix 1: cache for next step's state
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
        # v7: SpontaneousNanOracle marks the strongest real-bug suspects;
        # always persist them (regardless of new/old signature) into a
        # dedicated `real_bugs/` folder so they're easy to triage.
        is_real_bug = any(
            ("real_bug_candidate" in (v.tags or [])) for v in verdicts
        )
        if is_real_bug:
            write_json(os.path.join(self.out_dir, "real_bugs", f"{tc_id}.json"),
                       log_entry)
            # Also reward strongly so RL is pulled toward this region.
            extra = float(self.cfg.reward.get("real_bug_candidate", 20.0))
            rb.components["real_bug_candidate"] = extra
            rb.total += extra
            self.last_reward = rb.total

        if kind == "crash" or result.timeout:
            bucket = "crashes"
        elif kind in ("warning_only", "runtime"):
            bucket = "warnings"
        elif kind == "inconsistency":
            bucket = "interesting"
        elif kind == "invalid":
            # v7: invalid is fuzzer-internal noise — never persist as a finding.
            bucket = None
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
