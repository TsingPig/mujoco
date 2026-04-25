"""LLM Policy: zero-shot prompted (default) + interfaces for SFT / DPO finetune.

Heavy deps (`transformers`, `peft`) are imported lazily; if unavailable the
policy degrades to RandomPolicy and logs a warning. Action space and reward
are SHARED with ActorCriticPolicy via the same `policy_api` and `features`.

Output format the model is REQUIRED to produce:
    {"mutator": "GEOM_PERTURB", "param_seed_idx": 7, "rollout_idx": 2}

If JSON parse fails -> fallback to random; track parse-failure rate.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Optional

from ..mutations.registry import MUTATOR_IDS
from .actor_critic_policy import PARAM_SEED_TABLE
from .policy_api import Action, Transition

_PROMPT_TEMPLATE = """You are a mutation selector for a MuJoCo XML fuzzer.
Choose ONE mutator and ONE param_seed bucket and ONE rollout bucket.

Available mutators (pick one by exact name):
{mutators}

param_seed_idx is an integer in [0, {n_param}).
rollout_idx is an integer in [0, {n_roll}).

Current state:
{state_text}

Reply ONLY with one line of valid JSON:
{{"mutator": "...", "param_seed_idx": int, "rollout_idx": int}}
"""


class LLMPolicy:
    name = "llm_finetune"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B",
                 n_rollout_buckets: int = 5, max_new_tokens: int = 64,
                 device: str = "cpu", lora_path: Optional[str] = None,
                 enable: bool = False, seed: int = 0):
        self.model_name = model_name
        self.n_rollout_buckets = max(1, n_rollout_buckets)
        self.n_param = len(PARAM_SEED_TABLE)
        self.n_mut = len(MUTATOR_IDS)
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.lora_path = lora_path
        self._rng = random.Random(seed)
        self.parse_fail = 0
        self.calls = 0
        self._pipe = None
        if enable:
            self._init_pipeline()
        else:
            print("[LLMPolicy] enable=False -> running as random fallback "
                  "(set llm.enabled=true in config to load real model).")

    # ------------- model loading -------------

    def _init_pipeline(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.lora_path:
                from peft import PeftModel
                mdl = PeftModel.from_pretrained(mdl, self.lora_path)
            self._pipe = pipeline("text-generation", model=mdl, tokenizer=tok,
                                  device_map=self.device)
            print(f"[LLMPolicy] loaded {self.model_name} (lora={self.lora_path})")
        except Exception as e:
            print(f"[LLMPolicy] failed to load {self.model_name!r}: {e}. "
                  "Falling back to random.")
            self._pipe = None

    # ------------- decision -------------

    def _build_prompt(self, state_text: str) -> str:
        return _PROMPT_TEMPLATE.format(
            mutators="\n".join(f"- {m}" for m in MUTATOR_IDS),
            n_param=self.n_param,
            n_roll=self.n_rollout_buckets,
            state_text=state_text or "(empty)",
        )

    def _parse(self, text: str) -> Optional[dict]:
        m = re.search(r"\{.*?\}", text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if not isinstance(obj, dict):
                return None
            return obj
        except Exception:
            return None

    def select(self, state_vec, state_text: str,
               action_mask: Optional[list[bool]] = None) -> Action:
        self.calls += 1
        # Random fallback path
        if self._pipe is None:
            choices = [i for i, ok in enumerate(action_mask or [True] * self.n_mut) if ok]
            mut_idx = self._rng.choice(choices) if choices else 0
            return Action(
                mutator_idx=mut_idx,
                param_seed=PARAM_SEED_TABLE[self._rng.randrange(self.n_param)],
                rollout_bucket_idx=self._rng.randrange(self.n_rollout_buckets),
            )
        prompt = self._build_prompt(state_text)
        try:
            out = self._pipe(prompt, max_new_tokens=self.max_new_tokens,
                             do_sample=True, temperature=0.7)
            text = out[0]["generated_text"][len(prompt):]
        except Exception:
            text = ""
        obj = self._parse(text)
        if obj is None:
            self.parse_fail += 1
            return self.select_random_fallback(action_mask)
        try:
            mid = obj["mutator"]
            mut_idx = MUTATOR_IDS.index(mid)
            param_idx = int(obj.get("param_seed_idx", 0)) % self.n_param
            roll_idx = int(obj.get("rollout_idx", 0)) % self.n_rollout_buckets
        except Exception:
            self.parse_fail += 1
            return self.select_random_fallback(action_mask)
        # Honour mask
        if action_mask and not action_mask[mut_idx]:
            return self.select_random_fallback(action_mask)
        return Action(
            mutator_idx=mut_idx,
            param_seed=PARAM_SEED_TABLE[param_idx],
            rollout_bucket_idx=roll_idx,
        )

    def select_random_fallback(self, action_mask) -> Action:
        choices = [i for i, ok in enumerate(action_mask or [True] * self.n_mut) if ok]
        mut_idx = self._rng.choice(choices) if choices else 0
        return Action(
            mutator_idx=mut_idx,
            param_seed=PARAM_SEED_TABLE[self._rng.randrange(self.n_param)],
            rollout_bucket_idx=self._rng.randrange(self.n_rollout_buckets),
        )

    # ------------- finetune hooks (interface only) -------------

    def update(self, batch: list[Transition]) -> dict[str, float]:
        # Online finetune is intentionally NOT done here. Use:
        #   tools/llm_collect_traces.py  -> dump (state_text, action) demonstrations
        #   tools/llm_sft.py             -> LoRA SFT on demonstrations
        #   tools/llm_dpo.py             -> DPO on (high-reward, low-reward) pairs
        # Then re-instantiate LLMPolicy with `lora_path=<sft_or_dpo_dir>`.
        return {"calls": self.calls, "parse_fail": self.parse_fail}
