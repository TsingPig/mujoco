"""Collect (state_text, action) demonstrations from RL/random fuzzing run logs.

Reads the JSONL produced by FuzzRunner (`logs/run_*.jsonl`) and emits a
dataset suitable for SFT / DPO:

  out_sft.jsonl :  {"prompt": "...", "completion": "{\"mutator\":...}"}
  out_dpo.jsonl :  {"prompt": "...", "chosen": "...", "rejected": "..."}

Demonstrations are filtered by reward percentile so the LLM only mimics
high-reward steps. DPO pairs are formed within the same (seed, mutator
group) bucket: high-reward = chosen, low-reward = rejected.

Usage:
    python -m tools.llm_collect_traces --in logs/run_rl_ppo.jsonl \
        --out-sft data/sft.jsonl --out-dpo data/dpo.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from src.mutations.registry import MUTATOR_IDS
from src.rl.actor_critic_policy import PARAM_SEED_TABLE


PROMPT_TEMPLATE = """You are a mutation selector for a MuJoCo XML fuzzer.
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


def _build_prompt(state_text: str, n_roll: int) -> str:
    return PROMPT_TEMPLATE.format(
        mutators="\n".join(f"- {m}" for m in MUTATOR_IDS),
        n_param=len(PARAM_SEED_TABLE),
        n_roll=n_roll,
        state_text=state_text or "(empty)",
    )


def _completion(action: dict) -> str:
    return json.dumps({
        "mutator": MUTATOR_IDS[action["mutator_idx"]],
        "param_seed_idx": PARAM_SEED_TABLE.index(action["param_seed"])
            if action["param_seed"] in PARAM_SEED_TABLE else 0,
        "rollout_idx": action["rollout_bucket_idx"],
    }, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL run log produced by FuzzRunner.")
    ap.add_argument("--out-sft", default=None)
    ap.add_argument("--out-dpo", default=None)
    ap.add_argument("--n-roll", type=int, default=5)
    ap.add_argument("--top-pct", type=float, default=0.3,
                    help="SFT keeps top X percentile by reward.")
    ap.add_argument("--bot-pct", type=float, default=0.3,
                    help="DPO uses bottom X percentile as 'rejected'.")
    args = ap.parse_args(argv)

    rows = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows:
        print("[collect] no rows", file=sys.stderr); return 1

    # Required fields: state_text, action {mutator_idx,param_seed,rollout_bucket_idx}, reward
    rows = [r for r in rows if "state_text" in r and "action" in r and "reward" in r]
    rows.sort(key=lambda r: r["reward"], reverse=True)
    n = len(rows)
    n_top = max(1, int(n * args.top_pct))
    n_bot = max(1, int(n * args.bot_pct))
    top, bot = rows[:n_top], rows[-n_bot:]

    n_sft = n_dpo = 0
    if args.out_sft:
        os.makedirs(os.path.dirname(args.out_sft) or ".", exist_ok=True)
        with open(args.out_sft, "w", encoding="utf-8") as f:
            for r in top:
                f.write(json.dumps({
                    "prompt": _build_prompt(r["state_text"], args.n_roll),
                    "completion": _completion(r["action"]),
                }, ensure_ascii=False) + "\n")
                n_sft += 1

    if args.out_dpo:
        os.makedirs(os.path.dirname(args.out_dpo) or ".", exist_ok=True)
        # Bucket by mutator group (rough novelty bucket) so chosen/rejected are comparable
        by_seed = defaultdict(list)
        for r in top: by_seed[r.get("seed_id", "_")].append(("chosen", r))
        for r in bot: by_seed[r.get("seed_id", "_")].append(("rejected", r))
        with open(args.out_dpo, "w", encoding="utf-8") as f:
            for seed, items in by_seed.items():
                chosen = [r for tag, r in items if tag == "chosen"]
                rejected = [r for tag, r in items if tag == "rejected"]
                for c, rj in zip(chosen, rejected):
                    f.write(json.dumps({
                        "prompt": _build_prompt(c["state_text"], args.n_roll),
                        "chosen": _completion(c["action"]),
                        "rejected": _completion(rj["action"]),
                    }, ensure_ascii=False) + "\n")
                    n_dpo += 1

    print(f"[collect] in={n} rows -> sft={n_sft}  dpo={n_dpo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
