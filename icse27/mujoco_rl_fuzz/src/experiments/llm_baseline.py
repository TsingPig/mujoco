"""LLM baseline experiment (zero-shot prompted by default).

Usage:
    python -m src.experiments.llm_baseline --config configs/default.yaml --budget 200
    python -m src.experiments.llm_baseline --enable --model Qwen/Qwen2.5-1.5B
"""
from __future__ import annotations

import argparse
import os
import sys

from ..config import load_config
from ..rl.llm_policy import LLMPolicy
from ..runner import FuzzRunner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--enable", action="store_true",
                    help="Actually load the LLM (requires transformers + GPU recommended).")
    ap.add_argument("--model", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    budget = args.budget if args.budget is not None else cfg.run.budget
    root = args.root or os.path.abspath(os.path.dirname(os.path.dirname(args.config)))

    pol = LLMPolicy(
        model_name=args.model or cfg.llm.model_name,
        n_rollout_buckets=len(cfg.rollout.steps_buckets),
        max_new_tokens=cfg.llm.max_new_tokens,
        lora_path=args.lora,
        enable=args.enable or cfg.llm.enabled,
        seed=cfg.run.seed,
    )
    runner = FuzzRunner(cfg, pol, project_root=root)
    print(f"[llm_baseline] root={root}  budget={budget}  enable_model={pol._pipe is not None}")
    summary = runner.loop(budget)
    print("\n========== SUMMARY ==========")
    print(f"raw     : {summary['n_raw']}")
    print(f"unique  : {summary['n_unique']}")
    print(f"by_kind : {summary.get('by_kind_unique')}")
    print(f"calls={pol.calls}  parse_fail={pol.parse_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
