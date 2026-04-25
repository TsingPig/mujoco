"""Multi-policy benchmark: run random / a2c / ppo (and optionally llm) under
identical budget and seed, then print a side-by-side table.

Usage:
    python -m src.experiments.benchmark --config configs/default.yaml --budget 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from ..config import load_config
from ..rl.actor_critic_policy import ActorCriticPolicy
from ..rl.llm_policy import LLMPolicy
from ..rl.random_policy import RandomPolicy
from ..runner import FuzzRunner


def _run(name, policy, cfg, root, budget):
    t0 = time.time()
    runner = FuzzRunner(cfg, policy, project_root=root)
    summary = runner.loop(budget)
    summary["elapsed"] = round(time.time() - t0, 2)
    summary["policy"] = name
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--policies", nargs="+",
                    default=["random", "a2c", "ppo"],
                    choices=["random", "a2c", "ppo", "llm"])
    ap.add_argument("--llm-enable", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = os.path.abspath(os.path.dirname(os.path.dirname(args.config)))

    summaries = []
    for name in args.policies:
        print(f"\n========== running policy={name} budget={args.budget} ==========")
        if name == "random":
            pol = RandomPolicy(n_rollout_buckets=len(cfg.rollout.steps_buckets),
                               seed=cfg.run.seed)
        elif name in ("a2c", "ppo"):
            pol = ActorCriticPolicy(
                algorithm=name,
                history_k=cfg.rl.history_k,
                n_rollout_buckets=len(cfg.rollout.steps_buckets),
                hidden_dim=cfg.rl.hidden_dim,
                lr=cfg.rl.lr, gamma=cfg.rl.gamma,
                entropy_coef=cfg.rl.entropy_coef, value_coef=cfg.rl.value_coef,
                rollout_per_update=cfg.rl.rollout_per_update,
                seed=cfg.run.seed,
            )
        elif name == "llm":
            pol = LLMPolicy(model_name=cfg.llm.model_name,
                            n_rollout_buckets=len(cfg.rollout.steps_buckets),
                            enable=args.llm_enable, seed=cfg.run.seed)
        else:
            continue
        summaries.append(_run(name, pol, cfg, root, args.budget))

    # Table
    print("\n========== BENCHMARK RESULTS ==========")
    cols = ["policy", "n_raw", "n_unique", "elapsed"]
    print("  ".join(f"{c:>10}" for c in cols))
    for s in summaries:
        print("  ".join(f"{s.get(c, ''):>10}" for c in cols))
    print("\nby_kind_unique per policy:")
    for s in summaries:
        print(f"  {s['policy']:>10}  {s.get('by_kind_unique')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
