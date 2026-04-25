"""RL-guided fuzzing experiment (Actor-Critic, default = PPO).

Usage:
    python -m src.experiments.rl_guided_fuzz --config configs/default.yaml --budget 10000
    python -m src.experiments.rl_guided_fuzz --algorithm a2c --budget 5000
"""
from __future__ import annotations

import argparse
import os
import sys

from ..config import load_config
from ..runner import FuzzRunner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--algorithm", choices=["reinforce", "vanilla_ac", "a2c", "ppo"],
                    default=None,
                    help="Override cfg.rl.policy. Defaults to ppo if unset.")
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    budget = args.budget if args.budget is not None else cfg.run.budget
    root = args.root or os.path.abspath(os.path.dirname(os.path.dirname(args.config)))

    algo = args.algorithm or ("ppo" if cfg.rl.policy in ("actor_critic", "ppo") else cfg.rl.policy)
    if algo not in ("ppo", "a2c", "vanilla_ac", "reinforce"):
        algo = "ppo"

    try:
        from ..rl.actor_critic_policy import ActorCriticPolicy
    except RuntimeError as e:
        print(f"[rl_guided_fuzz] {e}", file=sys.stderr)
        return 2

    pol = ActorCriticPolicy(
        algorithm=algo,
        history_k=cfg.rl.history_k,
        n_rollout_buckets=len(cfg.rollout.steps_buckets),
        hidden_dim=cfg.rl.hidden_dim,
        lr=cfg.rl.lr, gamma=cfg.rl.gamma,
        entropy_coef=cfg.rl.entropy_coef, value_coef=cfg.rl.value_coef,
        rollout_per_update=cfg.rl.rollout_per_update,
        seed=cfg.run.seed,
    )
    runner = FuzzRunner(cfg, pol, project_root=root)
    print(f"[rl_guided_fuzz] root={root}  algo={algo}  budget={budget}  "
          f"seeds={len(runner.pool)}")
    summary = runner.loop(budget)
    print("\n========== SUMMARY ==========")
    print(f"raw     : {summary['n_raw']}")
    print(f"unique  : {summary['n_unique']}")
    print(f"by_kind : {summary.get('by_kind_unique')}")
    print(f"elapsed : {summary['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
