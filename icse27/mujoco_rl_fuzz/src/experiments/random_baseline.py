"""Random baseline experiment.

Usage:
    python -m src.experiments.random_baseline --config configs/default.yaml --budget 1000
"""
from __future__ import annotations

import argparse
import os
import sys

from ..config import load_config
from ..rl.random_policy import RandomPolicy
from ..runner import FuzzRunner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--root", default=None,
                    help="Project root (defaults to parent of --config dir).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    budget = args.budget if args.budget is not None else cfg.run.budget
    root = args.root or os.path.abspath(os.path.dirname(os.path.dirname(args.config)))
    if not os.path.isabs(root):
        root = os.path.abspath(root)

    pol = RandomPolicy(n_rollout_buckets=len(cfg.rollout.steps_buckets),
                       seed=cfg.run.seed)
    runner = FuzzRunner(cfg, pol, project_root=root)
    print(f"[random_baseline] root={root}  budget={budget}  seeds={len(runner.pool)}  "
          f"mutators={len(runner.allowed)}")
    summary = runner.loop(budget)
    print("\n========== SUMMARY ==========")
    print(f"raw     : {summary['n_raw']}")
    print(f"unique  : {summary['n_unique']}")
    print(f"by_kind : {summary.get('by_kind_unique')}")
    print(f"elapsed : {summary['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
