"""CLI entry: dryrun (Step 1 done) | random | rl (Step 5 / Step 7+).

Usage:
    python -m src.main --help
    python -m src.main dryrun --config configs/default.yaml --seed seeds/pendulum.xml
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import load_config
from .seed_pool import SeedPool
from .utils import make_rng


def cmd_dryrun(args) -> int:
    cfg = load_config(args.config)
    pool = SeedPool(args.seeds_dir or cfg.seeds.dir, cfg.seeds.pattern)
    print(f"[dryrun] config            : {args.config}")
    print(f"[dryrun] seeds dir         : {pool.root}")
    print(f"[dryrun] seeds found       : {len(pool)}")
    for r in pool.list():
        print(f"           - {r.name}  id={r.seed_id}")
    print(f"[dryrun] mutators          : {cfg.mutators}")
    print(f"[dryrun] rl.policy         : {cfg.rl.policy}")
    print(f"[dryrun] llm.enabled       : {cfg.llm.enabled}")
    if args.seed:
        print(f"[dryrun] explicit seed file: {args.seed}  exists={os.path.exists(args.seed)}")
    print("[dryrun] OK (Step 1 skeleton). Subprocess harness coming in Step 2.")
    return 0


def cmd_random(args) -> int:
    print("[random] Not implemented yet (Step 5). See guide §18.")
    return 1


def cmd_rl(args) -> int:
    print("[rl] Not implemented yet (Step 7). See guide §18.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mujoco_rl_fuzz",
                                description="MuJoCo RL-guided fuzzing prototype (ICSE'27).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dry = sub.add_parser("dryrun", help="Validate config + list seeds + nothing else.")
    p_dry.add_argument("--config", default="configs/default.yaml")
    p_dry.add_argument("--seeds-dir", default=None)
    p_dry.add_argument("--seed", default=None, help="Optional single XML to check.")
    p_dry.set_defaults(func=cmd_dryrun)

    p_rand = sub.add_parser("random", help="Run random-baseline fuzzing.")
    p_rand.add_argument("--config", default="configs/default.yaml")
    p_rand.add_argument("--budget", type=int, default=None)
    p_rand.set_defaults(func=cmd_random)

    p_rl = sub.add_parser("rl", help="Run RL-guided fuzzing (actor-critic).")
    p_rl.add_argument("--config", default="configs/default.yaml")
    p_rl.add_argument("--budget", type=int, default=None)
    p_rl.set_defaults(func=cmd_rl)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
