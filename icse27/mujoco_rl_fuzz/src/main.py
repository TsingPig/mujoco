"""CLI entry: dryrun | random | rl | llm.

Usage:
    python -m src.main --help
    python -m src.main dryrun  --config configs/default.yaml
    python -m src.main random  --config configs/default.yaml --budget 200
    python -m src.main rl      --config configs/default.yaml --budget 1000 --algorithm ppo
    python -m src.main llm     --config configs/default.yaml --budget 100
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import load_config
from .seed_pool import SeedPool


def _resolve_root(config_path: str) -> str:
    return os.path.abspath(os.path.dirname(os.path.dirname(config_path)))


def cmd_dryrun(args) -> int:
    cfg = load_config(args.config)
    root = _resolve_root(args.config)
    pool = SeedPool(os.path.join(root, cfg.seeds.dir), cfg.seeds.pattern)
    print(f"[dryrun] root={root}")
    print(f"[dryrun] seeds_dir={pool.root}  found={len(pool)}")
    for r in pool.list()[:10]:
        print(f"           - {r.name}  id={r.seed_id}")
    print(f"[dryrun] mutators={cfg.mutators}")
    print(f"[dryrun] rl.policy={cfg.rl.policy}  llm.enabled={cfg.llm.enabled}")
    return 0


def cmd_random(args) -> int:
    from .experiments.random_baseline import main as m
    forward = ["--config", args.config]
    if args.budget is not None:
        forward += ["--budget", str(args.budget)]
    return m(forward)


def cmd_rl(args) -> int:
    from .experiments.rl_guided_fuzz import main as m
    forward = ["--config", args.config]
    if args.budget is not None: forward += ["--budget", str(args.budget)]
    if args.algorithm:           forward += ["--algorithm", args.algorithm]
    return m(forward)


def cmd_llm(args) -> int:
    from .experiments.llm_baseline import main as m
    forward = ["--config", args.config]
    if args.budget is not None: forward += ["--budget", str(args.budget)]
    if args.enable:              forward += ["--enable"]
    if args.model:               forward += ["--model", args.model]
    if args.lora:                forward += ["--lora", args.lora]
    return m(forward)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mujoco_rl_fuzz",
                                description="MuJoCo RL-guided fuzzing prototype (ICSE'27).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("dryrun", help="Validate config and list seeds.")
    pa.add_argument("--config", default="configs/default.yaml")
    pa.set_defaults(func=cmd_dryrun)

    pa = sub.add_parser("random", help="Random baseline.")
    pa.add_argument("--config", default="configs/default.yaml")
    pa.add_argument("--budget", type=int, default=None)
    pa.set_defaults(func=cmd_random)

    pa = sub.add_parser("rl", help="RL-guided fuzzing (Actor-Critic, default PPO).")
    pa.add_argument("--config", default="configs/default.yaml")
    pa.add_argument("--budget", type=int, default=None)
    pa.add_argument("--algorithm",
                    choices=["reinforce", "vanilla_ac", "a2c", "ppo"],
                    default=None)
    pa.set_defaults(func=cmd_rl)

    pa = sub.add_parser("llm", help="LLM baseline (zero-shot by default; --enable to load model).")
    pa.add_argument("--config", default="configs/default.yaml")
    pa.add_argument("--budget", type=int, default=None)
    pa.add_argument("--enable", action="store_true")
    pa.add_argument("--model", default=None)
    pa.add_argument("--lora", default=None)
    pa.set_defaults(func=cmd_llm)

    pa = sub.add_parser("bench", help="Run multiple policies side-by-side.")
    pa.add_argument("--config", default="configs/default.yaml")
    pa.add_argument("--budget", type=int, default=200)
    pa.add_argument("--policies", nargs="+", default=["random", "a2c", "ppo"])
    pa.add_argument("--llm-enable", action="store_true")
    pa.set_defaults(func=lambda a: __import__(
        "src.experiments.benchmark", fromlist=["main"]).main(
        ["--config", a.config, "--budget", str(a.budget),
         "--policies", *a.policies] + (["--llm-enable"] if a.llm_enable else [])))

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
