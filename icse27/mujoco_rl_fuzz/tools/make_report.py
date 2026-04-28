"""Read run_*.jsonl + summary_*.json under logs/ and produce:

- outputs/report/cumulative_unique.png  : unique signatures vs step (one line/policy)
- outputs/report/by_kind_stacked.png    : unique counts by kind (ok/compile/warning_only)
- outputs/report/mutator_usage.png      : per-policy mutator histogram
- outputs/report/reward_curve.png       : smoothed step reward over time
- outputs/report/REPORT.md              : full markdown report w/ algo descriptions

Usage:
    python -m tools.make_report
    python -m tools.make_report --policies random reinforce vanilla_ac a2c ppo
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- algorithm names + filenames ---------------------------------------------

POLICY_FILE = {
    "random":     "random",
    "reinforce":  "actor_critic_reinforce",
    "vanilla_ac": "actor_critic_vanilla_ac",
    "a2c":        "actor_critic_a2c",
    "ppo":        "actor_critic_ppo",
}

POLICY_DESC = {
    "random":     ("Random baseline",
                   "Uniformly samples (mutator, param_seed, rollout_bucket). "
                   "No learning, no state — pure exploration baseline."),
    "reinforce":  ("REINFORCE (Williams 1992)",
                   "Monte-Carlo policy gradient with batch-mean baseline. "
                   "Uses the value head only as a constant baseline; full-episode "
                   "returns; high variance."),
    "vanilla_ac": ("Vanilla 1-step Actor-Critic (GzFuzz-style)",
                   "TD(0) bootstrap: delta = r + gamma*V(s') - V(s). "
                   "Single gradient step per buffer flush; no minibatching, "
                   "no trust region. The simplest 'real' AC."),
    "a2c":        ("A2C (synchronous N-step)",
                   "N-step return + advantage normalization + global-norm clip. "
                   "One gradient step per rollout flush."),
    "ppo":        ("PPO + GAE (Schulman 2017)",
                   "Clipped surrogate objective (eps=0.2), GAE(lambda=0.95), "
                   "4 epochs over MB=32 — recommended main algorithm."),
}

# --- mutators (kept in canonical order matching src/mutations/registry.py) ----
MUTATOR_IDS = [
    "STRUCT_GROW", "STRUCT_SHRINK", "STRUCT_REWIRE",
    "GEOM_PERTURB", "JOINT_PERTURB", "INERTIAL_PERTURB", "CONTACT_PERTURB",
    "ACTUATOR_EDIT", "SOLVER_TOGGLE", "STATE_PERTURB",
]


# ---- IO ----------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- analysis helpers --------------------------------------------------------

def cumulative_unique_series(rows):
    """Return (steps, unique_count) cumulative."""
    seen = set()
    xs, ys = [], []
    for i, r in enumerate(rows, start=1):
        sig = r.get("signature")
        if sig is not None:
            seen.add(sig)
        xs.append(i)
        ys.append(len(seen))
    return xs, ys


def by_kind_unique_dict(summary):
    """v7: split into real-finding kinds and `invalid` (fuzzer-internal)."""
    if summary is None:
        return {"ok": 0, "warning_only": 0, "runtime": 0, "crash": 0,
                "inconsistency": 0, "invalid": 0, "compile": 0}
    d = summary.get("by_kind_unique") or {}
    out = {k: int(d.get(k, 0)) for k in ("ok", "warning_only", "runtime",
                                          "crash", "inconsistency",
                                          "invalid", "compile")}
    return out


def real_bug_count(rows):
    """Count testcases tagged `real_bug_candidate` by SpontaneousNanOracle."""
    n = 0
    sigs: set = set()
    for r in rows:
        for v in r.get("verdicts") or []:
            if "real_bug_candidate" in (v.get("tags") or []):
                n += 1
                if r.get("signature"):
                    sigs.add(r["signature"])
                break
    return n, len(sigs)


def mutator_usage(rows):
    c = Counter()
    for r in rows:
        for m in r.get("mutations", []) or []:
            mid = m.get("mutator_id")
            if mid:
                c[mid] += 1
    return c


def smoothed_reward(rows, win=50):
    rewards = [float(r.get("reward", 0.0)) for r in rows]
    if not rewards:
        return [], []
    out = []
    s, q = 0.0, []
    for x in rewards:
        q.append(x)
        s += x
        if len(q) > win:
            s -= q.pop(0)
        out.append(s / len(q))
    return list(range(1, len(out) + 1)), out


# ---- plotting ----------------------------------------------------------------

PLT_COLOR = {
    "random":     "#888888",
    "reinforce":  "#1f77b4",
    "vanilla_ac": "#2ca02c",
    "a2c":        "#ff7f0e",
    "ppo":        "#d62728",
}


def plot_cumulative_unique(per_policy, out_png):
    plt.figure(figsize=(8, 5))
    for name, data in per_policy.items():
        xs, ys = data["cum"]
        plt.plot(xs, ys, label=name, color=PLT_COLOR.get(name, None), linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("cumulative unique signatures")
    plt.title("Cumulative unique signatures vs. step (budget=1000, 4 seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def plot_by_kind_stacked(per_policy, out_png):
    names = list(per_policy.keys())
    ok = [per_policy[n]["by_kind"]["ok"] for n in names]
    wo = [per_policy[n]["by_kind"]["warning_only"] for n in names]
    rt = [per_policy[n]["by_kind"]["runtime"] for n in names]
    cr = [per_policy[n]["by_kind"]["crash"] for n in names]
    inv = [per_policy[n]["by_kind"]["invalid"] +
           per_policy[n]["by_kind"]["compile"]  # legacy logs may still use "compile"
           for n in names]
    x = list(range(len(names)))
    plt.figure(figsize=(8, 5))
    plt.bar(x, inv, label="invalid (fuzzer noise)", color="#cccccc")
    base = list(inv)
    plt.bar(x, ok, bottom=base, label="ok", color="#9ecae1")
    base = [a + b for a, b in zip(base, ok)]
    plt.bar(x, wo, bottom=base, label="warning_only", color="#a1d99b")
    base = [a + b for a, b in zip(base, wo)]
    plt.bar(x, rt, bottom=base, label="runtime", color="#fd8d3c")
    base = [a + b for a, b in zip(base, rt)]
    plt.bar(x, cr, bottom=base, label="crash", color="#d62728")
    plt.xticks(x, names)
    plt.ylabel("unique signatures")
    plt.title("Unique signatures by kind (v7: invalid separated)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def plot_mutator_usage(per_policy, out_png):
    names = list(per_policy.keys())
    n_mut = len(MUTATOR_IDS)
    width = 0.8 / max(1, len(names))
    plt.figure(figsize=(11, 5.5))
    for i, name in enumerate(names):
        c = per_policy[name]["mut_usage"]
        total = sum(c.values()) or 1
        ys = [c.get(m, 0) / total for m in MUTATOR_IDS]
        xs = [j + i * width for j in range(n_mut)]
        plt.bar(xs, ys, width=width, label=name, color=PLT_COLOR.get(name, None))
    plt.xticks([j + 0.4 - width / 2 for j in range(n_mut)],
               MUTATOR_IDS, rotation=30, ha="right")
    plt.ylabel("usage fraction")
    plt.title("Mutator usage distribution per policy")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def plot_reward_curve(per_policy, out_png):
    plt.figure(figsize=(8, 5))
    for name, data in per_policy.items():
        xs, ys = data["reward"]
        plt.plot(xs, ys, label=name, color=PLT_COLOR.get(name, None), linewidth=1.2)
    plt.xlabel("step")
    plt.ylabel("reward (window=50)")
    plt.title("Smoothed per-step reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


# ---- markdown report ---------------------------------------------------------

def build_markdown(per_policy, out_md, budget):
    lines = []
    lines.append("# MJ-Fuzz benchmark report")
    lines.append("")
    lines.append(f"- Budget per policy: **{budget}**")
    lines.append("- Seeds: pendulum, double_pendulum, box_stack, slider_crank")
    lines.append("- MuJoCo: 3.2.3   |   PyTorch: 2.4.1+cpu   |   Python: 3.8.6")
    lines.append("- Subprocess-isolated executor; reward & triage per `configs/default.yaml`.")
    lines.append("")
    lines.append("## 1. Algorithms compared")
    lines.append("")
    lines.append("| Policy | Family | One-line description |")
    lines.append("|---|---|---|")
    for name in per_policy:
        title, desc = POLICY_DESC[name]
        lines.append(f"| `{name}` | {title} | {desc} |")
    lines.append("")
    lines.append("## 2. Headline results")
    lines.append("")
    lines.append("**`real_findings` excludes fuzzer-internal `invalid` inputs (compile-time "
                 "failures from malformed MJCF). `real_bugs` = testcases flagged by "
                 "`SpontaneousNanOracle` (finite input \u2192 NaN/Inf or spontaneous "
                 "BADQPOS/BADQVEL/BADQACC; not user-injected).**")
    lines.append("")
    lines.append("| Policy | raw | real_unique | invalid | ok | warn | runtime | crash | real_bugs (raw / sig) | elapsed (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, d in per_policy.items():
        s = d["summary"] or {}
        bk = d["by_kind"]
        elapsed = float(s.get("elapsed_sec", 0.0))
        invalid_cnt = bk.get("invalid", 0) + bk.get("compile", 0)
        real_unique = (bk.get("ok", 0) + bk.get("warning_only", 0) +
                       bk.get("runtime", 0) + bk.get("crash", 0) +
                       bk.get("inconsistency", 0))
        rb_raw, rb_sig = d.get("real_bug", (0, 0))
        lines.append(f"| `{name}` | {s.get('n_raw','?')} | **{real_unique}** | {invalid_cnt} | "
                     f"{bk.get('ok',0)} | {bk.get('warning_only',0)} | "
                     f"{bk.get('runtime',0)} | {bk.get('crash',0)} | "
                     f"{rb_raw} / {rb_sig} | {elapsed:.1f} |")
    lines.append("")
    lines.append("## 3. Plots")
    lines.append("")
    lines.append("### 3.1 Cumulative unique signatures vs. step")
    lines.append("![cumulative](cumulative_unique.png)")
    lines.append("")
    lines.append("### 3.2 Unique signatures by kind")
    lines.append("![by_kind](by_kind_stacked.png)")
    lines.append("")
    lines.append("### 3.3 Mutator usage distribution")
    lines.append("![mut](mutator_usage.png)")
    lines.append("")
    lines.append("### 3.4 Smoothed reward (window=50)")
    lines.append("![reward](reward_curve.png)")
    lines.append("")
    lines.append("## 4. Top-10 real-finding signatures per policy")
    lines.append("")
    lines.append("*(`invalid` signatures \u2014 MutationSkip / compile failures from "
                 "malformed MJCF \u2014 are excluded; they are fuzzer-internal noise, "
                 "not MuJoCo bugs.)*")
    lines.append("")
    for name, d in per_policy.items():
        s = d["summary"] or {}
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| sig | count | kind | exception | warnings |")
        lines.append("|---|---|---|---|---|")
        rows_sig = (s.get("top_signatures_real")
                    or [r for r in (s.get("top_signatures") or [])
                        if r.get("kind") not in ("invalid", "compile")])
        for row in rows_sig[:10]:
            warns = ",".join(row.get("warnings") or []) or "-"
            etype = row.get("etype") or "-"
            lines.append(f"| `{row.get('sig')}` | {row.get('count')} | "
                         f"{row.get('kind')} | {etype} | {warns} |")
        if not rows_sig:
            lines.append("| _(none \u2014 only invalid inputs were produced)_ | | | | |")
        lines.append("")
    lines.append("## 5. Notes")
    lines.append("")
    lines.append("- **`real_unique`** = ok + warning_only + runtime + crash + inconsistency. "
                 "Invalid (compile-failures from malformed mutator output) are EXCLUDED "
                 "and shown separately as `invalid` \u2014 they are not MuJoCo bugs.")
    lines.append("- **`real_bugs`** comes from `SpontaneousNanOracle`: testcases where MuJoCo "
                 "produced NaN/Inf/BAD-warnings WITHOUT us injecting NaN/Inf into the state. "
                 "`raw` = #testcases, `sig` = #distinct signatures.")
    lines.append("- Each step pre-compiles the mutated XML in the main process and resamples "
                 "params up to N times if compile fails (GZFuzz-style validity gate, "
                 "`run.validity_retries`). This collapses the old flood of trivial "
                 "compile-fail signatures.")
    lines.append("- See `outputs/real_bugs/*.json` for full diagnostics of each "
                 "`real_bug_candidate` testcase.")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---- main --------------------------------------------------------------------

def main(argv=None):
    # Resolve defaults relative to the project root (parent of tools/)
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _default_logs = os.path.join(_project_root, "logs")
    _default_out  = os.path.join(_project_root, "outputs", "report")

    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default=_default_logs)
    ap.add_argument("--out-dir", default=_default_out)
    ap.add_argument("--policies", nargs="+",
                    default=["random", "reinforce", "vanilla_ac", "a2c", "ppo"])
    ap.add_argument("--budget", type=int, default=1000,
                    help="Annotated in the report header only.")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    per = {}
    missing = []
    for name in args.policies:
        fbase = POLICY_FILE[name]
        jp = os.path.join(args.logs_dir, f"run_{fbase}.jsonl")
        sp = os.path.join(args.logs_dir, f"summary_{fbase}.json")
        if not os.path.exists(jp):
            missing.append(name)
            continue
        rows = load_jsonl(jp)
        # JSONL is append-mode in runner.py; keep only the most recent `budget`
        # rows so re-runs don't double-count older sessions.
        if len(rows) > args.budget:
            rows = rows[-args.budget:]
        per[name] = {
            "rows": rows,
            "summary": load_summary(sp),
            "cum": cumulative_unique_series(rows),
            "by_kind": by_kind_unique_dict(load_summary(sp)),
            "mut_usage": mutator_usage(rows),
            "reward": smoothed_reward(rows, win=50),
            "real_bug": real_bug_count(rows),
        }
        print(f"[loaded] {name}: rows={len(rows)}")

    if missing:
        print(f"[warn] missing logs for: {missing}")
    if not per:
        print("[error] no logs found.")
        return 1

    plot_cumulative_unique(per, os.path.join(args.out_dir, "cumulative_unique.png"))
    plot_by_kind_stacked(per, os.path.join(args.out_dir, "by_kind_stacked.png"))
    plot_mutator_usage(per, os.path.join(args.out_dir, "mutator_usage.png"))
    plot_reward_curve(per, os.path.join(args.out_dir, "reward_curve.png"))
    build_markdown(per, os.path.join(args.out_dir, "REPORT.md"), args.budget)

    print(f"[done] report written to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
