# A Reinforcement-Learning-Guided Fuzzer for the MuJoCo Physics Engine

**Technical Report — ICSE'27 prototype (`icse27/mujoco_rl_fuzz/`)**

---

## Abstract

We present **MJ-Fuzz**, a feedback-driven fuzzing framework targeting the
[MuJoCo](https://github.com/google-deepmind/mujoco) physics engine. MJ-Fuzz
treats MJCF (the XML-based scene description language) and the simulator's
runtime state jointly as the input space, applies a fixed catalogue of
**ten high-level mutation operators** inspired by GzFuzz, and uses a
**hierarchical Actor–Critic policy** (with REINFORCE / vanilla-AC / A2C /
PPO interchangeable training algorithms) to maximise a triage-aware reward
signal: new failure signatures, NaN / Inf propagation warnings,
solver-divergence warnings, and consistency violations. The same state
featurizer is shared by an LLM-based mutation selector that we use as a
baseline; the LLM head is trained with LoRA-SFT on high-reward traces and
DPO on (high-reward, low-reward) pairs. The framework is implemented in
~3.5 kLOC of Python, runs every test-case in an isolated sub-process, and
has been verified end-to-end on MuJoCo 3.2.3 / Windows 10.

---

## 1. Background and threat model

### 1.1 What is MuJoCo and why fuzz it?

[MuJoCo](https://mujoco.readthedocs.io/en/stable/) is the de-facto physics
backbone for robotics and embodied-AI research (DeepMind Suite, MuJoCo MPC,
Isaac Lab, Robosuite). Its C/C++ core is exposed to Python through pybind11
and mediates a complex, optimisation-heavy pipeline: XML compilation
(`mj_loadXML` / `mj_compile`), constraint factorisation, integrator stepping,
solver iteration. Bugs in this stack can corrupt training data of
downstream RL pipelines, so **finding crashes, simulator divergence, and
silent NaN propagation is research-relevant**.

We target three classes of defect:

| Defect class | Manifestation captured by MJ-Fuzz |
|---|---|
| **Compilation faults**   | C-level exceptions surfaced through the `mujoco` Python wrapper (e.g. *"mass and inertia of moving bodies must be larger than mjMINVAL"*). |
| **Runtime divergence**   | `BAD*` warning codes — `BADCTRL`, `BADQACC`, `BADQVEL`, `BADQPOS`, `WARNINEQ`, `CNSTRFULL`, `CONTACTFULL` — captured both via `mjData.warning[i].number` (counts) and `mujoco.set_mju_user_warning` (messages). |
| **Inconsistency**         | Two MjData instances initialised identically and stepped identically diverge in `qpos` beyond a numerical tolerance (`1e-9`). |

### 1.2 Threat model

We assume the user has (i) the upstream MuJoCo Python wheel, (ii) a corpus
of seed MJCF scenes, and (iii) an attacker who can supply arbitrary MJCF
+ initial state. The attacker's goal is to drive MuJoCo into one of the
defect classes above. The fuzzer's goal is to find such inputs efficiently.

---

## 2. System architecture

```
                ┌─────────────────────────────────────────────┐
                │                Fuzz Runner                  │
                │    (orchestrates one fuzzing iteration)     │
                └────────────────────┬────────────────────────┘
                                     │
         ┌───────────────┬───────────┼─────────────┬───────────────┐
         │               │           │             │               │
         ▼               ▼           ▼             ▼               ▼
┌────────────────┐ ┌──────────┐ ┌─────────┐ ┌────────────┐ ┌──────────────┐
│  Seed Pool     │ │  Policy  │ │ Mutator │ │ Subprocess │ │   Triage     │
│ (MJCF corpus)  │ │ (RL/LLM/ │ │ Library │ │  Worker    │ │ + Oracles    │
│                │ │  Random) │ │  (×10)  │ │ (isolated) │ │              │
└────────────────┘ └────┬─────┘ └────┬────┘ └─────┬──────┘ └──────┬───────┘
                        │            │            │               │
                        │  Action    │  Apply     │  Result       │  Verdict
                        │  ──────►   │  ──────►   │  ──────►      │  ──────►
                        │            │            │               │
                        └────────────┴────────────┴───────────────┘
                                                           │
                                                           ▼
                                                    Reward & Featurizer
                                                           │
                                                           ▼
                                                    Policy update
```

### 2.1 Module map

| Module | Path | Role |
|---|---|---|
| Seed pool       | `src/seed_pool.py`               | sample / refresh MJCF seeds |
| Mutators (×10)  | `src/mutations/xml_mutators.py`  | high-level XML / runtime ops |
| Subprocess worker | `src/engine/subprocess_worker.py` | crash-isolated execution |
| Oracles         | `src/oracles/`                   | compile / runtime / consistency / solver-diff |
| Triage          | `src/triage.py`                  | blake2s signature → unique-bug counter |
| Featurizer      | `src/rl/features.py`             | shared state vector + textual rendering |
| Reward          | `src/rl/reward.py`               | weighted multi-component scalar |
| RL networks     | `src/rl/networks.py`             | shared trunk + 3-head categorical policy + value head |
| RL algorithms   | `src/rl/algorithms/`             | REINFORCE / vanilla-AC / A2C / PPO |
| LLM policy      | `src/rl/llm_policy.py`           | zero-shot prompt + LoRA SFT/DPO interface |
| Runner          | `src/runner.py`                  | main loop (≈250 LOC) |

### 2.2 Sub-process isolation

Every test-case is dispatched to `python -m src.engine.subprocess_worker` via
`subprocess.run(timeout=…)`. A worker (a) parses the mutated XML, (b)
applies optional runtime *directives* (state perturbation, `disableflags`
toggles, integrator/solver overrides), (c) installs a user-warning hook
*before* compilation, (d) steps the simulation N times, (e) optionally
runs a second instance for consistency checking, and (f) writes a
structured JSON result. **No exception ever reaches the parent process**;
all NaN/Inf values are encoded as the strings `"nan"`, `"+inf"`, `"-inf"`
to survive JSON transport. This costs ~50 ms per test case but enables
unattended overnight runs.

---

## 3. Mutation catalogue (the "10 high-level operators")

We deliberately follow GzFuzz's design philosophy: **a small, fixed,
human-readable action space**. Each mutator is a deterministic function of
(seed XML, parameter dict). The parameter dict is constrained to a finite
**bucket grid** (see §6.1) so a discrete-action policy can address it.

| ID | Class | One-line semantics |
|---|---|---|
| `STRUCT_GROW`     | structural | append `<body><joint><geom>` subtree under a parent |
| `STRUCT_SHRINK`   | structural | delete a non-root `<body>` |
| `STRUCT_REWIRE`   | structural | move a body to a different (cycle-checked) parent |
| `GEOM_PERTURB`    | parameter  | flip `type`/`size` (skips planes) |
| `JOINT_PERTURB`   | parameter  | flip `type`/`range`/`damping` |
| `INERTIAL_PERTURB`| parameter  | rewrite `<inertial>` (mass / diaginertia buckets) |
| `CONTACT_PERTURB` | parameter  | set `friction` and `condim` |
| `ACTUATOR_EDIT`   | config     | add / delete / mutate a `<motor>` with `ctrlrange` |
| `SOLVER_TOGGLE`   | config     | rewrite `<option integrator solver iterations>` AND emit a runtime directive (`mjtIntegrator`/`mjtSolver` enums) |
| `STATE_PERTURB`   | runtime    | XML unchanged; emits a runtime directive that sets `qpos`/`qvel`/`ctrl` to {NaN, +Inf, ×scale}. **Auto-sets `disableflags |= mjDSBL_CLAMPCTRL`** when ctrl is the target so silent clamping does not mask `BADCTRL`. |

The order above is the *canonical action index* — the single source of
truth for both RL action heads and the LLM prompt enumeration.

---

## 4. Oracles and triage

### 4.1 Oracles

| Oracle | Trigger condition |
|---|---|
| `CompileOracle`     | `mj_loadXML` raises any exception |
| `RuntimeOracle`     | sub-process timeout, non-zero return code, NaN/Inf in `qpos`/`qvel`/`ctrl`, or any `BAD*` warning |
| `ConsistencyOracle` | `‖qpos_run1 − qpos_run2‖∞ > 1e-9` |
| `SolverDiffOracle`  | (consumed when the runner produces `solver_diff` payloads — present as a stub for cross-solver experiments) |

### 4.2 Triage

Each result is reduced to an `IssueSignature`

```python
sig = blake2s(failure_kind ‖ exception_type ‖ first_warning_code ‖ topology_hash, 8)
```

The `Triage` class maintains `n_raw`, `n_unique`, `by_kind_unique`, and
`top_signatures`. **Novelty (`is_new = sig not in seen`) feeds back into
the reward**, biasing the policy toward exploring untouched failure
modes.

---

## 5. State featurizer (the "shared head")

The featurizer is the **single component shared by RL and LLM** policies.
It produces both:

1. a **dense vector** `s ∈ R^32` for the neural policy (5 model dims +
   3 magnitude statistics + 4 boolean flags + 2 reward/novelty + 8 warning
   multi-hot + 10 mutation-history one-hot average), and
2. a **textual rendering** `state_text(s)` for the LLM prompt.

Both views are **deterministic functions of the same `RawState`** — this
is the technical lever that makes the RL-vs-LLM comparison fair.

---

## 6. Action space and policy head

### 6.1 Hierarchical discrete action

We factor every mutation step as a triple

$$
a = (m, p, k), \quad m \in \{1,\dots,10\},\quad p \in \{1,\dots,16\},\quad k \in \{1,\dots,K_{\text{roll}}\}
$$

* $m$ — mutator index (one of the 10).
* $p$ — index into a fixed table `PARAM_SEED_TABLE` of 16 prime ints
  (`13, 29, 47, …, 577`); each mutator maps it to its own bucket grid
  (e.g. `STATE_PERTURB.target ∈ {qpos,qvel,ctrl}` × `scale ∈ {nan,inf,1e1,1e3}`).
* $k$ — rollout-length bucket (config `[10, 50, 100, 500, 1000]` steps).

This factorisation collapses what would be a ~$10\times16\times5 = 800$-way
flat softmax into three small heads with shared trunk; on a 32-d input the
total parameter count is ≤ 50 k.

### 6.2 Network

```
RawState  ─►  Featurize  ─►  Linear(32→128) ─► GELU ─► Linear(128→128) ─► GELU
                                                        │
                       ┌────────────────────────────────┼────────────────────────────────┐
                       ▼                                ▼                                ▼
                Head_mut: Linear(128→10)        Head_param: Linear(128→16)         Head_value: Linear(128→1)
                       │                                │
                       ▼                                ▼
                       a_m  ── condition ──►  Head_roll: Linear(128+10 → K_roll)
                                                        │
                                                        ▼
                                                       a_k
```

Conditional sampling: $a_p \sim \text{Cat}(\text{softmax}(\text{Head}_p(h)))$
*after* $a_m$ is drawn; $a_k$ is conditioned on the chosen mutator
(simple concat). `evaluate(s, mask, a)` recomputes `log_prob`, `entropy`,
`value` for arbitrary $(m,p,k)$ — required by every algorithm in §7.

### 6.3 Action masking

If a mutator declares `applicable(tree) = False` for the current seed
(e.g. `STRUCT_SHRINK` on a single-body scene), the corresponding logit is
set to $-\infty$ before softmax. The policy never wastes a step on an
inapplicable operator.

---

## 7. Training algorithms — what is implemented

> **Yes**, the simplest GzFuzz-style 1-step actor-critic is supported as
> the `vanilla_ac` algorithm. Below is the complete catalogue.

All four algorithms share the same network (§6.2) and the same buffered
caller (`ActorCriticPolicy.update`) — they differ only in the loss formula.

### 7.1 Algorithm matrix

| Algorithm | Module | Uses critic? | TD horizon | Trust region | Recommended |
|---|---|---|---|---|---|
| `reinforce`  | `src/rl/algorithms/reinforce.py`  | only as constant baseline | Monte-Carlo (full episode) | none | ablation only |
| `vanilla_ac` | `src/rl/algorithms/vanilla_ac.py` | yes (TD-bootstrap) | 1-step | none | **GzFuzz baseline** |
| `a2c`        | `src/rl/algorithms/a2c.py`        | yes | N-step (= buffer size) | grad-clip + adv-norm | strong baseline |
| `ppo`        | `src/rl/algorithms/ppo.py`        | yes | N-step + GAE(λ=0.95) | clipped surrogate ε=0.2, 4 epochs × MB=32 | **main, recommended** |
| `impala`     | (placeholder)                      | — | — | V-trace | reserved for v0.3 |
| `a3c`        | (intentionally rejected)           | — | — | — | Windows-fork unfriendly + sub-process contention |

### 7.2 Loss formulas

**REINFORCE with mean baseline** (Williams, 1992):

$$
\mathcal{L} = -\frac{1}{T}\sum_{t} \log \pi_\theta(a_t \mid s_t) \cdot (R_t - \bar R) - \beta\, \mathbb{E}\,[H(\pi_\theta)]
$$

where $R_t = \sum_{k\geq t} \gamma^{k-t} r_k$ and $\bar R$ is the batch mean.

**Vanilla 1-step Actor–Critic** (the GzFuzz-style baseline):

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$
$$
\mathcal{L} = -\log\pi_\theta(a_t\mid s_t)\,\overline{\delta_t}
\;+\; c_v\,\tfrac{1}{2}\delta_t^2 \;-\; \beta\,H(\pi_\theta)
$$

Single gradient step per buffer flush; no minibatching. (Overline = stop-gradient.)

**A2C (synchronous N-step)**:

$$
R_t = \sum_{k=t}^{T-1}\gamma^{k-t} r_k, \quad
A_t = R_t - V_\phi(s_t),\ \tilde A_t = (A_t-\mu_A)/(\sigma_A+\varepsilon)
$$
$$
\mathcal{L} = -\log\pi(a_t\mid s_t)\,\tilde A_t + c_v(R_t - V_\phi(s_t))^2 - \beta H(\pi)
$$

Single gradient step + global-norm clip (0.5).

**PPO + GAE** (the recommended main algorithm; Schulman 2017):

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),\quad
\hat A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}
$$
$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$
$$
\mathcal{L}_{\text{CLIP}} = \mathbb{E}\,\bigl[\min(r_t \hat A_t,\;
\text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t)\bigr]
$$

We optimise $\mathcal{L} = -\mathcal{L}_{\text{CLIP}} + c_v(R_t - V_\phi)^2 - \beta H(\pi)$
for 4 epochs over minibatches of 32, with $\lambda = 0.95$, $\epsilon = 0.2$,
$\gamma = 0.99$.

### 7.3 Why PPO is the recommended main

In our setting an *action* is a mutation that takes 0.3–1.0 s of physics
simulation. Each gradient step is precious. PPO (i) is robust to learning
rate, (ii) samples are reused 4 times via the importance ratio + clipping,
which roughly quadruples sample efficiency vs A2C, and (iii) is the
strongest on-policy method whose update fits in ~80 LOC. Deeper algorithms
(SAC, IMPALA) are not justified at our buffer size (~64 transitions).

### 7.4 Verified runs (RTX 3060 Laptop, 4 seeds, budget=80)

| Algorithm   | n_raw | n_unique | by_kind |
|---|---|---|---|
| `reinforce`  | 80 | 13 | ok:8, compile:2, warning_only:3 |
| `vanilla_ac` | 80 | 14 | ok:9, compile:2, warning_only:3 |
| `a2c`        | 80 | 12 | ok:9, compile:2, warning_only:1 |
| `ppo`        | 80 | 12 | ok:8, compile:2, warning_only:2 |

Differences are within noise at 80 steps; the published comparison should
use ≥ 5 000 steps per algorithm.

---

## 8. Reward shaping

A scalar reward is computed per step from a `RewardBreakdown` whose components are weighted in `configs/default.yaml`:

| Component | Triggered when | Default weight |
|---|---|---|
| `crash`                  | exception in worker (compile or runtime) | +5.0 |
| `fatal_error`            | unrecoverable C-level abort | +10.0 |
| `timeout`                | sub-process `TimeoutExpired` | +2.0 |
| `new_warning`            | new `BAD*` code never seen before | +3.0 |
| `known_warning`          | repeat warning | +0.3 |
| `inconsistency`          | `‖Δqpos‖∞ > 1e-9` | +2.0 |
| `novel_state_bucket`     | `is_new` signature | +1.0 |
| `trivial_compile_fail`   | well-formed-XML rejected before any step | -0.2 (penalty) |
| `duplicate_signature`    | repeat of an old signature | -0.1 (penalty) |

The two negative terms are critical: without `trivial_compile_fail` the
policy collapses onto producing arbitrary garbage that fails to compile;
without `duplicate_signature` it gets stuck in any one failure mode.

---

## 9. LLM policy and finetune pipeline

### 9.1 Zero-shot policy

`LLMPolicy` builds a JSON-output prompt:

```
You are a mutation selector for a MuJoCo XML fuzzer.
Available mutators (pick one by exact name):
- STRUCT_GROW
- STRUCT_SHRINK
…
Current state:
<state_text(s)>

Reply ONLY with one line of valid JSON:
{"mutator": "...", "param_seed_idx": int, "rollout_idx": int}
```

The model output is parsed; `parse_fail` is tracked. Failure → random
fallback. Action space is **identical** to RL (same `MUTATOR_IDS`, same
`PARAM_SEED_TABLE`, same `n_rollout_buckets`) so the comparison is
controlled.

### 9.2 Three-phase finetune

```
RL run (PPO budget≥1000)
        │  logs/run_actor_critic_ppo.jsonl  (state_text + action + reward)
        ▼
tools/llm_collect_traces.py
   --top-pct 0.3   ──►  data/sft.jsonl   (high-reward demonstrations)
   --bot-pct 0.3   ──►  data/dpo.jsonl   (chosen/rejected pairs)
        │
        ▼
tools/llm_sft.py     (LoRA + optional 4-bit QLoRA + grad-ckpt)
   --model Qwen/Qwen2.5-1.5B  --quant 4bit  --grad-accum 4
        │
        ▼  checkpoints/sft_qwen15
tools/llm_dpo.py     (DPOTrainer; reuses SFT ckpt as both init and ref)
        │
        ▼  checkpoints/dpo_qwen15
src.main llm  --enable  --model checkpoints/dpo_qwen15
```

### 9.3 What you need to prepare to start an LLM-finetune experiment

**(a) Software** — install in this order on the training host:

```
pip install -r requirements.txt        # MuJoCo engine
pip install -r requirements-rl.txt     # GPU PyTorch (CUDA 12.x wheel for RTX cards)
pip install -r requirements-llm.txt    # transformers + peft + trl + datasets + bitsandbytes
```

**(b) Data** — produce a JSONL trace before you can SFT:

```
python -m src.main rl --config configs/default.yaml --budget 2000 --algorithm ppo
python -m tools.llm_collect_traces --in logs/run_actor_critic_ppo.jsonl \
       --out-sft data/sft.jsonl --out-dpo data/dpo.jsonl --top-pct 0.3 --bot-pct 0.3
```

A 2 000-step PPO run yields ~600 SFT and ~600 DPO rows, which is
**barely sufficient** for LoRA convergence on a 1.5 B model. For the
paper, plan for **≥ 10 000 RL steps** before collecting.

**(c) Hardware** — the 6 GB VRAM ceiling on a 3060 forces specific
hyperparameters. The recipe below has been validated to fit:

| Model | quant | batch | grad-accum | max_len | grad-ckpt | Approx VRAM |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | none | 2 | 1 | 1024 | off | ~3.0 GB |
| Qwen2.5-1.5B | 4-bit | 1 | 4  | 512  | on  | ~5.0 GB |
| Llama-3.2-3B | 4-bit | 1 | 8  | 384  | on  | ~5.5 GB (tight) |
| Qwen2.5-7B   | n/a   | — | —  | —    | —   | requires AutoDL 4090 (24 GB) |
| Qwen2.5-14B  | n/a   | — | —  | —    | —   | requires AutoDL A100 |

**(d) Tokenizer / chat template** — every model requires its tokenizer's
own template. The current SFT script appends `tokenizer.eos_token` to the
training text so generation halts cleanly; for instruct-tuned models
you may want to switch to `tokenizer.apply_chat_template`.

**(e) Storage** — ~6 GB per LoRA adapter checkpoint (full fp16 base + small LoRA);
plan ~20 GB free disk per training run for two epochs of checkpointing.

**(f) Hyperparameter starting points (LoRA)**:

| Hyperparam | Value | Notes |
|---|---|---|
| `lora_r` | 8 | bump to 16 if loss plateaus |
| `lora_alpha` | 16 | keep 2× `lora_r` |
| `target_modules` | `["q_proj","v_proj"]` | Qwen / Llama; for other archs check the model card |
| LR (SFT) | 2e-4 | classic LoRA setting |
| LR (DPO) | 5e-5 | smaller — DPO is sharper |
| `beta` (DPO) | 0.1 | start small to keep close to SFT init |
| epochs (SFT) | 3 | overfits quickly past 5 |
| epochs (DPO) | 2 | enough |

**(g) AutoDL workflow** — for ≥ 7 B models, a complete recipe is
documented in `EXECUTION.md` §4 and packaged as
`tools/autodl/{sync_to_remote.ps1, bootstrap.sh, train_remote.sh}`:

```
$env:AUTODL_HOST = "connect.westa.seetacloud.com"
$env:AUTODL_PORT = "12345"
.\tools\autodl\sync_to_remote.ps1                   # rsync push code
ssh -p 12345 root@$AUTODL_HOST                      # then on the remote:
  bash tools/autodl/bootstrap.sh                   # one-time env setup
  bash tools/autodl/train_remote.sh Qwen/Qwen2.5-7B ppo 5000
.\tools\autodl\sync_to_remote.ps1 -Pull            # rsync pull checkpoints
```

---

## 10. Experimental protocol for the paper

The minimum table to reproduce in the paper:

| Baseline | Setting | Budget |
|---|---|---|
| **B-rand**     | random mutator + uniform parameter | 10 000 |
| **B-greedy**   | epsilon-greedy on signature novelty | 10 000 |
| **B-LLM-zs**   | Qwen-1.5B zero-shot | 10 000 |
| **B-LLM-sft**  | + LoRA SFT on B-PPO traces | 10 000 |
| **B-LLM-dpo**  | + DPO | 10 000 |
| **A-reinf**    | REINFORCE | 10 000 |
| **A-vac**      | Vanilla 1-step AC (GzFuzz-style) | 10 000 |
| **A-a2c**      | A2C | 10 000 |
| **A-ppo**      | PPO + GAE (main) | 10 000 |

Three random seeds per row. Primary metrics:

* **Cumulative unique signatures** vs. budget (RQ1: efficiency)
* **Time-to-first-bug per failure_kind** (RQ2: triage-aware coverage)
* **Mutator-usage entropy** over time (RQ3: exploration profile)
* **LLM `parse_fail` rate** over training (RQ4: how much SFT/DPO improves
  format compliance)

Ablations:

* Reward components on/off (`new_warning`, `inconsistency`, novelty bonus)
* Action masking on/off
* Featurizer: drop warning multi-hot / drop history average
* PPO clip ε ∈ {0.1, 0.2, 0.3}; GAE λ ∈ {0.9, 0.95, 0.99}

---

## 11. Limitations and threats to validity

* **Single language target** — we fuzz only the public Python wrapper, not
  the C/C++ entry points; bugs only reachable through e.g. `mjMODEL`
  manual construction are out of scope.
* **Action space is fixed** — the 10 mutators are an *expert-designed*
  abstraction. New defect classes outside their span (e.g. mesh-decoder
  bugs in STL parsing) are reachable only via the structural mutators'
  side effects.
* **Reward weights are hand-tuned** — automated weight tuning (e.g.
  PBT or population search) is future work.
* **Single-machine experiments** — the framework spawns one worker at a
  time to avoid OOM on a 6 GB laptop GPU; throughput on a server-class
  CPU+GPU should scale linearly with `ProcessPoolExecutor`.

---

## 12. Reproducibility

| | |
|---|---|
| OS verified  | Windows 10 + PowerShell 5.1, Ubuntu 22.04 (AutoDL) |
| Python       | 3.10 |
| MuJoCo       | 3.2.3 |
| PyTorch      | 2.1+ (CUDA 12.x wheel for GPU) |
| Cmd ‒ smoke  | `python demo_smoke.py` (4/4 PASS) |
| Cmd ‒ random | `python -m src.main random --config configs/default.yaml --budget 200` |
| Cmd ‒ RL     | `python -m src.main rl --config configs/default.yaml --budget 1000 --algorithm ppo` |
| Cmd ‒ bench  | `python -m src.main bench --config configs/default.yaml --budget 500 --policies random vanilla_ac a2c ppo` |
| Cmd ‒ LLM    | see `EXECUTION.md` §3 |

Full installation, hyperparameter, and remote-training instructions are in
[`mujoco_rl_fuzz/EXECUTION.md`](mujoco_rl_fuzz/EXECUTION.md).

---

## Appendix A. File index

```
icse27/mujoco_rl_fuzz/
├── EXECUTION.md                         # operations manual
├── configs/default.yaml                 # all knobs
├── demo_smoke.py                        # 4-stage MuJoCo signal-capture demo
├── seeds/                               # 4 hand-written + curated MJCF
├── src/
│   ├── main.py                          # CLI: dryrun / random / rl / llm / bench
│   ├── runner.py                        # main fuzz loop
│   ├── triage.py                        # signature de-duplication
│   ├── seed_pool.py
│   ├── engine/
│   │   ├── subprocess_worker.py         # crash-isolated executor
│   │   ├── compile_and_run.py           # dispatcher
│   │   └── state_ops.py                 # NaN/Inf JSON encoding
│   ├── mutations/                       # 10 high-level mutators
│   ├── oracles/                         # 4 oracles
│   ├── rl/
│   │   ├── features.py                  # SHARED featurizer (RL + LLM)
│   │   ├── reward.py
│   │   ├── networks.py                  # Actor-Critic with 3 heads
│   │   ├── actor_critic_policy.py
│   │   ├── llm_policy.py
│   │   ├── random_policy.py
│   │   └── algorithms/                  # reinforce, vanilla_ac, a2c, ppo
│   └── experiments/
│       ├── random_baseline.py
│       ├── rl_guided_fuzz.py
│       ├── llm_baseline.py
│       └── benchmark.py                 # multi-policy side-by-side
├── tools/
│   ├── fetch_assets.py                  # curated upstream MJCF crawler
│   ├── llm_collect_traces.py            # build SFT + DPO datasets from RL logs
│   ├── llm_sft.py                       # LoRA SFT (+ optional 4-bit QLoRA)
│   ├── llm_dpo.py                       # DPO continue from SFT ckpt
│   └── autodl/
│       ├── sync_to_remote.ps1
│       ├── bootstrap.sh
│       └── train_remote.sh
└── tests/test_mutations.py
```
