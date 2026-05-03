# MuJoCo Fuzz 2.0 下一阶段：GzFuzz 风格模型搭建与 Actor-Critic 算法设计

更新时间：2026-04-30。

本文用于承接当前 12 类 seeds、operators、oracles 设计，并评估它们是否足以推进到 GzFuzz 风格的强化学习 fuzzing 框架。结论是：**当前 corpus 和 operator 设计已经具备进入 actor-critic 阶段的基础，但需要增加一个“语义动作层”和一个“episode runner + oracle feedback 层”，把研究级 operator 标签落到可执行动作、状态特征与奖励函数上。**

## 1. 对当前 seeds / operators 的适配性评估

当前实现相当适合继续沿 GzFuzz 的思路推进，理由如下。

第一，seed pool 已经不是零散 XML，而是带 bug taxonomy 的结构化 corpus。`seeds2` 共扫描 214 个 seeds，12 类累计成员约 493，平均每个 seed 命中约 2.3 类。这个多标签结构非常适合 RL：同一个模型可以在不同类别上下文中触发不同动作空间，避免把所有 mutator 混成一个无语义的大动作集合。

第二，bug taxonomy 来自真实历史 issue / PR，而不是只从 MJCF grammar 枚举。`_docs/mujoco_bug_reports_80cases.md` 中 109 条真实报告已经被映射到 C1-C12，每类都有可解释的 failure mode、trigger operation 和 oracle。这一点比单纯 grammar fuzzing 更接近 GzFuzz 的核心思路：利用领域知识把搜索空间压缩到更可能出 bug 的行为区域。

第三，底层 mutator 已经具备“可训练动作”的雏形。`src/mutations/registry.py` 中有 23 个 canonical mutator，`intensity.py` 给每个 mutator 维护固定 intensity mode。也就是说，当前动作已经可以编码为 `(mutator_id, intensity_mode)`，并能保证分布可复现。

第四，categories.yaml 中的 `operators:` 与 `oracles:` 字段提供了语义动作空间。它们目前更像研究级 macro-operator 标签，例如 `zero_control_rollout`、`kp_kd_sweep`、`mjx_classic_diff`、`scripted reach-close-lift`。下一步不应把这些标签直接塞给 RL，而应映射到一个可执行 action schema。

需要补齐的关键缺口也很明确：

| 缺口 | 当前状态 | 下一步需要做什么 |
|---|---|---|
| Episode runner | README 中已有 P2 计划，但尚未实现 | 实现 `src/runner/episode.py`，统一执行 compile、rollout、runtime directives、backend diff |
| Oracle feedback | `src/triage` 只有 signature | 实现静态、动态、差分 oracle，并输出结构化 failure signal |
| 语义动作映射 | `categories.yaml` 有 operator 标签，`registry.py` 有原子 mutator | 建立 `semantic_operator -> atomic mutator/runtime script/backend test` 映射 |
| State encoder | MANIFEST 中已有基础 metadata | 补充 compiled model features、历史动作、上次 oracle 信号、coverage/novelty 特征 |
| Reward | 还没有训练闭环 | 设计新 failure signature、oracle 异常强度、覆盖增量、有效编译率、时间成本组成的奖励 |

因此，我建议不要推翻当前设计。更稳的路线是：**保留 12 类 taxonomy 作为高层上下文，把现有 23 个 mutator 作为 L1/L2 原子动作，再为 C7-C12 增加环境级 runtime/backend operator。**

## 2. 与 GzFuzz 思路的对应关系

可以把当前系统映射为以下闭环：

```text
Seed corpus + bug taxonomy
        |
        v
Category-aware action space
        |
        v
Actor selects semantic operator / atomic mutator / intensity / target
        |
        v
Executor mutates MJCF or emits runtime directive
        |
        v
Compile + rollout + differential backend execution
        |
        v
Oracle produces failure signal + novelty signature
        |
        v
Critic estimates value; replay buffer updates policy
```

GzFuzz 风格的关键不是“必须照搬某个具体 RL 算法”，而是以下四点：

1. 把 fuzzing 看成 sequential decision-making，而不是每轮独立随机变异。
2. 让模型根据当前 seed、历史变异、执行反馈选择下一个 operator。
3. 用 failure novelty、coverage / state novelty、执行有效性作为奖励。
4. 把新发现的高价值变体回灌到 seed pool，形成增量 corpus。

当前 MuJoCo 项目的优势是 oracle 更强：物理仿真天然能提供 qpos、qvel、qacc、contact、sensor、reward、render frame、backend residual 等连续反馈。这比一般程序 fuzzing 只有 crash / coverage 的信号更丰富，也更适合 actor-critic。

## 3. 建议的总体模型

我建议采用两层策略：

1. **Category scheduler**：选择当前 episode 关注哪个 bug 类别或 seed-category pair。
2. **Operator actor-critic**：在该类别的动作空间内选择语义 operator、原子 mutator、intensity、target selector 和 runtime script。

第一阶段可以先固定 category，从每个 seed 的 labels 中均匀采样，直接训练一个共享 actor-critic。等 runner 和 oracle 稳定后，再加入 category scheduler。

### 3.1 MDP 定义

一个 episode 从 `(seed_id, category_id)` 开始，最多执行 `T` 个 step。每个 step 选择一个 operator，并得到一次 compile / rollout / oracle feedback。

**状态 `s_t`** 建议由五部分拼接：

| 特征组 | 内容 |
|---|---|
| Seed metadata | source、category multi-hot、nq、nv、nbody、ngeom、nu、是否包含 actuator / sensor / plugin / equality / tendon / mesh-heavy assets |
| Compiled model features | mass range、inertia condition proxy、geom type histogram、joint type histogram、actuator type histogram、contact param summary、solver / timestep / integrator |
| Mutation history | 最近 K 个 `(mutator_id, intensity_mode, target_type)`，累计 step 数，连续 compile failure 数 |
| Oracle feedback | 上一步 finite check、contact force max、energy spike、trajectory residual、backend diff residual、render pixel hash diff、reward residual |
| Search bookkeeping | 已见 signature 数、是否命中过历史 bug pattern、当前预算、耗时、seed 是否来自 replay pool |

**动作 `a_t`** 不建议只有 `mutator_id`，而应是参数化动作：

```text
a_t = (
  semantic_operator,
  atomic_operator,
  intensity_mode,
  target_selector,
  runtime_profile,
  backend_profile
)
```

其中 `semantic_operator` 来自 `categories.yaml`，`atomic_operator` 来自 `registry.py` 或 runner 内置 runtime script。第一版可以把动作离散化为合法组合表，例如：

```text
C3.mass_inertia_action_space = [
  (MUTATE_INERTIAL, near_zero_mass, random_body),
  (MUTATE_INERTIAL, anisotropic, high_mass_body),
  (SET_QVEL_RUNTIME, large, base_or_heavy_link),
  (runtime_off_center_impulse, medium, largest_body),
]
```

**转移 `P(s_{t+1}|s_t,a_t)`** 由 executor 决定：如果 XML mutation 编译失败，则回滚并进入 penalty state；如果编译通过，则运行 rollout / differential test，收集 oracle signal。

**终止条件** 包括：达到最大 step、发现新高危 signature、连续编译失败过多、模型被 shrink 判定不可继续、超时。

## 4. 动作层：从现有 mutator 到语义 operator

当前 23 个 mutator 可以作为 L1/L2 原子动作，但 C7-C12 需要额外 runtime / system-level operator。推荐建立一个单独文件，例如 `src/rl/action_space.py` 或 `src/operators/semantic.py`，维护如下映射：

| 语义 operator | 已有原子动作 | 还需补充的 executor |
|---|---|---|
| `mass_scale` / `diaginertia_scale` | `MUTATE_INERTIAL` | body target selector、small impulse rollout |
| `collision_geom_mutation` | `MUTATE_GEOM_SIZE`、`MUTATE_GEOM_SHAPE`、`MUTATE_CONTACT_MARGIN` | visual-collision AABB compare、penetration oracle |
| `ctrlrange_mutation` | `ACTUATOR_MUTATE_RANGE`、`MUTATE_JOINT_LIMIT` | target qpos replay、single actuator sweep |
| `timestep_integrator_sweep` | `MUTATE_TIMESTEP`、`MUTATE_INTEGRATOR`、`MUTATE_SOLVER_ITER` | repeated rollout residual |
| `state_reset_replay` | `SET_QPOS_RUNTIME`、`SET_QVEL_RUNTIME`、`SET_CTRL_RUNTIME` | reset / clone / warm-start differential |
| `mjx_classic_diff` | 无直接 XML mutator | classic vs MJX runner、trajectory residual oracle |
| `scripted_grasp_lift` | 部分使用 actuator/runtime directives | task script executor、object pose oracle |
| `render_backend_stress` | 无直接 XML mutator | EGL/OSMesa/GLFW/render loop executor |

这层很重要，因为 actor 不应该被迫学习“哪个底层 XML 属性等于哪个 bug 类别”。类别知识已经在 taxonomy 里，RL 应该学习在某个类别下如何排序、组合和调参。

## 5. Reward 设计

奖励需要兼顾 bug 发现、有效探索和运行成本。建议第一版使用如下形式：

```text
r_t =
  + R_new_bug      * I[new_failure_signature]
  + R_hist_match   * I[matched_historical_bug_pattern]
  + R_oracle_mag   * normalized_oracle_severity
  + R_novel_state  * state_or_coverage_novelty
  + R_valid        * I[compile_and_rollout_success]
  - P_compile      * I[compile_failure]
  - P_duplicate    * I[duplicate_signature]
  - P_timeout      * normalized_runtime_cost
```

推荐初始权重：

| 项 | 建议值 | 理由 |
|---|---:|---|
| `R_new_bug` | 10.0 | 新 signature 是核心目标 |
| `R_hist_match` | 3.0 | 历史 bug pattern 可作为早期 dense reward |
| `R_oracle_mag` | 0.5-2.0 | 连续异常强度有助于 critic 学习 |
| `R_novel_state` | 0.1-1.0 | 避免只围绕已知 crash 打转 |
| `R_valid` | 0.05 | 鼓励可执行而非乱改 XML |
| `P_compile` | 0.5-2.0 | 不能过大，否则模型不敢探索边界 |
| `P_duplicate` | 0.5 | 抑制重复发现 |
| `P_timeout` | 0.1-1.0 | 控制昂贵 backend / render test |

注意：编译失败不应一律视为无价值。C1/C12 的目标本来包含 load / build / import failure。但对于 C2-C11 的物理类 episode，编译失败应作为负奖励，因为它没有进入目标 oracle。

## 6. Actor-Critic 选择

第一版建议使用 PPO，而不是更复杂的 off-policy 方法。理由是动作空间可以先离散化，episode 较短，reward sparse 但 oracle severity 可提供 dense signal，PPO 工程风险低。

模型结构建议：

```text
Input state vector
    |
Shared MLP backbone
    |
    +-- Actor head: categorical distribution over legal action ids
    +-- Critic head: V(s)
```

如果后续加入参数化动作，可扩展为多头 actor：

```text
P(semantic_operator | s)
P(atomic_operator | s, semantic_operator)
P(intensity | s, semantic_operator, atomic_operator)
P(target_selector | s, semantic_operator, atomic_operator)
```

但第一版不要过早复杂化。建议先把合法 action tuple 预展开为离散 action id，并用 category mask 屏蔽不适用动作。

## 7. 训练流程伪代码

```text
Initialize policy pi_theta and value V_phi
Initialize replay seed pool S with curated seeds and historical regression seeds
Initialize signature database D

for iteration = 1..N:
    trajectories = []
    for worker = 1..W:
        seed = sample_seed(S)
        category = sample_category(seed.labels)
        model = load(seed)
        state = encode(seed, category, model, history=[])

        for t = 1..T:
            legal_actions = action_mask(category, model, state)
            action = sample(pi_theta(state), legal_actions)
            result = executor.apply(model, action)

            if result.compile_ok:
                feedback = runner.rollout_or_diff(model, action.runtime_profile)
                oracle = oracle_suite.check(seed, category, model, feedback)
            else:
                oracle = compile_failure_feedback(result)

            signature = triage_signature(seed, category, action, oracle)
            reward = compute_reward(signature, oracle, D, category)
            next_state = encode(seed, category, model, history + [action], oracle)

            trajectories.append(state, action, reward, next_state)

            if is_new_high_value(signature, D):
                D.add(signature)
                S.maybe_add(model, signature)
                dump_finding(seed, category, action_history, oracle)

            if terminal(result, oracle, t):
                break

            state = next_state

    update PPO(theta, phi, trajectories)
```

## 8. 实验设计建议

建议论文实验分为四组 RQ。

| RQ | 问题 | 指标 | 对比方法 |
|---|---|---|---|
| RQ1 | 12 类 seed/operator/oracle 是否覆盖真实历史 bug pattern | historical bug replay rate、oracle match rate | 手工 regression、随机 seed selection |
| RQ2 | Actor-critic 是否比随机/规则 fuzzing 更快发现独特 failure | unique signatures、time-to-first-bug、AUC | random operator、uniform category、coverage-guided、flat GzFuzz-style baseline |
| RQ3 | 语义 taxonomy 是否提升搜索效率 | valid mutation ratio、category hit rate、duplicate ratio | 无 taxonomy、只用 23 个 flat mutator |
| RQ4 | 三层 oracle 是否减少误报并提高可解释性 | confirmed bugs、false positive review、ablation success | crash-only oracle、NaN-only oracle、dynamic-only oracle |

对比方法建议包含：

1. `Random-Mutator`：均匀选择 seed、mutator、intensity。
2. `Rule-Based`：按类别固定 operator 顺序，不训练策略。
3. `Flat-AC`：actor-critic 只看 23 个 mutator，不使用 category mask。
4. `Taxonomy-AC`：本文主方法。
5. `Taxonomy-AC w/o Differential Oracle`：去掉 backend / version / parameter diff。
6. `Historical Regression Only`：只跑已知 regression 脚本，作为低探索 baseline。

## 9. 建议的实现里程碑

### M1: Runner 与 Finding Schema

实现：

- `src/runner/episode.py`
- `src/runner/executor.py`
- `src/oracles/static.py`
- `src/oracles/dynamic.py`
- `src/oracles/differential.py`
- `findings/<signature>/manifest.json`

最小输出字段沿用现有文档：`seed_id`、`category_id`、`operator_sequence`、`oracle_name`、`failure_signal`、`minimal_reproducer`、`suspected_root_cause`、`ablation_result`、`whether_historical_pattern_matched`。

### M2: Action Space 与 State Encoder

实现：

- `src/rl/action_space.py`
- `src/rl/state_encoder.py`
- `src/rl/reward.py`

先把每个 category 的合法 action tuple 离散展开，并用 mask 控制 actor 输出。

### M3: PPO 训练与并行采样

实现：

- `src/rl/ppo.py`
- `src/rl/train.py`
- `configs/rl_ppo.yaml`
- `experiments/run_taxonomy_ac.yaml`

第一版可以只跑 C1-C6，等稳定后扩展 C7-C12。

### M4: Replay Pool 与 Shrinking

实现：

- `src/seedpool/replay_pool.py`
- `tools/repro.py`
- `tools/shrink.py`

新 signature 对应 mutated XML 如果稳定复现，则进入 replay pool；重复发现时只更新统计，不重复保存。

## 10. 当前最适合先做的范围

为了尽快形成论文可跑实验，我建议第一轮聚焦 C1-C6：

| 类别 | 推荐原因 |
|---|---|
| C1 | preflight / compile / asset oracle 成本低，易自动化 |
| C2 | rollout 稳定性信号强，CoM drift / qacc spike 容易量化 |
| C3 | 现有 `MUTATE_INERTIAL` 已覆盖核心动作 |
| C4 | contact / penetration / force 是 MuJoCo 高价值缺陷面 |
| C5 | actuator range、gear、tracking 可用动作和 oracle 都清晰 |
| C6 | reset / replay / warm-start 与 RL 环境正确性高度相关 |

C7-C12 保留为扩展实验或 case study：它们很有论文价值，但对依赖、后端、渲染、JAX/GPU 环境要求更高，不适合作为第一版训练闭环的主路径。

## 11. 论文主张建议

论文可以不只说“我们把 GzFuzz 搬到 MuJoCo”。更好的主张是：

> MuJoCo 生态的 fuzzing 难点不是输入 grammar，而是物理模型、任务组合、后端差异与 oracle 语义。我们将真实历史 bug 映射为 category-specific seed/operator/oracle 三元组，并用 actor-critic 在该结构化动作空间中学习测试序列。相比 flat mutator fuzzing，该方法能把探索预算集中到高风险物理模式，并通过三层 oracle 生成可解释、可复现的 failure report。

这个主张和当前仓库状态是匹配的：已有分类、seed pool、原子 mutator、可视化 GUI 和历史 bug 索引；缺的是 runner、oracle、RL 训练与实验记录。
