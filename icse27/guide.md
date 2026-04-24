# MuJoCo RL-Guided Fuzzing 原型 — 完整实施指南

```
问: 你对 MuJoCo / fuzzing / RL 三者的当前掌握程度？
答: MuJoCo 零基础，fuzzing 零基础，RL 有经验（grid-world A3C 那种）
问: RL vs LLM-finetune 的对比，你希望共享到哪一层？
答: 共享 state featurizer（结构特征+warning+数值统计），RL/LLM 只换 policy head
问: LLM baseline 你设想的具体形态？
答: 小模型微调（Qwen2.5-1.5B / CodeLlama-7B）做 mutation 选择 + 参数预测
问: 可行性 demo 的具体程度？
答: 单脚本：1 seed + 1 mutation + mj_step，打印 warning/异常 → 证链路通
问: 现在就在当前 rl-game 仓库里搭可行性 demo？
答: 是，独立子目录搭 demo + 写指南，让我亲眼看到 warning/异常被抓到
问: ICSE'27 novelty claim 倾向？
答: RL vs LLM-finetune 在 simulator fuzzing 上的系统对比 + 共享 featurizer 统一框架
```


> 本文档面向**下一个 MuJoCo 仓库中的 Copilot Agent**，作为"如何从零做出一个最小可跑、可扩展为论文原型的 MuJoCo RL fuzzing 系统"的工作手册。
>
> **角色定位**：你是一个偏工程落地的研究型 Copilot。你的最高优先级是：
> 1. 让用户尽快跑起来
> 2. 让用户尽快有 random baseline
> 3. 让用户尽快有异常样本
> 4. 让用户后续容易加 RL 和写论文
>
> 不要做学术综述，不要写空泛设计图，每一步必须产出**可运行代码**。

---

## 0. 阅读须知

- 本文档是**单一事实来源 (Single Source of Truth)**，所有决策以此为准。
- 不要假设你看过用户之前任何对话或任何 RL/Fuzzing 项目。
- 不要假设你读过 GzFuzz / 任何论文。本文已包含所需启发。
- 严格按 §18 的实现顺序工作，不要跳步。
- 每一步都要 commit 可运行代码，不要只给设计图。

---

## 1. 项目目标

构建一个**针对 MuJoCo 上游主链路 (upstream)** 的 RL-guided fuzzing 原型。

**测试对象（按优先级）**：
1. `MJCF/XML` 的加载、解析、编译
2. `MjSpec` 程序化建模与结构变异
3. `mjModel` / `mjData` 的创建、重置、状态设置
4. `mj_step` 驱动下的运行时异常、warning、数值问题

**一句话目标**：

> 做一个能对 MuJoCo 官方主链路进行**结构化变异、执行、判错、学习变异策略**的最小 RL fuzzing 系统。

**当前阶段不测**：第三方下游项目（任务/控制器/reward）的逻辑 bug。

---

## 2. 严格的"不要做"清单

为了尽快落地，**当前阶段绝不要做**：

- ❌ 复杂物理正确性证明 / 能量守恒验证
- ❌ 一次性覆盖所有 MuJoCo API
- ❌ 先写庞大框架再慢慢填空
- ❌ viewer / rendering / GUI
- ❌ 多进程分布式训练
- ❌ 纯生成式 MJCF grammar fuzzing（字符串级随机生成）
- ❌ 第三方下游项目的 reward / task / controller bug
- ❌ 复杂深度网络（GNN、Transformer 大模型等）
- ❌ 跨版本差分测试（暂不做）
- ❌ C++ 实现（先纯 Python 原型）

要的是：**小而完整、能跑通、能持续产出 testcase 和异常信号。**

---

## 3. 关键背景与设计权衡

### 3.1 为什么测上游而非下游
MuJoCo 主链路被广泛复用，一个 compile/runtime bug 影响所有下游。下游 task 的 reward bug 是单点问题，价值低。

### 3.2 为什么先做结构化变异，不做纯随机 XML
纯字符串级随机生成 → 大量无效样本（无法 parse）→ 浪费算力。
结构化变异（基于 `MjSpec` 或 XML AST）→ 大多数样本可编译 → 信号密度高。

### 3.3 为什么必须先 random baseline，再 RL
如果 random 都产生不出异常信号，**RL 没有学习信号**，等于在零梯度上瞎学。
正确路线：harness → mutators → oracle → random baseline → RL。

### 3.4 GzFuzz 的核心启发（无需读原论文）
- 不直接生成完整复杂输入。
- 由一个**策略模块**决定"选哪类 generator/mutator"。
- 如该 mutator 需要参数，再由**参数选择模块**离散补全。
- 执行后根据反馈更新策略。

映射到 MuJoCo：
- 主策略 → 选哪个 mutation generator
- 参数选择 → 选参数 bucket
- 执行 testcase → 收集 reward
- 更新策略

---

## 4. 系统主循环

每轮 fuzzing iteration 的标准流程：

```
1.  从种子池选 seed model
2.  加载为 MjSpec / 可变异表示
3.  执行 1 次或多次 mutation
4.  得到 mutated model
5.  尝试 compile / load
6.  若 compile 成功 → 创建 mjData
7.  reset + 可选 state perturbation
8.  跑若干步 mj_step（在子进程内）
9.  收集 oracle 信号
10. 记录 testcase / 结果 / 日志
11. 给策略反馈 reward
12. 进入下一轮
```

### 必备三层能力

| 层 | 职责 |
|---|---|
| **Fuzzing Harness** | 加载种子、变异、编译、执行（子进程）、收集结果、保存失败样本 |
| **Mutation Library** | 一组离散、可控、可记录的结构化 mutator |
| **Oracle + Reward** | 把执行结果转为：异常类别 + 统计信号 + RL reward |

---

## 5. 技术栈

**强制使用**：
- Python 3.10+
- 官方 `mujoco` Python 包
- `numpy`
- `torch`（仅用于轻量 RL）
- `lxml` 或 `xml.etree.ElementTree`
- `multiprocessing` / `subprocess`
- `pyyaml`（配置）

**可选**：
- `gymnasium`（不是必须）

**禁止**：第三方重型 RL 框架（Ray、Stable-Baselines3 等），先手写小循环。

---

## 6. 项目目录结构

```text
mujoco_rl_fuzz/
  README.md
  requirements.txt
  configs/
    default.yaml
  seeds/
    README.md
    pendulum.xml
    cartpole.xml
    ...                     # 5~20 个种子
  logs/
  outputs/
    crashes/
    warnings/
    interesting/
    minimized/
  src/
    __init__.py
    main.py                 # CLI 入口
    runner.py               # 主循环
    config.py               # YAML → dataclass
    utils.py
    seed_pool.py
    testcase.py             # TestCase 数据结构
    result.py               # ExecutionResult
    triage.py               # 去重 / signature
    reducers.py             # 失败样本最小化（先留接口）
    oracles/
      __init__.py
      base.py
      compile_oracle.py
      runtime_oracle.py
      consistency_oracle.py
    mutations/
      __init__.py
      base.py
      registry.py
      xml_mutators.py
      spec_mutators.py
      param_buckets.py
    engine/
      __init__.py
      compile_and_run.py
      subprocess_worker.py  # ★ 子进程执行入口
      state_ops.py
    rl/
      __init__.py
      features.py
      reward.py
      random_policy.py
      bandit_policy.py
      actor_critic.py
      replay_buffer.py
      trainer.py
    experiments/
      random_baseline.py    # 模式 A
      rl_guided_fuzz.py     # 模式 B
  tests/
    test_mutations.py
    test_oracles.py
    test_runner.py
```

**要求**：
- 架构清晰，可直接 `python -m src.main` 启动。
- random baseline 与 RL-guided 两个模式必须分开脚本可独立跑。

---

## 7. Seed 设计

### 7.1 来源
优先选官方 / 高质量 MuJoCo XML，覆盖结构差异：
- 简单 pendulum / cartpole
- 多关节机械臂
- quadruped
- 带 actuator 的模型
- 带 contact / geom 的模型

### 7.2 数量
**第一版严格控制在 5~20 个种子**。先保证每个都能稳定 compile + run。

### 7.3 Seed Pool 接口约束
```python
class SeedPool:
    def list(self) -> list[str]: ...
    def sample(self, rng) -> SeedRecord: ...
    def add(self, xml_path: str, meta: dict) -> None: ...
```

---

## 8. Mutation Generators（最小集合）

每个 mutator 都必须：
- 有唯一 ID（字符串，如 `"add_body"`）
- 参数可序列化（dict[str, Any]，全部可 JSON）
- 输出人类可读日志
- 实现 `apply(testcase, params) -> MutationResult`
- 失败时返回明确原因（不抛异常吞掉信息）

### 8.1 结构级（必须 8 个）
| ID | 说明 | 关键参数 |
|---|---|---|
| `add_body` | 在已有 body 下加子 body | parent, pos_bucket, quat_bucket |
| `remove_body` | 删非根 body | target_id |
| `add_geom` | 给 body 加 geom | body, shape_type, size_bucket, pos_bucket |
| `remove_geom` | 删 geom | target_id |
| `add_joint` | 给 body 加 joint | body, jtype, axis_bucket, range_bucket |
| `remove_joint` | 删 joint | target_id |
| `add_actuator` | 加 actuator（绑定已有 joint） | joint, ctrlrange_bucket |
| `remove_actuator` | 删 actuator | target_id |

### 8.2 参数级（必须 8 个）
`mutate_geom_size`, `mutate_geom_pos`, `mutate_body_pos`, `mutate_joint_range`,
`mutate_mass_or_density`, `mutate_friction`, `mutate_damping`, `mutate_ctrlrange`

### 8.3 配置级（必须 3 个）
`toggle_integrator`, `toggle_solver_or_flags`, `state_perturbation`（compile 后扰动 qpos/qvel/ctrl）

> 共 19 个 mutator。第一轮 PR 至少实现 8~10 个（覆盖结构 + 参数 + 配置三大类）。

---

## 9. 参数桶（Bucketization）

**严禁一上来就用连续空间**。所有参数必须先离散化。

| 参数 | 推荐 buckets |
|---|---|
| position | `[-1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1]` |
| size | `small / medium / large` |
| joint range | `narrow / medium / wide` |
| mass | `tiny / normal / heavy` |
| friction | `low / medium / high` |
| damping | `low / medium / high` |
| ctrlrange | `tight / normal / wide` |
| rollout steps | `[1, 5, 10, 50, 100]` |

要求：
- 全部从 `configs/default.yaml` 可改。
- 默认值少而稳，不要爆炸组合。

---

## 10. Oracle 设计

所有 oracle 输出**统一结构化**（dict / dataclass），便于 triage 与 reward。

### 10.1 Compile Oracle
检测：XML parse failure / compile failure / model creation failure / API error。

### 10.2 Runtime Oracle
检测：`FatalError` / Python exception / 子进程异常退出码 / timeout / NaN / Inf / 数值爆炸 / runtime warning。

### 10.3 Warning Oracle
**必须捕获并分类的 warning**：
- `BADQPOS`, `BADQVEL`, `BADQACC`, `BADCTRL`
- `INERTIA`
- `CONTACTFULL`, `CNSTRFULL`
- 以及其他可暴露的 warning（用 `mujoco.set_mju_user_warning` 或读取 `mjData.warning`）

### 10.4 Consistency Oracle（最小版）
对同一 `mjModel` 创建两份 `mjData`，设相同初态与 control，跑相同步数，比较：
- `qpos`, `qvel`, `act`, `ctrl`
- 阈值差异 → suspicious inconsistency

### 10.5 当前阶段不做
能量守恒、严格物理不变量、跨版本差分。

---

## 11. ★ 子进程执行（强制要求）

MuJoCo 可能 crash / deadlock / timeout / 直接带走 Python 进程。

**每个 testcase 的 compile + run 必须在子进程中执行**。

### 设计
- 主进程：调度 / 选 seed / 选 mutator / 记日志 / triage / 训练策略。
- 子进程：`engine/subprocess_worker.py`，做 compile + run。
- 子进程返回**结构化 JSON**（stdout 或临时文件）。
- 子进程挂掉 → 主进程捕获 returncode → 标记为 crash → 继续。

### 推荐实现
```python
# 主进程
proc = subprocess.run(
    [sys.executable, "-m", "src.engine.subprocess_worker", "--input", json_path],
    timeout=TIMEOUT_SEC,
    capture_output=True,
)
result = parse_result(proc, json_path)  # 包含 returncode / stdout / timeout 标记
```

不要用 `multiprocessing.Pool` 共享 `mjModel`，跨进程序列化不安全。

---

## 12. 日志与结果保存

每个 testcase 必须保存：
- testcase ID（hash 或 UUID）
- parent seed ID
- mutation 序列（list of `{id, params}`）
- compile 是否成功
- runtime 是否成功
- warning 列表
- exception / traceback
- subprocess returncode
- timeout 标记
- rollout 步数
- 关键状态统计（qpos/qvel/ctrl 的 max abs，是否 NaN/Inf）
- reward
- novelty signature

### 文件布局
- 每个 testcase → 一个 JSON 元数据 + 一个可复现的 mutated XML。
- 异常样本分目录：`outputs/crashes/`, `outputs/warnings/`, `outputs/interesting/`。

---

## 13. Triage 与去重

**绝不允许同类 bug 重复计数几千次**。

### Signature（最小版）
```
signature = hash((
    exception_type,
    traceback_top_frame_summary,
    sorted(warning_types),
    compile_or_runtime_failure_kind,
    returncode,
    model_shape_summary,    # (nq, nv, nu, nbody, ngeom)
))
```

### 输出两层统计
1. **raw cases** 总数
2. **unique issue signatures** 数

报告同时打印两者。

---

## 14. Reward 设计（RL 用）

**先实现可配置函数**，能拆解输出 total + 各分量。

### 推荐初版
```text
+10  crash / subprocess abnormal exit
+8   FatalError
+5   timeout
+4   compile success 且出现 new runtime warning
+3   suspicious inconsistency
+2   出现 runtime warning（已知类型）
+1   达到 novel state bucket / model bucket
-1   mutation 立即 trivial compile failure
-2   exact duplicate of known signature
```

### 接口要求
```python
@dataclass
class RewardBreakdown:
    total: float
    components: dict[str, float]

def compute_reward(result: ExecutionResult, triage_state) -> RewardBreakdown: ...
```

---

## 15. RL 设计

### 15.1 阶段一：Random Baseline（必须先）
- 随机选 seed
- 随机选 mutator
- 随机选参数 bucket
- 随机选 rollout steps

### 15.2 阶段二：轻量 RL
**优先级**：
1. Contextual bandit（先做）
2. 简单 actor-critic（再做）
3. PPO（最后考虑）

### 15.3 状态特征（低维 vector，不要 GNN）
- 模型形状：`nq, nv, nu, nbody, ngeom, njnt, nsite, nactuator`
- 最近 compile / runtime 是否成功（bool）
- warning count
- warning type one-hot / multi-hot
- max abs `qpos / qvel / ctrl`
- 是否 NaN / Inf
- contact 数量统计
- 最近 reward
- 最近 novelty
- mutation history 简要编码（last K 个 mutator ID）

### 15.4 动作空间
- mutator ID（离散）
- 参数 bucket ID（离散）
- rollout steps bucket（离散）

### 15.5 ★ 核心实现原则
> **先让 RL 学"选哪个 mutation generator 更容易出问题"，而不是让 RL 直接生成完整 XML。**

---

## 16. 运行模式

### 模式 A：Random Baseline
```bash
python -m src.experiments.random_baseline --config configs/default.yaml --budget 10000
```
**输入**：seeds + mutation budget + rollout config
**输出**：raw cases / unique issues / 统计报告

### 模式 B：RL-guided Fuzzing
```bash
python -m src.experiments.rl_guided_fuzz --config configs/default.yaml --budget 10000
```
**输入**：seeds + training budget + model config
**输出**：训练日志 / reward 曲线 / 问题发现曲线 / unique issues

---

## 17. 最小实验目标

### 实验 1：可运行性
证明系统能：load seed → mutate → compile → run → 记录异常。

### 实验 2：Random baseline 有效性
证明 random 版本能产出**非平凡**异常信号（至少有 warning / inconsistency / crash 中的一类）。

### 实验 3：RL > Random
比较指标：
- 每 1000 testcase 的 unique issues 数
- 首次发现 crash / FatalError 的平均步数
- 累积 reward
- 新 warning pattern 数量

---

## 18. ★ 实现顺序（严格遵守）

| Step | 任务 | 完成判据 |
|---|---|---|
| 1 | 项目骨架 + 依赖 + 配置系统 | `python -m src.main --help` 可运行 |
| 2 | Seed loader + compile/run harness（含子进程） | 能跑 1 个 seed + 0 mutation 出结果 |
| 3 | 8~10 个 mutation generators | 单元测试通过 |
| 4 | compile / runtime / warning oracle | 能输出结构化结果 |
| 5 | Random baseline | 能产出异常样本 |
| 6 | 最小 triage / dedup | unique vs raw 两层统计输出 |
| 7 | Bandit 或小型 actor-critic | reward 曲线非平 |
| 8 | RL-guided 实验脚本 + 统计 | 出对比图 |

> **每一步都要 commit 可运行代码，不要只给设计图。**

---

## 19. 代码质量要求

- 模块化，单文件不超过 ~400 行
- 关键函数带 docstring（说明输入输出 + 失败模式）
- 不要过度抽象（不要给一次性逻辑造工厂模式）
- 不要为"优雅"牺牲可调试性
- 日志清晰（每个 testcase ID 可追溯）
- 异常处理明确（不要裸 `except:`）
- 一切配置化（魔法数字进 yaml）
- 默认参数能直接 `python -m ...` 跑

### MuJoCo API 不确定时
1. 先查官方 Python API（`mujoco.MjModel.from_xml_path`, `mujoco.MjSpec`, `mj_step` 等）
2. 选最简单稳妥的入口
3. 不要依赖实验性 / 内部 API

---

## 20. ★ 第一轮输出形式（不要直接写所有代码）

收到任务后，**先按下面六部分输出**，等用户确认再继续生成代码：

### 第一部分：Implementation Plan
非常具体的分步实现计划，细到模块级。

### 第二部分：Project Skeleton
建议的目录树 + 每个文件的职责（一行说明即可）。

### 第三部分：Core Data Structures
列出核心数据结构（dataclass 草稿）：
- `TestCase`
- `ExecutionResult`
- `MutationRecord`
- `IssueSignature`
- `RLState`

### 第四部分：Mutation API Design
统一的 mutator 接口（base class + 一个示例实现的伪代码）。

### 第五部分：Oracle API Design
统一的 oracle 接口（base class + 输出结构）。

### 第六部分：First Runnable Milestone
"第一批必须先写出来"的文件清单 + 顺序 + 每个文件的预计代码量。

---

## 21. 生成代码时的硬性约束

- 优先**最小可跑**版本
- 每次只生成**少量、完整、可粘贴**的文件（不要半成品）
- 文件之间 import 关系**自洽**（不要 import 不存在的符号）
- 给出运行命令
- 标出需要用户手动准备的 seeds 或依赖
- 不要默认存在用户没提供的本地路径
- 不要在代码里写绝对路径
- 子进程入口必须可独立 `python -m` 调用

---

## 22. 角色再强调

你是**偏工程落地的研究型 Copilot**。

最高优先级（按顺序）：
1. 让用户尽快跑起来
2. 让用户尽快有 random baseline
3. 让用户尽快有异常样本
4. 让用户后续容易加 RL 和写论文

任何与上述四点冲突的"优雅"、"通用"、"完备"诉求，**全部让位**。

---

## Appendix A：核心数据结构参考草稿

```python
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class MutationRecord:
    mutator_id: str
    params: dict[str, Any]
    success: bool
    reason: Optional[str] = None

@dataclass
class TestCase:
    tc_id: str
    parent_seed_id: str
    mutations: list[MutationRecord]
    xml_path: str               # 持久化的 mutated XML
    rollout_steps: int

@dataclass
class WarningRecord:
    wtype: str                  # BADQPOS / BADCTRL / ...
    count: int
    message: str = ""

@dataclass
class ExecutionResult:
    tc_id: str
    compile_ok: bool
    runtime_ok: bool
    warnings: list[WarningRecord]
    exception_type: Optional[str]
    traceback_summary: Optional[str]
    returncode: int
    timeout: bool
    steps_done: int
    state_stats: dict[str, float]   # max_abs_qpos, has_nan, ...
    consistency_diff: Optional[float]

@dataclass
class IssueSignature:
    sig_hash: str
    exception_type: Optional[str]
    warning_types: tuple[str, ...]
    failure_kind: str
    returncode: int
    model_shape: tuple[int, ...]    # (nq, nv, nu, nbody, ngeom)

@dataclass
class RLState:
    model_shape: tuple[int, ...]
    last_compile_ok: bool
    last_runtime_ok: bool
    warning_multi_hot: list[int]
    max_abs_qpos: float
    max_abs_qvel: float
    max_abs_ctrl: float
    has_nan: bool
    has_inf: bool
    last_reward: float
    last_novelty: float
    mutation_history_ids: list[int]   # last K
```

## Appendix B：Mutator 基类参考

```python
class BaseMutator:
    id: str = "abstract"

    def sample_params(self, testcase: TestCase, rng) -> dict: ...

    def apply(self, testcase: TestCase, params: dict) -> "MutationApplyResult":
        """
        Returns MutationApplyResult(ok, new_xml_path|None, reason|None).
        Must NOT raise on expected failures; only raise on programming bugs.
        """
        raise NotImplementedError
```

## Appendix C：Oracle 基类参考

```python
class BaseOracle:
    name: str = "abstract"

    def evaluate(self, raw_run_output: dict) -> "OracleVerdict":
        """
        raw_run_output: subprocess 返回的 JSON dict.
        Returns OracleVerdict(triggered, severity, tags, details).
        """
        raise NotImplementedError
```

---

**END OF GUIDE**

按 §18 顺序开干，按 §20 形式先给出第一轮输出，等用户确认后再生成代码。
