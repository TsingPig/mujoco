# MuJoCo RL-Fuzz — v8 实验报告

**日期**：2026-04-28　　**MuJoCo**：3.2.3　　**Python**：3.8.6　　**OS**：Windows 11

---

## 一、核心结论

经过两轮重构（v7：结构化输入有效性门控；v8：差分 solver 测试），模糊测试器在 MuJoCo 3.2.3 中发现了 **6 个可复现的 solver 不一致案例**。

测试规模：1,000 个测试用例（random + PPO 各 500），25 个种子 XML（4 个玩具模型 + 21 个来自 `mujoco/model/` 的真实模型）。

每个 "real_bug_candidate" 均经过独立复现验证：迭代次数上限提至 **20,000**，收敛容差设为 **1e-12**。27 个候选中有 6 个在该预算下 Newton / CG / PGS 三个求解器仍输出不同不动点——即求解器收敛到了不同的物理状态，而非仅仅迭代次数不够。

**最强一例**（`outputs/tc/666c282aa06fe800.xml`，slider-crank + equality 约束 + 与地面相交的 64 kg 球形体）：PGS 的最终 `qpos` 偏差随迭代次数**增大而扩大**，明确排除收敛速度因素：

| 迭代次数 | solver_only_max_diff |
|:--------:|---------------------:|
|      200 |              2.87e+0 |
|     1000 |              2.82e+0 |
|     5000 |              3.71e+0 |
|    20000 |              **3.85e+0** |

这正是 MuJoCo 文献中凸求解器历史上发生分歧的典型场景（接触 + 等式约束同时激活），是黑盒模糊测试得到的可信 bug 类信号。

---

## 二、各版本主要变更

原始报告中 `unique=191` 的数字有 97% 是噪声：fuzzer 自身生成的无效 XML 编译失败被计入"唯一签名"。v7 消除了这一膨胀；v8 加入了唯一能从外部暴露求解器级 bug 的 oracle。

### v7 — 有效性门控 + invalid 类别

- `triage.classify` 新增 `invalid` 类型，编译失败与 `MutationSkip` 均归入此类，不计入 `n_unique_real`。
- `runner.py` 参照 GZFuzz 设计，在主进程通过 `mujoco.MjModel.from_xml_path` 预编译验证；每个 action 最多重采样 `cfg.run.validity_retries=3` 次后才判 `invalid`。
- 奖励调整：移除 `novel_compile=+1.5`，新增 `invalid_input=-0.5` 与 `real_bug_candidate=+20.0`。RL 不再因生成无效 XML 而获奖。
- Mutator 修复：`STRUCT_GROW`/`GEOM_PERTURB` 按形状类型使用正确 size 维数；`STRUCT_GROW` 始终生成显式 `<inertial>`；`JOINT_PERTURB` 跳过 ball/free 关节；`ACTUATOR_EDIT add` 使用唯一名称。
- 新增 `SpontaneousNanOracle`：触发条件 `compile_ok ∧ ¬injected ∧ (NaN|Inf|BADQPOS|BADQVEL|BADQACC|early-abort)`。
- v7 全量 bench（5 策略 × 1000）：0 个真实 bug。11 个候选均为注入型误报，已通过 `injected_nan_inf` / `injected_bad` 标记过滤。

### v8 — 差分 solver 测试 + 多样化种子

- **Worker 差分测试**（`src/engine/subprocess_worker.py`）：完成主 rollout 后，以 5 种 (solver, integrator) 组合（`Newton/CG/PGS+Euler`、`Newton+implicit`、`Newton+implicitfast`）重跑同一 XML + 同一扰动，参数 `iterations=200, tolerance=1e-10`。计算两类差值：`max_diff`（相对参考）与 `solver_only_max_diff`（同 integrator 三元组两两最大差值）。
- **SolverDiffOracle**（`src/oracles/diff_oracle.py`）：当 `solver_only_max_diff > 1e-3` 时触发。跨 integrator 的差异（Euler vs implicit 在数学上本就不同）仅保留在诊断字段，不触发报警。
- **种子池扩展**至 25 个，加入了 `humanoid_v2`、`car_v2`、`balloons_v2`、`slidercrank_v2`、`mug_v2` 等真实 mujoco 模型文件。
- `src/result.py` + `src/engine/compile_and_run.py` 将 `solver_diff` 字段透传至持久化 JSON 记录。

---

## 三、v8 Bench 结果

```
python -m src.main bench --budget 500 --policies random ppo
```

| 策略   | n_raw | n_unique | by_kind                                | 耗时    |
|--------|------:|---------:|----------------------------------------|--------:|
| random |   500 |        7 | ok=1, invalid=2, warning_only=4        |  418 s  |
| ppo    |   500 |        9 | ok=1, invalid=3, warning_only=5        |  412 s  |

**实时 bug 候选（real_bug_candidate）统计：**

| 类型                       | 数量 |
|----------------------------|-----:|
| 总计持久化                 |   27 |
| 注入型误报（injected_*）   |    0 |
| 自发 NaN/BAD（clean 输入） |    0 |
| Solver 不一致（纯信号）    |   **27** |
| Crash                      |    0 |

---

## 四、27 个候选的复现分类

`python tools/reproduce_top_finds.py` 以 `iters ∈ [200, 1000, 5000, 20000]`、`tol=1e-12` 重跑每个保存的 XML，重测 `solver_only_max_diff`：

| 分类                     | 数量 | 含义                                                  |
|--------------------------|-----:|-------------------------------------------------------|
| `convergence_artifact`   |   18 | iters≤5000 时差值降至 1e-3 以下——PGS 慢而已，非 bug |
| `STABLE_DISAGREEMENT`    |  **6** | iters=20000 时差值仍≥原始值 50%——求解器不动点不同   |
| `PARTIAL`                |    3 | 差值减小但仍高于 1e-3——边界情况                      |

### 6 个 STABLE_DISAGREEMENT 案例

| tc_id | 原始 diff | iters=20000 diff | 种子系列 |
|:------|----------:|----------------:|:---------|
| `666c282aa06fe800` | 1.625e+0 | **3.852e+0**（随迭代增大！） | slider_crank + equality |
| `325f61c751e6052b` | 2.600e+0 | 2.600e+0 | slider_crank 变体 |
| `05e5a50eadb2a332` | 2.688e-1 | 7.917e-1 | slider_crank 变体 |
| `8a3d79ca8e924c02` | 5.504e-2 | 7.429e-2 | — |
| `2bb30b2fa81644b2` | 9.333e-3 | 8.821e-3 | — |
| `de2cd51e56918572` | 2.289e-3 | 2.878e-3 | — |

**最强案例 XML 摘要**（`666c282aa06fe800.xml`）：在原有 `rod_slider` 等式约束的 slider-crank 基础上，fuzzer 追加了一个通过 ball 关节连接、质量 64 kg 的胶囊体，该体与地面相交。50 步仿真后：

- `Newton+Euler` 与 `Newton+implicit` 之间差值为 1.93——两种 integrator 本就不同，属预期。
- `PGS+Euler` 随迭代次数增加**向远离 Newton/CG 方向漂移**，表明 PGS 在该接触流形上收敛到了一个错误（或振荡）不动点。

### 对论文论点的意义

模糊测试器现在能产生**不平凡、可复现**的 MuJoCo 信号，且已明确排除：

1. **结构无效 XML**（归入 `invalid` 类，不计入 bug 统计）
2. **自注入 NaN/Inf**（`injected_nan_inf` / `injected_bad` 标记过滤）
3. **求解器收敛慢**（通过迭代次数扫描 `tools/reproduce_top_finds.py` 过滤）

6 个 STABLE_DISAGREEMENT 案例是可信的 MuJoCo solver bug 候选，应附上最小复现 XML 向上游提 issue，迭代次数-差值表格便于上游快速验证。

---

## 五、复现步骤

```powershell
cd icse27/mujoco_rl_fuzz

# 运行 Bench（约 14 分钟）
python -m src.main bench --budget 500 --policies random ppo

# 分类 real_bug 候选
python tools/analyze_v8_bugs.py

# 迭代次数扫描（全量复现）
python tools/reproduce_top_finds.py > outputs/analysis/v8_repro_all.txt 2>&1
```

- Bug 候选 JSON 记录：`outputs/real_bugs/<tc_id>.json`
- 最小复现 XML：`outputs/tc/<tc_id>.xml`
- 分析汇总：`outputs/analysis/`

---

## 六、文件归档结构

```
icse27/mujoco_rl_fuzz/
├── archive/               ← 历史 bench 日志与报告（v1–v7）
│   ├── logs_v1/ … logs_v8/
│   └── reports_v1/ … reports_v7/
├── logs/                  ← 当前版本 bench 产物（v8，已归入 archive/logs_v8）
├── outputs/
│   ├── real_bugs/         ← 27 个 real_bug_candidate JSON
│   ├── tc/                ← 对应最小复现 XML
│   └── analysis/          ← v8_repro_all.txt / v8_bug_analysis.txt
├── seeds/                 ← 25 个种子 XML
├── src/                   ← 框架源码
│   └── oracles/diff_oracle.py   ← SolverDiffOracle（v8 新增）
├── tools/
│   ├── analyze_v8_bugs.py       ← 候选分类脚本
│   └── reproduce_top_finds.py   ← 迭代次数扫描脚本
└── REPORT_v8.md           ← 本报告
```
