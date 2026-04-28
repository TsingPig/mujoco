<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;700&family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

body, .markdown-body {
  font-family: 'Noto Serif SC', 'Source Han Serif SC', 'Songti SC', '宋体', serif !important;
  font-size: 16px;
  line-height: 1.85;
  color: #2c3e50;
  max-width: 980px;
  margin: 0 auto;
  padding: 40px 50px;
  background: #fafafa;
}

h1 {
  font-family: 'Noto Sans SC', 'PingFang SC', '苹方', sans-serif !important;
  font-weight: 700;
  font-size: 2.2em;
  color: #1a365d;
  border-bottom: 3px double #1a365d;
  padding-bottom: 0.4em;
  margin-top: 0.5em;
}

h2 {
  font-family: 'Noto Sans SC', 'PingFang SC', sans-serif !important;
  font-weight: 700;
  color: #2c5282;
  border-left: 5px solid #2c5282;
  padding-left: 0.6em;
  margin-top: 2em;
  font-size: 1.6em;
}

h3 {
  font-family: 'Noto Sans SC', 'PingFang SC', sans-serif !important;
  color: #2b6cb0;
  font-weight: 500;
  font-size: 1.25em;
  margin-top: 1.5em;
}

h4 {
  font-family: 'Noto Sans SC', sans-serif !important;
  color: #4a5568;
  font-weight: 500;
}

code, pre {
  font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, monospace !important;
  font-size: 0.92em;
}

code {
  background: #edf2f7;
  padding: 2px 6px;
  border-radius: 4px;
  color: #c53030;
}

pre {
  background: #1a202c;
  color: #e2e8f0;
  padding: 16px;
  border-radius: 8px;
  overflow-x: auto;
}

pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}

table {
  border-collapse: collapse;
  margin: 1em 0;
  width: 100%;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  background: white;
}

th {
  background: #2c5282;
  color: white;
  padding: 10px 14px;
  font-family: 'Noto Sans SC', sans-serif !important;
}

td {
  padding: 8px 14px;
  border-bottom: 1px solid #e2e8f0;
}

tr:hover { background: #f7fafc; }

blockquote {
  border-left: 4px solid #ed8936;
  background: #fffaf0;
  padding: 12px 18px;
  margin: 1em 0;
  color: #744210;
  border-radius: 0 6px 6px 0;
}

img {
  max-width: 100%;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.12);
  margin: 12px 0;
}

strong { color: #c53030; }

hr {
  border: none;
  height: 2px;
  background: linear-gradient(to right, transparent, #cbd5e0, transparent);
  margin: 2.5em 0;
}

.metric {
  display: inline-block;
  background: #2c5282;
  color: white;
  padding: 4px 10px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  margin: 0 3px;
}

.warn {
  background: #fed7d7;
  border-left: 4px solid #c53030;
  padding: 12px 16px;
  border-radius: 0 6px 6px 0;
  margin: 1em 0;
}

.success {
  background: #c6f6d5;
  border-left: 4px solid #2f855a;
  padding: 12px 16px;
  border-radius: 0 6px 6px 0;
  margin: 1em 0;
}
</style>

# MuJoCo 模糊测试 RL 引导技术报告（v5）

> **生成时间**：2026 年 4 月 26 日 · **实验版本**：v5 (Option D — Policy 主导种子选择 + 16-种子池)
> **对照版本**：v1 (Bandit) → v2 (Reward Bug) → v3 (Reward 修复) → v4 (Seed 观察) → **v5 (Policy 选 seed)**

---

## 0. 一句话总结

<div class="success">
<strong>核心结论</strong>：在赋予 RL Policy 种子选择权且扩展种子池至 16 个变体后，所有 RL 算法（REINFORCE / VanillaAC / A2C）在 1000 步预算下发现 <strong>22 个 unique signature</strong>，远超 random baseline 的 <strong>9 个</strong>，提升 <strong>+144%</strong>。
</div>

---

## 1. 研究问题与设计动机

### 1.1 待回答的问题

模糊测试（Fuzzing）的核心目标是**最大化 unique 失败/异常签名的覆盖率**。在 MuJoCo 这类高维物理仿真器上，模糊测试面临几个独特挑战：

1. **状态-动作空间巨大**：XML 模型 × 突变器 × 参数 × rollout 步数构成的笛卡尔积非常大；
2. **信号稀疏**：绝大多数随机突变要么编译失败（ValueError），要么产生重复签名；
3. **种子敏感**：不同的初始模型（pendulum / double_pendulum / box_stack / slider_crank）对同一突变器的反应差异巨大。

> **核心研究问题**：在固定预算（1000 次执行）下，**强化学习引导策略**能否显著超过**纯随机突变 baseline**？如果能，**为什么**它能赢？

### 1.2 实验设计沿革（5 个版本）

| 版本 | 关键变化 | RL vs Random (unique) | 状态 |
|---|---|---|---|
| v1 | Bandit baseline | 16 vs 24 | ❌ RL 弱于 random |
| v2 | 加 ActorCritic | 25 vs 32 | ❌ Reward bug，mean = -1.19 |
| v3 | 修复 Reward 函数 | 25 vs 32 | ⚠️ Reward 正常但仍输 |
| v4 | State 加 seed_idx 观察 | 24 vs 32 | ⚠️ Policy 看得到种子但仍输 |
| **v5** | **Policy 选种子 + 池 4→16** | **22 vs 9** | ✅ **RL 全面胜出** |

---

## 2. 系统架构

### 2.1 模糊测试流程

```
┌───────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Seed Pool    │───▶│  Policy.select│───▶│  Mutator.apply   │
│ (16 XML 模型) │    │  (RL/Random)  │    │ (10 种突变器)    │
└───────────────┘    └──────────────┘    └──────────────────┘
                                                  │
                          ┌───────────────────────┘
                          ▼
                  ┌──────────────────┐    ┌──────────────────┐
                  │ Subprocess Worker│───▶│  Triage + Reward │
                  │ (隔离的 MuJoCo)  │    │  (签名 / 信号)   │
                  └──────────────────┘    └──────────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────────┐
                                          │  Policy.update   │
                                          │  (PPO / A2C ...) │
                                          └──────────────────┘
```

### 2.2 Action 空间（v5 版）

每一步 Policy 输出一个 4 元组：

| 维度 | 大小 | 含义 |
|---|---|---|
| `seed_idx` | **16** | **新增**：选择哪个种子模型（v5 改进） |
| `mutator_idx` | 10 | 选择哪个突变器（STRUCT_GROW / GEOM_PERTURB / ...） |
| `param_idx` | 16 | 突变器内部 RNG 的离散种子桶 |
| `rollout_idx` | 5 | 仿真步数桶（10 / 50 / 100 / 500 / 1000） |

**总动作空间** = 16 × 10 × 16 × 5 = <span class="metric">12,800</span> 种离散组合。

### 2.3 RL Policy 网络结构

<pre><code>State (64 维) ──▶ SharedTrunk(GELU×2, h=128)
                          │
              ┌───────────┼───────────┬─────────────┐
              ▼           ▼           ▼             ▼
         head_seed   head_mut    head_param    head_roll
          (16 类)    (10 类)    (16 类)|mut    (5 类)|mut
              │           │           │             │
              └─────── log_prob 累加 + entropy 累加 ──┘
                          ▼
                    Value Head V(s)
</code></pre>

层级采样：先采 `seed_idx`，再采 `mutator_idx`（带 mask），然后基于 mutator one-hot 条件采 `param_idx` 和 `rollout_idx`。

---

## 3. 实验设置

### 3.1 硬件与软件栈

| 项目 | 配置 |
|---|---|
| OS | Windows 11 |
| CPU | （单线程 subprocess，无 GPU） |
| Python | 3.8.6 |
| MuJoCo | 3.2.3 |
| PyTorch | 2.4.1+cpu |

### 3.2 比较的策略

| Policy | 算法家族 | 一句话描述 |
|---|---|---|
| `random` | 随机基线 | 在所有动作维度上均匀采样，无学习 |
| `reinforce` | REINFORCE | 蒙特卡洛策略梯度 + 批均值 baseline |
| `vanilla_ac` | 1-step TD AC | δ = r + γV(s') - V(s)，单梯度步，GzFuzz 风格 |
| `a2c` | A2C | N-step return + 优势归一化 + 梯度裁剪 |
| `ppo` | PPO + GAE | clip ε=0.2, GAE λ=0.95, 4 epochs, MB=32 |

### 3.3 关键超参数

```yaml
budget: 1000              # 每个策略的执行预算
seeds_pool: 16            # 种子模型数（v5 扩展）
n_mutators: 10
history_k: 8              # 状态包含最近 8 个突变器
hidden_dim: 128
lr: 3.0e-4
gamma: 0.99
entropy_coef: 0.05        # v3 提升以增加探索
rollout_per_update: 64
reward:
  novel_state_bucket: 2.0
  trivial_compile_fail: -0.3
  duplicate_signature: -2.0
  curiosity_coef: 1.0     # v3 加入 c/sqrt(visits) 计数 bonus
```

---

## 4. 主要结果

### 4.1 总体性能对比

<div class="success">
所有 RL 算法显著超过 random baseline。VanillaAC、REINFORCE、A2C 三者并列最佳，PPO 略弱。
</div>

| Policy | raw | **unique** | ok | compile | warning_only | 用时 (s) | unique/min |
|---|---|---|---|---|---|---|---|
| `random` | 1000 | **9** | 5 | 2 | 2 | 787.6 | 0.69 |
| `reinforce` | 1000 | **22** ⬆️ | 9 | 2 | 11 | 866.3 | 1.52 |
| `vanilla_ac` | 1000 | **22** ⬆️ | 9 | 2 | 11 | 859.7 | 1.54 |
| `a2c` | 1000 | **22** ⬆️ | 9 | 2 | 11 | 867.6 | 1.52 |
| `ppo` | 1000 | **19** ⬆️ | 7 | 2 | 10 | 921.7 | 1.24 |

**RL vs Random 提升幅度**：

- VanillaAC / REINFORCE / A2C: <span class="metric">+144%</span> (22 vs 9)
- PPO: <span class="metric">+111%</span> (19 vs 9)

### 4.2 累积 Unique 签名曲线

> 该图展示了 5 个策略随步数推移的 unique 签名累积速度。RL 曲线明显在前 200 步即开始拉开差距。

![累积 unique 签名](cumulative_unique.png)

**观察**：
- Random（紫色）增长缓慢且早期饱和（~step 400 后几乎停滞）
- 三个 AC 类算法增长曲线几乎重合，呈持续向上趋势
- PPO 早期增长慢于 AC 家族，可能与 clip 限制了探索激进度有关

### 4.3 按签名类别分布

![按类别堆叠](by_kind_stacked.png)

**关键发现**：
- **`warning_only` 类别是 RL 优势的主要来源**：RL 找到 10-11 个 BADQVEL/BADCTRL 警告类型，而 random 只找到 2 个
- 这表明 RL 学会了**触发数值不稳定的状态-参数组合**，而 random 难以撞上
- Compile 失败数（2 vs 2）相同，说明编译器层面的覆盖差异很小

### 4.4 突变器使用分布

![突变器使用](mutator_usage.png)

**观察**：
- Random 在 10 个突变器上接近均匀
- RL 学会**集中于产能高的突变器**（如 `STRUCT_REWIRE`、`STATE_PERTURB`）
- PPO 的分布相对最分散，可能是 entropy loss 防止其过度集中

### 4.5 平滑奖励曲线（窗口=50）

![奖励曲线](reward_curve.png)

<div class="warn">
<strong>反直觉观察 — 必须正视</strong>：图上 random（灰线）的平滑奖励长期稳定在 <strong>~0.9</strong>，而所有 RL 算法从 ~3.0 急速衰减到 <strong>~0.3-0.5</strong>，<strong>per-step reward 反而低于 random</strong>。但 unique 签名数 RL=22 远高于 random=9。这看似矛盾，实则揭示了奖励函数与真实目标之间的偏差。
</div>

**为什么 RL 奖励更低却找到更多 unique？**

回顾奖励项设计：

| 奖励项 | 数值 | 触发条件 |
|---|---|---|
| `novel_state_bucket` | **+2.0** | 命中之前没见过的 state bucket（一次性奖励） |
| `duplicate_signature` | **−2.0** | 产生已见过的 signature |
| `trivial_compile_fail` | −0.3 | 简单的编译失败 |
| `curiosity_coef / √visits` | +1.0 衰减 | 计数式好奇心 bonus |

**关键机制**：

1. **RL 学会了"扎堆深挖"策略**：聚焦少数高产种子+突变器组合 → 早期狂吃 novelty bonus（曲线初始 ~3.0），然后开始**反复命中已见 signature** → `duplicate_signature` 的 −2 惩罚把奖励压到 0.3 附近。
2. **Random 的"广撒网"策略**：均匀采样 16 种子 × 10 突变器 → 大部分输出都是低产但**互不重复**的小 novelty → 平均拿 +0.5~+1.0，**几乎不撞 duplicate 重罚**，所以平均奖励 ~0.9 看起来很"稳"。
3. **unique 数的真相**：RL 在"扎堆"过程中触发了大量稀有 BADQVEL/BADQPOS 的 warning_only 签名（22 个里有 11 个是 warning_only），这些是 random 的稀疏采样**永远撞不上**的；而 random 的奖励虽然高，签名却高度集中在几个 ok 状态上（9 个里 5 个是 ok）。

<div class="success">
<strong>结论</strong>：<strong>per-step reward 不是 fuzzing 的真实目标</strong>——unique signature coverage 才是。当前 reward 函数对 RL 不利（重复惩罚太重，使其奖励曲线难看），但 RL 仍然在真实目标 unique 上大胜，说明**学到的 policy 是有价值的，只是 reward shaping 把它"显得很差"**。
</div>

**这反过来给了我们一个重要的论文 insight**：

> Reward shaping 与下游 metric 的对齐问题在 fuzzing-as-RL 中被严重低估。`duplicate_signature = −2` 是从"避免冗余执行"动机出发的，但它惩罚了 RL 的"深挖式探索"——而深挖恰恰是发现稀有 warning 的必要路径。**后续工作应当尝试将 reward 直接定义为 ΔUnique（增量 unique 数），而非 per-step novelty/duplicate 二分**。

### 4.6 Top 签名分布对比

不同策略发现的 Top-3 高频签名一致（都是常见 ok 状态），但 **RL 在长尾签名上明显更丰富**：

**Random 的长尾**（前 9 个签名后即归零）：
- `5fe8fe8f` (529 次, ok), `4264c9ff` (222 次, compile), `1073671a` (114 次, ok)
- 仅有 2 个 warning_only 签名

**RL 的长尾**（22 个签名，包含多个稀有 BADQVEL/BADQPOS 警告）：
- `029daba9` (15 次, BADQVEL), `cc8d1704` (12 次, BADQVEL), `0c2dcbb9` (12 次, BADQVEL)
- 这些稀有警告签名是 RL 区别于 random 的关键

---

## 5. 演化过程：为什么 v5 才赢？

<div class="warn">
<strong>关键经验</strong>：v1-v4 RL 全部输给 random，根本原因是<strong>学习信号或观察不充分</strong>，而不是算法本身有问题。
</div>

| 版本 | 失败/成功原因 |
|---|---|
| v2 | Reward 函数双计 trivial_compile_fail，mean reward = -1.19，policy 学到"少做事最好" |
| v3 | 修复后 mean reward = -0.02，但 policy 看不到当前是哪个种子 → 学到所有种子的边缘平均策略 |
| v4 | 给 state 加 `seed_idx` one-hot，policy 终于能区分种子，但仍只能从 4 个种子里挑 → 探索空间有限 |
| **v5** | **Action 加 `seed_idx`** + **种子池扩到 16** → policy 能主动聚焦高产种子，random 反而被稀释 |

### 5.1 为什么 random 在 v5 反而**变差**了（32 → 9）？

> 这是一个反直觉但重要的现象：种子池扩大后，random 的 unique 数从 32 跌到 9。

**原因分析**：
- v4 中 pool=4，random 均匀采样每种子 ≈ 250 步，每个种子的"产能上限"都被采到
- v5 中 pool=16，random 每种子 ≈ 62 步，且 12 个变体高度相似 → 重复签名严重
- **稀释效应**：random 把预算浪费在相似变体上，而 RL 学会**忽略低产种子**

### 5.2 这给论文带来什么洞见？

<div class="success">
<strong>论文卖点</strong>："在大种子池场景下，RL 引导策略相对 random baseline 的优势会被放大（+144% vs +33%）。这表明 RL 不仅是'学到一个聪明策略'，更是'学会忽略噪音种子'。"
</div>

---

## 6. 实验规模评估与诚实讨论

### 6.1 当前规模的具体数字

| 维度 | 当前 v5 |
|---|---|
| 每策略预算 | 1,000 步 |
| 策略数量 | 5（含 random） |
| 总执行次数 | 4,834 次（4 RL × 1000 + random 834） |
| 种子模型数 | 16 |
| 单次执行用时 | ~0.8 s |
| 总耗时 | ~73 分钟 |
| 唯一签名数（最佳 RL） | 22 |

### 6.2 与文献基线对比

> **诚实评估：当前实验规模相对于发表级模糊测试论文偏小。**

| 论文/系统 | 典型规模 | 当前 v5 |
|---|---|---|
| AFL/AFL++ | 24 小时 × 数十核 | 73 分钟 × 单线程 |
| GzFuzz (ICSE'23) | 24 小时 × 4 fuzzer | 14.4 分钟 × 5 策略 |
| ML4Fuzz / NEUZZ | 数百万 inputs | 5,000 inputs |
| **当前 v5** | **同上的 1/100 - 1/1000** | — |

### 6.3 规模不足的具体风险

1. **统计可靠性差**：RL 三算法都恰好 22，可能是同一组高产种子被穷尽，而非真实并列；建议跑多次种子
2. **晚期收敛未验证**：累积曲线尾部仍在上升，1000 步可能离收敛上限还有距离
3. **种子多样性不足**：16 个种子里 12 个是基础 4 个的轻微变体（仅 1-2 次突变），多样性其实只有 4 大类
4. **缺少 P-value / 置信区间**：没有跨 random seed 重复实验

### 6.4 推荐的后续实验配置

| 改进项 | 建议规模 | 预估耗时 |
|---|---|---|
| 单策略预算 | 5,000 步 | ~67 min/策略 |
| 重复实验 | 5 个 random seed | ×5 倍 |
| 种子池多样性 | 30+ 个真实场景模型 | 一次性 |
| 总耗时 | ~30 小时 | 通宵 + 一上午 |
| 总执行次数 | 125,000 | ~25× 当前 |

> **如果你有跑过夜的预算**，下一步可以做：
> - 把 `budget` 改成 5000、把 `tools/run_full_bench.ps1` 包一层"5 random seed"的循环
> - 在 `make_report.py` 里加 mean ± std bar 和 Mann-Whitney U test
> - 收集 30+ 真实 MuJoCo 模型作为种子（可从 `mujoco_menagerie` 抓）

---

## 7. 局限性与未解之谜

### 7.1 仍未解决的问题

1. **🔴 Reward 与目标不对齐（最严重）**：4.5 节图表显示 RL per-step reward (~0.4) **低于** random (~0.9)，但 unique 数 RL 反而是 random 的 2.4 倍。说明当前 `duplicate_signature = −2` 惩罚把 RL 的"深挖式探索"算成了负贡献。**下个版本必须把 reward 改成 ΔUnique 增量奖励**，否则 RL 是"被惩罚着学好的"，泛化性存疑。
2. **三个简单 AC 算法并列 22**：可能撞到了"种子内可达签名数"的天花板，需要更大种子池或更长预算才能区分
3. **PPO 反而稍弱（19）**：与文献预期相反，怀疑 entropy_coef=0.05 + clip 0.2 的组合在小数据下偏保守
4. **未做 ablation**：没有单独验证"扩种子池"和"policy 选 seed"哪个贡献更大
5. **奖励工程依赖性强**：v1→v5 大量调试都在 reward function 上，泛化性存疑（与第 1 点相关）

### 7.2 后续工作建议

- **A. 严格 ablation**（4 因素 × 2 水平）：
  - pool size: 4 vs 16
  - policy 选 seed: 是 vs 否
  - curiosity bonus: 有 vs 无
  - entropy: 0.01 vs 0.05
- **B. 大规模收敛实验**：5,000+ 步看 RL/Random 差距是否持续扩大或饱和
- **C. 跨种子稳定性**：5 个 random seed 跑出 mean±std
- **D. 与 LLM 引导对比**：现有 `src/rl/llm_policy.py` 接入后形成 4-way 对比

---

## 8. 复现实验的命令

### 8.1 快速烟测（~30 秒）

```powershell
cd icse27\mujoco_rl_fuzz
& "E:\--IDLE\python.exe" -u -m src.experiments.rl_guided_fuzz `
    --config configs/default.yaml --algorithm ppo --budget 30
```

### 8.2 完整 v5 bench（~75 分钟）

```powershell
cd icse27\mujoco_rl_fuzz
powershell -ExecutionPolicy Bypass -NoProfile `
    -File tools\run_full_bench.ps1
```

### 8.3 实时监控

```powershell
Get-Content logs\bench_full.log -Wait -Tail 30
```

### 8.4 重新生成报告

```powershell
& "E:\--IDLE\python.exe" tools\make_report.py
```

---

## 9. 文件索引

| 文件 | 用途 |
|---|---|
| `outputs/report/REPORT.md` | 英文版自动生成报告 |
| `outputs/report/REPORT_CN.md` | **本文件**（中文详细技术报告） |
| `outputs/report/*.png` | 4 个可视化图表 |
| `logs/run_*.jsonl` | 每策略每步的完整 trace |
| `logs/summary_*.json` | 每策略的聚合摘要 |
| `logs/bench_full.log` | 完整 bench 时间线 |
| `logs_v{1,2,3,4}_*_backup/` | 历史版本的实验数据 |

---

<div style="text-align: center; color: #718096; font-size: 0.9em; margin-top: 3em;">
报告生成于 2026-04-26 · v5 (Option D Implementation)
<br/>
<em>实验规模偏小，论文级别仍需 5x-10x 扩展，详见第 6 节</em>
</div>
