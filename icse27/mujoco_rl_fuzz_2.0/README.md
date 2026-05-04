# MuFuzz (mujoco_rl_fuzz_2.0)

**MuFuzz** 是一个面向 MuJoCo 物理引擎的、**RL-guided** 的 fuzzing 框架。
RL 不负责判断 bug，**bug 判断由 static / dynamic / differential oracles 完成**；
RL 学习的是 *exploration policy*（"下一步用哪个 seed × scene × 轨迹原语，
触达哪条 oracle"）。

本目录与 `../mujoco_rl_fuzz/`（v1）**完全隔离**，除 `src/triage/signature.py`
是 v1 triage 哈希的硬拷贝外，没有任何从 v1 的导入。

---

## 版本路线

- **v2.0-α** ✅ — 单层 actor 种子 + 23 个 mutator + 双层验证门。
- **v2.0-β** ✅ — Layered Seed Corpus (L0–L3)、Scene Composer、Env Adapters、
  Trajectory Protocol、Episodic Runner、6 个 oracles、random / rule baseline。
- **v2.0-γ** ⏳ — PPO actor-critic exploration policy（基于 β 同一 runner+oracle 接口）、
  finding shrinker、MJX backend differential oracle、experiment harness。

> Per project policy: PPO 不在 β 中实现。β 提供完整 substrate，γ 仅替换 selector。

---

## 一、v2.0-β 快速上手

```powershell
# 1. （已有）拉 L0 actor 种子
python tools/fetch_actor_seeds.py            # 等价 tools/fetch_seeds.py

# 2. L1 合成场景（actor + arena/props 包装）
python tools/build_synthetic_scenes.py --count 50

# 3. L2 开源环境（缺依赖会优雅降级为 metadata_only）
python tools/fetch_env_scenes.py --no-smoke
# 装好 robosuite / dm_control / ... 后再去掉 --no-smoke

# 4. L3 轨迹种子（基于 L0/L1/L2 生成可重放协议）
python tools/build_trajectory_seeds.py

# 5. 校验四层语料分布
python tools/validate_layered_corpus.py

# 6. 列表 / 详情可视化
python tools/visualize_corpus.py --list --layer synthetic_scene
python tools/visualize_corpus.py --show <seed_id> --rollout

# 7. 跑 baseline
python tools/run_random_fuzz.py --budget 50
python tools/run_rule_fuzz.py   --budget 50
# 输出 findings/random/*.json, findings/rule/*.json
```

**严禁声明任何"已发现真实 bug"**。所有 finding 仅是 oracle severity 信号，
需 γ 阶段的 reproducer + shrinker 才能上报。

---

## 二、v2.0-β 模块结构

```
src/
  corpus/        layered seed schema + manifest + provenance
  scene/         L1 SyntheticSceneSeed composer (10 templates)
  env_adapters/  L2 OpenEnvSeed adapters (robosuite / gym_robotics /
                 dm_control / mujoco_playground / myosuite)
  trajectory/    L3 TrajectoryProtocol + recorder + replay
  runner/        episodic dispatcher (RunRequest / RunResult)
  oracles/       finite_state / instability / contact_force /
                 reset_repro / sensor_reward / backend_diff
  triage/        signature.py (v1 hard-copy) + finding.py
  viz/           tabular corpus viewer
tools/
  build_synthetic_scenes.py    fetch_env_scenes.py
  build_trajectory_seeds.py    visualize_corpus.py
  validate_layered_corpus.py   run_random_fuzz.py / run_rule_fuzz.py
seeds/
  curated/             L0 actor seeds (α 阶段产物)
  synthetic_scenes/    L1 manifest + per-seed scene.xml
  open_envs/           L2 manifest (jsonl, no payload)
  trajectory_seeds/    L3 manifest
  _quarantine/         compile-failed payloads
```

每个层都有统一的 `LayeredSeed` schema 与 JSONL `manifest.jsonl`，
runner 只看 isinstance dispatch。

---

## 三、v2.0-α 已完成（保留）

α 阶段只实现“**能产生合法可编译模型 + 能落地的种子库**”，RL 部分推到 β。

### 1. 种子收集 — `tools/fetch_seeds.py`
- 浅 clone 7 个上游仓库（`mujoco`、`menagerie`、`dm_control`、`gym_robotics`、
  `mujoco_mpc`、`robosuite`、`mjx`），按 `configs/default.yaml` 的 cap 采样。
- 输出到 `seeds/curated/<source>__<slug>/model.xml`，自动镜像同目录的
  `assets/`、`meshes/`、`textures/`，并复制 `LICENSE`。
- 当前仅拉取了 `mujoco` 源 ⇒ **10/10 模型 `mj_loadXML` 通过**。

### 2. 安全模型包装 — `src/mjcf/`
- `spec_loader.SafeModel`：lxml 解析 + `snapshot()`/`restore()` 字节级回滚 +
  `compile()` 走 `chdir(asset_dir) → mj_loadXML` 解析资源。
- `invariants.py`：`all_joints` / `all_geoms` 严格限定为 `<body>` 直接子元素，
  `hingelike_joints` 要求显式 `type=hinge|slide`（避免类继承 ball）。

### 3. 23 个高语义 Mutator — `src/mutations/ops_*.py`
所有取值来自 `intensity.py::INTENSITY_TABLE`（**禁止随机字符串生成**），
按类别分 6 个文件：

| 文件 | Mutator |
|---|---|
| `ops_struct.py` | `STRUCT_GROW_LINK`、`STRUCT_SHRINK`、`STRUCT_DUPLICATE_SUBTREE` |
| `ops_geom.py` | `MUTATE_GEOM_SHAPE`、`MUTATE_GEOM_SIZE`、`MUTATE_INERTIAL`、`MUTATE_FRICTION`、`MUTATE_SOLREF_SOLIMP`、`MUTATE_CONTACT_MARGIN` |
| `ops_joint.py` | `MUTATE_JOINT_TYPE`、`MUTATE_JOINT_LIMIT`、`MUTATE_DAMPING_FRICTION` |
| `ops_actuator.py` | `ACTUATOR_ADD`、`ACTUATOR_MUTATE_RANGE`、`ACTUATOR_DELETE`、`EQUALITY_ADD`、`TENDON_ADD` |
| `ops_solver.py` | `MUTATE_TIMESTEP`、`MUTATE_INTEGRATOR`、`MUTATE_SOLVER_ITER` |
| `ops_state.py` | `SET_QPOS_RUNTIME`、`SET_QVEL_RUNTIME`、`SET_CTRL_RUNTIME`（仅 runtime） |

每个 mutator 区分 *合法 intensity_modes* 与 *invalid_parseable_modes*
（`nan_inf` / `negative_principal` / `singular_zero_range` 等故意触发拒绝路径的）。

### 4. 双层验证门
- **In-process**：`apply()` 后立即 `safe_compile()`，失败回滚。
- **CI Gate** `tools/validate_mutators.py`：
  `seed × mutator × intensity × reps` 笛卡尔，跳过 `invalid_parseable`，
  全部 legal cell 必须 compile 通过。

**当前实测结果**

| Gate | 命令 | 结果 |
|---|---|---|
| 合成种子冒烟 | `python tools/smoke_mutators.py` | **88 / 88 OK** |
| 真实种子加载 | `python tools/validate_seeds.py` | **10 / 10 OK** |
| 真实种子 mutator | `python tools/validate_mutators.py --reps 1` | **880 cells / 0 fail** |

> `pytest` 因网络受限暂时装不上，等价的 `tools/smoke_mutators.py` 已覆盖
> `tests/test_intensity_table.py` + `tests/test_each_mutator_validity.py` 的全部断言。

---

## 二、目录结构

```
configs/default.yaml          # seed cap / mutator 列表 / invalid 白名单
seeds/curated/<src>__<slug>/  # 种子 + 镜像 assets
seeds/_quarantine/            # mj_loadXML 失败的种子（隔离）
src/
  mjcf/        spec_loader.py / invariants.py
  mutations/   base / intensity / ops_*.py / registry
  triage/      signature.py（v1 拷贝）
tools/
  fetch_seeds.py            # 拉种子
  validate_seeds.py         # 加载验证
  validate_mutators.py      # CI 门
  smoke_mutators.py         # 合成种子冒烟（无依赖）
  visualize_seed.py         # 【新】可视化单个种子
  visualize_mutator.py      # 【新】可视化 mutator 前/后
tests/                      # pytest 文件（待装环境）
```

---

## 三、可视化

需要 `mujoco>=3.2`（已在 `pyproject.toml` 中）。两个工具都基于 `mujoco.viewer`，
首次启动会弹出原生窗口；按窗口的关闭按钮或 `ESC` 进入下一步 / 退出。

### 3.1 看一个种子跑起来

```powershell
# 列出全部种子
python tools/visualize_seed.py --list

# 跑某个种子（默认实时步进直到关闭窗口）
python tools/visualize_seed.py mujoco__model_humanoid_humanoid

# 只静态看（不步进）
python tools/visualize_seed.py mujoco__model_car_car --static
```

### 3.2 看一个 mutator 的效果

```powershell
# 把 MUTATE_GEOM_SIZE 的 huge 强度作用到 humanoid 上，先后弹出 Before / After
python tools/visualize_mutator.py MUTATE_GEOM_SIZE `
    --seed mujoco__model_humanoid_humanoid --intensity huge

# 不带 --intensity 则在该 mutator 的所有合法强度上挨个展示
python tools/visualize_mutator.py STRUCT_GROW_LINK --seed mujoco__model_car_car

# 不带 --seed 时使用合成种子（与 smoke_mutators.py 同一份）
python tools/visualize_mutator.py MUTATE_JOINT_TYPE
```

工具流程：
1. 弹出 **Before** 窗口（原始模型，关掉窗口进入下一步）；
2. 内存中执行 mutator → 写出临时 XML → `mj_loadXML` 编译；
3. 弹出 **After** 窗口（标题里带变更摘要）；
4. mutated XML 同时落盘到 `.cache/viz/<mutator>__<seed>__<intensity>.xml` 便于 diff。

---

## 四、TODO — v2.0-γ（PPO + 真正的 finding pipeline）

需要继续推进时，把对应 prompt 直接抛给我即可。

### Prompt G1 — Episode RL Loop & Reward
> 在 `src/rl/` 实现 actor-critic：state = (current_layer, last_K_actions,
> trace summary stats)，action = (s_actor, s_scene, s_traj, o_atom, q, i, r, b)
> 八元组采样器；reward = oracle severity delta + 新 signature 增益 −
> ε·compile_fail。先封装 selector 接口，PPO 留给 G2。

### Prompt G2 — PPO Implementation
> 用 cleanrl 风格的 single-file PPO 实现 selector backbone，与
> `src/runner.run_seed` 同一接口。

### Prompt G3 — Reproducer + Shrinker
> `tools/repro.py findings/<id>` 用 mjpython 单独跑 dump 出来的 protocol；
> `tools/shrink.py` 用 delta-debugging 缩减 horizon / action_seq /
> mutator history。

### Prompt G4 — MJX Backend Differential
> 接 mjx，扩展 `BackendDiffOracle.evaluate`：用同一 protocol 在 classic 和 mjx
> 上各跑一次，比较 qpos/qvel residual。

### Prompt G5 — Experiment Harness
> `experiments/` + hydra/yaml sweep；记录 (config_hash, seed, signatures,
> wallclock, oracle_severity_sum)。
