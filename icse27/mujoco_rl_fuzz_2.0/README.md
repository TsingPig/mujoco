# mujoco_rl_fuzz_2.0

GzFuzz 风格、面向 MuJoCo 物理引擎的 RL fuzzer —— v2.0 重设计。

本目录与 `../mujoco_rl_fuzz/`（v1）**完全隔离**，除 `src/triage/signature.py`
是 v1 triage 哈希的硬拷贝外，没有任何从 v1 的导入。

---

## 一、当前已完成（v2.0-α）

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

## 四、TODO（v2.0-β / 后续 prompt 模板）

需要继续推进时，把对应 prompt 直接抛给我即可。

### Prompt P1 — 拉齐其余 6 个种子源
> “网络已经可用，请运行 `python tools/fetch_seeds.py`（不带 `--only`），
> 拉齐 menagerie / dm_control / gym_robotics / mujoco_mpc / robosuite / mjx
> 共 7 个源；之后跑 `validate_seeds.py` + `validate_mutators.py --reps 1`，
> 把新出现的 post_compile_fail 按 v2.0-α 一样的方式逐个收敛到 0，
> 并把通过的种子总数追加到 README §一·1 末尾。”

### Prompt P2 — Episodic Runner & Oracle
> “实现 `src/runner/episode.py`：给定 (model_xml, runtime_directives, n_steps)，
> 调用 `mj_step` 循环；oracle 检测 NaN/Inf qacc、mjWARN_BADCTRL、
> contact 数突变、能量爆增；命中后 dump 到 `findings/<sig>/`。
> Sig 复用 `src/triage/signature.py`。”

### Prompt P3 — Actor-Critic + Sequential Mutator Selector
> “按 GzFuzz ISSTA'25 的思路实现 `src/rl/`：state = (current_model_features,
> last_K_mutators)，action = mutator_id ∪ intensity_mode，
> reward = +1 新签名 / -ε 编译失败 / +λ·branch_cov_delta（覆盖率挂 v1 已有的
> mjMARKLINE 桩）。先用 PPO，policy/value 共享 MLP backbone。”

### Prompt P4 — 增量种子 / Pool / Replay
> “实现 `src/seedpool/`：episode 找到的 mutated XML 若新增 issue 签名，
> 按概率回灌到 curated pool；保留滚动 LRU。”

### Prompt P5 — 复现脚本与最小化
> “实现 `tools/repro.py findings/<sig>`：用 mjpython 单独跑 dump 出来的
> XML+directive，确认稳定复现；再实现 `tools/shrink.py` 用 delta-debugging
> 缩减 XML 与 directive。”

### Prompt P6 — 实验记录
> “给 ICSE'27 起一个 `experiments/` 目录，配 hydra/yaml 的 sweep 脚本，
> 每次跑落盘 (config_hash, seed, found_sigs, wallclock)。”
