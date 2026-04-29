# MuJoCo 生态 Bug Taxonomy 与 Seed / Operator / Oracle 设计清单（给 AI Agent）

更新时间：2026-04-29。目标：根据 MuJoCo 生态真实 issue / PR / commit fix 的模式，指导 AI agent 自动选择 seed、operator 与 oracle。

## 总体设计原则

1. 不把 RL 当作 oracle。RL 只负责在 operator 空间中学习选择更可能暴露异常的测试序列。
2. 每类 bug 都应有自己的 seed pool、operator subset 与 oracle。不同类别可以复用 operator，但 oracle 必须对应真实失效现象。
3. 优先使用 differential testing：原模型 vs 参数修正模型、classic MuJoCo vs MJX、同 seed 同 action 重放、同模型不同版本/渲染后端。
4. 对模型 bug 不只检查 crash/NaN，还要检查物理合理性、API 语义、任务成功、观测/奖励一致性。


## Operator 分层建议

### L0：Corpus / metadata / preflight operators（进入仿真前）
- `O0.1` cold clone 后扫描 XML、mesh、texture、skin、checkpoint、include 是否存在。
- `O0.2` 对每个 seed 跑 `MjModel.from_xml_path/string`、`mj_saveLastXML`、PyMJCF load、backend load。
- `O0.3` 做 MuJoCo version matrix、Python version matrix、backend matrix（classic MuJoCo / MJX / mujoco-py legacy / robosuite wrapper）。
- `O0.4` 生成 seed metadata：body/joint/geom/actuator/sensor/plugin/contact 数量、是否包含 freejoint、mocap、equality、tendon、muscle、flexcomp、skin、mesh-heavy assets。

### L1：Static model mutation / inspection operators（细粒度模型层）
- `O1.1` mass / diaginertia / inertial pos / inertial quat perturbation。
- `O1.2` geom size / pos / quat / type / contype / conaffinity / margin / group mutation。
- `O1.3` joint range / damping / armature / limited / autolimits mutation。
- `O1.4` actuator ctrlrange / gear / gainprm / biasprm / actdim / plugin mutation。
- `O1.5` visual-collision AABB / centroid / mesh principal-axis comparison。
- `O1.6` left-right symmetry checker and mutation。
- `O1.7` include path、meshdir、texture dir、asset duplication、many-mesh stress。

### L2：Runtime state/control/solver operators（中粒度仿真层）
- `O2.1` zero-control、home-control、random small control、single-actuator sweep。
- `O2.2` qpos/qvel perturbation、base tilt、object pose perturbation。
- `O2.3` external force/torque、off-center impulse、small push、contact impulse。
- `O2.4` timestep / integrator / solver / iterations / cone / Jacobian / impratio / disable flags sweep。
- `O2.5` warm-start clear/preserve、mj_resetData vs manual qpos/qvel/act restore。
- `O2.6` classic MuJoCo vs MJX vs wrapper differential rollout。
- `O2.7` sensor readback、contact force readback、inverse dynamics replay。

### L3：Task/composition/backend operators（粗粒度系统层）
- `O3.1` attach arm+gripper、hand+object、mobile base+arm、drone+payload。
- `O3.2` scripted reach-close-lift、push-slide、open-door、hold-stationary、hover-yaw、grasp-release sequences。
- `O3.3` RecordVideo / rgb_array / human / offscreen / vectorized env rendering stress。
- `O3.4` JAX grad/jacfwd/jacrev、PPO training smoke、batch/vmap action injection。
- `O3.5` package install/import/build matrix as CI preflight。

## 更新后的 Bug 分类

### C1. MJCF 加载 / schema / include / asset 引用类

**定义**：模型还没有进入动力学仿真阶段就失败，或不同解析器/版本对同一 MJCF/URDF/include/assets 的解释不一致。

**代表性历史报告 ID**：`MNG-008`, `MNG-013`, `GYM-011`, `DMC-003`, `DMC-004`, `DMC-009`, `MYO-004`, `MPG-007`。

**建议 seed 列表**：
- `mujoco_menagerie/booster_t1`
- `mujoco_menagerie/ufactory_lite6`
- `mujoco_menagerie/unitree_g1`
- `mujoco_menagerie/robotiq_2f85_v4`
- `dm_control nested include toy scenes`
- `dm_control PyMJCF many-mesh models`
- `Gymnasium Hopper-v4 XML`
- `MyoSuite myoLeg tutorial assets`
- `MuJoCo URDF-imported models`
- `custom XML with nested include`
- `custom XML with missing mesh`
- `custom XML with flexcomp`
- `custom XML with thousands of meshes`

**建议 operator**：
- file existence scan
- meshdir/texturedir mutation
- include path rewrite
- compiler autolimits toggle
- deprecated attribute injection/removal
- URDF->MJCF roundtrip
- PyMJCF/native MuJoCo differential load
- version matrix compile

**必须配套的 oracle**：
- `MjModel.from_xml_path/string` 是否成功
- 错误类型是否为 schema violation / FileNotFoundError / XML Error
- native MuJoCo 与 PyMJCF 是否加载一致
- 同一模型在不同 MuJoCo 版本上的编译结果是否一致
- 所有 mesh/texture/hfield/skin/checkpoint 文件是否可达

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C2. 启动稳定性 / 默认控制器发散类

**定义**：模型能加载，但在默认 keyframe、zero-control、home-control 或小扰动下出现 CoM drift、振荡增长、摔倒、qvel/qacc spike。

**代表性历史报告 ID**：`MNG-002`, `MNG-003`, `MNG-005`, `MJC-014`, `MJC-016`。

**建议 seed 列表**：
- `mujoco_menagerie/booster_t1`
- `mujoco_menagerie/kinova_gen3`
- `mujoco_menagerie/unitree_go2`
- `mujoco_menagerie/unitree_h1`
- `mujoco_menagerie/unitree_g1`
- `mujoco_menagerie/agility_cassie`
- `mujoco_menagerie/talos`
- `mujoco_menagerie/franka_emika_panda`
- `mujoco_menagerie/kuka_iiwa_14`
- `mujoco_menagerie/universal_robots_ur5e`
- `mujoco_menagerie/universal_robots_ur10e`
- `dm_control/humanoid_stand`
- `dm_control/walker_walk`
- `dm_control/quadruped_walk`
- `Gymnasium/Humanoid-v5`
- `Gymnasium/Ant-v5`
- `Gymnasium/Hopper-v5`
- `Gymnasium/Walker2d-v5`
- `MuJoCo core humanoid MJX examples`

**建议 operator**：
- zero-control rollout
- home-keyframe reset
- small ctrl pulse
- single-joint target sweep
- kp/kd sweep
- damping/armature scale
- timestep/integrator sweep
- solver iteration sweep
- contact exclusion toggle

**必须配套的 oracle**：
- NaN/Inf in qpos/qvel/qacc
- base height drop / fall detector
- CoM drift over threshold
- joint oscillation amplitude growth
- contact force spike
- kinetic energy spike
- same test after parameter patch 是否恢复稳定

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C3. 质量 / 惯量 / COM / mass matrix 类

**定义**：质量、惯量、惯性坐标系、左右对称性或组合后的质量矩阵存在不合理性；这类 bug 往往不是 NaN，而是轻微受力后出现不合理动力学响应。

**代表性历史报告 ID**：`MNG-004`, `MNG-015`, `MNG-016`, `MNG-018`, `RSU-008`。

**建议 seed 列表**：
- `mujoco_menagerie/stanford_tidybot`
- `mujoco_menagerie/robotiq_2f85_v4`
- `mujoco_menagerie/talos`
- `mujoco_menagerie/leap_hand`
- `mujoco_menagerie/shadow_dexee`
- `mujoco_menagerie/allegro_hand`
- `mujoco_menagerie/franka_emika_panda`
- `mujoco_menagerie/kuka_iiwa_14`
- `robosuite NullGripper + Panda`
- `robosuite Robotiq85 + arm`
- `MyoSuite myoArm`
- `MyoSuite myoLeg`
- `custom symmetric biped`
- `custom mobile base`

**建议 operator**：
- mass scale
- diaginertia scale
- inertial pos/quat perturbation
- left-right mirror comparison
- off-center impulse
- small torque
- base push
- mass matrix readback
- inverse dynamics compare

**必须配套的 oracle**：
- mass-bbox-inertia plausibility
- radius of gyration 是否过小/过大
- COM 与 mesh/collision centroid 偏移
- 左右对称 body 的 mass/inertia/COM 是否一致
- mass matrix 是否 finite/symmetric/positive definite
- 小扰动后的 angular velocity / rotational energy 是否异常
- 修复参数前后 differential testing

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C4. 碰撞 / 接触几何 / 穿透 / contact force 类

**定义**：碰撞几何、接触列表、穿透深度、contact force、contact sensor 或 contact backend 在不同设置下不一致。

**代表性历史报告 ID**：`MNG-011`, `MNG-015`, `MJC-003`, `MJC-005`, `MJC-009`, `MJC-010`, `MJC-012`, `MJC-016`, `GYM-013`。

**建议 seed 列表**：
- `mujoco_menagerie/universal_robots_ur10e`
- `mujoco_menagerie/universal_robots_ur5e`
- `mujoco_menagerie/robotiq_2f85_v4`
- `mujoco_menagerie/kinova_gen3`
- `Gymnasium/Pusher-v4`
- `Gymnasium/Ant-v5`
- `Gymnasium/Humanoid-v5`
- `Gymnasium/InvertedDoublePendulum-v4`
- `MuJoCo core humanoid MJX`
- `MuJoCo core cylinder-plane minimal model`
- `dm_control/quadruped`
- `robosuite/Lift`
- `robosuite/PickPlace`
- `Gymnasium-Robotics/AdroitHandRelocate`
- `Gymnasium-Robotics/FetchPickAndPlace`

**建议 operator**：
- collision size/pos/quat mutation
- visual-collision AABB compare
- floor/table/object pose mutation
- contact margin/friction/solref/solimp mutation
- gravity on/off
- contact-rich state reset
- MJX/classic backend differential
- cylinder/plane boundary sweep

**必须配套的 oracle**：
- 非预期 initial contact
- penetration depth
- contact pair count/order/geom pair consistency
- contact force magnitude and variance
- floor tunneling detector
- self-intersection detector
- MJX vs classic MuJoCo contact differential
- contact sensor / mj_contactForce consistency

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C5. 执行器 / 控制范围 / gear / force direction / tracking 类

**定义**：执行器目标、ctrlrange、gear 符号、力矩方向、肌肉/逆动力学控制或 tracking 行为与期望不一致。

**代表性历史报告 ID**：`MNG-001`, `MNG-006`, `MNG-007`, `MNG-017`, `MYO-005`, `MYO-007`, `MPG-008`。

**建议 seed 列表**：
- `mujoco_menagerie/flexiv_rizon4`
- `mujoco_menagerie/kuka_iiwa_14`
- `mujoco_menagerie/leap_hand`
- `mujoco_menagerie/skydio_x2`
- `mujoco_menagerie/shadow_dexee`
- `mujoco_menagerie/allegro_hand`
- `mujoco_menagerie/franka_emika_panda`
- `mujoco_menagerie/universal_robots_ur5e`
- `mujoco_menagerie/universal_robots_ur10e`
- `MyoSuite/myoArm`
- `MyoSuite/myoElbowPose1D6MExoRandom-v0`
- `MyoSuite/myoLegWalk`
- `MuJoCo Playground/LeapCubeReorientation`
- `robosuite IK_POSE envs`
- `robosuite Franka Panda`

**建议 operator**：
- ctrlrange mutation
- joint range mutation
- single-actuator sweep
- target qpos replay
- gear sign flip
- gear magnitude scale
- kp/kd sweep
- actuator force perturbation
- inverse dynamics ctrl reconstruction
- hold-stationary test

**必须配套的 oracle**：
- target-vs-actual qpos steady-state error
- end-effector pose error
- overshoot/settling time
- single actuator 是否只影响预期 joint/axis
- drone yaw/roll/pitch net torque consistency
- qfrc_actuator 与 inverse dynamics residual
- controller zero target drift

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C6. 状态 reset / reproducibility / warm-start / clone 类

**定义**：同 seed、同 state、同 action 后轨迹不一致；state-setting 在 contact-rich 状态下破坏物体位置；mocap weld reset 写错。

**代表性历史报告 ID**：`GRB-002`, `GRB-003`, `GRB-004`, `GRB-005`, `GYM-014`, `DMC-005`。

**建议 seed 列表**：
- `Gymnasium-Robotics/FetchPickAndPlace-v2`
- `Gymnasium-Robotics/FetchPickAndPlace-v3`
- `Gymnasium-Robotics/FetchReach`
- `Gymnasium-Robotics/FetchPush`
- `Gymnasium-Robotics/FetchSlide`
- `Gymnasium-Robotics/AdroitHandRelocate`
- `Gymnasium-Robotics/AdroitHandDoor`
- `Gymnasium-Robotics/AdroitHandPen`
- `Gymnasium MuJoCo envs`
- `dm_control custom mocap env`
- `robosuite Lift/PickPlace with hard_reset`
- `custom mocap-weld models`

**建议 operator**：
- same seed repeated reset
- same action replay
- warm-start clear/preserve toggle
- get_state/set_state replay
- contact-rich state capture
- mocap weld eq_data reset
- deepcopy/serialization clone
- hard_reset toggle

**必须配套的 oracle**：
- obs/trajectory equality
- initial qpos/qvel/act equality
- object teleport detector
- penetration after set_state
- mocap-body pose residual
- state-space key consistency
- cloned env trajectory divergence

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C7. 传感器 / 观测 / info / reward 语义类

**定义**：仿真本身可能没有崩溃，但 API 暴露给 RL agent 的 observation/info/reward/sensor 与模型状态或文档语义不一致。

**代表性历史报告 ID**：`MJC-008`, `GYM-007`, `GYM-008`, `GYM-009`, `GYM-010`, `GYM-012`, `GYM-015`, `GRB-001`, `RSU-006`, `MPG-002`。

**建议 seed 列表**：
- `Gymnasium/Ant-v5`
- `Gymnasium/Humanoid-v5`
- `Gymnasium/Reacher-v5`
- `Gymnasium/Pusher-v5`
- `Gymnasium/InvertedDoublePendulum-v4`
- `Gymnasium-Robotics/FetchPickAndPlace`
- `Gymnasium-Robotics custom obstacle Fetch`
- `robosuite Franka Panda F/T sensor`
- `MuJoCo core acceleration sensor models`
- `MuJoCo Playground vision envs`
- `dm_control pixel observation envs`

**建议 operator**：
- info-observation consistency check
- reward pre/post transition differential
- healthy range boundary sweep
- sensor force perturbation
- FK recomputation
- observation shape mutation
- custom site/body injection
- docs/API parameter scan

**必须配套的 oracle**：
- info position == underlying qpos/site xpos
- reward component identity
- reward timing equals post-transition state
- constant-observation feature detector
- sensor_acc backend residual
- FK/sensor residual
- observation shape/schema oracle
- documented argument 是否实际生效

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C8. 渲染 / 相机 / headless / vectorized rendering 类

**定义**：MuJoCo 生态大量 bug 发生在 offscreen rendering、EGL/OSMesa/GLFW、RecordVideo、多环境相机 buffer、deformable/skin rendering 上。

**代表性历史报告 ID**：`DMC-002`, `DMC-006`, `DMC-007`, `DMC-008`, `GYM-001`, `GYM-003`, `GYM-004`, `GYM-005`, `GYM-016`, `RSU-001`, `RSU-002`, `RSU-003`, `RSU-011`, `RSU-012`, `RSU-013`, `MYO-003`。

**建议 seed 列表**：
- `dm_control humanoid/camera envs`
- `Gymnasium HalfCheetah-v5`
- `Gymnasium Ant-v5`
- `Gymnasium Humanoid-v5`
- `Gymnasium FetchSlide-v4 with play`
- `robosuite Lift/PickPlace/Wipe cameras`
- `robosuite SawyerPickPlace`
- `robosuite deformable/skin env`
- `MyoSuite installation example`
- `MuJoCo Playground vision tasks`
- `custom multi-camera scene`
- `custom vectorized env camera stress`

**建议 operator**：
- GL backend switch EGL/OSMesa/GLFW
- same-state multi-render
- RecordVideo wrapper
- multi-episode render context reuse
- SyncVectorEnv render
- multiple cameras
- unrelated import/context pollution
- deformable skin render
- Python version matrix
- GPU device selection

**必须配套的 oracle**：
- framebuffer completeness
- black frame detector
- pixel hash determinism
- image flip/mirror detector
- cross-env frame contamination
- segfault/process crash
- render exception classification
- camera frame shape/value schema

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C9. MJX / backend differential / plugin / action dimension 类

**定义**：同一模型在 classic MuJoCo、MJX、Brax、madrona-mjx、robosuite wrapper 中维度、contact、sensor、state update 或 plugin 语义不同。

**代表性历史报告 ID**：`MNG-009`, `MJC-001`, `MJC-003`, `MJC-006`, `MJC-007`, `MJC-008`, `MJC-010`, `MJC-012`, `MJC-013`, `MJC-016`, `RSU-007`, `MPG-001`, `MPG-003`。

**建议 seed 列表**：
- `mujoco_menagerie/shadow_dexee`
- `mujoco_menagerie/robotiq_2f85`
- `MuJoCo core humanoid`
- `MuJoCo core cartpole`
- `MuJoCo core plane-cylinder`
- `MuJoCo Playground cable env`
- `MuJoCo Playground vision randomization env`
- `robosuite robotiq_gripper_85 with touch_grid`
- `Gymnasium MuJoCo classic envs converted to MJX`
- `custom equality constraint models`
- `custom plugin actuator models`

**建议 operator**：
- classic vs MJX differential
- plugin on/off
- actdim/action vector mutation
- batched/vmap action injection
- eq_active toggle
- contact pair generation
- sensor readback
- MJX state get/put
- renderer randomization function mutation

**必须配套的 oracle**：
- model.nu/action_space/actdim consistency
- state trajectory residual classic-vs-MJX
- contact count/force differential
- sensor backend residual
- shape mismatch / IndexError / broadcast error
- plugin recognition error
- same seed backend divergence

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C10. JAX/MJX differentiability / GPU numerical / precision 类

**定义**：面向 GPU 批量训练或 differentiable simulation 时，gradient、jacobian、XLA/cusolver、precision 出现 NaN 或硬错误。

**代表性历史报告 ID**：`MJC-002`, `MJC-011`, `MJC-015`, `MPG-004`, `MPG-005`。

**建议 seed 列表**：
- `MuJoCo MJX ball-plane contact`
- `MuJoCo MJX contact transition models`
- `MuJoCo MJX CG solver models`
- `MuJoCo Playground PPO training envs`
- `MuJoCo Playground humanoid/walker/go2/leap tasks`
- `custom differentiable contact toy models`
- `custom contact leaving-ground model`

**建议 operator**：
- jax.grad/jacfwd/jacrev
- contact/no-contact boundary sweep
- solver switch CG/Newton
- x64 on/off
- batch size sweep
- CUDA/JAX version matrix
- PPO reset/train smoke
- seed sweep

**必须配套的 oracle**：
- gradient finite check
- jacobian NaN/Inf
- XlaRuntimeError/cusolver internal error
- precision mismatch logs
- training abort
- same model CPU/GPU differential

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C11. 任务级组合 / grasp / attachment / manipulation 类

**定义**：单个模型可用，但 arm+gripper+object/table/controller 组合后发生抓取失败、finger tilt、gripper collapse、attachment pose 错误。

**代表性历史报告 ID**：`MNG-010`, `MNG-012`, `MNG-014`, `RSU-005`, `RSU-009`。

**建议 seed 列表**：
- `mujoco_menagerie/ur5e + robotiq_2f85 + cube`
- `mujoco_menagerie/ur10e + robotiq_2f85`
- `mujoco_menagerie/franka_emika_panda + robotiq`
- `mujoco_menagerie/kuka_iiwa + gripper`
- `robosuite Lift/PickPlace/Stack/NutAssembly`
- `robosuite Door/Wipe/ToolHang`
- `robosuite BDGripper`
- `robosuite SchunkSvhLeftHand`
- `robosuite Robotiq85`
- `Gymnasium-Robotics FetchPickAndPlace/FetchPush/FetchSlide`
- `Gymnasium-Robotics AdroitHandRelocate`
- `custom arm+gripper+object scenes`

**建议 operator**：
- attach-site mutation
- mount pose mutation
- object mass/size/friction mutation
- reach-close-lift sequence
- gripper force sweep
- finger actuator sweep
- controller switch
- table/floor pose mutation
- integrator switch
- task reset mutation

**必须配套的 oracle**：
- grasp success/lift height
- object slip/drop
- finger symmetry / angle deviation
- gripper collapse detector
- contact stability
- unexpected arm joint motion after gripper command
- task success metric regression

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

### C12. 版本 / 安装 / 构建 / 绑定选择类

**定义**：这一类不一定是物理 bug，但对 seed corpus 采集、CI、可复现和大规模 fuzzing 很关键；可作为 preflight/operator，不作为核心物理贡献。

**代表性历史报告 ID**：`DMC-001`, `DMC-010`, `GYM-002`, `GYM-006`, `MPY-001`, `MPY-003`, `MPY-004`, `MPY-005`, `MPY-006`, `MPY-007`, `MPY-008`, `MYO-001`, `MYO-002`, `MYO-006`, `MPG-006`, `MPC-003`, `MPC-004`, `MPC-005`, `MPC-006`, `MPC-007`, `MPC-008`。

**建议 seed 列表**：
- `mujoco-py install/import`
- `Gymnasium MuJoCo v4/v5`
- `Gymnasium-Robotics mujoco-py legacy envs`
- `dm_control source install`
- `MyoSuite tutorials`
- `MuJoCo Playground install`
- `mujoco_mpc pip/source/Windows build`
- `all selected seed repos`

**建议 operator**：
- Python version matrix
- MuJoCo version matrix
- mujoco vs mujoco-py backend toggle
- Cython version matrix
- GL backend env var
- compiler/OS matrix
- package import smoke
- CI cold install
- LD_LIBRARY_PATH/PATH mutation

**必须配套的 oracle**：
- import/env creation success
- build wheel success
- dependency resolver success
- API field existence
- wrong backend selected
- platform-specific compile errors
- installation timeout
- runtime ABI mismatch

**AI agent 执行提示**：先从代表性 seed 生成最小测试序列；如果 oracle 只发现异常但不能归因，需要再做 ablation，例如关闭 contact、修正 inertial、切换 backend、清空 warm-start、修改 timestep 或替换 renderer。

## 推荐的最小落地顺序

第一阶段只做 6 类：C1 加载/schema、C2 启动稳定性、C3 mass/inertia、C4 碰撞/contact、C5 actuator tracking、C6 reset/reproducibility。它们最容易自动化、最容易复现，也最贴近 MuJoCo model fuzzing 的论文主线。

第二阶段扩展 C7 observation/reward、C8 rendering、C9 MJX differential、C10 differentiability、C11 composition task。它们更能展示系统覆盖面，但工程成本更高。

C12 build/install/version 适合作为 corpus preflight 和 CI 贡献，不建议作为论文的核心物理 bug claim。

## 输出格式建议

AI agent 每次发现异常时，应输出：`seed_id`、`operator_sequence`、`oracle_name`、`failure_signal`、`minimal_reproducer`、`suspected_root_cause`、`ablation_result`、`whether_historical_pattern_matched`。
