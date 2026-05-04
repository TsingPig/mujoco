"""Local web GUI for seeds 2.0 — categorized & collapsible.

Same backend pattern as `viz_gui.py` (stdlib http.server, spawns
`visualize_seed.py` / `visualize_mutator.py` subprocesses), but the seed
table is grouped by the 12 bug-taxonomy categories defined in
`seeds2/categories.yaml`. Every category renders inside a `<details>` block
so the page is short by default. A toolbar lets you expand/collapse all.

Usage:
    python tools/viz_gui2.py            # serves http://127.0.0.1:9100/
    python tools/viz_gui2.py --port 9100 --no-open
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mutations.registry import MUTATORS, MUTATOR_IDS  # noqa: E402

SEEDS_DIR = ROOT / "seeds" / "curated"
SEEDS2_DIR = ROOT / "seeds2"
TOOLS_DIR = ROOT / "tools"
PYTHON = sys.executable

# Source -> {license SPDX, upstream repo, 中文用途说明}. Mirrors viz_gui.py.
SOURCE_INFO: dict[str, dict[str, str]] = {
    "mujoco": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco",
        "desc":    "MuJoCo 官方仓库示例模型（humanoid、car、cards、tendon_arm 等），覆盖引擎核心特性的最小可复现场景。",
    },
    "menagerie": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco_menagerie",
        "desc":    "DeepMind 维护的高质量真实机器人模型库（Franka、UR5e、Spot、ANYmal、ALOHA、Shadow Hand 等），对几何/惯量/驱动器参数做过校准。",
    },
    "dm_control": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/dm_control",
        "desc":    "DeepMind Control Suite——经典 RL benchmark（cartpole、cheetah、walker、quadruped 等），任务难度可控，适合控制类策略 fuzz。",
    },
    "gym_robotics": {
        "license": "MIT",
        "repo":    "github.com/Farama-Foundation/Gymnasium-Robotics",
        "desc":    "Farama Gymnasium-Robotics（Fetch、HandManipulate、PointMaze 等），稀疏/密集奖励的目标到达类操作任务。",
    },
    "mujoco_mpc": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco_mpc",
        "desc":    "DeepMind MJPC 任务模型（Cartpole-Swingup、Quadruped、Humanoid-Walk 等），约束/接触组合较多。",
    },
    "robosuite": {
        "license": "MIT",
        "repo":    "github.com/ARISE-Initiative/robosuite",
        "desc":    "ARISE 模块化机械臂操作框架（Lift、Stack、PickPlace、Door 等），单/双臂 + 多种 gripper。",
    },
    "mjx": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco",
        "desc":    "MJX（MuJoCo 的 JAX 后端）官方示例模型，适合 GPU 批量并行的几何/约束子集。",
    },
    "robocasa": {
        "license": "MIT",
        "repo":    "github.com/robocasa/robocasa",
        "desc":    "RoboCasa 大规模厨房模拟器：100+ 原子任务 + long-horizon composite，最接近真实机械臂做饭的开源场景。",
    },
    "libero": {
        "license": "MIT",
        "repo":    "github.com/Lifelong-Robot-Learning/LIBERO",
        "desc":    "LIBERO 130 个长程操作任务（kitchen / study / living），专为 lifelong learning 设计。",
    },
    "mimicgen": {
        "license": "MIT",
        "repo":    "github.com/NVlabs/mimicgen",
        "desc":    "NVIDIA MimicGen：Coffee、Stack-Three、Threading、Square 等多步操作场景。",
    },
    "safety_gymnasium": {
        "license": "Apache-2.0",
        "repo":    "github.com/PKU-Alignment/safety-gymnasium",
        "desc":    "PKU 安全 RL benchmark（Car/Point/Doggo + hazards/pillars/vases）。",
    },
    "myosuite": {
        "license": "Apache-2.0",
        "repo":    "github.com/MyoHub/myosuite",
        "desc":    "MyoSuite 高保真生物力学模型，含大量 tendon/site/equality 约束。",
    },
    "mujoco_playground": {
        "license": "Apache-2.0",
        "repo":    "github.com/google-deepmind/mujoco_playground",
        "desc":    "MuJoCo Playground（DeepMind）：dm_control_suite 标准化版本，便于做后端兼容性 fuzz。",
    },
    "composed": {
        "license": "inherits parent",
        "repo":    "tools/compose_arena.py",
        "desc":    "compose_arena.py 用 <replicate> 把若干父种子拼成的多实例 arena。",
    },
    "unknown": {"license": "?", "repo": "?", "desc": "未识别来源。"},
}


def _upstream_url(repo: str | None, commit: str | None,
                  src_rel_path: str | None) -> str | None:
    """GitHub blob URL for the seed's original XML, or None when unknown."""
    if not repo or not repo.startswith("github.com/"):
        return None
    base = f"https://{repo.rstrip('/')}"
    if not commit or not src_rel_path:
        return base
    rel = str(src_rel_path).replace("\\", "/")
    return f"{base}/blob/{commit}/{rel}"


# --------------------------------------------------------------------------
# 中文释义字典：列头 / operator / oracle。供前端浮层弹窗使用。
# --------------------------------------------------------------------------
COLUMN_TIPS: dict[str, str] = {
    "name":   "种子名（curated 文件夹名）。点击 ▶ 实时跑、◻ 静态查看、⧉ 复制路径、M 送入 mutator、🌐 打开上游 GitHub。",
    "source": "种子来源仓库。鼠标移到 source 标签上可看仓库背景介绍。",
    "nq":     "<b>nq</b>：广义坐标维度（自由度配置数）。每个 free joint +7、ball +4、hinge/slide +1。",
    "nv":     "<b>nv</b>：广义速度维度。free joint +6、ball +3、hinge/slide +1。一般 nv ≤ nq。",
    "nbody":  "<b>nbody</b>：&lt;body&gt; 节点总数（含 worldbody）。反映场景物体数量与运动学树规模。",
    "ngeom":  "<b>ngeom</b>：&lt;geom&gt; 总数。包含可视和碰撞几何体；越大越接近真实，但接触求解越慢。",
    "nu":     "<b>nu</b>：actuator 数量（电机/位置/速度/通用控制器）。决定 ctrl 向量长度。",
    "act":    "操作动作（实时跑 / 静态查看 / 复制路径 / 送入 mutator / 打开上游 GitHub）。",
}

OPERATOR_TIPS: dict[str, str] = {
    # —— XML / 资源 / 加载链路 ——
    "file_existence_scan": "扫描 XML 引用的资源文件是否存在",
    "meshdir_texturedir_mutation": "改写 compiler 的 meshdir/texturedir 路径",
    "include_path_rewrite": "改写 &lt;include&gt; 的相对/绝对路径",
    "autolimits_toggle": "切换 compiler.autolimits（自动 joint range）",
    "urdf_mjcf_roundtrip": "URDF→MJCF→URDF 往返一致性",
    "pymjcf_native_diff_load": "dm_control PyMJCF vs 原生 mujoco 加载差异",
    "version_matrix_compile": "多 mujoco 版本下编译同一 XML",
    "asset_reachability": "<i>(此处亦作 operator)</i> 检查所有 asset 引用",
    # —— 物理参数变异 ——
    "gravity_toggle": "开关重力或翻转 g 向量",
    "timestep_integrator_sweep": "扫描 timestep + integrator 组合",
    "integrator_switch": "切换积分器（Euler/RK4/implicit）",
    "solver_switch": "切换求解器（PGS/CG/Newton）",
    "solver_iter_sweep": "扫描求解器迭代次数",
    "warm_start_toggle": "开关求解器 warm-start",
    "x64_toggle": "float32 / float64 精度切换",
    # —— 质量 / 惯量 / 几何 ——
    "mass_scale": "整体或单 body 质量缩放",
    "diaginertia_scale": "对角惯量缩放",
    "damping_armature_scale": "joint damping / armature 缩放",
    "inertial_pos_quat_perturb": "惯性元 pos / quat 微扰",
    "mass_matrix_readback": "读取并校验质量矩阵 M",
    "object_mass_size_friction_mutation": "物体质量 / 尺寸 / 摩擦三联变异",
    "diaginertia_scale ": "对角惯量缩放",
    # —— 驱动器 / 控制 ——
    "gear_magnitude_scale": "actuator gear 幅值缩放",
    "gear_sign_flip": "actuator gear 符号反转",
    "ctrlrange_mutation": "actuator ctrlrange 变异",
    "joint_range_mutation": "joint range 变异",
    "actdim_action_mutation": "actuator actdim / action 维度变异",
    "single_actuator_sweep": "逐 actuator 扫描激励",
    "single_joint_target_sweep": "逐 joint 目标位置扫描",
    "finger_actuator_sweep": "手指 actuator 力扫描",
    "gripper_force_sweep": "夹爪力扫描",
    "kp_kd_sweep": "PD 增益 kp/kd 扫描",
    "healthy_range_sweep": "扫描 healthy_range（locomotion 任务）",
    "controller_switch": "切换 controller 类型",
    "small_ctrl_pulse": "注入小幅控制脉冲",
    "small_torque": "注入小力矩",
    "off_center_impulse": "施加偏心冲量",
    "base_push": "对 base/torso 施加推力",
    "hold_stationary": "保持静止状态",
    "zero_control_rollout": "零控制信号 rollout",
    "actuator_force_perturb": "对 actuator 力做小扰动",
    "sensor_force_perturb": "对力传感器读数做扰动",
    "left_right_mirror_compare": "左右镜像对比",
    # —— 接触 ——
    "contact_margin_friction_solref_solimp_mutation": "接触 margin/friction/solref/solimp 变异",
    "contact_pair_generation": "显式生成 &lt;pair&gt; 接触对",
    "contact_exclusion_toggle": "开关 &lt;exclude&gt; 接触排除",
    "contact_boundary_sweep": "刚体在接触边界附近的位姿扫描",
    "cylinder_plane_boundary_sweep": "圆柱-平面接触边界扫描",
    "collision_geom_mutation": "替换 collision geom 类型(sphere/box/capsule)",
    "visual_collision_aabb_compare": "可视 vs 碰撞 geom AABB 对比",
    # —— 状态 / 重置 / 回放 ——
    "get_set_state_replay": "set_state→get_state 回放",
    "contact_rich_state_capture": "保存接触丰富状态快照",
    "contact_rich_state_reset": "从接触丰富状态恢复",
    "target_qpos_replay": "用 target qpos 回放",
    "same_action_replay": "同一动作序列重放",
    "same_seed_repeated_reset": "同 seed 重复 reset",
    "seed_sweep": "seed 扫描",
    "hard_reset_toggle": "软 / 硬 reset 切换",
    "home_keyframe_reset": "用 home keyframe reset",
    "task_reset_mutation": "任务 reset 选项变异",
    "deepcopy_clone": "deepcopy 克隆 env",
    "mocap_weld_reset": "mocap / weld reset 后位姿",
    "eq_active_toggle": "&lt;equality&gt; active 开关",
    # —— 场景 / 位姿 ——
    "floor_table_object_pose_mutation": "floor / table / object 位姿变异",
    "table_floor_pose_mutation": "桌面 / 地面位姿变异",
    "mount_pose_mutation": "末端 / 机座 mount 位姿变异",
    "attach_site_mutation": "attach site 位置变异",
    "custom_site_body_inject": "注入自定义 site / body",
    "obs_shape_mutation": "改变 observation 形状",
    "info_obs_consistency": "info dict 与 obs 一致性",
    "sensor_readback": "sensor 读回",
    # —— 动力学 / 数值 ——
    "fk_recompute": "多次重算前向运动学",
    "inverse_dynamics_compare": "正 / 逆动力学交叉验证",
    "inv_dyn_ctrl_reconstruction": "由 qacc 反推 ctrl 并重放",
    "jax_grad_jac": "JAX 求梯度 / 雅可比",
    # —— MJX / 后端 ——
    "mjx_classic_diff": "MJX vs Classic 后端差分",
    "classic_vs_mjx_diff": "Classic vs MJX 后端差分（同上别名）",
    "mjx_state_get_put": "MJX 状态 put / get 往返",
    "batched_vmap_action": "vmap 批量执行同一动作",
    "batch_size_sweep": "扫描 batch size",
    "gpu_device_select": "选择不同 GPU 设备",
    # —— 安装 / 版本矩阵 ——
    "cuda_jax_version_matrix": "CUDA + JAX 版本矩阵",
    "cython_matrix": "Cython 编译矩阵",
    "python_version_matrix": "Python 版本矩阵",
    "mujoco_version_matrix": "MuJoCo 版本矩阵",
    "compiler_os_matrix": "编译器 / OS 矩阵",
    "ci_cold_install": "CI 冷安装(无缓存)",
    "package_import_smoke": "仅 import 包做最小 smoke",
    "import_context_pollution": "检查 import 顺序污染",
    "plugin_toggle": "开关 mujoco plugin",
    # —— 渲染 ——
    "gl_backend_env": "改 GL backend 环境变量(MUJOCO_GL)",
    "gl_backend_switch": "切换 GL backend(EGL/GLFW/OSMesa)",
    "multi_camera": "渲染多相机",
    "multi_episode_render_context": "跨 episode 渲染上下文复用",
    "record_video_wrapper": "VideoRecorder wrapper",
    "renderer_randomization_mutation": "渲染随机化(光照/材质)",
    "same_state_multi_render": "同状态多次渲染",
    "deformable_skin_render": "渲染可变形 skin",
    "sync_vector_env_render": "SyncVectorEnv 渲染",
    # —— 任务 / RL ——
    "ppo_smoke": "PPO 训练 smoke run",
    "reach_close_lift_seq": "reach→close→lift 抓取序列",
    "reward_pre_post_diff": "reward 转移前后差分",
    "mujoco_vs_mujoco_py": "mujoco vs mujoco-py 一致性",
    "docs_api_param_scan": "文档 API 参数扫描",
}

ORACLE_TIPS: dict[str, str] = {
    "action_space_consistency": "动作空间形状/dtype 一致",
    "actuator_axis_isolation": "actuator 轴独立(无串扰)",
    "api_field_existence": "API 字段是否存在",
    "asset_reachability": "&lt;mesh/texture/include&gt; 资源可达",
    "base_height_drop": "base / torso 高度跌落检测",
    "black_frame": "渲染输出全黑帧",
    "build_wheel_success": "编译 wheel 成功",
    "camera_frame_schema": "相机输出 frame 形状 / dtype",
    "cloned_env_divergence": "deepcopy 后行为发散",
    "com_drift": "重心漂移",
    "com_vs_mesh_centroid": "COM 与 mesh 几何中心一致",
    "constant_observation_detector": "obs 始终为常量(传感器死)",
    "contact_count_force_diff": "接触数与合力差分",
    "contact_force_magnitude": "接触力幅值合理",
    "contact_force_spike": "接触力毛刺",
    "contact_pair_consistency": "接触对一致性",
    "contact_sensor_consistency": "接触传感器一致性",
    "contact_stability": "接触稳定性",
    "cpu_gpu_diff": "CPU / GPU 后端差分",
    "cross_env_frame_contamination": "跨 env 渲染帧污染",
    "cross_version_compile_consistency": "跨版本编译一致性",
    "dependency_resolver": "pip 依赖解析",
    "documented_arg_effective": "文档参数实际生效",
    "drone_yaw_roll_pitch_torque_balance": "无人机 yaw/roll/pitch 力矩平衡",
    "end_effector_pose_error": "末端位姿误差",
    "finger_symmetry": "手指左右对称",
    "fk_sensor_residual": "FK 与传感器残差",
    "floor_tunneling": "物体穿透地面",
    "framebuffer_completeness": "framebuffer 完整性",
    "from_xml_path_success": "from_xml_path 加载成功",
    "gradient_finite": "梯度有限(无 NaN/Inf)",
    "grasp_success": "抓取是否成功",
    "gripper_collapse": "夹爪坍缩(自穿透)",
    "image_flip": "图像上下颠倒",
    "import_success": "import 是否成功",
    "info_pos_eq_xpos": "info.pos == data.xpos",
    "initial_state_equality": "初始状态相等",
    "install_timeout": "安装超时",
    "jacobian_finite": "雅可比有限",
    "kinetic_energy_spike": "动能毛刺",
    "left_right_param_consistency": "左右参数一致",
    "mass_bbox_inertia_plausibility": "质量 / 包围盒 / 惯量合理",
    "mass_matrix_finite_psd": "质量矩阵 M 有限且半正定",
    "mjx_vs_classic_contact_diff": "MJX vs Classic 接触差分",
    "mocap_pose_residual": "mocap 位姿残差",
    "native_vs_pymjcf_consistency": "原生 vs PyMJCF 一致性",
    "object_slip": "物体滑动",
    "object_teleport_after_set_state": "set_state 后物体瞬移",
    "obs_schema": "observation 模式(shape/dtype)",
    "obs_trajectory_equality": "obs 轨迹相等",
    "oscillation_amplitude": "振荡幅度",
    "overshoot_settling": "超调与稳定时间",
    "param_patch_diff": "参数补丁前后差分",
    "penetration_after_set_state": "set_state 后穿透",
    "penetration_depth": "穿透深度",
    "pixel_hash_determinism": "像素 hash 确定性",
    "platform_compile_error": "平台编译错误",
    "plugin_recognition": "plugin 识别",
    "precision_mismatch": "精度不匹配",
    "qfrc_actuator_inv_dyn_residual": "qfrc_actuator 与逆动力学残差",
    "qpos_qvel_qacc_finite": "qpos/qvel/qacc 有限",
    "radius_of_gyration": "回转半径合理",
    "render_exception_class": "渲染异常类型",
    "reward_component_identity": "reward 分量恒等",
    "reward_timing_post_transition": "reward 时序在 transition 后",
    "runtime_abi_mismatch": "运行时 ABI 不匹配",
    "same_seed_backend_divergence": "同 seed 不同后端发散",
    "schema_violation_class": "schema 违反类型",
    "segfault": "段错误",
    "self_intersection": "自交",
    "sensor_acc_residual": "加速度传感器残差",
    "sensor_backend_residual": "传感器后端残差",
    "shape_mismatch": "形状不匹配",
    "small_perturb_response": "小扰动响应",
    "state_space_key_consistency": "状态字典 key 一致",
    "state_trajectory_residual": "状态轨迹残差",
    "target_vs_actual_steady_state": "目标 vs 实际稳态",
    "task_metric_regression": "任务指标回归",
    "training_abort": "训练中止",
    "unexpected_arm_motion": "机械臂异常运动",
    "unexpected_initial_contact": "初始非预期接触",
    "wrong_backend": "后端被错误选中",
    "xla_runtime_error": "XLA 运行时错误",
    "zero_target_drift": "零目标漂移",
}


# ---- mutator 中文说明 + 分组（用于种子页底部"算子可视化"） ----
MUTATOR_GROUPS: list[tuple[str, str, tuple[str, ...]]] = [
    ("struct",   "结构 STRUCT",   ("STRUCT_GROW_LINK", "STRUCT_SHRINK", "STRUCT_DUPLICATE_SUBTREE")),
    ("geom",     "几何 GEOM",     ("MUTATE_GEOM_SHAPE", "MUTATE_GEOM_SIZE", "MUTATE_INERTIAL",
                                    "MUTATE_FRICTION", "MUTATE_SOLREF_SOLIMP", "MUTATE_CONTACT_MARGIN")),
    ("joint",    "关节 JOINT",    ("MUTATE_JOINT_TYPE", "MUTATE_JOINT_LIMIT", "MUTATE_DAMPING_FRICTION")),
    ("actuator", "驱动器 / 约束 ACTUATOR", ("ACTUATOR_ADD", "ACTUATOR_MUTATE_RANGE", "ACTUATOR_DELETE",
                                            "EQUALITY_ADD", "TENDON_ADD")),
    ("solver",   "求解器 SOLVER", ("MUTATE_TIMESTEP", "MUTATE_INTEGRATOR", "MUTATE_SOLVER_ITER")),
    ("runtime",  "运行时状态 RUNTIME", ("SET_QPOS_RUNTIME", "SET_QVEL_RUNTIME", "SET_CTRL_RUNTIME")),
]

MUTATOR_TIPS: dict[str, str] = {
    "STRUCT_GROW_LINK":         "在现有 body 树末端追加一节子 body（含 joint+geom），扩大自由度。常用来探"
                                 "<b>触结构变化后的编译/求解稳定性</b>。",
    "STRUCT_SHRINK":            "随机摘除一个叶子或整棵子树，缩小模型规模——验证<b>结构裁剪</b>是否仍可"
                                 "编译并保持自洽。",
    "STRUCT_DUPLICATE_SUBTREE": "把现有子树原地复制 N 次，制造多实例 / 多智能体场景，验证<b>命名空间冲突、"
                                 "ID 复用</b>等问题。",
    "MUTATE_GEOM_SHAPE":        "把 geom 的 type 改成另一种基本几何（sphere/capsule/box/cylinder/ellipsoid），"
                                 "测试不同 type 的 size 维数与碰撞行为。",
    "MUTATE_GEOM_SIZE":         "按强度档位放缩 geom 的 size 数组，触发 tiny/large/near_zero 等极端尺寸。",
    "MUTATE_INERTIAL":          "改写 body 的 mass / 对角惯量，验证<b>三角不等式 / PSD / 负主轴</b>等校验。",
    "MUTATE_JOINT_TYPE":        "在 hinge / slide / ball / free 之间互换 joint type，触发 nq/nv 维数变化。",
    "MUTATE_JOINT_LIMIT":       "调整 joint range，覆盖 narrow / wide / inverted(lo&gt;hi) / 单点等用例。",
    "MUTATE_DAMPING_FRICTION":  "调整 joint 的 damping / armature / frictionloss，验证阻尼 / 干摩擦数值"
                                 "稳定性。",
    "MUTATE_FRICTION":          "调整 geom 的 friction 三元组（sliding, torsional, rolling），覆盖光滑 / "
                                 "高摩擦 / 负值非法等。",
    "MUTATE_SOLREF_SOLIMP":     "调整接触 solref / solimp 的求解参数，触发不同接触刚度与脉冲响应。",
    "MUTATE_CONTACT_MARGIN":    "调整 geom 的 contact margin / gap，影响接触检测裕量与穿透行为。",
    "ACTUATOR_ADD":             "在现有 joint 上追加 actuator（motor / position / velocity 等），验证"
                                 "<b>添加新驱动器</b>后控制维数与编译。",
    "ACTUATOR_MUTATE_RANGE":    "改写 actuator 的 ctrlrange / forcerange，覆盖 narrow / wide / inverted "
                                 "等区间。",
    "ACTUATOR_DELETE":          "随机删除一个 actuator，缩减 nu，验证<b>控制维数减少</b>是否仍合法。",
    "EQUALITY_ADD":             "添加 equality 约束（connect / weld / joint 等），引入额外刚性约束。",
    "TENDON_ADD":               "添加 tendon 约束（fixed / spatial），引入耦合关节运动的张力链。",
    "MUTATE_TIMESTEP":          "改写 option/timestep，覆盖 tiny / huge / nan_inf / negative 等离散步长。",
    "MUTATE_INTEGRATOR":        "切换积分器（Euler / RK4 / implicit / implicitfast），验证<b>不同积分器</b>"
                                 "下的稳定性差异。",
    "MUTATE_SOLVER_ITER":       "改写 option/iterations 与 tolerance，触发求解器收敛 / 非收敛分支。",
    "SET_QPOS_RUNTIME":         "<i>(runtime-only)</i> 在第 0 步前直接覆写 mjData.qpos——不改 XML，验证"
                                 "<b>初始位形</b>越界行为。",
    "SET_QVEL_RUNTIME":         "<i>(runtime-only)</i> 在第 0 步前覆写 mjData.qvel——验证<b>初始速度</b>"
                                 "极端值（巨大、NaN 等）。",
    "SET_CTRL_RUNTIME":         "<i>(runtime-only)</i> 每步注入 ctrl 信号——验证 actuator 在饱和 / NaN / "
                                 "震荡输入下的行为。",
}

INTENSITY_TIPS: dict[str, str] = {
    # geom size / 通用幅度
    "tiny":                "极小档：1e-3 量级。",
    "small":               "小档：~0.01 量级。",
    "medium":              "中等档：~0.1 量级（常作为基线）。",
    "large":               "大档：~0.5 量级。",
    "huge":                "超大档：&gt;=1.0，常会触发数值警告。",
    "near_zero":           "接近 0 但 &gt; mjMINVAL：编译通过、运行时风险高。",
    # inertial
    "near_zero_mass":      "质量接近 0：易触发 1/m 爆炸。",
    "small_mass":          "小质量。",
    "default_mass":        "缺省质量 1.0。",
    "huge_mass":           "巨大质量：1e3 量级。",
    "near_zero_inertia":   "惯量接近 0：易触发 1/I 爆炸。",
    "anisotropic":         "各向异性主轴（满足三角不等式）。",
    "negative_principal":  "<b>非法</b>：负主轴惯量，编译应当报错。",
    # joint
    "narrow":              "极窄区间：[-0.05, 0.05]。",
    "wide":                "大范围：约 [-π, π]。",
    "inverted":            "<b>非法</b>：lo &gt; hi。",
    "singular_zero_range": "<b>非法</b>：零宽度区间（lo == hi）。",
    # solver / numeric
    "negative":            "<b>非法</b>：负数（应被 schema 拒绝）。",
    "zero":                "0 值：边界条件。",
    "nan_inf":             "<b>非法</b>：NaN / ±Inf。",
    "extreme":             "极端档：超过常规 1~2 个数量级。",
    # integrator
    "euler":               "Euler 积分器（缺省）。",
    "rk4":                 "RK4 四阶积分器：较准但较慢。",
    "implicit":            "Implicit 积分器：刚性系统更稳定。",
    "implicitfast":        "Implicit-fast 变体：implicit 的近似版本。",
    # geom shape
    "sphere":              "球体：size 维数 = 1。",
    "capsule":             "胶囊：size 维数 = 2（半径、半长）。",
    "box":                 "盒子：size 维数 = 3（半边长）。",
    "cylinder":            "圆柱：size 维数 = 2。",
    "ellipsoid":           "椭球：size 维数 = 3。",
    # joint type
    "hinge":               "铰链关节：1 自由度旋转。",
    "slide":               "滑动关节：1 自由度平移。",
    "ball":                "球关节：3 自由度旋转（四元数表示）。",
    "free":                "自由关节：6 自由度（位置 + 四元数）。",
    # struct
    "leaf_only":           "仅删除叶子 body。",
    "random_subtree":      "随机选一棵子树整体删除。",
    "single":              "复制 1 份。",
    "chain_x3":            "复制 3 份（链式）。",
    "chain_x10":           "复制 10 份。",
    "simple_pendulum":     "追加一节单摆。",
    "compound_arm":        "追加一节复合臂。",
    "cluster_balls":       "追加一团球。",
    "_default":            "无强度档（mutator 不接受 intensity 参数）。",
}


def _mutator_group(mid: str) -> tuple[str, str]:
    for gid, gname, ids in MUTATOR_GROUPS:
        if mid in ids:
            return gid, gname
    return "other", "其他 OTHER"


def _seed_kind(name: str, source: str) -> str:
    """对 seed 名称做启发式归类，返回 1 句中文模型类型说明。"""
    n = name.lower()
    rules: list[tuple[tuple[str, ...], str]] = [
        (("humanoid", "h1", "g1", "atlas"),                "人形 humanoid 全身铰链模型"),
        (("walker", "hopper", "cheetah", "ant"),           "经典 RL locomotion 基准（DM Control / Gym）"),
        (("quadruped", "anymal", "spot", "go1", "go2", "b1", "aliengo", "laikago"),
                                                            "四足机器人 locomotion 模型"),
        (("cartpole", "pendulum", "acrobot", "lqr",
          "ball_in_cup", "swimmer", "reacher", "fish",
          "point_mass"),                                    "经典控制 / 低维 RL benchmark"),
        (("crazyflie", "skydio", "x2", "drone", "quadcopter", "aerial",
          "bitcraze"),                                      "无人机 / 旋翼 aerial 模型"),
        (("franka", "panda", "ur5", "ur10", "kuka", "iiwa",
          "lite6", "xarm", "sawyer", "jaco", "kinova",
          "widowx", "viperx", "aloha", "baxter"),           "工业 / 协作机械臂模型"),
        (("hand", "shadow", "mpl", "leap", "robotiq",
          "gripper", "finger"),                             "手 / 夹爪 / 多指操作模型"),
        (("kitchen", "stove", "cabinet", "faucet", "drawer",
          "microwave", "fridge", "sink", "basin"),          "厨房 / 家居室内操作场景"),
        (("myo",),                                          "MyoSuite 高保真生物力学肌骨模型"),
        (("dog", "fruitfly"),                               "DM Control 生物形态模型"),
        (("car", "vehicle"),                                "车辆 / 移动平台模型"),
        (("cards", "tendon", "net", "cloth", "rope",
          "duplo", "balloon", "cube"),                      "MuJoCo 引擎特性 demo（柔体 / 张拉 / 小物件）"),
        (("composed", "arena"),                             "compose_arena 拼装多实例场景"),
        (("scene", "world"),                                "场景级 XML（含地面 / 灯光 / 多物体）"),
    ]
    for keys, desc in rules:
        if any(k in n for k in keys):
            return desc
    return f"{source} 来源的 MuJoCo MJCF 模型"


def _seed_tip_html(m: dict) -> str:
    """构造 seed 行的浮层 HTML（标题 + 简介 + 统计 + 路径 + 上游链接）。"""
    def _esc(s: object) -> str:
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))
    name = m.get("name", "?")
    src = m.get("source") or "?"
    kind = _seed_kind(name, src)
    nq = m.get("nq"); nv = m.get("nv")
    nbody = m.get("nbody"); ngeom = m.get("ngeom"); nu = m.get("nu")
    parts = [
        f'<div class="tip-h"><b>{_esc(name)}</b></div>',
        f'<div class="tip-line">{_esc(kind)}</div>',
        '<div class="tip-sep"></div>',
        f'<div class="tip-line"><b>来源</b>：{_esc(src)}'
        f' &nbsp;<span class="tip-dim">({_esc(m.get("license") or "?")})</span></div>',
    ]
    if m.get("source_desc"):
        parts.append(f'<div class="tip-line tip-dim">{_esc(m["source_desc"])}</div>')
    parts.append(
        f'<div class="tip-line"><b>规模</b>：'
        f'nq={_esc(nq)} · nv={_esc(nv)} · nbody={_esc(nbody)}'
        f' · ngeom={_esc(ngeom)} · nu={_esc(nu)}</div>'
    )
    xml_path = m.get("xml_path") or f"seeds/curated/{name}/model.xml"
    parts.append(f'<div class="tip-line tip-dim">本地：<code>{_esc(xml_path)}</code></div>')
    if m.get("commit"):
        parts.append(f'<div class="tip-line tip-dim">commit：<code>{_esc(m["commit"])[:12]}</code></div>')
    if m.get("upstream_url"):
        parts.append(f'<div class="tip-line">🌐 <code>{_esc(m["upstream_url"])}</code></div>')
    return "".join(parts)


_CACHE: dict | None = None
_BUGS_CACHE: dict | None = None

# --------------------------------------------------------------------------
# 真实历史 bug：解析 _docs/mujoco_bug_reports_80cases.md（表格）+
# _docs/mujoco_bug_taxonomy_seed_operator_oracle.md（每个 Cn 的代表性 ID 列表）
# 让前端在每个分类卡片下渲染对应的真实 bug 小按钮。
# --------------------------------------------------------------------------
_BUG_ID_RE = __import__("re").compile(r"^[A-Z]{2,4}-\d{3}$")
_REPO_SHORT_RE = __import__("re").compile(r"`([^`]+)`")
_BACKTICK_ID_RE = __import__("re").compile(r"`([A-Z]{2,4}-\d{3})`")
_LINK_RE = __import__("re").compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
_CAT_HEADER_RE = __import__("re").compile(r"^###\s+(C\d{1,2})\.\s+(.+?)$")


def _load_bug_reports() -> dict:
    """Parse the 80-case bug-report markdown + taxonomy doc.

    Returns ``{"bugs": {id: {...}}, "cat_bugs": {cat_id: [id, ...]}}``.
    Cached. Both files are optional; missing files yield empty maps so the
    GUI degrades gracefully.
    """
    global _BUGS_CACHE
    if _BUGS_CACHE is not None:
        return _BUGS_CACHE
    docs = ROOT / "_docs"
    reports_md = docs / "mujoco_bug_reports_80cases.md"
    taxonomy_md = docs / "mujoco_bug_taxonomy_seed_operator_oracle.md"
    bugs: dict[str, dict] = {}
    if reports_md.is_file():
        for raw in reports_md.read_text(encoding="utf-8").splitlines():
            if not raw.startswith("|"):
                continue
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            if len(cells) < 10:
                continue
            bid = cells[0]
            if not _BUG_ID_RE.match(bid):
                continue
            repo_m = _REPO_SHORT_RE.search(cells[1] or "")
            repo = repo_m.group(1) if repo_m else cells[1]
            source_label = cells[2]                       # e.g. "Issue #251"
            title_zh = cells[3]                           # 中英混合的标题/现象
            link_m = _LINK_RE.search(cells[9] or "")
            url = link_m.group(1) if link_m else ""
            bugs[bid] = {
                "id": bid,
                "repo": repo,
                "source_label": source_label,
                "title": title_zh,
                "url": url,
            }
    cat_bugs: dict[str, list[str]] = {}
    if taxonomy_md.is_file():
        cur_cid: str | None = None
        for raw in taxonomy_md.read_text(encoding="utf-8").splitlines():
            mh = _CAT_HEADER_RE.match(raw)
            if mh:
                cur_cid = mh.group(1)
                cat_bugs.setdefault(cur_cid, [])
                continue
            if cur_cid and "代表性历史报告" in raw:
                ids = _BACKTICK_ID_RE.findall(raw)
                # de-dup, preserve order
                seen: set[str] = set()
                for x in ids:
                    if x not in seen:
                        seen.add(x); cat_bugs[cur_cid].append(x)
    _BUGS_CACHE = {"bugs": bugs, "cat_bugs": cat_bugs}
    return _BUGS_CACHE


def _load() -> dict:
    """Load (and cache) the seeds2 manifest + categories.yaml + per-cat memberships."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    cats_path = SEEDS2_DIR / "categories.yaml"
    manifest_path = SEEDS2_DIR / "MANIFEST.json"
    if not cats_path.is_file() or not manifest_path.is_file():
        print(f"[viz_gui2] missing seeds2 files; run "
              f"`python tools/build_seeds2.py` first", file=sys.stderr)
        cats: list = []
        manifest: list = []
    else:
        cats = yaml.safe_load(cats_path.read_text(encoding="utf-8")).get("categories", [])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in manifest}
    # Enrich every manifest entry with license / repo / upstream_url so the
    # frontend can render the cloud-link button + hover tooltip.
    for m in manifest:
        info = SOURCE_INFO.get(m.get("source") or "unknown", SOURCE_INFO["unknown"])
        m["license"] = info["license"]
        m["repo"] = info["repo"]
        m["source_desc"] = info["desc"]
        m["upstream_url"] = _upstream_url(info["repo"], m.get("commit"),
                                          m.get("src_rel_path"))
        m["tip_html"] = _seed_tip_html(m)
    grouped: dict[str, list[dict]] = {c["id"]: [] for c in cats}
    for m in manifest:
        for cid in m.get("categories", []):
            if cid in grouped:
                grouped[cid].append(m)
    # Sort each category by complexity heuristic, descending.
    def _complexity(m: dict) -> int:
        return ((m.get("nq") or 0) + (m.get("nv") or 0)
                + 2 * (m.get("nbody") or 0) + 2 * (m.get("ngeom") or 0)
                + 5 * (m.get("nu") or 0))
    for cid in grouped:
        grouped[cid].sort(key=lambda m: -_complexity(m))
    _CACHE = {"cats": cats, "manifest": manifest, "grouped": grouped,
              "by_name": by_name, "uncategorized": [m for m in manifest
                                                    if not m.get("categories")]}
    return _CACHE


def mutator_catalog() -> list[dict]:
    out = []
    for mid in MUTATOR_IDS:
        m = MUTATORS[mid]
        legal = [x for x in m.intensity_modes if x not in m.invalid_parseable_modes]
        invalid = list(m.invalid_parseable_modes)
        gid, gname = _mutator_group(mid)
        out.append({
            "id": mid, "runtime_only": bool(getattr(m, "runtime_only", False)),
            "legal": legal, "invalid": invalid,
            "group_id": gid, "group_name": gname,
            "tip": MUTATOR_TIPS.get(mid, ""),
        })
    return out


def spawn(cmd: list[str]) -> int:
    print(f"[spawn] {' '.join(cmd)}")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    proc = subprocess.Popen(cmd, cwd=str(ROOT),
                            creationflags=creationflags,
                            stdout=None, stderr=None)
    return proc.pid


# ---- 分层语料库 + Findings 加载 (③ ④) ---------------------------------

_CORPUS_CACHE: dict | None = None
_FINDINGS_CACHE: list | None = None


def _load_corpus_layers(force: bool = False) -> dict:
    """Read L1/L2/L3 manifests and return a dict of lists (JSON-serialisable)."""
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None and not force:
        return _CORPUS_CACHE
    try:
        from src.corpus import iter_manifest
    except ImportError:
        _CORPUS_CACHE = {"synthetic_scenes": [], "open_envs": [], "trajectory_seeds": []}
        return _CORPUS_CACHE

    def _read(rel: str, fields: list[str]) -> list[dict]:
        import dataclasses
        p = ROOT / rel
        if not p.exists():
            return []
        out = []
        for row in iter_manifest(p):
            if dataclasses.is_dataclass(row):
                d = dataclasses.asdict(row)
            elif hasattr(row, "__dict__"):
                d = vars(row)
            else:
                d = dict(row) if hasattr(row, "items") else {}
            out.append({f: d.get(f, "") for f in fields})
        return out

    _CORPUS_CACHE = {
        "synthetic_scenes": _read(
            "seeds/synthetic_scenes/manifest.jsonl",
            ["seed_id", "template_name", "actor_seed_id",
             "compile_status", "nq", "nbody", "nu"]),
        "open_envs": _read(
            "seeds/open_envs/manifest.jsonl",
            ["env_id", "adapter_name", "dependency_status",
             "runnable_status", "tags"]),
        "trajectory_seeds": _read(
            "seeds/trajectory_seeds/manifest.jsonl",
            ["seed_id", "parent_seed_id", "parent_layer",
             "replay_status", "action_kind", "horizon"]),
    }
    return _CORPUS_CACHE


def _load_findings(force: bool = False) -> list:
    """Scan findings/**/*.json and return list sorted by severity desc."""
    global _FINDINGS_CACHE
    if _FINDINGS_CACHE is not None and not force:
        return _FINDINGS_CACHE
    findings_root = ROOT / "findings"
    out: list[dict] = []
    if findings_root.exists():
        for fpath in sorted(findings_root.rglob("*.json")):
            try:
                d = json.loads(fpath.read_text(encoding="utf-8"))
                sev = sum(o.get("severity", 0) for o in d.get("oracle_signals", []))
                out.append({
                    "finding_id": d.get("finding_id", fpath.stem),
                    "seed_id":    d.get("seed_id", ""),
                    "layer":      d.get("layer", ""),
                    "signature":  d.get("signature", ""),
                    "severity":   round(sev, 2),
                    "oracle_signals": d.get("oracle_signals", []),
                    "_path": str(fpath.relative_to(ROOT)).replace("\\", "/"),
                })
            except Exception:
                continue
    out.sort(key=lambda x: -x["severity"])
    _FINDINGS_CACHE = out
    return out


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>seeds 2.0 · 按 bug 分类的可视化入口</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: ui-sans-serif, system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; padding: 18px 24px; max-width: 1500px; }
  h1 { margin: 0 0 4px 0; }
  .sub { color: #888; margin-bottom: 14px; font-size: 13px; }
  .card { border: 1px solid #8884; border-radius: 10px; padding: 12px 16px;
          margin-bottom: 14px; background: #ffffff08; }
  details { border: 1px solid #8883; border-radius: 8px; padding: 6px 12px;
            margin-bottom: 8px; background: #ffffff04; }
  details > summary { cursor: pointer; font-weight: 600; font-size: 15px;
                      list-style: none; padding: 6px 0; user-select: none; }
  details > summary::before { content: '▸'; display: inline-block; width: 1.2em;
                              transition: transform .12s; }
  details[open] > summary::before { content: '▾'; }
  details > summary:hover { color: #2563eb; }
  .cat-id    { display: inline-block; min-width: 36px; padding: 1px 7px;
               border-radius: 6px; background: #2563eb; color: white;
               font-family: ui-monospace, Consolas, monospace; font-size: 12px;
               text-align: center; margin-right: 8px; }
  .cat-meta  { color: #888; font-weight: 400; font-size: 12px; margin-left: 8px; }
  .cat-desc  { color: #555; font-size: 13px; margin: 6px 0 10px 44px;
               line-height: 1.45; }
  .ops-block { margin: 6px 0 4px 44px; padding: 8px 12px;
               border-left: 3px solid #2563eb55; border-radius: 4px;
               background: linear-gradient(90deg, #2563eb0d, transparent); }
  .ops-block.oracle-block { border-left-color: #16a34a88;
               background: linear-gradient(90deg, #16a34a10, transparent); }
  .ops-block.bugs-block   { border-left-color: #dc262688;
               background: linear-gradient(90deg, #dc262610, transparent); }
  .ops-label { display: inline-block; font-size: 11px; font-weight: 700;
               text-transform: uppercase; letter-spacing: .04em;
               color: #2563eb; margin-right: 6px; }
  .ops-block.oracle-block .ops-label { color: #16a34a; }
  .ops-block.bugs-block   .ops-label { color: #dc2626; }
  .ops-chip  { display: inline-block; font-family: ui-monospace, Consolas, monospace;
               font-size: 12px; padding: 2px 8px; margin: 2px 4px 2px 0;
               background: #ffffff10; border: 1px solid #8884; border-radius: 5px;
               color: inherit; }
  .ops-chip:hover { border-color: #2563eb; color: #2563eb; }
  .oracle-block .ops-chip:hover { border-color: #16a34a; color: #16a34a; }
  /* —— bugs chip：双拼小按钮（ID 段 + ↗ 跳转段） —— */
  .bug-chip { display: inline-flex; align-items: stretch; margin: 2px 4px 2px 0;
              border: 1px solid #dc262655; border-radius: 5px; overflow: hidden;
              font-family: ui-monospace, Consolas, monospace; font-size: 11px;
              line-height: 1.4; cursor: help; background: #dc262610; }
  .bug-chip .bug-id   { padding: 2px 7px; color: #b91c1c; font-weight: 600; }
  .bug-chip .bug-link { padding: 2px 6px; border-left: 1px solid #dc262633;
                        color: #b91c1c; text-decoration: none; cursor: pointer; }
  .bug-chip:hover { border-color: #dc2626; }
  .bug-chip:hover .bug-id, .bug-chip:hover .bug-link { color: #7f1d1d; }
  .bug-chip .bug-link:hover { background: #dc2626; color: white; }
  .bug-chip.dead .bug-link  { opacity: .35; pointer-events: none; }
  table.seeds { width: 100%; border-collapse: collapse; font-size: 12px;
                margin-top: 4px; }
  table.seeds th { text-align: left; font-weight: 500; color: #888;
                   border-bottom: 1px solid #8883; padding: 4px 6px;
                   cursor: pointer; user-select: none; }
  table.seeds td { padding: 4px 6px; border-bottom: 1px solid #8881;
                   vertical-align: middle; }
  table.seeds tr:hover td { background: #8881; }
  .seed-name { font-family: ui-monospace, Consolas, monospace; font-size: 11px; }
  .num { text-align: right; font-variant-numeric: tabular-nums;
         font-family: ui-monospace, Consolas, monospace; }
  button, select, input[type=text] {
    font-size: 12px; padding: 4px 8px; border-radius: 6px;
    border: 1px solid #8884; background: #ffffff10; color: inherit;
  }
  button { cursor: pointer; }
  button.primary { background: #2563eb; color: white; border-color: #2563eb; }
  button.primary:hover { background: #1d4ed8; }
  button.ghost { background: transparent; }
  .iconbtn { padding: 1px 6px; font-size: 11px; }
  .pill { display: inline-block; font-size: 10px; padding: 1px 6px;
          border-radius: 999px; margin-left: 4px; line-height: 1.5; }
  .pill.src    { background: #2563eb22; color: #2563eb; }
  .pill.tier   { background: #16a34a22; color: #16a34a; }
  .pill.cnt    { background: #f59e0b22; color: #b45309; font-variant-numeric: tabular-nums; }
  .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
             margin-bottom: 8px; }
  .toolbar input[type=text] { min-width: 220px; }
  #log { font-family: ui-monospace, Consolas, monospace; font-size: 12px;
         background: #0001; padding: 8px; border-radius: 6px; max-height: 160px;
         overflow: auto; white-space: pre-wrap; }
  mark { background: #fde047; color: inherit; padding: 0 1px; border-radius: 2px; }

  /* ---- 浮层弹窗（替代浏览器原生 title） ---- */
  #tip-pop { position: fixed; display: none; z-index: 9999;
             max-width: 420px; padding: 10px 12px;
             background: #1f2937; color: #f3f4f6;
             border: 1px solid #4b5563; border-radius: 8px;
             box-shadow: 0 8px 28px #0008;
             font-size: 12px; line-height: 1.55; pointer-events: none;
             font-family: ui-sans-serif, system-ui, "Segoe UI", "Microsoft YaHei", sans-serif; }
  #tip-pop b      { color: #93c5fd; font-weight: 600; }
  #tip-pop code   { background: #374151; padding: 1px 5px; border-radius: 3px;
                    font-family: ui-monospace, Consolas, monospace; font-size: 11px;
                    color: #fde68a; word-break: break-all; }
  #tip-pop .tip-h { font-size: 13px; margin-bottom: 4px; color: #e5e7eb; }
  #tip-pop .tip-line { margin: 2px 0; }
  #tip-pop .tip-dim  { color: #9ca3af; }
  #tip-pop .tip-sep  { height: 1px; background: #4b5563; margin: 6px 0; }
  [data-tip]      { cursor: help; }
  .ops-chip[data-tip] { cursor: help; }
  table.seeds th[data-tip]    { border-bottom-style: dashed; }
  table.seeds tr[data-tip] td:first-child { position: relative; }

  /* ---- mutator 可视化 v2 ---- */
  .mut-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
                 padding: 8px 10px; background: #2563eb0a; border-radius: 8px;
                 border: 1px solid #2563eb33; margin-bottom: 10px; }
  .mut-toolbar label { font-size: 12px; color: #555; }
  .mut-seed-card { display: inline-flex; align-items: center; gap: 8px;
                   padding: 4px 10px; border-radius: 6px;
                   background: #16a34a18; border: 1px solid #16a34a55;
                   font-family: ui-monospace, Consolas, monospace; font-size: 11px;
                   color: #15803d; min-height: 22px; }
  .mut-seed-card.empty { background: #8881; color: #888; border-color: #8884; }
  .mut-group { margin-bottom: 12px; border: 1px solid #8883; border-radius: 8px;
               background: #ffffff04; }
  .mut-group > .mut-group-h { padding: 8px 12px; font-weight: 600; font-size: 13px;
                              border-bottom: 1px solid #8882;
                              display: flex; align-items: center; gap: 8px;
                              background: linear-gradient(90deg, #2563eb12, transparent); }
  .mut-group .mut-group-h .pill { background: #2563eb22; color: #2563eb; }
  .mut-group-body { display: grid; gap: 10px; padding: 10px;
                    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
  .mut-card  { border: 1px solid #8884; border-radius: 8px; padding: 8px 10px;
               background: #ffffff08; display: flex; flex-direction: column; gap: 6px; }
  .mut-card-h { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .mut-id    { font-family: ui-monospace, Consolas, monospace; font-size: 12px;
               font-weight: 700; color: #2563eb; cursor: help;
               border-bottom: 1px dashed #2563eb88; }
  .mut-runtime-pill { background: #f59e0b22; color: #b45309;
                      font-size: 10px; padding: 1px 6px; border-radius: 999px; }
  .mut-row   { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }
  .mut-row-label { font-size: 10px; font-weight: 700; color: #888;
                   text-transform: uppercase; letter-spacing: .04em; min-width: 52px; }
  .intens-chip { font-family: ui-monospace, Consolas, monospace; font-size: 11px;
                 padding: 2px 8px; border-radius: 999px; cursor: pointer;
                 border: 1px solid #2563eb55; background: #2563eb14; color: #2563eb;
                 user-select: none; }
  .intens-chip:hover { background: #2563eb; color: white; }
  .intens-chip.invalid { border-color: #dc262655; background: #dc262614; color: #dc2626; }
  .intens-chip.invalid:hover { background: #dc2626; color: white; }
  .intens-chip.all   { border-style: dashed; }
  .mut-empty { color: #aaa; font-size: 11px; }
  /* ---- corpus tab panes ---- */
  .corpus-tab-btn.on { background: #2563eb; color: white; border-color: #2563eb; }
</style>
</head>
<body>

<h1>seeds 2.0 · 按 MuJoCo bug 分类的可视化入口</h1>
<div class="sub">
  分类规则定义于
  <code>seeds2/categories.yaml</code>；映射详见
  <code>_docs/mujoco_bug_taxonomy_seed_operator_oracle.md</code>。
  种子文件实际仍存放于 <code>seeds/curated/&lt;name&gt;/model.xml</code>，本页只做索引。
  默认全部分类折叠。
</div>

<div class="card">
  <div class="toolbar">
    <button id="expand-all" class="ghost">▾ 全部展开</button>
    <button id="collapse-all" class="ghost">▸ 全部折叠</button>
    <input type="text" id="filter" placeholder="过滤种子名 / 来源..."/>
    <span style="color:#888;font-size:12px">共 <b id="total-cnt">0</b> 类 · <b id="total-seeds">0</b> 种子</span>
    <button class="ghost" id="refresh">⟳ 刷新</button>
  </div>
</div>

<div id="cats"></div>

<div class="card">
  <details>
    <summary>未归类种子（uncategorized）<span class="pill cnt" id="uncat-cnt">0</span></summary>
    <div id="uncat-body" style="margin-top:8px"></div>
  </details>
</div>

<div class="card">
  <details open>
    <summary>② Operator 算子可视化（点击 intensity 即跑 Before/After 对比）</summary>
    <div class="mut-toolbar">
      <label>种子：</label>
      <input type="text" id="mut-seed" list="mut-seed-list" placeholder="可输入或从上方 M 按钮选" style="min-width:280px"/>
      <datalist id="mut-seed-list"></datalist>
      <button class="ghost iconbtn" id="mut-seed-clear" data-tip="清空已选种子">✕</button>
      <span id="mut-seed-info" class="mut-seed-card empty">尚未选择种子（可留空，使用 mutator 默认 seed）</span>
      <span style="flex:1"></span>
      <label data-tip="勾选后只跑 AFTER 单视图，跳过 BEFORE 对照">
        <input type="checkbox" id="mut-no-before"/> 跳过 BEFORE
      </label>
      <input type="text" id="mut-filter" placeholder="过滤 mutator id..." style="min-width:200px"/>
    </div>
    <div id="muts"></div>
  </details>
</div>

<div class="card">
  <details open id="sec-corpus">
  <summary>③ 分层语料库 <small style="font-weight:400;color:#888">(L1 合成场景 / L2 开源环境 / L3 轨迹种子)</small></summary>
  <div class="toolbar" style="margin-top:8px">
    <button class="ghost corpus-tab-btn" data-tab="l1">L1 合成场景 <span class="pill cnt" id="l1-count"></span></button>
    <button class="ghost corpus-tab-btn" data-tab="l2">L2 开源环境 <span class="pill cnt" id="l2-count"></span></button>
    <button class="ghost corpus-tab-btn" data-tab="l3">L3 轨迹种子 <span class="pill cnt" id="l3-count"></span></button>
    <button class="ghost" id="refresh-corpus">⟳ 刷新</button>
    <span style="margin-left:auto;display:flex;gap:6px">
      <button class="ghost" id="btn-validate">📋 验证语料库</button>
      <button class="ghost" id="btn-rand-fuzz">🎲 随机 Fuzz×20</button>
      <button class="ghost" id="btn-rule-fuzz">📐 规则 Fuzz×20</button>
    </span>
  </div>
  <div class="corpus-tab-pane" data-tab="l1">
    <div class="toolbar"><input type="text" id="l1-filter" placeholder="过滤 seed_id / template..." style="min-width:260px"/></div>
    <table class="seeds"><thead><tr>
      <th>seed_id</th><th>template</th><th>actor</th><th>compile</th>
      <th class="num">nq</th><th class="num">nbody</th><th class="num">nu</th><th>动作</th>
    </tr></thead><tbody id="l1-tbody"></tbody></table>
  </div>
  <div class="corpus-tab-pane" data-tab="l2" style="display:none">
    <div class="toolbar"><input type="text" id="l2-filter" placeholder="过滤 env_id / adapter..." style="min-width:260px"/></div>
    <table class="seeds"><thead><tr>
      <th>env_id</th><th>adapter</th><th>依赖</th><th>可运行</th><th>tags</th>
    </tr></thead><tbody id="l2-tbody"></tbody></table>
  </div>
  <div class="corpus-tab-pane" data-tab="l3" style="display:none">
    <div class="toolbar"><input type="text" id="l3-filter" placeholder="过滤 seed_id / parent..." style="min-width:260px"/></div>
    <table class="seeds"><thead><tr>
      <th>seed_id</th><th>parent</th><th>layer</th><th>replay</th><th>action_kind</th><th class="num">horizon</th>
    </tr></thead><tbody id="l3-tbody"></tbody></table>
  </div>
  </details>
</div>

<div class="card">
  <details open id="sec-findings">
  <summary>④ Findings <span class="pill cnt" id="findings-count"></span></summary>
  <div class="toolbar" style="margin-top:8px">
    <input type="text" id="findings-filter" placeholder="过滤 finding_id / seed_id / layer / signature..." style="min-width:300px"/>
    <button class="ghost" id="refresh-findings" style="margin-left:auto">⟳ 刷新</button>
  </div>
  <table class="seeds"><thead><tr>
    <th>finding_id</th><th>seed_id</th><th>layer</th><th>severity</th>
    <th>failed oracles</th><th>signature</th><th>path</th>
  </tr></thead><tbody id="findings-tbody"></tbody></table>
  <div style="font-size:11px;color:#888;margin-top:6px">findings/ 目录下所有 .json 文件自动扫描，按 severity 降序排列。</div>
  </details>
</div>

<div class="card">
  <h3 style="margin:0 0 6px 0;font-size:13px;color:#888">日志</h3>
  <div id="log">(等待操作)</div>
</div>

<script>
const STATE = __STATE__;          // {cats, grouped, manifest, uncategorized}
const MUTS = __MUTS__;
const TIPS = __TIPS__;            // {col, op, or, src}
let CORPUS = __CORPUS__;
let FINDINGS = __FINDINGS__;

// ---- 浮层弹窗：监听全局 mouseover/mouseout，定位跟随鼠标 ----
const $tip = document.createElement("div");
$tip.id = "tip-pop";
document.body.appendChild($tip);
function _placeTip(evt) {
  const pad = 14, w = $tip.offsetWidth, h = $tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth  - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  if (x < 8) x = 8;
  if (y < 8) y = 8;
  $tip.style.left = x + "px";
  $tip.style.top  = y + "px";
}
document.addEventListener("mouseover", e => {
  const el = e.target.closest("[data-tip]");
  if (!el) return;
  const html = el.getAttribute("data-tip");
  if (!html) return;
  $tip.innerHTML = html;
  $tip.style.display = "block";
  _placeTip(e);
});
document.addEventListener("mousemove", e => {
  if ($tip.style.display === "block") _placeTip(e);
});
document.addEventListener("mouseout", e => {
  const el = e.target.closest("[data-tip]");
  if (el) $tip.style.display = "none";
});
window.addEventListener("scroll", () => { $tip.style.display = "none"; }, true);
const $log = document.getElementById("log");
const $catsRoot = document.getElementById("cats");
const $filter = document.getElementById("filter");
const LS_KEY = "mjfuzz_viz2_state_v1";
const UI = Object.assign({open: {}, filter: ""}, (() => {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
  catch { return {}; }
})());
$filter.value = UI.filter || "";

function saveUI() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(UI)); } catch {}
}
function log(m) {
  const t = new Date().toLocaleTimeString();
  $log.textContent += `\\n[${t}] ${m}`;
  $log.scrollTop = $log.scrollHeight;
}
async function launch(payload) {
  log("launching " + JSON.stringify(payload));
  try {
    const r = await fetch("/launch", {method:"POST",
      headers:{"content-type":"application/json"}, body: JSON.stringify(payload)});
    const j = await r.json();
    if (j.ok) log("→ pid=" + j.pid + "  " + j.cmd.join(" "));
    else log("✗ " + j.error);
  } catch (e) { log("✗ network: " + e); }
}
function escHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
}
function highlight(text, q) {
  const t = escHtml(text);
  if (!q) return t;
  const i = t.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return t;
  return t.slice(0,i) + "<mark>" + t.slice(i,i+q.length) + "</mark>" + t.slice(i+q.length);
}

function buildSeedTable(seeds, q) {
  const filtered = q
    ? seeds.filter(s => s.name.toLowerCase().includes(q.toLowerCase())
                     || (s.source||"").toLowerCase().includes(q.toLowerCase()))
    : seeds;
  if (!filtered.length) {
    return '<div style="color:#888;font-size:12px;padding:6px 0">(过滤后无匹配)</div>';
  }
  const rows = filtered.map(s => {
    const xmlPath = s.xml_path || `seeds/curated/${s.name}/model.xml`;
    const tier = s.tier === "asset" ? '<span class="pill tier">asset</span>' : '';
    const seedTip = s.tip_html || (`<b>${escHtml(s.name)}</b>`);
    const srcTip = TIPS.src[s.source] || "(未知来源)";
    const linkBtn = s.upstream_url
      ? `<button class="ghost iconbtn" data-act="link" data-url="${escHtml(s.upstream_url)}" data-tip="在 GitHub 打开上游 XML：<br><code>${escHtml(s.upstream_url)}</code>">&#127760;</button>`
      : `<button class="ghost iconbtn" disabled data-tip="未知上游链接" style="opacity:.35">&#127760;</button>`;
    return `
      <tr data-tip="${escHtml(seedTip)}">
        <td><span class="seed-name">${highlight(s.name, q)}</span> ${tier}</td>
        <td><span class="pill src" data-tip="<b>${escHtml(s.source||"?")}</b> &nbsp;<span class='tip-dim'>(${escHtml(s.license||"?")})</span><div class='tip-sep'></div>${escHtml(srcTip)}">${escHtml(s.source||"?")}</span></td>
        <td class="num">${s.nq ?? "?"}</td>
        <td class="num">${s.nv ?? "?"}</td>
        <td class="num">${s.nbody ?? "?"}</td>
        <td class="num">${s.ngeom ?? "?"}</td>
        <td class="num">${s.nu ?? "?"}</td>
        <td>
          <button class="primary iconbtn" data-act="run"    data-name="${escHtml(s.name)}" data-tip="▶ 实时跑（mujoco.viewer.launch）">▶</button>
          <button class="ghost iconbtn"   data-act="static" data-name="${escHtml(s.name)}" data-tip="◻ 静态查看（不步进）">◻</button>
          <button class="ghost iconbtn"   data-act="copy"   data-path="${escHtml(xmlPath)}" data-tip="⧉ 复制 XML 路径：<br><code>${escHtml(xmlPath)}</code>">⧉</button>
          <button class="ghost iconbtn"   data-act="mut"    data-name="${escHtml(s.name)}" data-tip="M 送入下方 mutator 区作为种子">M</button>
          ${linkBtn}
        </td>
      </tr>`;
  }).join("");
  const COL = TIPS.col;
  return `<table class="seeds">
    <thead><tr>
      <th data-tip="${escHtml(COL.name||'')}">name</th>
      <th data-tip="${escHtml(COL.source||'')}">source</th>
      <th class="num" data-tip="${escHtml(COL.nq||'')}">nq</th>
      <th class="num" data-tip="${escHtml(COL.nv||'')}">nv</th>
      <th class="num" data-tip="${escHtml(COL.nbody||'')}">nbody</th>
      <th class="num" data-tip="${escHtml(COL.ngeom||'')}">ngeom</th>
      <th class="num" data-tip="${escHtml(COL.nu||'')}">nu</th>
      <th data-tip="${escHtml(COL.act||'')}">动作</th>
    </tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function bindRowActions(scope) {
  scope.querySelectorAll('button[data-act="run"]').forEach(b => {
    b.onclick = () => launch({tool:"seed", seed:b.dataset.name, static:false});
  });
  scope.querySelectorAll('button[data-act="static"]').forEach(b => {
    b.onclick = () => launch({tool:"seed", seed:b.dataset.name, static:true});
  });
  scope.querySelectorAll('button[data-act="copy"]').forEach(b => {
    b.onclick = async () => {
      try { await navigator.clipboard.writeText(b.dataset.path); log("已复制: " + b.dataset.path); }
      catch { log("复制失败: " + b.dataset.path); }
    };
  });
  scope.querySelectorAll('button[data-act="mut"]').forEach(b => {
    b.onclick = () => {
      document.getElementById("mut-seed").value = b.dataset.name;
      log("mutator 种子 → " + b.dataset.name);
      try { refreshSeedInfo(); } catch (e) {}
      const m = document.getElementById("muts");
      if (m) m.scrollIntoView({behavior:"smooth", block:"start"});
    };
  });  scope.querySelectorAll('button[data-act="link"]').forEach(b => {
    b.onclick = () => {
      window.open(b.dataset.url, '_blank', 'noopener');
      log('[link] ' + b.dataset.url);
    };
  });}

function renderCats() {
  const q = $filter.value.trim();
  $catsRoot.innerHTML = "";
  let totalSeeds = 0;
  STATE.cats.forEach(c => {
    const seeds = (STATE.grouped[c.id] || []);
    totalSeeds += seeds.length;
    const isOpen = !!UI.open[c.id];
    const det = document.createElement("details");
    if (isOpen) det.setAttribute("open", "");
    det.dataset.cid = c.id;
    const ops = (c.operators || []).map(o => {
      const tip = TIPS.op[o] || "(暂无说明)";
      return `<span class="ops-chip" data-tip="<b>${escHtml(o)}</b><div class='tip-sep'></div>${escHtml(tip)}">${escHtml(o)}</span>`;
    }).join("");
    const oracles = (c.oracles || []).map(o => {
      const tip = TIPS.or[o] || "(暂无说明)";
      return `<span class="ops-chip" data-tip="<b>${escHtml(o)}</b><div class='tip-sep'></div>${escHtml(tip)}">${escHtml(o)}</span>`;
    }).join("");
    const bugs = (c.bugs || []).map(b => {
      const tipParts = [
        `<div class='tip-h'><b>${escHtml(b.id)}</b>`
        + (b.repo ? ` <span class='tip-dim'>· ${escHtml(b.repo)}</span>` : "")
        + (b.source_label ? ` <span class='tip-dim'>· ${escHtml(b.source_label)}</span>` : "")
        + `</div>`,
        `<div class='tip-line'>${escHtml(b.title || '(无简介)')}</div>`,
      ];
      if (b.url) tipParts.push(`<div class='tip-sep'></div><div class='tip-line tip-dim'>🌐 <code>${escHtml(b.url)}</code></div>`);
      const tip = tipParts.join("");
      const linkPart = b.url
        ? `<a class="bug-link" href="${escHtml(b.url)}" target="_blank" rel="noopener" data-tip="跳转 GitHub：<br><code>${escHtml(b.url)}</code>" onclick="event.stopPropagation();">↗</a>`
        : `<span class="bug-link" title="无链接">↗</span>`;
      const cls = b.url ? "bug-chip" : "bug-chip dead";
      return `<span class="${cls}" data-tip="${escHtml(tip)}"><span class="bug-id">${escHtml(b.id)}</span>${linkPart}</span>`;
    }).join("");
    det.innerHTML = `
      <summary>
        <span class="cat-id">${c.id}</span>${escHtml(c.name_zh || "")}
        <span class="cat-meta">· ${seeds.length} seeds</span>
      </summary>
      <div class="cat-desc">${escHtml(c.description || "")}</div>
      <div class="ops-block"><span class="ops-label">Operators</span>${ops || '<span class="ops-chip">(none)</span>'}</div>
      <div class="ops-block oracle-block"><span class="ops-label">Oracles</span>${oracles || '<span class="ops-chip">(none)</span>'}</div>
      <div class="ops-block bugs-block"><span class="ops-label">Bugs (real, ${(c.bugs||[]).length})</span>${bugs || '<span class="ops-chip">(none)</span>'}</div>
      <div class="seed-table"></div>
    `;
    det.querySelector(".seed-table").innerHTML = buildSeedTable(seeds, q);
    bindRowActions(det);
    det.addEventListener("toggle", () => {
      UI.open[c.id] = det.open; saveUI();
    });
    $catsRoot.appendChild(det);
  });
  document.getElementById("total-cnt").textContent = STATE.cats.length;
  document.getElementById("total-seeds").textContent = STATE.manifest.length;

  // Uncategorized.
  const $uncatBody = document.getElementById("uncat-body");
  $uncatBody.innerHTML = buildSeedTable(STATE.uncategorized, q);
  document.getElementById("uncat-cnt").textContent = STATE.uncategorized.length;
  bindRowActions($uncatBody);
}

document.getElementById("expand-all").onclick = () => {
  document.querySelectorAll("#cats details").forEach(d => {
    d.setAttribute("open", ""); UI.open[d.dataset.cid] = true;
  });
  saveUI();
};
document.getElementById("collapse-all").onclick = () => {
  document.querySelectorAll("#cats details").forEach(d => {
    d.removeAttribute("open"); UI.open[d.dataset.cid] = false;
  });
  saveUI();
};
$filter.oninput = () => { UI.filter = $filter.value; saveUI(); renderCats(); };
document.getElementById("refresh").onclick = async () => {
  const r = await fetch("/api/state?refresh=1"); const j = await r.json();
  STATE.cats = j.state.cats; STATE.grouped = j.state.grouped;
  STATE.manifest = j.state.manifest; STATE.uncategorized = j.state.uncategorized;
  renderCats(); rebuildMutDatalist(); log("reloaded.");
};

// ---- mutator section ----
const $muts = document.getElementById("muts");
const $mutFilter = document.getElementById("mut-filter");
const $mutSeed = document.getElementById("mut-seed");
const $mutSeedInfo = document.getElementById("mut-seed-info");

function rebuildMutDatalist() {
  const list = document.getElementById("mut-seed-list");
  list.innerHTML = "";
  STATE.manifest.slice().sort((a,b) => a.name.localeCompare(b.name)).forEach(s => {
    const o = document.createElement("option");
    o.value = s.name; o.label = `[${s.source}] nq=${s.nq ?? '?'} nbody=${s.nbody ?? '?'}`;
    list.appendChild(o);
  });
}

function _byName(name) {
  if (!name) return null;
  for (const m of STATE.manifest) if (m.name === name) return m;
  return null;
}
function refreshSeedInfo() {
  const v = $mutSeed.value.trim();
  const m = _byName(v);
  if (!v) {
    $mutSeedInfo.className = "mut-seed-card empty";
    $mutSeedInfo.textContent = "尚未选择种子（可留空，使用 mutator 默认 seed）";
  } else if (m) {
    $mutSeedInfo.className = "mut-seed-card";
    $mutSeedInfo.innerHTML = `✓ <b>${escHtml(m.name)}</b>
      <span class='tip-dim'>[${escHtml(m.source||'?')}]</span>
      nq=${m.nq ?? '?'} · nv=${m.nv ?? '?'} · nbody=${m.nbody ?? '?'} · ngeom=${m.ngeom ?? '?'} · nu=${m.nu ?? '?'}`;
    $mutSeedInfo.setAttribute("data-tip", m.tip_html || ("<b>" + escHtml(m.name) + "</b>"));
  } else {
    $mutSeedInfo.className = "mut-seed-card empty";
    $mutSeedInfo.textContent = "⚠ 该名字未在 manifest 中找到";
    $mutSeedInfo.removeAttribute("data-tip");
  }
}

function _runMut(mid, intensity) {
  launch({
    tool: "mutator", mutator: mid,
    seed: $mutSeed.value.trim() || null,
    intensity: intensity || null,
    no_before: document.getElementById("mut-no-before").checked,
  });
}

function _intensTip(mode) {
  const t = TIPS.intens[mode];
  return `<b>${escHtml(mode)}</b><div class='tip-sep'></div>${t || "(暂无说明)"}<div class='tip-sep'></div><span class='tip-dim'>点击 = 立即跑此强度的 Before/After</span>`;
}

function renderMuts() {
  const filt = $mutFilter.value.trim().toLowerCase();
  $muts.innerHTML = "";
  // 按组归并
  const groups = {};
  MUTS.forEach(m => {
    if (filt && !m.id.toLowerCase().includes(filt)) return;
    const gid = m.group_id || "other";
    (groups[gid] = groups[gid] || {name: m.group_name || "其他", items: []}).items.push(m);
  });
  // 保持声明顺序
  const order = ["struct","geom","joint","actuator","solver","runtime","other"];
  let total = 0;
  order.forEach(gid => {
    const g = groups[gid]; if (!g) return;
    total += g.items.length;
    const sec = document.createElement("div");
    sec.className = "mut-group";
    const head = document.createElement("div");
    head.className = "mut-group-h";
    head.innerHTML = `<span>${escHtml(g.name)}</span>
      <span class="pill cnt">${g.items.length} mutator</span>`;
    sec.appendChild(head);
    const body = document.createElement("div");
    body.className = "mut-group-body";
    g.items.forEach(m => body.appendChild(_renderMutCard(m)));
    sec.appendChild(body);
    $muts.appendChild(sec);
  });
  if (!total) {
    $muts.innerHTML = '<div class="mut-empty" style="padding:12px">(过滤后无匹配 mutator)</div>';
  }
}

function _renderMutCard(m) {
  const card = document.createElement("div");
  card.className = "mut-card";
  const tip = TIPS.mut[m.id] || "(暂无说明)";
  const runPill = m.runtime_only
    ? `<span class="mut-runtime-pill" data-tip="runtime-only：不改 XML，只在运行期注入指令（qpos/qvel/ctrl）">runtime-only</span>` : "";
  // 头部
  const head = document.createElement("div");
  head.className = "mut-card-h";
  head.innerHTML = `<span class="mut-id" data-tip="<b>${escHtml(m.id)}</b><div class='tip-sep'></div>${tip}">${escHtml(m.id)}</span>${runPill}`;
  card.appendChild(head);
  // 合法 intensity 行
  const legalRow = document.createElement("div");
  legalRow.className = "mut-row";
  legalRow.innerHTML = `<span class="mut-row-label" data-tip="点击任一 intensity 立即跑 Before/After">合法</span>`;
  // "全部" 快捷
  const all = document.createElement("span");
  all.className = "intens-chip all";
  all.textContent = "▶ 全部合法";
  all.setAttribute("data-tip", "<b>不指定 intensity</b><div class='tip-sep'></div>由 mutator 自行随机选一个合法档位<div class='tip-sep'></div><span class='tip-dim'>点击 = 跑 Before/After</span>");
  all.onclick = () => _runMut(m.id, null);
  legalRow.appendChild(all);
  if (!m.legal.length) {
    const e = document.createElement("span"); e.className = "mut-empty"; e.textContent = "(无 intensity 档位)";
    legalRow.appendChild(e);
  }
  m.legal.forEach(mode => {
    const c = document.createElement("span");
    c.className = "intens-chip";
    c.textContent = mode;
    c.setAttribute("data-tip", _intensTip(mode));
    c.onclick = () => _runMut(m.id, mode);
    legalRow.appendChild(c);
  });
  card.appendChild(legalRow);
  // invalid 行（如有）
  if (m.invalid && m.invalid.length) {
    const invRow = document.createElement("div");
    invRow.className = "mut-row";
    invRow.innerHTML = `<span class="mut-row-label" data-tip="故意非法的 intensity（CI 跳过；点击仍可手动跑以观察 MuJoCo 的拒绝/警告）">非法</span>`;
    m.invalid.forEach(mode => {
      const c = document.createElement("span");
      c.className = "intens-chip invalid";
      c.textContent = mode;
      c.setAttribute("data-tip", _intensTip(mode));
      c.onclick = () => _runMut(m.id, mode);
      invRow.appendChild(c);
    });
    card.appendChild(invRow);
  }
  return card;
}

$mutFilter.oninput = renderMuts;
$mutSeed.oninput = refreshSeedInfo;
$mutSeed.onchange = refreshSeedInfo;
document.getElementById("mut-seed-clear").onclick = () => { $mutSeed.value = ""; refreshSeedInfo(); };

window.addEventListener("error", e => {
  log("[JS ERR] " + (e.error && e.error.stack || e.message || e));
});
try { renderCats(); } catch (e) { log("[renderCats] " + (e.stack||e)); }
try { rebuildMutDatalist(); } catch (e) { log("[rebuildMutDatalist] " + (e.stack||e)); }
try { renderMuts(); } catch (e) { log("[renderMuts] " + (e.stack||e)); }
try { refreshSeedInfo(); } catch (e) { log("[refreshSeedInfo] " + (e.stack||e)); }

// ─────────────────────────────────────────────
// ③ 分层语料库
// ─────────────────────────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function statusColor(v) {
  if (!v) return "#888";
  if (v === "ok" || v === "runnable") return "#16a34a";
  if (v === "partial" || v === "unknown") return "#d97706";
  if (v === "missing" || v === "failed" || v === "broken") return "#dc2626";
  return "#888";
}
function statusPill(v) {
  const c = statusColor(v);
  return `<span style="display:inline-block;padding:1px 7px;border-radius:999px;font-size:10px;background:${c}22;color:${c}">${escHtml(v||"?")}</span>`;
}

const $tabBtns = document.querySelectorAll(".corpus-tab-btn");
const $tabPanes = document.querySelectorAll(".corpus-tab-pane");
function switchCorpusTab(name) {
  $tabBtns.forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  $tabPanes.forEach(p => p.style.display = p.dataset.tab === name ? "" : "none");
}
$tabBtns.forEach(b => b.onclick = () => switchCorpusTab(b.dataset.tab));

function renderL1() {
  const tbody = document.getElementById("l1-tbody");
  if (!tbody) return;
  const filt = document.getElementById("l1-filter").value.trim().toLowerCase();
  const items = CORPUS.synthetic_scenes.filter(s =>
    !filt || s.seed_id.toLowerCase().includes(filt) || s.template_name.toLowerCase().includes(filt));
  tbody.innerHTML = "";
  for (const s of items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="seed-name">${escHtml(s.seed_id)}</td>
      <td>${escHtml(s.template_name)}</td>
      <td class="seed-name" style="font-size:10px">${escHtml(s.actor_seed_id)}</td>
      <td>${statusPill(s.compile_status)}</td>
      <td class="num">${s.nq}</td><td class="num">${s.nbody}</td><td class="num">${s.nu}</td>
      <td>
        <button class="ghost iconbtn" title="visualize_corpus --show">▶</button>
        <button class="ghost iconbtn" title="--rollout">↻</button>
      </td>`;
    tr.querySelectorAll("button")[0].onclick = () => launch({tool:"corpus_show", seed_id:s.seed_id});
    tr.querySelectorAll("button")[1].onclick = () => launch({tool:"corpus_show", seed_id:s.seed_id, rollout:true});
    tbody.appendChild(tr);
  }
  document.getElementById("l1-count").textContent = items.length + "/" + CORPUS.synthetic_scenes.length;
}

function renderL2() {
  const tbody = document.getElementById("l2-tbody");
  if (!tbody) return;
  const filt = document.getElementById("l2-filter").value.trim().toLowerCase();
  const items = CORPUS.open_envs.filter(s =>
    !filt || s.env_id.toLowerCase().includes(filt) || s.adapter_name.toLowerCase().includes(filt));
  tbody.innerHTML = "";
  for (const s of items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="seed-name" style="font-size:11px">${escHtml(s.env_id)}</td>
      <td><span class="pill src">${escHtml(s.adapter_name)}</span></td>
      <td>${statusPill(s.dependency_status)}</td>
      <td>${statusPill(s.runnable_status)}</td>
      <td style="font-size:11px;color:#888">${escHtml((s.tags||[]).join(", "))}</td>`;
    tbody.appendChild(tr);
  }
  document.getElementById("l2-count").textContent = items.length + "/" + CORPUS.open_envs.length;
}

function renderL3() {
  const tbody = document.getElementById("l3-tbody");
  if (!tbody) return;
  const filt = document.getElementById("l3-filter").value.trim().toLowerCase();
  const items = CORPUS.trajectory_seeds.filter(s =>
    !filt || s.seed_id.toLowerCase().includes(filt) || s.parent_seed_id.toLowerCase().includes(filt));
  tbody.innerHTML = "";
  for (const s of items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="seed-name" style="font-size:11px">${escHtml(s.seed_id)}</td>
      <td style="font-size:11px;color:#888">${escHtml(s.parent_seed_id)}</td>
      <td><span class="pill src" style="font-size:10px">${escHtml(s.parent_layer)}</span></td>
      <td>${statusPill(s.replay_status)}</td>
      <td>${escHtml(s.action_kind)}</td>
      <td class="num">${s.horizon}</td>`;
    tbody.appendChild(tr);
  }
  document.getElementById("l3-count").textContent = items.length + "/" + CORPUS.trajectory_seeds.length;
}

document.getElementById("l1-filter").oninput = renderL1;
document.getElementById("l2-filter").oninput = renderL2;
document.getElementById("l3-filter").oninput = renderL3;

document.getElementById("refresh-corpus").onclick = async () => {
  const r = await fetch("/api/corpus?refresh=1"); const j = await r.json();
  CORPUS = j.corpus; renderL1(); renderL2(); renderL3();
  log("corpus 已刷新  L1=" + CORPUS.synthetic_scenes.length +
      " L2=" + CORPUS.open_envs.length + " L3=" + CORPUS.trajectory_seeds.length);
};
document.getElementById("btn-validate").onclick = () =>
  launch({tool:"run_tool", script:"validate_layered_corpus.py"});
document.getElementById("btn-rand-fuzz").onclick = () =>
  launch({tool:"run_tool", script:"run_random_fuzz.py", args:["--budget","20"]});
document.getElementById("btn-rule-fuzz").onclick = () =>
  launch({tool:"run_tool", script:"run_rule_fuzz.py", args:["--budget","20"]});

switchCorpusTab("l1");
try { renderL1(); renderL2(); renderL3(); } catch(e) { log("[corpus] " + (e.stack||e)); }

// ─────────────────────────────────────────────
// ④ Findings
// ─────────────────────────────────────────────
function renderFindings() {
  const tbody = document.getElementById("findings-tbody");
  if (!tbody) return;
  const filt = document.getElementById("findings-filter").value.trim().toLowerCase();
  const items = FINDINGS.filter(f =>
    !filt || f.finding_id.toLowerCase().includes(filt) ||
    f.seed_id.toLowerCase().includes(filt) || f.layer.toLowerCase().includes(filt) ||
    f.signature.toLowerCase().includes(filt));
  document.getElementById("findings-count").textContent = items.length + "/" + FINDINGS.length;
  tbody.innerHTML = "";
  for (const f of items) {
    const tr = document.createElement("tr");
    const oracles = (f.oracle_signals || []).filter(o => o.failed).map(o => o.name).join(", ");
    const w = Math.min(60, f.severity * 10);
    const sevBar = f.severity > 0
      ? `<div style="display:inline-block;width:${w}px;height:7px;background:#dc2626;border-radius:4px;vertical-align:middle;margin-right:4px"></div>`
      : "";
    tr.innerHTML = `
      <td class="seed-name" style="font-size:11px">${escHtml(f.finding_id)}</td>
      <td style="font-size:11px">${escHtml(f.seed_id)}</td>
      <td><span class="pill src">${escHtml(f.layer)}</span></td>
      <td>${sevBar}<span class="num" style="font-size:11px">${f.severity.toFixed(1)}</span></td>
      <td style="font-size:11px;color:#dc2626">${escHtml(oracles||"—")}</td>
      <td style="font-size:10px;color:#888">${escHtml(f.signature)}</td>
      <td style="font-size:10px;color:#888">${escHtml(f._path)}</td>`;
    tbody.appendChild(tr);
  }
}

document.getElementById("findings-filter").oninput = renderFindings;
document.getElementById("refresh-findings").onclick = async () => {
  const r = await fetch("/api/findings?refresh=1"); const j = await r.json();
  FINDINGS = j.findings; renderFindings();
  log("findings 已刷新 (" + FINDINGS.length + " 条)");
};
try { renderFindings(); } catch(e) { log("[findings] " + (e.stack||e)); }
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _state_payload(self, refresh: bool) -> dict:
        global _CACHE, _BUGS_CACHE
        if refresh:
            _CACHE = None
            _BUGS_CACHE = None
        s = _load()
        bugdb = _load_bug_reports()
        bugs_map = bugdb["bugs"]; cat_bugs = bugdb["cat_bugs"]

        def _cat_with_bugs(c: dict) -> dict:
            out = {k: v for k, v in c.items() if k in
                   ("id", "name_zh", "description", "doc_anchor",
                    "operators", "oracles", "sources")}
            ids = cat_bugs.get(c["id"], [])
            out["bugs"] = [
                bugs_map[i] for i in ids if i in bugs_map
            ]
            # also surface ids that have no body (rare) so user still sees them
            for i in ids:
                if i not in bugs_map:
                    out["bugs"].append({"id": i, "title": "(详情未在 80cases 文档中)",
                                        "url": "", "repo": "", "source_label": ""})
            return out

        return {
            "cats": [_cat_with_bugs(c) for c in s["cats"]],
            "grouped": s["grouped"],
            "manifest": s["manifest"],
            "uncategorized": s["uncategorized"],
        }

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            state = self._state_payload(refresh=False)
            tips = {
                "col": COLUMN_TIPS,
                "op":  OPERATOR_TIPS,
                "or":  ORACLE_TIPS,
                "src": {k: v["desc"] for k, v in SOURCE_INFO.items()},
                "mut": MUTATOR_TIPS,
                "intens": INTENSITY_TIPS,
            }
            html = (INDEX_HTML
                    .replace("__STATE__", json.dumps(state, ensure_ascii=False))
                    .replace("__MUTS__", json.dumps(mutator_catalog()))
                    .replace("__TIPS__", json.dumps(tips, ensure_ascii=False))
                    .replace("__CORPUS__", json.dumps(_load_corpus_layers(), ensure_ascii=False))
                    .replace("__FINDINGS__", json.dumps(_load_findings(), ensure_ascii=False)))
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/state":
            return self._json(200, {"state": self._state_payload(refresh="refresh" in qs)})
        if parsed.path == "/api/corpus":
            return self._json(200, {"corpus": _load_corpus_layers(force="refresh" in qs)})
        if parsed.path == "/api/findings":
            return self._json(200, {"findings": _load_findings(force="refresh" in qs)})
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/launch":
            self.send_error(404); return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json: {e}"})
        tool = payload.get("tool")
        try:
            if tool == "seed":
                seed = payload.get("seed")
                if not seed:
                    raise ValueError("seed required")
                cmd = [PYTHON, str(TOOLS_DIR / "visualize_seed.py"), seed]
                if payload.get("static"):
                    cmd.append("--static")
            elif tool == "mutator":
                mid = payload.get("mutator")
                if not mid:
                    raise ValueError("mutator required")
                cmd = [PYTHON, str(TOOLS_DIR / "visualize_mutator.py"), mid]
                if payload.get("seed"):
                    cmd += ["--seed", payload["seed"]]
                if payload.get("intensity"):
                    cmd += ["--intensity", payload["intensity"]]
                if payload.get("no_before"):
                    cmd.append("--no-before")
            elif tool == "corpus_show":
                seed_id = payload.get("seed_id")
                if not seed_id:
                    raise ValueError("seed_id required")
                cmd = [PYTHON, str(TOOLS_DIR / "visualize_corpus.py"), "--show", seed_id]
                if payload.get("rollout"):
                    cmd.append("--rollout")
            elif tool == "run_tool":
                script = payload.get("script")
                if not script:
                    raise ValueError("script required")
                # Security: only allow scripts that live in tools/ directory.
                target = (TOOLS_DIR / script).resolve()
                if not str(target).startswith(str(TOOLS_DIR.resolve())):
                    raise ValueError(f"script must be inside tools/: {script!r}")
                cmd = [PYTHON, str(target)] + list(payload.get("args", []))
            else:
                raise ValueError(f"unknown tool: {tool!r}")
            return self._json(200, {"ok": True, "pid": spawn(cmd), "cmd": cmd})
        except Exception as e:
            return self._json(400, {"ok": False, "error": str(e)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[viz_gui2] serving {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz_gui2] bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
