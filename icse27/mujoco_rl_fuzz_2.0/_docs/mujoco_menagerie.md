https://github.com/google-deepmind/mujoco_menagerie


你更有机会在 `mujoco_menagerie` 里找到的是 **真实模型 bug / model-quality bug / simulation asset bug**，而不是 MuJoCo 引擎本体 bug。也就是说，你的工具如果设计得好，完全可能发现“某个官方/社区维护的 MJCF 模型存在惯量、碰撞体、执行器、关节范围、控制维度、稳定性、组合装配方面的问题”。这类 bug 在 Menagerie 的 issue、PR 和 commit history 里已经反复出现过。

## 1. 为什么 Menagerie 适合作为真实 bug case

Menagerie 的定位是 Google DeepMind curated 的高质量 MuJoCo 模型集合，但它的 README 里也直接说：MuJoCo 很强大、建模选项很多，因此很容易创建“不按预期行为”的 bad models；Menagerie 的目标是提供开箱即用的 well-designed models。更关键的是，README 又承认模型质量是 ongoing effort，当前很多模型“不一定已经足够好”，并计划用 A+ / A / B / C 的质量分级来表达不同模型的系统辨识、真实性和稳定性水平。([GitHub][1])

这意味着它不是一个“bug-free benchmark”，而是一个**权威但仍在演进的模型库**。对你的研究来说，这反而很重要：如果连这种 curated repo 里都持续出现模型质量问题，那么自动化 fuzzing / regression / oracle checking 的价值是成立的。

## 2. issue 里已经出现的真实 bug 类型

从 bug-labeled issues 看，Menagerie 里不是只有“加载失败”这种浅层问题，而是有不少与你的 MuJoCo fuzzing 很贴合的真实问题。

第一类是 **启动后物理不稳定 / exploding / jittering**。例如 Booster T1 的 issue #237 直接报告默认位置控制增益 `kp=75, kv=5` 会导致启动后很快不稳定，表现为 CoM drift、振荡增长、几秒内摔倒；报告者还做了 gain sweep，把默认参数与更稳定的 `kp=200~250, kv=6~8` 做了定量对比。([GitHub][2]) 这类问题非常适合你的 fuzzing：加载模型、reset 到 keyframe、无外部扰动 step 5 秒，然后检测 qpos/qvel/CoM/contact force 是否发散。

第二类是 **惯量/质量参数错误或不合理**。Stanford TidyBot base 的 issue #231 指出 60kg base 的 `diaginertia=0.001` 极不现实，会让机器人像点质量一样轻微受力就高速旋转，并建议改成约 `2.708 kg*m^2` 量级。([GitHub][3]) Kinova Gen3 的 issue #232 也报告机械臂 bare env 中刚 spawn 就出现 bouncing / violent oscillations，尤其 joint 3 和 joint 7，用户提出的修复包括降低 actuator gains、增加 armature/damping、添加相邻 body 的 contact exclusions。([GitHub][4]) 这说明“惯量—控制器—接触”之间的组合错误是真实存在的。

第三类是 **碰撞体配置错误**。UR10e 的 issue #192 报告 shoulder link 的 collision size 可能错误，会和地面产生异常碰撞，导致大 contact force 和仿真不稳定；报告者给出最小复现：注释 keyframe 后直接运行 simulate，即可看到 shoulder_link 与 ground 发生异常接触，修改 size 可解决。([GitHub][5]) 这类问题也很适合 fuzzing：你可以在 reset 后检查“非预期初始接触”“巨大 contact force”“visual-collision mismatch”“自碰撞/地面穿插”。

第四类是 **执行器定义、控制范围、控制维度错误**。KUKA iiwa14 的 issue #214 质疑 actuator `ctrlrange` 与 joint range 完全一致，可能与 position control 语义不匹配。([GitHub][6]) Shadow DEX-EE 的 issue #209 指出该模型用了 `mujoco.pid` plugin 和 `actdim=2`，导致 Brax vmap 后输出 action dimension = 12，但模型期望 24，从而触发维度 mismatch。([GitHub][7]) 这类问题可以通过“读取模型维度 + runtime action injection + backend compatibility oracle”检测。

第五类是 **执行器 gear / 力矩方向错误**。Skydio X2 的 issue #219 指出四个 thruster 的 gear 配置里，`thrust1` 和 `thrust4` 同向旋转但位于同一侧边；用户认为同向旋转的 rotor 应位于对角线，才能产生正确 yaw torque 而不引入其他轴的旋转。([GitHub][8]) 这说明 Menagerie 里会出现“模型能加载，但动力学语义错”的问题。普通 XML schema checker 找不到这种 bug，但 domain-aware oracle 有机会找。

## 3. PR 和 commit history 证明：这些不是空泛抱怨，确实有 bug 被修了

PR #252 在 2026-04-16 被合并，标题就是 “Fix base collision alignment and pad mass in robotiq_2f85_v4”。这个 PR 修了两个具体问题：一是 Robotiq 2F-85 V4 的 visual base geom 有 `pos` 和 `quat`，但 collision base geom 没有这些属性，导致 collision mesh 与 visual mesh 错位；二是 pad geom mass 原来是 `1e-6`，每个 pad body 约 0.002g，而原始 Robotiq 2F-85 模型是每个 pad body 3.5g，所以 PR 将每个 pad geom 改为 1.75g。维护者还评论说这看起来像 oversight。([GitHub][9])

PR #240 在 2026-03-17 被合并，修的是 Robotiq 2F-85 V4 的 inertial frame misalignment。PR 描述说，该 v4 模型把 body frame 移到了 joint location，mesh 通过 geom `pos/quat` 偏移，但 `<inertial>` 元素仍然从旧模型照搬，导致 linkage bodies 的惯性中心相对实际几何偏移 25–44 mm。这个 PR 把惯性位置和主轴修正到与编译后的 mesh centroid / principal axes 对齐。([GitHub][10])

PR #203 在 2025-09-14 被合并，修复 LEAP Hand 左手 `thumb_cmc` joint control range，报告者说明修复后 joint 行为正确。([GitHub][11]) PR #160 在 2025-04-10 被合并，修复 Talos 模型左腿第一 link 的惯性 `pos` 和 `quat` 与右腿相同导致的非真实不对称问题，审查者也确认“你是对的”。([GitHub][12])

commit history 也支持这一点：近期主分支里有 `Fix base collision geom alignment in robotiq_2f85_v4`、`Fix pad geom masses in robotiq_2f85_v4`、`Fix misaligned inertia frames in robotiq_2f85_v4`、`Recompute PD gains with armature and per-class natural frequencies` 等 commit。([GitHub][13]) 这说明 Menagerie 的真实维护历史里，模型 bug 修复集中在 **collision alignment、mass/inertia、PD gain、armature、joint range、symmetry** 这些方向，而不是只有文档或 README 小修。

## 4. 对你的 fuzzing 来说，最可能找到的 bug 是哪几类

我认为你最有机会找到的是真实 bug，按成功概率排序如下。

最高概率是 **stability bug**。例如模型 reset 后，在无控制、默认控制、随机小控制、keyframe reset、不同 timestep / solver setting 下出现 qpos/qvel 爆炸、CoM 大漂移、接触力异常、机器人立刻摔倒。这类 bug 已经在 Booster T1、Kinova Gen3、UR10e 等 issue 里出现过。你的 fuzzing 不需要特别复杂的物理知识，只要设计一组可复现的 rollout oracle 即可。

第二高概率是 **inertia / mass plausibility bug**。比如质量极小、惯量极小、惯性中心和 mesh centroid 偏离过大、左右对称 body 的惯性参数不对称、mass/inertia 与几何尺寸明显不匹配。Robotiq V4、TidyBot、Talos 的修复和 issue 都说明这类问题真实存在。你可以做静态 checker + 动态验证：先检查参数异常，再通过 rollout 看是否导致异常旋转、抖动或接触力。

第三高概率是 **visual-collision mismatch / initial contact bug**。PR #252 和 issue #192 都证明 collision geom 错位或尺寸错误会造成真实问题。你的 checker 可以比较 visual mesh 与 collision mesh 的 AABB、body-frame pose、初始状态下的地面穿插、自碰撞、contact force spike。

第四高概率是 **actuator / control semantic bug**。包括 ctrlrange 不合理、gear 符号错误、actdim 和 action dimension 不一致、position actuator 参数导致不稳定。这类 bug 更有研究价值，因为它不是单纯 XML 语法错误，而是“模型可加载但控制语义不对”。

第五类是 **model composition bug**。例如 UR5e/UR10e + Robotiq 2F-85 gripper 组合后出现手指异常、gripper glitching、末端执行器控制影响机械臂其他关节等问题。Issue #156 报告 UR5e + Robotiq 2F-85 抓 cube 时一侧手指倾斜；早期 issue #40 也报告 UR 系列机械臂和 2F85 gripper 组合后出现异常 abrupt movement。([GitHub][14]) 这类 bug 对你尤其重要，因为它接近 GzFuzz 的“系统级组合输入”，不是单个孤立 XML。

## 5. 你的系统应该怎么把它包装成有说服力的 case

不要写成：

> We fuzz MuJoCo Menagerie and find bugs in MuJoCo.

这会被审稿人打回来，因为很多问题不是 MuJoCo engine 本体，而是模型资产质量问题。

更准确的论文定位应该是：

> We target model-level and task-level faults in MuJoCo-based robotic simulation assets. Using MuJoCo Menagerie as a curated but evolving corpus, our fuzzer detects physically invalid, dynamically unstable, or semantically inconsistent MJCF models and model compositions.

也就是说，你是在做 **robotic simulation model fuzzing / model-quality testing / simulation asset regression testing**。这个方向是成立的，而且 Menagerie 的 issue/PR 历史可以作为 motivation evidence。

## 6. 我建议你把 oracle 设计成 5 组

第一组：**Load & compile oracle**。检查 XML 是否能被 `MjModel.from_xml_path` 编译，是否存在 missing mesh、invalid range、plugin/action dimension 不匹配、backend incompatibility。

第二组：**Static physical plausibility oracle**。检查 mass、diaginertia、inertial pos/quat、左右对称 body、mesh AABB、collision/visual alignment、joint range、actuator ctrlrange、gear sign。这个能覆盖 Talos、Robotiq inertia、TidyBot inertia、Skydio gear 这类问题。

第三组：**Initial-state contact oracle**。reset 后不施加控制，运行 `mj_forward` 和短步长 `mj_step`，检测不该有的 ground contact、自碰撞、巨大 contact force、penetration depth。这能覆盖 UR10e shoulder collision、Robotiq collision alignment。

第四组：**Stability rollout oracle**。对每个 model/scene 跑 1–5 秒，包含 no-control、zero-control、home-control、small-random-control、actuator sweep，检测 NaN/Inf、qpos/qvel 爆炸、CoM drift、falling、contact force spike、energy spike。这个能覆盖 Booster T1、Kinova Gen3 等问题。

第五组：**Composition oracle**。把 arm + gripper、hand + object、mobile base + object 组合起来，检查 attach site、actuator indexing、gripper closure symmetry、抓取接触稳定性、控制一个 actuator 是否异常影响其他 joint。这个最接近你想模仿 GzFuzz 的系统级 value。

## 7. 最终判断

我对这个 case 的判断是：

**可以做，而且比你随机抓 GitHub seed 更有说服力。** Menagerie 是权威 curated corpus，但 issue/PR/commit history 明确显示它仍然存在可被自动化检测的真实模型 bug，包括不稳定、惯量错误、碰撞体错位、控制范围错误、gear 方向错误、action dimension mismatch、组合模型异常等。

但你的 claim 要收敛：

不能主张“我能大量发现 MuJoCo 引擎 bug”；可以主张“我能在权威 MuJoCo 模型库中自动发现真实模型质量 bug，并用 issue/PR 历史验证这些 bug 类型真实存在”。

最适合你的论文卖点是：

> 现有 robotics simulation fuzzing 更多关注 simulator API 或 runtime crash；但实际 MuJoCo 用户大量依赖 MJCF/URDF-converted model assets，而这些 assets 即使来自 curated repositories，也会存在物理参数、碰撞几何、执行器语义、组合装配和 rollout 稳定性问题。我们提出面向 MuJoCo model assets 的 domain-aware fuzzing，并用 Menagerie 的 historical bug-fix commits 与新发现 case 证明其有效性。

这比单纯说“我 fuzz MuJoCo”更稳。

[1]: https://github.com/google-deepmind/mujoco_menagerie "GitHub - google-deepmind/mujoco_menagerie: A collection of high-quality models for the MuJoCo physics engine, curated by Google DeepMind. · GitHub"
[2]: https://github.com/google-deepmind/mujoco_menagerie/issues/237 "Booster T1 default position gains (kp=75, kv=5) cause immediate instability at startup · Issue #237 · google-deepmind/mujoco_menagerie · GitHub"
[3]: https://github.com/google-deepmind/mujoco_menagerie/issues/231 "Stanford tidybot base xml file base link inertia values · Issue #231 · google-deepmind/mujoco_menagerie · GitHub"
[4]: https://github.com/google-deepmind/mujoco_menagerie/issues/232 "Kinova Gen3 physical instability · Issue #232 · google-deepmind/mujoco_menagerie · GitHub"
[5]: https://github.com/google-deepmind/mujoco_menagerie/issues/192 "Collision size parameter may be wrong in ur10e · Issue #192 · google-deepmind/mujoco_menagerie · GitHub"
[6]: https://github.com/google-deepmind/mujoco_menagerie/issues/214 "actuator_ctrlrange for kuka iiwa14 · Issue #214 · google-deepmind/mujoco_menagerie · GitHub"
[7]: https://github.com/google-deepmind/mujoco_menagerie/issues/209 "Shadow_dexee action dimension mismatch due to activation dynamics · Issue #209 · google-deepmind/mujoco_menagerie · GitHub"
[8]: https://github.com/google-deepmind/mujoco_menagerie/issues/219 "Skydio X2 wrong gear for yaw · Issue #219 · google-deepmind/mujoco_menagerie · GitHub"
[9]: https://github.com/google-deepmind/mujoco_menagerie/pull/252 "Fix base collision alignment and pad mass in robotiq_2f85_v4 by mzamoramora-nvidia · Pull Request #252 · google-deepmind/mujoco_menagerie · GitHub"
[10]: https://github.com/google-deepmind/mujoco_menagerie/pull/240 "Fix misaligned inertia frames in robotiq_2f85_v4 by adenzler-nvidia · Pull Request #240 · google-deepmind/mujoco_menagerie · GitHub"
[11]: https://github.com/google-deepmind/mujoco_menagerie/pull/203 "Fix LEAP Hand's `left_hand.xml` thumb_cmc range by jonzamora · Pull Request #203 · google-deepmind/mujoco_menagerie · GitHub"
[12]: https://github.com/google-deepmind/mujoco_menagerie/pull/160 "Fixing asymmetry in Talos model by lorenzo96-cmd · Pull Request #160 · google-deepmind/mujoco_menagerie · GitHub"
[13]: https://github.com/google-deepmind/mujoco_menagerie/commits/main/ "Commits · google-deepmind/mujoco_menagerie · GitHub"
[14]: https://github.com/google-deepmind/mujoco_menagerie/issues/156 "Gripper of Robotiq 2f85 with UR5e do not work as supposed to · Issue #156 · google-deepmind/mujoco_menagerie · GitHub"
