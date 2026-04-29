# AI 自动场景构建与 Unity-GPT 协同渲染/仿真层技术方案

## 1. 问题背景与研究转向

此前基于 LLM 的 VR 开源项目 testing 主要面对的是“给定一个已有 Unity/VR 项目，让智能体自动探索并发现 bug”。这个方向的核心瓶颈不是 LLM 能否生成动作，而是测试对象本身往往缺乏足够密集、可复现、可验证的真实 bug；如果项目场景简单、交互链条短、oracle 薄弱，即使覆盖率提升，也很难产出有价值的缺陷发现证据。

因此新的切入点应从“被动测试已有场景”转向“主动构建可测试、可验证、可扩展的 Unity/XR 场景”。研究目标不是单纯让 LLM 写 C# 脚本，也不是只做 2D 图像到 3D 模型的外观生成，而是建立一个结构化工作流：由 LLM 负责需求理解、场景草案、约束分解、资产选择与修复决策；由 Unity 工具层负责确定性地落地 `.unity` 场景、Prefab、Collider、XR 组件、状态机与 PlayMode 验证。

该方向更适合作为 AI+SE/VR testing 的研究切入点，因为它可以直接解决原先 testing 中“缺少高质量测试对象、缺少复杂交互环境、缺少可执行 oracle”的问题。最终系统输出不应只是一个可看的 3D 场景，而应是一个可加载、可运行、可交互、可测试、可复现的 Unity/XR 场景工程。

## 2. 核心目标

系统输入可以是自然语言需求、2D 草图、参考图片、关卡描述或已有 Unity 项目的局部资产；系统输出应包含以下内容：

1. Unity 场景文件：包括房间结构、物体布局、灯光、材质、碰撞体、导航区域、XR 交互组件。
2. 场景语义图：记录每个对象的类别、功能、空间关系、交互属性与约束来源。
3. 交互逻辑配置：包括可抓取物、插槽、按钮、门禁、触发器、状态变量与任务链。
4. 测试计划：包括从初始状态到目标状态的动作序列、关键中间状态、预期 oracle。
5. 验证报告：包括构建是否成功、场景是否可加载、物体是否重叠、交互是否可触发、脚本是否报错、任务是否可完成。

一句话概括：系统不是“生成漂亮场景”，而是“生成可被 VR testing 使用的结构化交互场景”。

## 3. 总体架构

推荐将系统拆成六层，而不是让一个大模型直接控制 Unity 编辑器。

```text
User Requirement / Sketch / Reference Image
        │
        ▼
[1] Requirement Parser
    需求理解、任务目标、场景类型、对象清单、交互目标
        │
        ▼
[2] Scene Graph Planner
    生成 Semantic Scene Graph + Layout Constraints + Interaction Graph
        │
        ▼
[3] Asset Retrieval & Normalization
    检索本地/公开资产，统一格式、尺度、pivot、collider、license 信息
        │
        ▼
[4] Unity Scene Synthesizer
    通过 Editor Script / Batch Mode 确定性创建场景、Prefab、组件与脚本绑定
        │
        ▼
[5] Simulation & Interaction Validator
    编译、加载、碰撞检测、可达性检查、PlayMode 自动测试、日志采集
        │
        ▼
[6] Repair Loop
    将错误映射为结构化 patch，再回到 Planner / Synthesizer 层修复
```

关键设计原则是：LLM 只生成结构化计划，不直接手写大量 Unity 场景文件；Unity 端只执行合法 API，不让自然语言直接污染工程；Validator 必须把错误转换为可修复的结构化信息，而不是只返回一段日志。

## 4. 中间表示：不要让 LLM 直接生成 `.unity` 文件

Unity 的 `.unity` YAML 文件并不适合作为 LLM 的直接输出目标。更合理的方式是让模型生成一个中间表示，然后由确定性程序把它编译到 Unity 场景中。

### 4.1 SceneIntent

SceneIntent 表示用户到底想生成什么场景、用于什么测试。

```json
{
  "scene_type": "escape_room",
  "theme": "small laboratory",
  "target_platform": "Unity XR",
  "primary_goal": "player finds a keycard, unlocks a cabinet, retrieves a battery, powers the exit door",
  "expected_interactions": ["grab", "socket", "button", "door_unlock", "state_transition"],
  "difficulty": "medium",
  "view_mode": "first_person_vr"
}
```

### 4.2 SemanticSceneGraph

SemanticSceneGraph 描述对象、空间关系与语义角色。

```json
{
  "rooms": [
    {
      "id": "lab_room",
      "size": [8.0, 3.0, 6.0],
      "objects": ["desk_01", "cabinet_01", "exit_door", "power_panel"]
    }
  ],
  "objects": [
    {
      "id": "keycard_01",
      "category": "keycard",
      "affordances": ["grabbable"],
      "initial_location": "desk_01",
      "required_by": ["cabinet_unlock"]
    },
    {
      "id": "cabinet_01",
      "category": "storage",
      "affordances": ["openable", "lockable"],
      "contains": ["battery_01"]
    }
  ],
  "spatial_relations": [
    {"subject": "desk_01", "relation": "near", "object": "wall_north"},
    {"subject": "power_panel", "relation": "beside", "object": "exit_door"}
  ]
}
```

### 4.3 LayoutConstraint

LayoutConstraint 负责把语义需求变成几何约束，避免“看起来对但实际无法玩”。

```json
{
  "constraints": [
    {"type": "no_overlap", "objects": ["all_static_objects"]},
    {"type": "reachable", "object": "keycard_01", "actor": "vr_player", "max_distance": 1.2},
    {"type": "path_exists", "from": "spawn_point", "to": "exit_door", "min_width": 0.8},
    {"type": "support", "object": "keycard_01", "surface": "desk_01"},
    {"type": "visibility", "object": "exit_door", "from": "spawn_point"}
  ]
}
```

### 4.4 InteractionGraph

InteractionGraph 记录交互任务链，后续可以直接转成 Unity 组件与测试计划。

```json
{
  "states": [
    "has_keycard",
    "cabinet_unlocked",
    "has_battery",
    "power_enabled",
    "exit_unlocked"
  ],
  "transitions": [
    {
      "id": "pickup_keycard",
      "trigger": "grab(keycard_01)",
      "precondition": [],
      "effect": ["has_keycard"]
    },
    {
      "id": "unlock_cabinet",
      "trigger": "use(keycard_01, cabinet_01)",
      "precondition": ["has_keycard"],
      "effect": ["cabinet_unlocked"]
    },
    {
      "id": "power_exit",
      "trigger": "socket(battery_01, power_panel)",
      "precondition": ["has_battery"],
      "effect": ["power_enabled", "exit_unlocked"]
    }
  ]
}
```

## 5. 工作流一：从需求到场景草案

第一阶段不应直接生成 Unity 脚本，而应先把自然语言需求转成结构化设计。

输入示例：

```text
生成一个小型 VR 实验室逃脱场景。玩家从入口出生，需要在桌子上找到门禁卡，用门禁卡打开柜子，取出电池，把电池插入墙上的电源面板，最终打开出口门。场景需要包含可抓取、插槽、按钮和门禁状态变化。
```

LLM Planner 的输出应包括：

1. 房间数量与房间尺寸。
2. 静态对象清单：墙、地板、门、桌、柜、控制台、灯光。
3. 动态对象清单：门禁卡、电池、按钮、可抓取工具。
4. 交互链：grab → unlock → retrieve → socket → open。
5. 几何约束：物体不能重叠、路径可达、关键物体可见、抓取距离合理。
6. 测试目标：从 spawn 到 exit_unlock 的最短可执行动作序列。

这一阶段的产物是 JSON，不是 Unity 文件。

## 6. 工作流二：资产检索与标准化

场景质量不应完全依赖模型生成 3D mesh。更稳定的方案是“LLM 规划 + 资产库检索 + Unity 程序化拼装”。资产来源可以包括本地资产库、开源 Unity package、公开 3D 模型站点、团队自建 Prefab 仓库。为了避免后续不可控，建议先构建一个小型可控资产库。

每个资产需要维护如下元数据：

```json
{
  "asset_id": "lab_cabinet_A",
  "semantic_tags": ["cabinet", "storage", "laboratory", "openable"],
  "file_format": "fbx",
  "unity_prefab_path": "Assets/GeneratedAssets/Lab/CabinetA.prefab",
  "license": "free_for_research",
  "scale_hint": [1.2, 2.0, 0.5],
  "pivot_type": "bottom_center",
  "has_collider": true,
  "has_texture": true,
  "xr_ready": false,
  "recommended_components": ["BoxCollider", "XRSocketInteractable"]
}
```

资产标准化至少要处理五类问题：

1. 尺度不一致：同样是 door，有的模型高度 2m，有的高度 200 单位。
2. pivot 不一致：有的 pivot 在中心，有的在底部，有的偏离模型。
3. collider 缺失或过细：复杂 mesh collider 会拖慢运行，也可能导致抓取异常。
4. 材质缺失：导入后出现粉色材质或贴图丢失。
5. 语义标签缺失：模型文件名不能可靠表示对象功能，需要人工或模型辅助标注。

建议一开始不要追求无限模型库，而是先整理 50–100 个高质量、可复用、授权清晰的资产，包括门、桌、柜、按钮、钥匙、卡片、电池、工具、厨房物体、实验室物体、墙体模块、地板模块、灯光模块。这样更容易做出稳定 demo，也更容易做实验对比。

## 7. 工作流三：约束布局生成

场景生成的关键不在于“放很多物体”，而在于“放得符合游戏逻辑和 VR 交互约束”。推荐采用二阶段布局：先生成粗粒度 layout，再做局部修正。

### 7.1 粗布局

粗布局负责确定房间、门、主要家具、玩家出生点和目标点。可以用规则、搜索或轻量约束求解实现。

基本规则：

1. 房间坐标系固定为 Unity 坐标：X/Z 为水平面，Y 为高度。
2. 玩家出生点不能离墙太近，前方至少保留 1.5m 可视空间。
3. 主要路径宽度不小于 0.8m，VR 场景建议留到 1.0m 以上。
4. 关键交互物体高度应位于玩家可抓取范围，通常桌面物体高度 0.7–1.2m。
5. 门、按钮、电源面板等不能被家具遮挡。

### 7.2 局部布局修正

局部修正主要解决碰撞、遮挡、不可达和交互半径问题。

可使用如下检查：

```text
for each object:
    check bounding box overlap
    check support surface if object is portable
    check distance to required interaction partner
    check visibility from important waypoints
    check reachable distance for XR hand/controller
    check collider validity
```

如果发现错误，不应让 LLM 重新生成整个场景，而应生成局部 patch：

```json
{
  "patch_type": "move_object",
  "target": "keycard_01",
  "reason": "object is outside reachable distance from player path",
  "old_position": [2.4, 1.6, -1.0],
  "new_position": [1.8, 0.9, -0.8]
}
```

这样可以保证修复过程可控，并便于统计 repair 成功率。

## 8. 工作流四：Unity 场景合成层

Unity 场景合成层建议用 C# Editor Script 实现，不建议让 LLM 操作 GUI。外部 Python/Node 控制器将结构化 JSON 传给 Unity，Unity 通过 Batch Mode 或 Editor API 创建场景。

### 8.1 Unity 侧核心命令

建议实现一组确定性命令：

```json
{
  "commands": [
    {"op": "create_scene", "scene_name": "GeneratedLabEscape"},
    {"op": "create_room", "id": "lab_room", "size": [8, 3, 6]},
    {"op": "instantiate_prefab", "id": "desk_01", "prefab": "LabDeskA", "position": [1.5, 0, -1.0], "rotation": [0, 90, 0]},
    {"op": "add_component", "target": "keycard_01", "component": "XRGrabInteractable"},
    {"op": "add_component", "target": "power_panel", "component": "XRSocketInteractable"},
    {"op": "bind_state", "target": "exit_door", "state": "exit_unlocked"},
    {"op": "save_scene"}
  ]
}
```

### 8.2 Unity 侧组件模板

交互逻辑不要完全由模型即兴写 C#，而应沉淀为可组合模板。

推荐最小组件库：

1. `GeneratedSceneStateManager`：统一维护布尔状态与任务进度。
2. `XRGrabbableBinder`：把对象绑定为可抓取物。
3. `XRSocketRule`：定义某个 socket 只接受指定对象。
4. `DoorStateController`：根据状态打开/关闭门。
5. `ButtonTriggerController`：按钮触发状态变化。
6. `InventoryFlagController`：抓取某对象后设置状态。
7. `ValidationProbe`：运行时采集对象状态、触发日志和异常。

示例状态绑定：

```csharp
public class DoorStateController : MonoBehaviour
{
    public string requiredState = "exit_unlocked";
    public Transform doorTransform;
    public Vector3 closedPosition;
    public Vector3 openPosition;

    void Update()
    {
        if (GeneratedSceneStateManager.Instance.HasState(requiredState))
        {
            doorTransform.localPosition = Vector3.Lerp(
                doorTransform.localPosition,
                openPosition,
                Time.deltaTime * 3f
            );
        }
    }
}
```

这类模板的好处是：模型只负责配置 `requiredState`、对象绑定和任务链，不需要每次从零写一份不可控脚本。

## 9. 工作流五：渲染/仿真验证层

验证层是该研究区别于普通生成式 3D 场景工具的关键。没有验证层，系统只是“生成场景”；有验证层，系统才是“生成可测试场景”。

### 9.1 静态验证

静态验证在 PlayMode 前完成，主要检查工程结构与场景几何。

检查项包括：

1. 所有引用的 Prefab 是否存在。
2. 所有 GameObject ID 是否唯一。
3. 必要组件是否存在，如 Collider、Rigidbody、XRGrabInteractable、XRSocketInteractable。
4. 所有可抓取物是否有合理质量、碰撞体和抓取点。
5. 所有关键物体是否在场景边界内。
6. 所有支持关系是否成立，例如 keycard 在桌面上而不是悬空。
7. 玩家出生点是否可用，是否与物体重叠。
8. 任务路径是否存在，关键物体是否可达。

### 9.2 动态验证

动态验证在 PlayMode 中运行，检查运行时错误与交互链。

建议实现自动 PlayMode test：

```text
1. Load generated scene.
2. Wait N frames for physics settling.
3. Move virtual player to key waypoints.
4. Simulate grab(keycard_01).
5. Simulate use(keycard_01, cabinet_01).
6. Simulate grab(battery_01).
7. Simulate socket(battery_01, power_panel).
8. Check state exit_unlocked == true.
9. Check Unity Console has no Error/Exception.
10. Export validation_report.json.
```

动态验证至少输出：

```json
{
  "build_success": true,
  "scene_load_success": true,
  "console_errors": [],
  "physics_anomalies": [],
  "interaction_trace": [
    "pickup_keycard:success",
    "unlock_cabinet:success",
    "pickup_battery:success",
    "power_exit:success"
  ],
  "final_states": {
    "has_keycard": true,
    "cabinet_unlocked": true,
    "has_battery": true,
    "exit_unlocked": true
  },
  "task_success": true
}
```

## 10. Repair Loop：把错误变成可修复 patch

错误修复不能只把日志重新丢给 LLM。需要先把错误归类，再生成最小修复。

常见错误类型：

| 错误类型 | 例子 | 修复动作 |
|---|---|---|
| AssetMissing | Prefab 路径不存在 | 换资产、重新检索、使用 fallback prefab |
| ComponentMissing | 可抓取物没有 Collider | 添加 BoxCollider 或 Convex MeshCollider |
| ScaleInvalid | 门高 20m 或 keycard 过大 | 按类别 scale hint 归一化 |
| ObjectOverlap | 桌子和柜子重叠 | 局部移动或重新求解布局 |
| PathBlocked | 出生点到出口不可达 | 清除障碍、移动家具、扩大通道 |
| InteractionUnreachable | 手够不到按钮 | 调整高度或距离 |
| StateDeadlock | 任务链缺失前置状态 | 修复 InteractionGraph |
| ScriptCompileError | 自动生成脚本编译失败 | 回退到模板组件，不让 LLM 临时写复杂脚本 |

推荐修复流程：

```text
ValidationReport
    → ErrorClassifier
    → MinimalPatchGenerator
    → UnitySceneSynthesizer.apply_patch()
    → Validator rerun
    → stop if pass or max_retry reached
```

## 11. 与 Coplay / 纯 LLM Agent 的区别

该方案不应定位为“又一个让 LLM 控制 Unity 的聊天插件”。真正的研究点是：

1. 将自然语言需求转化为可验证的场景语义图。
2. 将 2D/文本草案转成带约束的 3D 布局，而不是只生成单个模型。
3. 将资产检索、组件绑定、状态机、测试计划统一到同一个可执行场景生成流程中。
4. 用 Unity 静态检查和 PlayMode 验证形成闭环，而不是只看主观效果。
5. 输出可以直接服务 VR testing 的场景与 oracle。

因此，即使已有工具能帮助写脚本或摆放简单对象，本方案仍然有研究空间，因为它关注的是“生成结果是否能被系统化测试和验证”。

## 12. 最小可行原型

建议第一阶段不要做大而全系统，而是做一个窄但完整的闭环。

### 12.1 场景范围

先支持三类场景：

1. 小型实验室逃脱场景。
2. 厨房物品操作场景。
3. 仓库/办公室任务场景。

### 12.2 交互范围

先支持 6 种基础交互：

1. 抓取 `grab`。
2. 放置 `socket/place`。
3. 按钮 `press`。
4. 门状态变化 `open/unlock`。
5. 容器开合 `open_container`。
6. 条件触发 `state_transition`。

### 12.3 资产规模

第一版资产库建议控制在 50–100 个可用 Prefab，而不是抓取大量质量不稳定的外部模型。每个资产都应有语义标签、尺度、碰撞体和推荐组件。

### 12.4 验证目标

最小 demo 必须满足：

1. 能从文本需求生成一个 Unity 场景。
2. 场景能在 Unity 中无错误加载。
3. 关键物体位置合理且不重叠。
4. 至少一条任务链能在 PlayMode test 中自动执行通过。
5. 失败时能产生结构化修复 patch。

## 13. 推荐模块划分

工程目录可以按如下方式组织：

```text
AI-UnitySceneBuilder/
  orchestrator/
    planner.py
    asset_retriever.py
    layout_solver.py
    repair_loop.py
    schemas/
      scene_intent.schema.json
      scene_graph.schema.json
      interaction_graph.schema.json
      validation_report.schema.json
  unity_project/
    Assets/
      GeneratedScenes/
      GeneratedAssets/
      SceneBuilder/
        Editor/
          SceneBuildCommandRunner.cs
          AssetImportNormalizer.cs
          StaticSceneValidator.cs
        Runtime/
          GeneratedSceneStateManager.cs
          XRSocketRule.cs
          DoorStateController.cs
          ValidationProbe.cs
      Tests/
        PlayMode/
          GeneratedScenePlayModeTest.cs
  benchmarks/
    tasks/
    assets_manifest.json
    results/
```

## 14. 可直接给开发助手的实现指令

下面这段可以直接交给 Copilot/Claude Code/其他开发助手作为第一阶段实现 prompt。

```text
请实现一个 Unity XR 自动场景生成原型，不要让 LLM 直接编辑 .unity YAML 文件，而是采用 JSON 中间表示 + Unity Editor Script 编译场景的方式。

目标：输入 scene_graph.json、layout_constraints.json、interaction_graph.json，自动在 Unity 中生成一个可加载的 XR 交互场景，并输出 validation_report.json。

第一阶段只需要支持：
1. 创建一个矩形房间，包括地板、墙、出口门和玩家出生点；
2. 从本地 Prefab manifest 中实例化桌子、柜子、门禁卡、电池、电源面板；
3. 自动添加 Collider、Rigidbody、XRGrabInteractable、XRSocketInteractable 等必要组件；
4. 根据 interaction_graph 绑定状态机，例如 grab keycard -> has_keycard，use keycard with cabinet -> cabinet_unlocked，socket battery into power_panel -> exit_unlocked；
5. 实现 StaticSceneValidator，检查 Prefab 是否存在、物体是否重叠、关键组件是否缺失、关键物体是否可达；
6. 实现 PlayMode test，自动模拟任务链并检查最终状态；
7. 如果失败，输出结构化错误类型，例如 AssetMissing、ComponentMissing、ObjectOverlap、PathBlocked、InteractionUnreachable、StateDeadlock、ScriptCompileError。

要求：
- 不要写复杂美术系统；
- 不要做开放世界；
- 不要依赖手动 Unity GUI 操作；
- 所有生成过程必须可以命令行复现；
- 所有 LLM 输出必须通过 JSON schema 校验后才能进入 Unity 执行层。
```

## 15. 最终可以形成的论文贡献

该方向可以凝练为三个贡献：

1. 提出一种面向 VR testing 的结构化 Unity/XR 场景生成框架，将自然语言/草图需求转化为可执行、可验证的交互场景。
2. 设计语义场景图、布局约束、交互图和 Unity 组件模板之间的编译流程，使 LLM 生成结果从“脚本草稿”变成“可运行工程资产”。
3. 构建自动验证与修复闭环，通过静态检查、PlayMode 测试和结构化 patch 提升生成场景的可用性、交互完成率与测试价值。
