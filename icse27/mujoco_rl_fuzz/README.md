# MuJoCo RL-Guided Fuzzing (ICSE'27 prototype)

> 见 [`../guide.md`](../guide.md) 为权威设计文档。本 README 只讲"怎么跑"。

## 安装

```powershell
# 推荐独立 venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `torch` 仅在跑 RL/Actor-Critic 时需要。Smoke demo 不依赖 torch。

## 第一步：可行性 Smoke Demo（必须先跑通）

目的：**亲眼看到 MuJoCo 的 warning / exception 被我们抓到**。

```powershell
python demo_smoke.py
```

期望输出（节选）：
- 阶段 1：正常 rollout，0 warning
- 阶段 2：超大 ctrl → 捕获 `BADCTRL` warning
- 阶段 3：qpos 注入 NaN → 捕获 `BADQPOS` warning + state has_nan=True
- 阶段 4：故意编译错误 XML → 捕获 `compile_error` exception type

如果四个阶段都能在终端打印结构化结果，链路通。

## 第二步：随机 baseline（待 Step 5 完成后启用）

```powershell
python -m src.main random --config configs/default.yaml --budget 1000
```

## 第三步：RL-guided（Actor-Critic，待 Step 7+ 完成后启用）

```powershell
python -m src.main rl --config configs/default.yaml --budget 10000
```

## 目录结构

见 `../guide.md` §6（已按 GzFuzz 风格 10 高级 mutator 调整）。

## 当前进度

- [x] Step 1 骨架 + 配置 + smoke demo
- [ ] Step 2 子进程 harness
- [ ] Step 3 10 个高级 mutator
- [ ] Step 4 oracle
- [ ] Step 5 random baseline
- [ ] Step 6 triage / dedup
- [ ] Step 7 Actor-Critic
- [ ] Step 8 LLM-finetune baseline + 对比
