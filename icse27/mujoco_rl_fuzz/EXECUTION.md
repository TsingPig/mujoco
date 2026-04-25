# 执行手册 (EXECUTION.md)

ICSE'27 MuJoCo RL-Guided Fuzzing — 从安装到训练的全流程。

---

## 0. 硬件与环境矩阵

| 场景 | 硬件 | 推荐路径 |
|---|---|---|
| 开发 / 调试 / 跑 RL 实验 | RTX 3060 Laptop (6 GB) | 本地 |
| LLM 微调 ≤ 1.5B (4-bit QLoRA) | RTX 3060 Laptop (6 GB) | 本地 (慢) |
| LLM 微调 1.5B–7B | AutoDL RTX 4090 / A5000 24 GB | 远程 |
| LLM 微调 ≥ 13B | AutoDL A100 40/80 GB | 远程 |

---

## 1. 本地安装 (Windows + RTX 3060)

```powershell
cd icse27\mujoco_rl_fuzz

# (1) Engine 必需依赖
pip install -r requirements.txt

# (2) RL 训练所需的 GPU 版 torch（当前装的是 CPU 版，需要替换）
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cu121 torch

# (3) LLM 微调依赖（仅在你打算跑 SFT/DPO 时安装）
pip install -r requirements-llm.txt
# Windows 上 bitsandbytes 装不上时改用：
pip install bitsandbytes --prefer-binary

# (4) 验证
python demo_smoke.py                                           # 4/4 PASS
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. RL 实验（本地 3060 即可）

### 2.1 单策略

```powershell
python -m src.main random --config configs/default.yaml --budget 200
python -m src.main rl     --config configs/default.yaml --budget 1000 --algorithm ppo
python -m src.main rl     --config configs/default.yaml --budget 1000 --algorithm a2c
```

### 2.2 多策略对比

```powershell
python -m src.main bench  --config configs/default.yaml --budget 500 --policies random a2c ppo
```

### 2.3 算法选择指南

| 算法 | 推荐预算 | 备注 |
|---|---|---|
| `random` | — | 永远跑作为基线 |
| `a2c`    | ≥ 500 | N-step return + 优势标准化 |
| `ppo`    | ≥ 1000 | **主推**。GAE λ=0.95 + clip ε=0.2 |
| `impala` / `a3c` | — | 故意未实现（Windows fork 问题 + 子进程争用） |

输出位置：
- `logs/run_<policy>.jsonl` — 每步轨迹（含 `state_text`/`reward`/`action`）
- `outputs/{crashes,warnings,interesting}/` — 触发异常的 testcase 持久化

---

## 3. LLM 实验

### 3.1 Zero-shot（不需训练）

```powershell
python -m src.main llm --config configs/default.yaml --budget 100 --enable --model Qwen/Qwen2.5-1.5B
```

`--enable` 才会真加载模型；不加就退化成 random fallback。

### 3.2 LLM 模型选择矩阵

| 模型 | 参数量 | 3060 6GB 可微调? | 推荐 quant |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` | 0.5 B | ✅ 全精度 | `none` |
| `Qwen/Qwen2.5-1.5B` | 1.5 B | ✅ QLoRA | `4bit` |
| `meta-llama/Llama-3.2-1B` | 1.0 B | ✅ QLoRA | `4bit` |
| `meta-llama/Llama-3.2-3B` | 3.0 B | ⚠️ 极限 | `4bit` + `--grad-ckpt` |
| `Qwen/Qwen2.5-7B`   | 7 B   | ❌ → AutoDL | — |
| `Qwen/Qwen2.5-14B`  | 14 B  | ❌ → AutoDL A100 | — |

### 3.3 训练流水线（本地 3060，Qwen2.5-1.5B 4-bit）

```powershell
# Step 1: 跑 RL 收集高/低奖励轨迹
python -m src.main rl --config configs/default.yaml --budget 1000 --algorithm ppo

# Step 2: 提取 SFT + DPO 数据集
python -m tools.llm_collect_traces `
  --in logs\run_actor_critic_ppo.jsonl `
  --out-sft data\sft.jsonl --out-dpo data\dpo.jsonl `
  --top-pct 0.3 --bot-pct 0.3

# Step 3: SFT (LoRA + 4bit)
python -m tools.llm_sft `
  --model Qwen/Qwen2.5-1.5B `
  --data data\sft.jsonl `
  --out  checkpoints\sft_qwen15 `
  --quant 4bit --grad-ckpt --grad-accum 4 --batch-size 1 --max-len 512 --epochs 3

# Step 4: DPO 在 SFT 之上
python -m tools.llm_dpo `
  --model checkpoints\sft_qwen15 `
  --data  data\dpo.jsonl `
  --out   checkpoints\dpo_qwen15 `
  --quant 4bit --grad-ckpt --grad-accum 4 --batch-size 1 --max-len 512 --epochs 2

# Step 5: 评估微调后的 LLM Policy
python -m src.main llm --config configs/default.yaml --budget 200 --enable `
  --model checkpoints\dpo_qwen15
```

3060 6GB + 1.5B + 4-bit + batch=1 + max_len=512 显存占用约 **4.5–5.2 GB**。
若 OOM，换 `Qwen/Qwen2.5-0.5B` 或把 `--max-len` 降到 384。

---

## 4. AutoDL 远程训练（推荐 ≥ 7B 模型）

### 4.1 一次性环境配置

1. AutoDL 控制台 → 租用容器实例
   - 镜像选 **PyTorch 2.x / CUDA 12.x**
   - GPU：1.5B–7B 选 RTX 4090（24 GB）；14B+ 选 A100
   - 数据盘 ≥ 50 GB
2. 拿到 SSH 连接信息（host / port / 密码），在本地 PowerShell 设环境变量：

```powershell
$env:AUTODL_HOST = "connect.westa.seetacloud.com"
$env:AUTODL_PORT = "12345"
$env:AUTODL_USER = "root"
```

3. 同步代码到远端（需要本机有 `rsync`，可用 Git-Bash 自带的）：

```powershell
.\tools\autodl\sync_to_remote.ps1
```

4. 远端首次 bootstrap（在 AutoDL 终端里）：

```bash
cd /root/mujoco_rl_fuzz
bash tools/autodl/bootstrap.sh
```

### 4.2 一键训练

```bash
# 在 AutoDL 终端
bash tools/autodl/train_remote.sh Qwen/Qwen2.5-7B ppo 5000

# 想用 4bit 节省显存：
QUANT=4bit bash tools/autodl/train_remote.sh Qwen/Qwen2.5-14B ppo 5000
```

参数：`<model> <algo> <budget>`，环境变量 `QUANT`/`EPOCHS_SFT`/`EPOCHS_DPO`。

### 4.3 拉回 checkpoints

```powershell
.\tools\autodl\sync_to_remote.ps1 -Pull
```

会把 `checkpoints/`、`logs/`、`outputs/` 拉回本地。

---

## 5. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `torch.cuda.is_available() = False` | 装的是 CPU 版 torch。按 §1 (2) 重装。 |
| 4-bit OOM @ 6GB | 降 `--max-len`、用 `Qwen2.5-0.5B`、或上 AutoDL |
| `bitsandbytes` Windows 装不上 | `pip install bitsandbytes --prefer-binary` 或 `bitsandbytes-windows` |
| `[collect] in=0 rows` | 旧日志没有 `state_text` 字段。删 `logs/*.jsonl` 重跑 RL |
| 远程 rsync 卡住 | 检查 SSH 端口；用 `-e "ssh -p $PORT -o StrictHostKeyChecking=no"` |
| `BADCTRL` 警告抓不到 | `STATE_PERTURB` 目标=ctrl 时必须设 `disable_clamp_ctrl`（已自动处理） |

---

## 6. 推荐论文实验配置

| 实验 | 命令 | 时长（3060） |
|---|---|---|
| 主对比 (random vs a2c vs ppo) | `bench --budget 5000` | ~70 min |
| 算法消融 (ppo, a2c) | `rl --algorithm {ppo,a2c} --budget 10000` | ~3 h 各 |
| LLM zero-shot | `llm --enable --budget 1000` | ~30 min（含模型加载） |
| LLM SFT+DPO 全管线 | `train_remote.sh` (AutoDL 4090) | ~1.5 h |
| LLM SFT+DPO (Qwen-7B) | `train_remote.sh ... 7B` (AutoDL 4090) | ~4 h |

收集完后用 `tools/llm_collect_traces.py` 生成训练集，建议 RL budget ≥ 5000 才能产出足够多的 high-reward 样本。
