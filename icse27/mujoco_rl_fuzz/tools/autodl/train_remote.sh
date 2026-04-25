#!/usr/bin/env bash
# train_remote.sh -- end-to-end training pipeline on the AutoDL machine.
#   1) generate RL traces (uses GPU torch)
#   2) collect SFT/DPO datasets
#   3) SFT (LoRA)
#   4) DPO (continues from SFT ckpt)
#
# Usage:
#   bash tools/autodl/train_remote.sh  Qwen/Qwen2.5-1.5B  ppo  2000
#                                       ^model            ^algo ^budget
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-1.5B}"
ALGO="${2:-ppo}"
BUDGET="${3:-2000}"
QUANT="${QUANT:-none}"          # "4bit" / "8bit" / "none". 24GB+ -> none.
EPOCHS_SFT="${EPOCHS_SFT:-3}"
EPOCHS_DPO="${EPOCHS_DPO:-2}"

cd "$(dirname "$0")/../.."

echo "============================================================"
echo " model    : $MODEL"
echo " algo     : $ALGO"
echo " budget   : $BUDGET"
echo " quant    : $QUANT"
echo "============================================================"

# 1) Collect RL traces -> logs/run_actor_critic_${ALGO}.jsonl
python -m src.main rl --config configs/default.yaml --budget "$BUDGET" --algorithm "$ALGO"

LOG="logs/run_actor_critic_${ALGO}.jsonl"
test -s "$LOG" || { echo "no log produced at $LOG"; exit 1; }

# 2) Build SFT + DPO datasets
mkdir -p data
python -m tools.llm_collect_traces \
  --in "$LOG" \
  --out-sft data/sft.jsonl \
  --out-dpo data/dpo.jsonl \
  --top-pct 0.3 --bot-pct 0.3

# 3) SFT (LoRA)
SFT_OUT="checkpoints/sft_$(echo "$MODEL" | tr '/' '_')"
EXTRA_SFT=""
[[ "$QUANT" != "none" ]] && EXTRA_SFT="--quant $QUANT --grad-ckpt --grad-accum 4"
python -m tools.llm_sft \
  --model "$MODEL" --data data/sft.jsonl --out "$SFT_OUT" \
  --epochs "$EPOCHS_SFT" --batch-size 2 --max-len 768 $EXTRA_SFT

# 4) DPO from SFT ckpt
DPO_OUT="checkpoints/dpo_$(echo "$MODEL" | tr '/' '_')"
EXTRA_DPO=""
[[ "$QUANT" != "none" ]] && EXTRA_DPO="--quant $QUANT --grad-ckpt --grad-accum 4"
python -m tools.llm_dpo \
  --model "$SFT_OUT" --data data/dpo.jsonl --out "$DPO_OUT" \
  --epochs "$EPOCHS_DPO" --batch-size 1 --max-len 768 $EXTRA_DPO

echo "============================================================"
echo " DONE  SFT -> $SFT_OUT"
echo "       DPO -> $DPO_OUT"
echo "============================================================"
