"""LoRA SFT for the LLM mutation policy.

Uses HF transformers + peft + trl.SFTTrainer. Designed for small models
(Qwen2.5-1.5B / Llama-3.2-1B). Run on a single GPU; falls back to CPU.

Usage:
    python -m tools.llm_sft \
        --model Qwen/Qwen2.5-1.5B \
        --data data/sft.jsonl \
        --out  checkpoints/sft_qwen15
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                    help="QLoRA quantization. Use 4bit on 6GB GPUs (RTX 3060).")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="Enable gradient checkpointing (saves VRAM, ~20%% slower).")
    ap.add_argument("--grad-accum", type=int, default=1)
    args = ap.parse_args(argv)

    try:
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
        import torch
    except ImportError as e:
        print(f"[sft] missing dep: {e}\n  See EXECUTION.md for install commands.",
              file=sys.stderr)
        return 2

    rows = _load_jsonl(args.data)
    if not rows:
        print("[sft] empty dataset", file=sys.stderr); return 1

    # Format as single text field "prompt + completion + EOS"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token or ""

    ds = Dataset.from_list([
        {"text": r["prompt"] + r["completion"] + eos} for r in rows
    ])

    quant_cfg = None
    if args.quant == "4bit":
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                       bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_use_double_quant=True)
    elif args.quant == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if args.quant != "none" else "auto",
        device_map="auto",
    )
    if args.quant != "none":
        model = prepare_model_for_kbit_training(model)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      target_modules=["q_proj", "v_proj"], bias="none",
                      task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_len,
        gradient_checkpointing=args.grad_ckpt,
        fp16=(args.quant != "none"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, train_dataset=ds, peft_config=lora,
                         tokenizer=tok, args=cfg, dataset_text_field="text")
    trainer.train()
    trainer.save_model(args.out)
    print(f"[sft] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
