"""DPO finetune for the LLM mutation policy.

Continues from an SFT checkpoint with preference pairs (chosen / rejected).

Usage:
    python -m tools.llm_dpo \
        --model checkpoints/sft_qwen15 \
        --data  data/dpo.jsonl \
        --out   checkpoints/dpo_qwen15
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--model", required=True, help="SFT checkpoint dir or HF id.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--grad-accum", type=int, default=1)
    args = ap.parse_args(argv)

    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
        import torch
    except ImportError as e:
        print(f"[dpo] missing dep: {e}\n  See EXECUTION.md for install commands.",
              file=sys.stderr)
        return 2

    rows = _load_jsonl(args.data)
    if not rows:
        print("[dpo] empty dataset", file=sys.stderr); return 1

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in rows if {"prompt", "chosen", "rejected"} <= r.keys()
    ])

    model = AutoModelForCausalLM.from_pretrained(args.model)
    ref   = AutoModelForCausalLM.from_pretrained(args.model)

    quant_cfg = None
    if args.quant == "4bit":
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                       bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_use_double_quant=True)
    elif args.quant == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    if quant_cfg is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_cfg,
            torch_dtype=torch.float16, device_map="auto")
        ref = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_cfg,
            torch_dtype=torch.float16, device_map="auto")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    cfg = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_len,
        gradient_checkpointing=args.grad_ckpt,
        fp16=(args.quant != "none"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(model=model, ref_model=ref, args=cfg,
                         train_dataset=ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[dpo] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
