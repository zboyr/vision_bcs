#!/usr/bin/env python3
"""Standalone eval for a Qwen LoRA checkpoint using training script's run_eval.

Usage:
    .venv/bin/python eval_qwen_checkpoint.py \
        --adapter outputs/qwen2_5_vl_3b_lora_bcs_0409_2019/epoch_3
"""
import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_qwen3_vl_4b_lora import (
    load_samples, load_system_prompt, run_eval, DEFAULT_PROMPTS_YAML,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (None for base)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--eval-csv", default="datasets/cat_10k/eval.csv")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--system-prompt", default=DEFAULT_PROMPTS_YAML)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    system_msg = load_system_prompt(os.path.join(base_dir, args.system_prompt))
    eval_set = load_samples(base_dir, args.eval_csv)
    print(f"Eval samples: {len(eval_set)}")

    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True, max_pixels=args.max_pixels)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Loading base {args.base_model}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, trust_remote_code=True,
        quantization_config=bnb, device_map="auto",
    )
    if args.adapter:
        print(f"Loading adapter {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    device = next(model.parameters()).device
    metrics = run_eval(model, processor, eval_set, device,
                       max_new_tokens=args.max_new_tokens, system_msg=system_msg)
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
