#!/usr/bin/env python3
"""Standalone eval for a Gemma 4 LoRA checkpoint with configurable max_new_tokens.

Usage:
    .venv/bin/python eval_gemma4_checkpoint.py \
        --adapter outputs/gemma4_e2b_lora_bcs_0411_1212/epoch_2 \
        --eval-csv datasets/cat_10k/eval.csv \
        --max-new-tokens 300
"""
import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

# Reuse functions from training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_gemma4_4b_lora import (
    load_samples,
    load_system_prompt,
    run_eval,
    DEFAULT_PROMPTS_YAML,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="LoRA adapter dir")
    parser.add_argument("--base-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--eval-csv", default="datasets/cat_10k/eval.csv")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--image-tokens", type=int, default=140)
    parser.add_argument("--system-prompt", default=DEFAULT_PROMPTS_YAML)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <adapter>/eval_outputs_reeval.json)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, args.system_prompt)
    system_msg = load_system_prompt(prompt_path)
    print(f"System prompt: {len(system_msg)} chars")

    eval_set = load_samples(base_dir, args.eval_csv)
    print(f"Eval samples: {len(eval_set)}")

    print(f"Loading processor...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.image_processor.image_seq_length = args.image_tokens
    processor.image_processor.max_soft_tokens = args.image_tokens

    print(f"Loading base model {args.base_model}...")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto",
    )
    if hasattr(model.model, "audio_tower"):
        del model.model.audio_tower
        if hasattr(model.model, "embed_audio"):
            del model.model.embed_audio
        torch.cuda.empty_cache()

    if args.adapter.lower() != "none":
        print(f"Loading adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        print("No adapter — running base model")

    model.eval()
    device = next(model.parameters()).device

    output_path = args.output or os.path.join(args.adapter, "eval_outputs_reeval.json")
    print(f"Running eval (max_new_tokens={args.max_new_tokens})...")
    metrics = run_eval(
        model, processor, eval_set, device,
        max_new_tokens=args.max_new_tokens,
        system_msg=system_msg,
        log_path=output_path,
    )
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
