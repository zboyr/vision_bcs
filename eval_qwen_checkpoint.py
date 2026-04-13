#!/usr/bin/env python3
"""Standalone eval for a Qwen LoRA checkpoint, saves per-sample log.

Usage:
    .venv/bin/python eval_qwen_checkpoint.py \
        --adapter outputs/qwen2_5_vl_3b_lora_bcs_0409_2019/epoch_3 \
        --output outputs/qwen_eval_m2.json
"""
import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_qwen3_vl_4b_lora import (
    load_samples, load_system_prompt, make_messages, parse_bcs,
    DEFAULT_PROMPTS_YAML,
)


def run_eval_with_log(model, processor, eval_set, device, max_new_tokens,
                      system_msg, log_path):
    """Same eval as training but saves per-sample records."""
    abs_errors, parsed = [], 0
    records = []
    for sample in eval_set:
        messages = make_messages(sample.image_path, None, system_msg)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, videos=vids,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)[0]
        pred = parse_bcs(output)
        records.append({
            "image_path": sample.image_path,
            "gt": sample.bcs_primary,
            "pred": pred,
            "output": output,
        })
        if pred is not None:
            parsed += 1
            abs_errors.append(abs(float(pred) - float(sample.bcs_primary)))

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    coverage = parsed / len(eval_set) if eval_set else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    return {"eval_count": len(eval_set), "parsed": parsed, "coverage": coverage, "mae": mae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--eval-csv", default="datasets/cat_10k/eval.csv")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--output", required=True, help="Per-sample log JSON output path")
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
    metrics = run_eval_with_log(
        model, processor, eval_set, device,
        max_new_tokens=args.max_new_tokens,
        system_msg=system_msg, log_path=args.output,
    )
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
