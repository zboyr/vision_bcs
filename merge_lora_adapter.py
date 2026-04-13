#!/usr/bin/env python3
"""Merge LoRA adapter into base model and save as a standalone HF model.

Usage:
    .venv/bin/python merge_lora_adapter.py \
        --base google/gemma-4-E2B-it \
        --adapter outputs/gemma4_e2b_lora_bcs_0411_1212/epoch_3 \
        --out outputs/gemma4_e2b_bcs_merged
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"Loading base model {args.base}...")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu",
    )
    print(f"Loading adapter {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging...")
    model = model.merge_and_unload()
    print(f"Saving merged model to {args.out}...")
    os.makedirs(args.out, exist_ok=True)
    # Force-clone shared storages so safetensors doesn't deduplicate normalization weights.
    state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.save_pretrained(args.out, safe_serialization=True, state_dict=state_dict)

    # Also save processor / tokenizer
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.out)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
