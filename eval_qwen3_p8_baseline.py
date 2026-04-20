#!/usr/bin/env python3
"""One-shot: evaluate base Qwen3-VL-4B-Instruct (no adapter) with P8 prompts
on the 10 split samples produced by the smoke-test run, for direct comparison
against the LoRA-fine-tuned MAE.

Usage:
    .venv/bin/python eval_qwen3_p8_baseline.py
"""
import json
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from finetune_common import Sample, run_eval
from finetune_qwen3_vl_4b_lora import make_prepare_inputs
from scoring.prompts import p8_vfewshot_prompts


SPLIT_JSON = "outputs/qwen3_vl_4b_p8_lora_bcs_0413_2349/eval_outputs_final_split.json"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LOG_PATH = "outputs/qwen3_vl_4b_p8_lora_bcs_0413_2349/eval_outputs_baseline_split.json"
MAX_NEW_TOKENS = 200
MAX_PIXELS = 401408


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, SPLIT_JSON), "r", encoding="utf-8") as f:
        records = json.load(f)
    samples = [
        Sample(image_path=r["image_path"], bcs_primary=int(r["gt"]), reasoning="")
        for r in records
    ]
    print(f"loaded {len(samples)} samples from {SPLIT_JSON}")

    system_msg, user_msg = p8_vfewshot_prompts()
    reference_image_paths = [
        os.path.join(base_dir, "prompts/cat_bcs.jpg"),
        os.path.join(base_dir, "prompts/dog_bcs.jpg"),
    ]
    print(f"P8 prompts loaded ({len(system_msg)} sys chars, "
          f"{len(reference_image_paths)} reference images)")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, max_pixels=MAX_PIXELS)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    prepare_inputs = make_prepare_inputs(user_msg, reference_image_paths)
    log_path = os.path.join(base_dir, LOG_PATH)
    metrics = run_eval(
        model, processor, samples, device, MAX_NEW_TOKENS, system_msg,
        prepare_inputs_fn=prepare_inputs, log_path=log_path,
    )
    print(f"baseline metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
