#!/usr/bin/env python3
"""
Score datasets/essay/dataset_full.csv with the fine-tuned LoRA adapter
and append one row to responses/essay_results.csv.

Usage:
    python3 score_essay_with_lora.py
    python3 score_essay_with_lora.py --adapter outputs/qwen2_5_vl_3b_lora_bcs_full
"""
import argparse
import csv
import datetime
import json
import os
import re

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


USER_MSG = (
    "Assess this cat's Body Condition Score. Examine the visible body shape, "
    "waist definition, abdominal profile, rib coverage, and overall fat/muscle "
    "distribution. Respond with JSON only."
)


def parse_bcs(output_text: str) -> int | None:
    match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        value = int(obj.get("bcs_primary"))
        if 1 <= value <= 9:
            return value
        return None
    except Exception:
        return None


def make_messages(image_path: str, system_msg: str):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_MSG},
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/qwen2_5_vl_3b_lora_bcs_full")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset", default="datasets/essay/dataset_full.csv")
    parser.add_argument("--results", default="responses/essay_results.csv")
    parser.add_argument("--image-dir", default="datasets/essay",
                        help="Directory for basename fallback lookup.")
    parser.add_argument("--image-col", default="path",
                        help="CSV column with image filename/path (e.g. 'path' or 'filename').")
    parser.add_argument("--system-prompt", default="prompts/bcs_prompts.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--source-name", default="Qwen/Qwen2.5-VL-3B-Instruct (lora-cat_10k-full)")
    parser.add_argument("--run", default="1")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load dataset (path,bcs)
    ds_path = os.path.join(base_dir, args.dataset)
    rows = []
    with open(ds_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Loaded {len(rows)} images from {args.dataset}")

    # Resolve image path: try as-is (relative to base_dir), else basename under --image-dir
    def resolve_image(csv_path: str) -> str:
        p = os.path.join(base_dir, csv_path)
        if os.path.exists(p):
            return p
        fallback = os.path.join(base_dir, args.image_dir, os.path.basename(csv_path))
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(f"image not found: {csv_path}")

    # Load system prompt
    import yaml
    prompt_path = os.path.join(base_dir, args.system_prompt)
    with open(prompt_path, "r", encoding="utf-8") as f:
        p = yaml.safe_load(f)
    system_msg = f"{p['role'].strip()}\n\n{p['bcs_scale'].strip()}\n\n{p['confidence_guide'].strip()}"

    # Load model + LoRA
    print(f"Loading base model: {args.model_id}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True, max_pixels=args.max_pixels
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, trust_remote_code=True, quantization_config=bnb, device_map="auto"
    )
    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, os.path.join(base_dir, args.adapter))
    model.eval()
    device = next(model.parameters()).device

    # Run inference
    preds = []
    deviations = []
    for i, r in enumerate(rows, start=1):
        img_path = resolve_image(r[args.image_col])
        gt = float(r["bcs"])
        messages = make_messages(img_path, system_msg)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        output = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        pred = parse_bcs(output)
        preds.append(pred)
        dev = abs(float(pred) - gt) if pred is not None else None
        deviations.append(dev)
        status = f"pred={pred} gt={int(gt)} dev={dev}" if pred is not None else f"PARSE FAIL: {output[:120]!r}"
        print(f"  [{i:2d}/{len(rows)}] {os.path.basename(img_path)}: {status}")

    # Compute mean deviation (skip None)
    valid = [d for d in deviations if d is not None]
    mean_dev = sum(valid) / len(valid) if valid else float("nan")
    print(f"\nmean_deviation={mean_dev:.4f}  parsed={len(valid)}/{len(preds)}")

    # Append row to results CSV. Column count = number of dataset rows.
    results_path = os.path.join(base_dir, args.results)
    n = len(preds)
    fieldnames = ["id", "source", "run", "mean_deviation"] + [f"bcs{i:02d}" for i in range(1, n + 1)]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    row_out = {"id": ts, "source": args.source_name, "run": args.run,
               "mean_deviation": f"{mean_dev:.4f}"}
    for i, p in enumerate(preds, start=1):
        row_out[f"bcs{i:02d}"] = f"{float(p):.1f}" if p is not None else ""

    with open(results_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row_out)
    print(f"Appended row id={ts} to {args.results}")


if __name__ == "__main__":
    main()
