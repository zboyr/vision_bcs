#!/usr/bin/env python3
"""Pre-fill experiment checkpoints from existing eval/scoring data.

- P3: from finetune eval JSON (same system_prompt_reasoning + user_prompt_reasoning)
- P4: from P1 results (majority-of-1 = same score)

Usage:
    .venv/bin/python prefill_checkpoints.py \
        --model-id M4 \
        --eval-json outputs/gemma4_e2b_lora_bcs_0412_1044/eval_outputs_epoch1_held_out.json \
        --dataset datasets/cat_10k/eval.csv \
        --log-dir responses/next_study/logs
"""
import argparse
import csv
import json
import os
from datetime import datetime


def load_dataset_id_map(csv_path: str) -> dict[str, int]:
    """Map image_path → sequential image_id (1-based, matching scoring/dataset.py)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(base_dir, csv_path)
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f), 1):
            path = row.get("path", "").strip()
            mapping[path] = idx
            # Also store by absolute path
            abs_path = os.path.join(base_dir, path)
            mapping[abs_path] = idx
    return mapping


def load_gt_map(csv_path: str) -> dict[int, float]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(base_dir, csv_path)
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f), 1):
            gt[idx] = float(row.get("bcs", 0))
    return gt


def prefill_p3(model_id: str, eval_json: str, dataset_csv: str, log_dir: str):
    """Write P3 checkpoint from finetune eval JSON."""
    id_map = load_dataset_id_map(dataset_csv)
    gt_map = load_gt_map(dataset_csv)
    eval_data = json.load(open(eval_json, encoding="utf-8"))

    cell_dir = os.path.join(log_dir, f"{model_id}_P3")
    os.makedirs(cell_dir, exist_ok=True)
    run_path = os.path.join(cell_dir, "run1.jsonl")

    # Don't overwrite if already exists
    if os.path.exists(run_path):
        with open(run_path) as f:
            existing = sum(1 for l in f if l.strip())
        if existing >= len(eval_data):
            print(f"P3: {run_path} already has {existing} entries, skip")
            return

    written = 0
    with open(run_path, "w", encoding="utf-8") as f:
        for rec in eval_data:
            img_path = rec["image_path"]
            img_id = id_map.get(img_path) or id_map.get(os.path.basename(img_path))
            if img_id is None:
                print(f"  WARN: no id for {img_path}")
                continue
            gt = gt_map.get(img_id, rec["gt"])
            bcs = rec["pred"]
            raw_output = rec["output"]
            entry = {
                "image_id": img_id,
                "bcs": bcs,
                "ground_truth": gt,
                "timestamp": datetime.now().isoformat(),
                "error": None if bcs is not None else "parse_fail",
                "raw": raw_output,
                "reasoning": "",
            }
            # Extract reasoning from output JSON
            try:
                obj = json.loads(raw_output)
                entry["reasoning"] = obj.get("reasoning", "")
            except (json.JSONDecodeError, TypeError):
                pass
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"P3: wrote {written} entries to {run_path}")


def prefill_p4(model_id: str, log_dir: str):
    """Write P4 (BO5) checkpoint from P1 results (majority of 1 = same score)."""
    p1_path = os.path.join(log_dir, f"{model_id}_P1", "run1.jsonl")
    if not os.path.exists(p1_path):
        print(f"P4: P1 checkpoint not found at {p1_path}, skip")
        return

    cell_dir = os.path.join(log_dir, f"{model_id}_P4")
    os.makedirs(cell_dir, exist_ok=True)
    run_path = os.path.join(cell_dir, "run1.jsonl")

    if os.path.exists(run_path):
        with open(run_path) as f:
            existing = sum(1 for l in f if l.strip())
        if existing > 0:
            print(f"P4: {run_path} already has {existing} entries, skip")
            return

    written = 0
    with open(run_path, "w", encoding="utf-8") as f:
        with open(p1_path) as pf:
            for line in pf:
                if not line.strip():
                    continue
                p1 = json.loads(line)
                bcs = p1.get("bcs")
                entry = {
                    "image_id": p1["image_id"],
                    "bcs": bcs,
                    "ground_truth": p1["ground_truth"],
                    "timestamp": datetime.now().isoformat(),
                    "error": p1.get("error"),
                    "votes": [bcs] * 5 if bcs is not None else [],
                    "raws": [p1.get("raw", "")] * 5,
                    "raw": p1.get("raw", ""),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

    print(f"P4: wrote {written} entries to {run_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--eval-json", required=True, help="Finetune eval JSON for P3")
    parser.add_argument("--dataset", default="datasets/cat_10k/eval.csv")
    parser.add_argument("--log-dir", default="responses/next_study/logs")
    args = parser.parse_args()

    prefill_p3(args.model_id, args.eval_json, args.dataset, args.log_dir)
    prefill_p4(args.model_id, args.log_dir)


if __name__ == "__main__":
    main()
