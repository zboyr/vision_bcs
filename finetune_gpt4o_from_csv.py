#!/usr/bin/env python3
"""
Fine-tune GPT-4o (vision) on the Purina 3D cat BCS dataset.

Input CSV columns: path, bcs
Each training example embeds the image as base64 so the model sees the photo.
Assistant output: single integer 1-9 (no JSON, no reasoning).

Usage:
    python finetune_gpt4o_from_csv.py
    python finetune_gpt4o_from_csv.py --input-csv purina_3d_dataset.csv --epochs 3
"""

import argparse
import base64
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert veterinary clinician specializing in feline medicine with extensive experience in Body Condition Scoring (BCS) of cats. You will be shown a photograph of a cat and asked to assess its Body Condition Score.

BCS uses a standardized 9-point scale (Association for Pet Obesity Prevention / veterinary consensus). Evaluate these key areas from the photo: ribs/chest, spine and back, hips, abdominal tuck (side view), and waist (top view). Use the following criteria:

Score 1 (Emaciated): Ribs project prominently with sharp edges; no fat layer; visible in short-haired cats. Spine and back project prominently. Hips clearly visible, sharp and protruding. Severe abdominal tuck, concave. Waist extremely narrow, hourglass.

Score 2 (Very Thin): Ribs easily felt with very minimal fat; visible on short-haired cats. Spine projects prominently with sharp edges. Hips visible and prominent. Very pronounced abdominal tuck, concave. Waist very narrow, extreme hourglass.

Score 3 (Thin): Ribs easily felt with slight fat covering; visible on short-haired cats. Spine projects prominently with sharp edges. Hips visible. Pronounced abdominal tuck, small abdominal fat. Visible waistline.

Score 4 (Moderately Thin): Ribs easily felt with light fat covering; not visible. Spine can be felt. Hips less bony, some fat; still visible in short-haired cats. Slight abdominal tuck, small fat pad. Noticeable waistline, not overly narrow.

Score 5 (Ideal): Ribs can be felt with slight fat covering. Spine smooth and easily felt, not sharp or bony. Hips rounded with fat layer. Slight abdominal tuck, small fat pad. Noticeable waistline behind ribs, gently curved hourglass from above.

Score 6 (Moderately Above Ideal): Ribs can still be felt but with more difficulty; not visible. Spine can still be felt. Hips can be felt, not visible; some fat. Minimal abdominal tuck, slight rounding. Waist not defined.

Score 7 (Overweight): Ribs difficult to feel under fat. Spine becoming difficult to feel. Hips felt with some pressure; not visible; fat deposits. No abdominal tuck, rounded abdomen. Waist barely visible or absent; body rounded from above.

Score 8 (Obese): Ribs very difficult to feel under fat. Spine very difficult to feel. Hips difficult to feel under fat, little definition. Pronounced rounding and distension of abdomen. No waist, broad and rounded from above. Notable fat deposits on body.

Score 9 (Severe Obesity): Ribs unable to feel. Spine difficult to feel. Hips heavily padded, significant fat; no definition. Large hanging abdominal fat pad. No waist, broad and rounded. Significant fat over lower spine, neck and chest.

Note: Long hair can obscure contours—infer fat coverage from visible landmarks (rib outline, waist, belly curve). Primordial pouch (loose skin under belly) is normal and not necessarily excess fat.

Please assess the cat's BCS based solely on visual cues from the photograph, using the above criteria. Output ONLY a single integer from 1 to 9. No explanation, no punctuation, just the number."""

USER_PROMPT = "What is the Body Condition Score (BCS) of this cat? Output only a single integer from 1 to 9."


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    image_path: str      # relative to project root
    ground_truth: int    # BCS label 1-9


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_rows(csv_path: Path) -> list[Row]:
    """
    Supports two CSV formats:
      - New (purina): columns  path, bcs
      - Legacy:       columns  image_id, image_path, ground_truth
    """
    rows: list[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            try:
                # New format
                if "path" in item and "bcs" in item:
                    rows.append(Row(
                        image_path=item["path"].strip(),
                        ground_truth=int(float(item["bcs"])),
                    ))
                # Legacy format
                elif "image_path" in item and "ground_truth" in item:
                    rows.append(Row(
                        image_path=item["image_path"].strip(),
                        ground_truth=int(round(float(item["ground_truth"]))),
                    ))
            except (KeyError, ValueError):
                continue
    return rows


def encode_image(image_path: Path) -> tuple[str, str]:
    """Return (base64_string, mime_type)."""
    ext = image_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(ext, "image/jpeg")
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return data, mime


# ── JSONL record builders ─────────────────────────────────────────────────────

def training_record(row: Row, base_dir: Path) -> dict[str, Any] | None:
    """Build one fine-tuning record with an embedded image."""
    img_path = base_dir / row.image_path
    if not img_path.exists():
        print(f"  [WARN] missing image: {img_path}")
        return None

    b64, mime = encode_image(img_path)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {"role": "assistant", "content": str(row.ground_truth)},
        ]
    }


def write_jsonl(path: Path, rows: list[Row], base_dir: Path) -> int:
    """Write JSONL, returns number of successfully written records."""
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            record = training_record(row, base_dir)
            if record is None:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_bcs_from_text(text: str) -> int | None:
    """Parse a BCS integer from model output (single integer or JSON fallback)."""
    text = text.strip()
    # Primary: bare integer
    if re.fullmatch(r"[1-9]", text):
        return int(text)
    # Grab first digit 1-9
    m = re.search(r"\b([1-9])\b", text)
    if m:
        return int(m.group(1))
    # JSON fallback
    block = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if block:
        try:
            obj = json.loads(block.group(0))
            value = int(obj.get("bcs", 0))
            if 1 <= value <= 9:
                return value
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return None


def clamp_bcs(value: float) -> int:
    return min(9, max(1, int(round(value))))


def class_from_bcs(value: float) -> str:
    if value <= 5:
        return "IW"
    if value <= 7:
        return "OW"
    return "OB"


# ── Fine-tuning job ───────────────────────────────────────────────────────────

def poll_job_until_terminal(client: OpenAI, job_id: str, interval: int, timeout: int) -> Any:
    start = time.time()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  job_status={job.status}", flush=True)
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        if time.time() - start > timeout:
            raise TimeoutError(f"fine-tuning job timed out after {timeout}s")
        time.sleep(interval)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-4o (vision) on cat BCS dataset")
    parser.add_argument("--input-csv",      default="purina_3d_dataset.csv")
    parser.add_argument("--work-dir",       default="outputs/gpt4o_ft_purina")
    parser.add_argument("--model",          default="gpt-4o-2024-08-06")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--train-ratio",    type=float, default=0.85)
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--poll-interval",  type=int,   default=30)
    parser.add_argument("--poll-timeout",   type=int,   default=21600)
    parser.add_argument("--max-validation", type=int,   default=0,
                        help="rows to eval post-training (0 = all)")
    parser.add_argument("--no-train",       action="store_true",
                        help="build JSONL only, skip API calls (dry run)")
    parser.add_argument("--skip-build",     action="store_true",
                        help="skip JSONL build if files already exist")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")

    # Load rows
    csv_path = base_dir / args.input_csv
    rows = read_rows(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    if len(rows) < 10:
        raise RuntimeError("too few rows in input CSV")

    random.seed(args.seed)
    random.shuffle(rows)

    split_idx = max(1, min(len(rows) - 1, int(len(rows) * args.train_ratio)))
    train_rows = rows[:split_idx]
    valid_rows = rows[split_idx:]
    print(f"Split: {len(train_rows)} train / {len(valid_rows)} validation")

    work_dir = base_dir / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build JSONL files
    train_jsonl = work_dir / "train.jsonl"
    valid_jsonl = work_dir / "valid.jsonl"
    if args.skip_build and train_jsonl.exists() and valid_jsonl.exists():
        print(f"--skip-build: reusing existing JSONL files")
        n_train = sum(1 for _ in train_jsonl.open())
        n_valid = sum(1 for _ in valid_jsonl.open())
        print(f"  {n_train} train / {n_valid} valid records")
    else:
        print("Building train JSONL (encoding images)...")
        n_train = write_jsonl(train_jsonl, train_rows, base_dir)
        print(f"  wrote {n_train} records → {train_jsonl}")
        print("Building validation JSONL...")
        n_valid = write_jsonl(valid_jsonl, valid_rows, base_dir)
        print(f"  wrote {n_valid} records → {valid_jsonl}")

    if args.no_train:
        print("--no-train: stopping before API calls.")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment / .env")

    client = OpenAI(api_key=api_key)

    # Use a client with a much longer timeout for large file uploads
    upload_client = OpenAI(api_key=api_key, timeout=600.0)

    def upload_with_retry(path: Path, purpose: str, max_tries: int = 5) -> Any:
        size_mb = path.stat().st_size / 1024 / 1024
        for attempt in range(1, max_tries + 1):
            try:
                print(f"  uploading {path.name} ({size_mb:.1f} MB, attempt {attempt}/{max_tries})...")
                with path.open("rb") as f:
                    return upload_client.files.create(file=f, purpose=purpose)
            except Exception as e:
                print(f"  upload error: {e}")
                if attempt < max_tries:
                    wait = min(30, 2 ** attempt)
                    print(f"  retrying in {wait}s...")
                    time.sleep(wait)
        raise RuntimeError(f"Failed to upload {path} after {max_tries} attempts")

    # Upload files
    print("Uploading files to OpenAI...")
    train_file = upload_with_retry(train_jsonl, "fine-tune")
    valid_file = upload_with_retry(valid_jsonl, "fine-tune")
    print(f"  train_file_id={train_file.id}")
    print(f"  valid_file_id={valid_file.id}")

    # Launch fine-tuning
    job = client.fine_tuning.jobs.create(
        model=args.model,
        training_file=train_file.id,
        validation_file=valid_file.id,
        hyperparameters={"n_epochs": args.epochs},
    )
    print(f"  job_id={job.id}")

    # Save job ID so llm_scoring can pick it up
    (work_dir / "job_id.txt").write_text(job.id, encoding="utf-8")

    # Poll until done
    print("Polling job status...")
    final_job = poll_job_until_terminal(client, job.id, args.poll_interval, args.poll_timeout)
    if final_job.status != "succeeded":
        report = {
            "job_id": final_job.id,
            "status": final_job.status,
            "model_requested": args.model,
            "error": getattr(final_job, "error", None),
        }
        (work_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report))
        return 2

    ft_model = final_job.fine_tuned_model
    print(f"fine_tuned_model={ft_model}")
    # Save fine-tuned model name for llm_scoring
    (work_dir / "ft_model.txt").write_text(ft_model, encoding="utf-8")

    # ── Post-training evaluation ──────────────────────────────────────────────
    eval_rows = rows if args.max_validation <= 0 else rows[: args.max_validation]
    print(f"Evaluating on {len(eval_rows)} rows...")

    predictions: list[dict[str, Any]] = []
    abs_errors: list[float] = []
    correct_class = 0
    parsed = 0

    for row in eval_rows:
        img_path = base_dir / row.image_path
        if not img_path.exists():
            continue
        b64, mime = encode_image(img_path)
        resp = client.chat.completions.create(
            model=ft_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
            max_completion_tokens=4,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        pred = parse_bcs_from_text(text)

        row_out: dict[str, Any] = {
            "image_path": row.image_path,
            "ground_truth": row.ground_truth,
            "pred_bcs": pred,
            "raw_output": text,
        }
        predictions.append(row_out)

        if pred is not None:
            parsed += 1
            abs_errors.append(abs(float(pred) - row.ground_truth))
            if class_from_bcs(float(pred)) == class_from_bcs(float(row.ground_truth)):
                correct_class += 1

    coverage  = parsed / len(eval_rows) if eval_rows else 0.0
    mae       = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    class_acc = (correct_class / parsed * 100.0) if parsed else 0.0

    # Save predictions CSV
    pred_csv = work_dir / "validation_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "ground_truth", "pred_bcs", "raw_output"])
        writer.writeheader()
        writer.writerows(predictions)

    report = {
        "job_id": final_job.id,
        "status": final_job.status,
        "model_requested": args.model,
        "fine_tuned_model": ft_model,
        "train_rows": n_train,
        "validation_rows": n_valid,
        "eval_rows": len(eval_rows),
        "parsed": parsed,
        "coverage": coverage,
        "mae": mae,
        "class_accuracy_pct": class_acc,
        "predictions_csv": str(pred_csv),
    }
    (work_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
