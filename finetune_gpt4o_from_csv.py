#!/usr/bin/env python3
import argparse
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


SYSTEM_PROMPT = (
    "You are an assistant for feline BCS scoring. "
    "Return strict JSON with keys: bcs, confidence, second_score, reasoning."
)


@dataclass
class Row:
    image_id: int
    image_path: str
    ground_truth: float


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
    rows: list[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            try:
                rows.append(
                    Row(
                        image_id=int(item["image_id"]),
                        image_path=item["image_path"],
                        ground_truth=float(item["ground_truth"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return rows


def clamp_bcs(value: float) -> int:
    return min(9, max(1, int(round(value))))


def training_record(row: Row) -> dict[str, Any]:
    target = {
        "bcs": clamp_bcs(row.ground_truth),
        "confidence": "A",
        "second_score": None,
        "reasoning": "Based on dataset annotation.",
    }
    user_prompt = (
        f"image_id={row.image_id}; image_path={row.image_path}. "
        "Predict cat Body Condition Score (1-9) and return JSON only."
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def write_jsonl(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(training_record(row), ensure_ascii=False) + "\n")


def class_from_bcs(value: float) -> str:
    if value <= 5:
        return "IW"
    if value <= 7:
        return "OW"
    return "OB"


def parse_bcs_from_text(text: str) -> int | None:
    block = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if block:
        try:
            obj = json.loads(block.group(0))
            value = int(obj.get("bcs"))
            if 1 <= value <= 9:
                return value
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    m = re.search(r'"bcs"\s*:\s*([1-9])', text)
    if m:
        return int(m.group(1))
    return None


def poll_job_until_terminal(client: OpenAI, job_id: str, interval: int, timeout: int) -> Any:
    start = time.time()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"job_status={job.status}", flush=True)
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        if time.time() - start > timeout:
            raise TimeoutError(f"fine-tuning job timeout after {timeout}s")
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-4o using deduplicated CSV and validate")
    parser.add_argument("--input-csv", default="dataset_nodup_min.csv")
    parser.add_argument("--work-dir", default="outputs/gpt4o_ft")
    parser.add_argument("--model", default="gpt-4o-2024-08-06")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--poll-timeout", type=int, default=21600)
    parser.add_argument("--max-validation", type=int, default=0, help="0 means validate on all rows")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in environment/.env")

    csv_path = base_dir / args.input_csv
    rows = read_rows(csv_path)
    if len(rows) < 10:
        raise RuntimeError("insufficient rows in input csv")

    random.seed(args.seed)
    random.shuffle(rows)

    split_idx = max(1, min(len(rows) - 1, int(len(rows) * args.train_ratio)))
    train_rows = rows[:split_idx]
    valid_rows = rows[split_idx:]

    work_dir = base_dir / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = work_dir / "train.jsonl"
    valid_jsonl = work_dir / "valid.jsonl"
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(valid_jsonl, valid_rows)

    client = OpenAI(api_key=api_key)

    with train_jsonl.open("rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    with valid_jsonl.open("rb") as f:
        valid_file = client.files.create(file=f, purpose="fine-tune")

    print(f"train_file_id={train_file.id}")
    print(f"valid_file_id={valid_file.id}")

    job = client.fine_tuning.jobs.create(
        model=args.model,
        training_file=train_file.id,
        validation_file=valid_file.id,
        hyperparameters={"n_epochs": args.epochs},
    )
    print(f"job_id={job.id}")

    final_job = poll_job_until_terminal(client, job.id, args.poll_interval, args.poll_timeout)
    if final_job.status != "succeeded":
        report = {
            "job_id": final_job.id,
            "status": final_job.status,
            "model_requested": args.model,
            "error": getattr(final_job, "error", None),
        }
        report_path = work_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return 2

    ft_model = final_job.fine_tuned_model
    print(f"fine_tuned_model={ft_model}")

    eval_rows = rows if args.max_validation <= 0 else rows[:args.max_validation]
    predictions: list[dict[str, Any]] = []
    parsed = 0
    abs_errors: list[float] = []
    correct_class = 0

    for row in eval_rows:
        prompt = (
            f"image_id={row.image_id}; image_path={row.image_path}. "
            "Predict cat Body Condition Score (1-9) and return JSON only."
        )
        resp = client.chat.completions.create(
            model=ft_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        pred = parse_bcs_from_text(text)

        row_out: dict[str, Any] = {
            "image_id": row.image_id,
            "image_path": row.image_path,
            "ground_truth": row.ground_truth,
            "pred_bcs": pred,
            "raw_output": text,
        }
        predictions.append(row_out)

        if pred is not None:
            parsed += 1
            abs_errors.append(abs(float(pred) - row.ground_truth))
            if class_from_bcs(float(pred)) == class_from_bcs(row.ground_truth):
                correct_class += 1

    coverage = parsed / len(eval_rows) if eval_rows else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    class_acc = (correct_class / parsed * 100.0) if parsed else 0.0

    pred_csv = work_dir / "validation_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "image_path", "ground_truth", "pred_bcs", "raw_output"],
        )
        writer.writeheader()
        writer.writerows(predictions)

    report = {
        "job_id": final_job.id,
        "status": final_job.status,
        "model_requested": args.model,
        "fine_tuned_model": ft_model,
        "train_rows": len(train_rows),
        "validation_rows_for_job": len(valid_rows),
        "eval_rows": len(eval_rows),
        "parsed": parsed,
        "coverage": coverage,
        "mae_vs_ground_truth": mae,
        "class_accuracy": class_acc,
        "train_jsonl": str(train_jsonl),
        "valid_jsonl": str(valid_jsonl),
        "predictions_csv": str(pred_csv),
    }

    report_path = work_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
