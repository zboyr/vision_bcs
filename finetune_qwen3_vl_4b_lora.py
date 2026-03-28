#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


SYSTEM_MSG = (
    "You are a veterinary clinician. Estimate feline body condition score (BCS) from image. "
    "Output strict JSON with keys: bcs, confidence, second_score, reasoning."
)
USER_MSG = "Assess this cat and return JSON only."


@dataclass
class Sample:
    image_path: str
    ground_truth: float


def load_samples(base_dir: str, dataset_csv: str) -> list[Sample]:
    path = os.path.join(base_dir, dataset_csv)
    rows: list[Sample] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    Sample(
                        image_path=os.path.join(base_dir, r["image_path"]),
                        ground_truth=float(r["ground_truth"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return rows


def target_json(gt: float) -> str:
    bcs = min(9, max(1, int(round(gt))))
    obj = {
        "bcs": bcs,
        "confidence": "A",
        "second_score": None,
        "reasoning": "Visual body shape indicates this BCS score.",
    }
    return json.dumps(obj, ensure_ascii=False)


def make_messages(image_path: str, answer_json: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_MSG},
            ],
        },
    ]
    if answer_json is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_json}]})
    return messages


def collate_train(processor: Any, batch: list[dict[str, Any]]) -> dict[str, Any]:
    messages = [make_messages(x["image_path"], target_json(float(x["ground_truth"]))) for x in batch]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
    image_inputs: list[Any] = []
    for m in messages:
        imgs, _ = process_vision_info(m)
        image_inputs.append(imgs[0])
    enc = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    enc["labels"] = labels
    return enc


def parse_bcs(output_text: str) -> int | None:
    match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        value = int(obj.get("bcs"))
        if 1 <= value <= 9:
            return value
        return None
    except Exception:
        return None


def run_eval(
    model: Any,
    processor: Any,
    eval_set: list[Sample],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    abs_errors: list[float] = []
    parsed = 0
    for sample in eval_set:
        messages = make_messages(sample.image_path, None)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        pred = parse_bcs(output)
        if pred is None:
            continue
        parsed += 1
        abs_errors.append(abs(float(pred) - sample.ground_truth))
    coverage = parsed / len(eval_set) if eval_set else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    return {"eval_count": len(eval_set), "parsed": parsed, "coverage": coverage, "mae": mae}


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-VL-4B on local BCS dataset")
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--output-dir", default="outputs/qwen3_vl_4b_lora_bcs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, args.output_dir), exist_ok=True)

    samples = load_samples(base_dir, args.dataset)
    if len(samples) < 10:
        raise RuntimeError("dataset too small")
    random.shuffle(samples)
    train_size = min(args.train_size, len(samples))
    train_set = samples[:train_size]
    eval_set = samples[train_size:]

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    train_records = [{"image_path": s.image_path, "ground_truth": s.ground_truth} for s in train_set]
    train_dataset = Dataset.from_list(train_records)

    adapter_dir = os.path.join(base_dir, args.output_dir)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    sft_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        data_collator=lambda b: collate_train(processor, b),
        processing_class=processor,
    )
    trainer.train()

    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    model.eval()
    device = next(model.parameters()).device
    metrics = run_eval(model, processor, eval_set, device, args.max_new_tokens)
    metrics["train_size"] = len(train_set)
    metrics["model_id"] = model_id
    metrics_path = os.path.join(adapter_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("saved_adapter=", adapter_dir)
    print("metrics=", json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
