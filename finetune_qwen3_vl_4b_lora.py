#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


SYSTEM_MSG = """You are an expert veterinary clinician specializing in feline medicine with extensive experience in Body Condition Scoring (BCS) of cats. You will be shown a photograph of a cat and asked to assess its Body Condition Score.

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

USER_MSG = "What is the Body Condition Score (BCS) of this cat? Output only a single integer from 1 to 9."


@dataclass
class Sample:
    image_path: str
    ground_truth: float


def load_samples(base_dir: str, dataset_csv: str) -> list[Sample]:
    """Load samples from CSV. Supports three formats:
    - Original: columns image_path, ground_truth
    - scores.csv: columns file, score (images in cat_data/final/)
    - purina_3d: columns path, bcs (path relative to CSV directory)
    """
    path = os.path.join(base_dir, dataset_csv)
    csv_dir = os.path.dirname(path)
    rows: list[Sample] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                if "path" in r and "bcs" in r:
                    # purina_3d format: path relative to CSV dir
                    bcs = r["bcs"].strip()
                    if not bcs:
                        continue
                    # Try path relative to CSV directory first
                    img = os.path.join(csv_dir, r["path"])
                    if not os.path.exists(img):
                        # Fallback: filename only in CSV directory
                        img = os.path.join(csv_dir, os.path.basename(r["path"]))
                    rows.append(Sample(image_path=img, ground_truth=float(bcs)))
                elif "file" in r and "score" in r:
                    # scores.csv format
                    score = r["score"].strip()
                    if not score:
                        continue
                    rows.append(
                        Sample(
                            image_path=os.path.join(base_dir, "cat_data", "final", r["file"]),
                            ground_truth=float(score),
                        )
                    )
                else:
                    # Original format
                    rows.append(
                        Sample(
                            image_path=os.path.join(base_dir, r["image_path"]),
                            ground_truth=float(r["ground_truth"]),
                        )
                    )
            except (KeyError, ValueError):
                continue
    return rows


def target_answer(gt: float) -> str:
    bcs = min(9, max(1, int(round(gt))))
    return str(bcs)


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
    messages = [make_messages(x["image_path"], target_answer(float(x["ground_truth"]))) for x in batch]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
    image_inputs: list[Any] = []
    for m in messages:
        imgs, _ = process_vision_info(m)
        image_inputs.append(imgs[0])
    enc = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    # Mask everything except the assistant's response:
    # Only compute loss on tokens AFTER "<|im_start|>assistant\n" up to "<|im_end|>"
    # This prevents the model from wasting capacity predicting prompt/image tokens.
    assistant_header = [151644, 77091, 198]  # <|im_start|> assistant \n
    for i in range(labels.shape[0]):
        ids = labels[i].tolist()
        # Find last occurrence of assistant header
        resp_start = -1
        for j in range(len(ids) - len(assistant_header), -1, -1):
            if ids[j:j + len(assistant_header)] == assistant_header:
                resp_start = j + len(assistant_header)
                break
        # Mask everything before assistant response
        if resp_start > 0:
            labels[i, :resp_start] = -100
    # Also mask padding
    labels[labels == processor.tokenizer.pad_token_id] = -100
    enc["labels"] = labels
    return enc


def parse_bcs(output_text: str) -> int | None:
    text = output_text.strip()
    # Bare integer
    if re.fullmatch(r"[1-9]", text):
        return int(text)
    # First digit 1-9 in text
    m = re.search(r"\b([1-9])\b", text)
    if m:
        return int(m.group(1))
    # JSON fallback
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            value = int(obj.get("bcs"))
            if 1 <= value <= 9:
                return value
        except Exception:
            pass
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
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
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
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-4B on BCS dataset")
    parser.add_argument("--dataset", default="cat_data/scores.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.1,
                        help="Fraction of data for evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--mode", choices=["projector", "lora", "projector+lora", "vit+projector"],
                        default="projector",
                        help="Training mode")
    parser.add_argument("--resume-adapter", type=str, default=None,
                        help="Path to existing LoRA adapter dir to resume from (merged before new LoRA init)")
    parser.add_argument("--unfreeze-vit-layers", type=int, default=4,
                        help="Number of ViT blocks to unfreeze from the end (for vit+projector mode)")
    args = parser.parse_args()

    # Mode-specific defaults
    if args.lr is None:
        args.lr = 1e-4 if args.mode == "vit+projector" else 2e-5
    if args.output_dir is None:
        args.output_dir = {
            "projector": "outputs/qwen3_vl_4b_projector_bcs",
            "lora": "outputs/qwen3_vl_4b_lora_bcs",
            "projector+lora": "outputs/qwen3_vl_4b_projlora_bcs",
            "vit+projector": "outputs/qwen3_vl_4b_vitproj_bcs",
        }[args.mode]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(adapter_dir, exist_ok=True)

    samples = load_samples(base_dir, args.dataset)
    if len(samples) < 10:
        raise RuntimeError(f"dataset too small: {len(samples)} samples")
    random.shuffle(samples)
    eval_size = max(10, int(len(samples) * args.eval_ratio))
    eval_set = samples[:eval_size]
    train_set = samples[eval_size:]
    print(f"Dataset: {len(samples)} total, {len(train_set)} train, {len(eval_set)} eval")

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True,
        min_pixels=256 * 28 * 28,   # ~200k pixels
        max_pixels=512 * 28 * 28,   # ~400k pixels — limit VRAM usage
    )

    use_lora = args.mode in ("lora", "projector+lora")
    train_projector = args.mode in ("projector", "projector+lora", "vit+projector")
    train_vit = args.mode == "vit+projector"

    if use_lora:
        # LoRA modes: load in 4-bit
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True,
            quantization_config=bnb, device_map="auto",
        )
        # Merge existing LoRA adapter before creating new one
        if args.resume_adapter:
            from peft import PeftModel
            print(f"Loading existing adapter from {args.resume_adapter}...")
            model = PeftModel.from_pretrained(model, args.resume_adapter)
            model = model.merge_and_unload()
            print("Existing adapter merged into base model")
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(model, lora_cfg)
    else:
        # Projector-only: load in 4-bit to save VRAM, merger will be upcast to fp32
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True,
            quantization_config=bnb, device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

    # --- Freeze / unfreeze based on mode ---
    if train_projector:
        if not use_lora:
            # Freeze everything first
            for param in model.parameters():
                param.requires_grad = False

        # Unfreeze merger (projector) layers
        # Must convert to float BEFORE setting requires_grad (bnb int4 can't have grads)
        merger_params = 0
        for name, module in model.named_modules():
            if "merger" in name:
                for pname, param in module.named_parameters(recurse=False):
                    param.data = param.data.float()
                    param.requires_grad = True
                    merger_params += param.numel()

        # Unfreeze last N ViT blocks
        vit_params = 0
        if train_vit:
            total_blocks = 24  # Qwen3-VL-4B has 24 ViT blocks
            first_unfrozen = total_blocks - args.unfreeze_vit_layers
            for name, module in model.named_modules():
                # Match model.visual.blocks.{N} (top-level block only)
                import re as _re
                m = _re.match(r"model\.visual\.blocks\.(\d+)$", name)
                if m and int(m.group(1)) >= first_unfrozen:
                    for pname, param in module.named_parameters(recurse=True):
                        param.data = param.data.float()
                        param.requires_grad = True
                        vit_params += param.numel()
            print(f"Unfroze ViT blocks {first_unfrozen}-{total_blocks-1} "
                  f"({args.unfreeze_vit_layers} blocks, {vit_params:,} params)")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Mode: {args.mode}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Merger (projector) params: {merger_params:,}")
        if train_vit:
            print(f"  ViT params: {vit_params:,}")

    model.gradient_checkpointing_enable()
    model.train()

    train_records = [{"image_path": s.image_path, "ground_truth": s.ground_truth} for s in train_set]
    train_dataset = Dataset.from_list(train_records)

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if use_lora:
        # Use SFTTrainer for LoRA modes
        sft_args = SFTConfig(
            output_dir=adapter_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            logging_steps=1,
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
    else:
        # Use standard Trainer for projector-only
        # Bypass Trainer's quantization validation: we have frozen quantized params
        # but trainable float merger params, which is safe.
        if hasattr(model, "is_quantized"):
            model.is_quantized = False
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
        training_args = TrainingArguments(
            output_dir=adapter_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            weight_decay=0.01,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            bf16=bf16_ok,
            fp16=not bf16_ok,
            gradient_checkpointing=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda b: collate_train(processor, b),
            processing_class=processor,
        )

    trainer.train()

    # Save
    if use_lora:
        model.save_pretrained(adapter_dir)
    else:
        # Save only actual trainable parameters (no bnb metadata)
        trainable_state = {n: p.data.clone() for n, p in model.named_parameters()
                           if p.requires_grad}
        save_path = os.path.join(adapter_dir, "finetuned_weights.pt")
        torch.save(trainable_state, save_path)
        print(f"Saved {len(trainable_state)} tensors to {save_path}")

    # Save merger/projector weights separately if they were trained
    if train_projector:
        merger_state = {}
        for name, param in model.named_parameters():
            if "merger" in name and param.requires_grad:
                # Strip peft prefix if present
                clean_name = name.replace("base_model.model.", "", 1) if name.startswith("base_model.model.") else name
                merger_state[clean_name] = param.data.clone().to(torch.bfloat16)
        if merger_state:
            merger_path = os.path.join(adapter_dir, "merger_weights.pt")
            torch.save(merger_state, merger_path)
            print(f"Saved {len(merger_state)} merger tensors to {merger_path}")

    processor.save_pretrained(adapter_dir)

    # Eval
    model.eval()
    device = next(model.parameters()).device
    metrics = run_eval(model, processor, eval_set, device, args.max_new_tokens)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_label = f"{model_id}_ft_{timestamp}"
    metrics["train_size"] = len(train_set)
    metrics["eval_size"] = len(eval_set)
    metrics["model_id"] = model_label
    metrics["mode"] = args.mode
    metrics_path = os.path.join(adapter_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"saved to: {adapter_dir}")
    print(f"metrics: {json.dumps(metrics, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
