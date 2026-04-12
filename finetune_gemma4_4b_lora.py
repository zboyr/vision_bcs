#!/usr/bin/env python3
"""LoRA fine-tune Gemma 4 E4B (multimodal) on local BCS dataset.

Adapted from finetune_qwen3_vl_4b_lora.py. Key differences vs Qwen:
- Uses AutoModelForMultimodalLM (Gemma 4 model class)
- No qwen_vl_utils dependency; PIL is used for image loading
- Message format uses {"type": "image", "url": <path>} per Gemma 4 chat template
- No max_pixels (Gemma 4 uses fixed visual token budget internally)
"""
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
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_PROMPTS_YAML = "prompts/bcs_prompts.yaml"


def _load_prompts_yaml():
    import yaml
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_PROMPTS_YAML)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_PROMPTS = _load_prompts_yaml()
USER_MSG = _PROMPTS["user_prompt_reasoning"].strip()


@dataclass
class Sample:
    image_path: str
    bcs_primary: int
    reasoning: str


def try_load_file(expected_filepath, dataset_csv_filepath):
    if os.path.exists(expected_filepath):
        return expected_filepath
    recoverePath = os.path.join(os.path.dirname(dataset_csv_filepath), os.path.basename(expected_filepath))
    if os.path.exists(recoverePath):
        return recoverePath
    raise ValueError(f"{expected_filepath} or {recoverePath} don't exist, check your dataset.csv")


def load_samples(base_dir: str, dataset_csv: str) -> list[Sample]:
    path = os.path.join(base_dir, dataset_csv)
    rows: list[Sample] = []
    print(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                if r.get("error"):
                    continue
                img_path = os.path.join(base_dir, r["path"])
                corrected_path = try_load_file(img_path, dataset_csv)
                rows.append(
                    Sample(
                        image_path=corrected_path,
                        bcs_primary=int(r.get("bcs") or r.get("bcs_primary")),
                        reasoning=r.get("reasoning", ""),
                    )
                )
            except (KeyError, ValueError) as e:
                print(f"skip row id={r.get('id')} err={e}")
                continue
    return rows


def target_json_from_row(r: dict[str, Any]) -> str:
    obj = {
        "reasoning": r.get("reasoning", "") or "",
        "bcs": int(r["bcs_primary"]),
    }
    return json.dumps(obj, ensure_ascii=False)


def load_system_prompt(path: str) -> str:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        p = yaml.safe_load(f)
    return p["system_prompt_reasoning"].strip()


def make_messages(image_path: str, answer_json: str | None, system_msg: str) -> list[dict[str, Any]]:
    # Gemma 4 uses {"type": "image", "url": <path>}; "url" can be a local file path.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": USER_MSG},
            ],
        },
    ]
    if answer_json is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_json}]})
    return messages


def extract_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    """Load PIL images from message content (works for both 'image' and 'url' keys)."""
    imgs: list[Image.Image] = []
    for m in messages:
        content = m.get("content", [])
        if not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "image":
                src = c.get("url") or c.get("image") or c.get("path")
                if isinstance(src, str):
                    imgs.append(Image.open(src).convert("RGB"))
                else:
                    imgs.append(src)
    return imgs


def collate_train(processor: Any, batch: list[dict[str, Any]], system_msg: str) -> dict[str, Any]:
    full_msgs = [make_messages(x["image_path"], target_json_from_row(x), system_msg) for x in batch]
    prompt_msgs = [m[:-1] for m in full_msgs]

    enc = processor.apply_chat_template(
        full_msgs, tokenize=True, return_dict=True, return_tensors="pt",
        padding=True, add_generation_prompt=False,
    )
    prompt_enc = processor.apply_chat_template(
        prompt_msgs, tokenize=True, return_dict=True, return_tensors="pt",
        padding=True, add_generation_prompt=True,
    )

    pad_id = processor.tokenizer.pad_token_id
    prompt_lens = (prompt_enc["input_ids"] != pad_id).sum(dim=1).tolist()

    labels = enc["input_ids"].clone()
    labels[labels == pad_id] = -100
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100
    enc["labels"] = labels
    return dict(enc)


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
    system_msg: str,
    log_path: str | None = None,
) -> dict[str, Any]:
    abs_errors: list[float] = []
    parsed = 0
    records: list[dict[str, Any]] = []
    model_dtype = next(model.parameters()).dtype
    for sample in eval_set:
        messages = make_messages(sample.image_path, None, system_msg)
        inputs = processor.apply_chat_template(
            [messages], tokenize=True, return_dict=True, return_tensors="pt",
            padding=True, add_generation_prompt=True,
        )
        inputs = inputs.to(device, dtype=model_dtype)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        pred = parse_bcs(output)
        records.append({
            "image_path": sample.image_path,
            "gt": sample.bcs_primary,
            "pred": pred,
            "output": output,
        })
        if pred is None:
            continue
        parsed += 1
        abs_errors.append(abs(float(pred) - float(sample.bcs_primary)))
    coverage = parsed / len(eval_set) if eval_set else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  raw outputs logged to {log_path}")
    return {"eval_count": len(eval_set), "parsed": parsed, "coverage": coverage, "mae": mae}


class EpochEvalCallback(TrainerCallback):
    def __init__(self, split_eval_set, held_out_eval_set, processor, max_new_tokens, system_msg, adapter_dir):
        self.split_eval_set = split_eval_set
        self.held_out_eval_set = held_out_eval_set
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.system_msg = system_msg
        self.adapter_dir = adapter_dir
        self.history: list[dict[str, Any]] = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        epoch = round(state.epoch) if state.epoch is not None else 0
        was_training = model.training

        ckpt_dir = os.path.join(self.adapter_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)
        print(f"[epoch {epoch}] saved checkpoint to {ckpt_dir}")

        model.eval()
        device = next(model.parameters()).device
        m = {"epoch": round(state.epoch, 3) if state.epoch is not None else None}
        for name, eset in [("split", self.split_eval_set), ("held_out", self.held_out_eval_set)]:
            if not eset:
                continue
            log_path = os.path.join(self.adapter_dir, f"eval_outputs_epoch{epoch}_{name}.json")
            r = run_eval(model, self.processor, eset, device, self.max_new_tokens, self.system_msg, log_path=log_path)
            print(f"[epoch {epoch}] {name}: mae={r['mae']:.3f} coverage={r['coverage']:.2f} parsed={r['parsed']}/{r['eval_count']}")
            m[name] = r
        self.history.append(m)
        if was_training:
            model.train()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Gemma 4 E4B on local BCS dataset")
    parser.add_argument("--dataset", default="datasets/cat_10k/train.csv")
    parser.add_argument("--eval-dataset", default="datasets/cat_10k/eval.csv",
                        help="Dedicated eval CSV for held-out evaluation.")
    parser.add_argument("--no-held-out-eval", action="store_true",
                        help="Skip held-out eval (much faster smoke tests).")
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap total samples loaded (0=all). Useful for smoke tests.")
    parser.add_argument("--baseline-eval", action="store_true",
                        help="Run eval before training to get a baseline MAE.")
    parser.add_argument("--output-dir", default="outputs/gemma4_e2b_lora_bcs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--image-tokens", type=int, default=140,
                        help="Visual token budget per image (Gemma 4 supports 70/140/280/560/1120). Lower=less VRAM.")
    parser.add_argument("--resume-adapter", default="")
    parser.add_argument("--system-prompt", default=DEFAULT_PROMPTS_YAML)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, args.system_prompt) if not os.path.isabs(args.system_prompt) else args.system_prompt
    system_msg = load_system_prompt(prompt_path)
    print(f"Loaded system prompt from: {prompt_path} ({len(system_msg)} chars)")
    args.output_dir = args.output_dir + "_" + datetime.now().strftime("%m%d_%H%M")
    os.makedirs(os.path.join(base_dir, args.output_dir), exist_ok=True)

    samples = load_samples(base_dir, args.dataset)
    if len(samples) < 10:
        raise RuntimeError("dataset too small")
    random.shuffle(samples)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    train_size = int(len(samples) * (args.train_size / 100))
    train_set = samples[:train_size]
    split_eval_set = samples[train_size:]
    held_out_eval_set = load_samples(base_dir, args.eval_dataset) if (args.eval_dataset and not args.no_held_out_eval) else []
    print(f"samples: train={len(train_set)}, split_eval={len(split_eval_set)}, held_out_eval={len(held_out_eval_set)}")

    model_id = args.model_id
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # Lower visual token budget to reduce LLM context and KV cache memory.
    processor.image_processor.image_seq_length = args.image_tokens
    processor.image_processor.max_soft_tokens = args.image_tokens
    print(f"Set Gemma 4 visual token budget to {args.image_tokens} per image")
    # Note: 4bit quantization breaks Gemma 4 vision tower (Gemma4ClippableLinear).
    # Load in bf16 instead.
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    # Freeze base params; LoRA will add trainable adapters.
    for p in model.parameters():
        p.requires_grad = False

    # Drop unused audio tower to free ~150 MB (BCS only needs vision).
    if hasattr(model.model, "audio_tower"):
        del model.model.audio_tower
        if hasattr(model.model, "embed_audio"):
            del model.model.embed_audio
        torch.cuda.empty_cache()
        print("Dropped audio_tower / embed_audio (unused for BCS).")
    if args.resume_adapter:
        resume_path = os.path.join(base_dir, args.resume_adapter) if not os.path.isabs(args.resume_adapter) else args.resume_adapter
        print(f"Resuming from LoRA adapter: {resume_path}")
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
    else:
        # Regex restricts LoRA to language_model layers only.
        # Vision tower uses Gemma4ClippableLinear which PEFT does not support.
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r".*language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj|per_layer_input_gate|per_layer_projection)$",
        )
        model = get_peft_model(model, lora_cfg)

    adapter_dir = os.path.join(base_dir, args.output_dir)
    baseline_metrics = None
    if args.baseline_eval and (split_eval_set or held_out_eval_set):
        print("Running baseline eval (pre-training)...")
        model.eval()
        device = next(model.parameters()).device
        baseline_metrics = {}
        for name, eset in [("split", split_eval_set), ("held_out", held_out_eval_set)]:
            if not eset:
                continue
            log_path = os.path.join(adapter_dir, f"eval_outputs_baseline_{name}.json")
            r = run_eval(model, processor, eset, device, args.max_new_tokens, system_msg, log_path=log_path)
            baseline_metrics[name] = r
            print(f"baseline {name}: {r}")

    model.train()

    train_records = [
        {
            "image_path": s.image_path,
            "bcs_primary": s.bcs_primary,
            "reasoning": s.reasoning,
        }
        for s in train_set
    ]
    train_dataset = Dataset.from_list(train_records)

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
        optim="adamw_torch",
        dataloader_num_workers=2,
        dataloader_persistent_workers=True,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    epoch_eval_cb = EpochEvalCallback(split_eval_set, held_out_eval_set, processor, args.max_new_tokens, system_msg, adapter_dir)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        data_collator=lambda b: collate_train(processor, b, system_msg),
        processing_class=processor,
        callbacks=[epoch_eval_cb],
    )
    trainer.train()

    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    model.eval()
    device = next(model.parameters()).device
    metrics = {"train_size": len(train_set), "model_id": model_id}
    for name, eset in [("split", split_eval_set), ("held_out", held_out_eval_set)]:
        if eset:
            log_path = os.path.join(adapter_dir, f"eval_outputs_final_{name}.json")
            metrics[f"final_{name}"] = run_eval(model, processor, eset, device, args.max_new_tokens, system_msg, log_path=log_path)
    if baseline_metrics:
        metrics["baseline"] = baseline_metrics
    metrics["epoch_history"] = epoch_eval_cb.history
    metrics_path = os.path.join(adapter_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("saved_adapter=", adapter_dir)
    print("metrics=", json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
