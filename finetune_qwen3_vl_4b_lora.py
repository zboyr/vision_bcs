
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
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
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
    # the working directory of this script sometimes different, so if
    # dataset only record relative path of it's current location, it will
    # fail. this func normalizes the path by either try to load the
    # file as it is, or try to normalize path to where dataset is
    # if both are not there, error
    if (os.path.exists(expected_filepath)):
        return expected_filepath
    recoverePath = os.path.join(os.path.dirname(dataset_csv_filepath), os.path.basename(expected_filepath))
    if (os.path.exists(recoverePath)):
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
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
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


def collate_train(processor: Any, batch: list[dict[str, Any]], system_msg: str) -> dict[str, Any]:
    full_msgs = [make_messages(x["image_path"], target_json_from_row(x), system_msg) for x in batch]
    prompt_msgs = [m[:-1] for m in full_msgs]  # drop assistant turn

    full_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in full_msgs]
    prompt_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompt_msgs]

    image_inputs: list[Any] = []
    for m in full_msgs:
        imgs, _ = process_vision_info(m)
        image_inputs.append(imgs[0])

    enc = processor(text=full_texts, images=image_inputs, return_tensors="pt", padding=True)
    prompt_enc = processor(text=prompt_texts, images=image_inputs, return_tensors="pt", padding=True)

    pad_id = processor.tokenizer.pad_token_id
    prompt_lens = (prompt_enc["input_ids"] != pad_id).sum(dim=1).tolist()

    labels = enc["input_ids"].clone()
    labels[labels == pad_id] = -100
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100
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
    system_msg: str,
) -> dict[str, Any]:
    abs_errors: list[float] = []
    parsed = 0
    for sample in eval_set:
        messages = make_messages(sample.image_path, None, system_msg)
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
        abs_errors.append(abs(float(pred) - float(sample.bcs_primary)))
    coverage = parsed / len(eval_set) if eval_set else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    return {"eval_count": len(eval_set), "parsed": parsed, "coverage": coverage, "mae": mae}


class EpochEvalCallback(TrainerCallback):
    def __init__(self, eval_set, processor, max_new_tokens, system_msg):
        self.eval_set = eval_set
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.system_msg = system_msg
        self.history: list[dict[str, Any]] = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or not self.eval_set:
            return
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        m = run_eval(model, self.processor, self.eval_set, device, self.max_new_tokens, self.system_msg)
        m["epoch"] = round(state.epoch, 3) if state.epoch is not None else None
        self.history.append(m)
        print(f"[epoch {m['epoch']}] eval: mae={m['mae']:.3f} coverage={m['coverage']:.2f} parsed={m['parsed']}/{m['eval_count']}")
        if was_training:
            model.train()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-VL-4B on local BCS dataset")
    parser.add_argument("--dataset", default="datasets/cat_10k/train.csv")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap total samples loaded (0=all). Useful for smoke tests.")
    parser.add_argument("--baseline-eval", action="store_true",
                        help="Run eval before training to get a baseline MAE.")
    parser.add_argument("--output-dir", default="outputs/qwen2_5_vl_3b_lora_bcs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # this is now percentage
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Generation cap. Target JSON length p99=145, max=174 in cat_10k annotations.")
    parser.add_argument("--max-pixels", type=int, default=401408,
                        help="Processor max_pixels (512*28*28=401408). Caps visual tokens per image.")
    parser.add_argument("--resume-adapter", default="",
                        help="Path to existing LoRA adapter to continue training from (instead of fresh init).")
    parser.add_argument("--system-prompt", default=DEFAULT_PROMPTS_YAML,
                        help="Path to prompts YAML file")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, args.system_prompt) if not os.path.isabs(args.system_prompt) else args.system_prompt
    system_msg = load_system_prompt(prompt_path)
    print(f"Loaded system prompt from: {prompt_path} ({len(system_msg)} chars)")
    os.makedirs(os.path.join(base_dir, args.output_dir), exist_ok=True)

    samples = load_samples(base_dir, args.dataset)
    if len(samples) < 10:
        raise RuntimeError("dataset too small")
    random.shuffle(samples)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    train_size = int(len(samples) * (args.train_size / 100))
    train_set = samples[:train_size]
    eval_set = samples[train_size:]
    print(f"samples: total={len(samples)}, train={len(train_set)}, eval={len(eval_set)}")

    model_id = args.model_id
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, max_pixels=args.max_pixels
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    if args.resume_adapter:
        resume_path = os.path.join(base_dir, args.resume_adapter) if not os.path.isabs(args.resume_adapter) else args.resume_adapter
        print(f"Resuming from LoRA adapter: {resume_path}")
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    baseline_metrics = None
    if args.baseline_eval and eval_set:
        print("Running baseline eval (pre-training)...")
        model.eval()
        device = next(model.parameters()).device
        baseline_metrics = run_eval(model, processor, eval_set, device, args.max_new_tokens, system_msg)
        print(f"baseline: {baseline_metrics}")

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
    epoch_eval_cb = EpochEvalCallback(eval_set, processor, args.max_new_tokens, system_msg)
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
    metrics = run_eval(model, processor, eval_set, device, args.max_new_tokens, system_msg)
    metrics["train_size"] = len(train_set)
    metrics["model_id"] = model_id
    if baseline_metrics is not None:
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