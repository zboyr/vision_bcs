#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from finetune_common import (
    SYSTEM_MSG,
    USER_MSG,
    EpochEvalCallback,
    load_samples,
    run_eval,
    target_json_from_row,
)
from scoring.prompts import p8_vfewshot_prompts


def make_messages(
    image_path: str,
    answer_json: str | None,
    system_msg: str,
    user_msg: str,
    reference_image_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    for ref in reference_image_paths or []:
        user_content.append({"type": "image", "image": ref})
    user_content.append({"type": "image", "image": image_path})
    user_content.append({"type": "text", "text": user_msg})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user", "content": user_content},
    ]
    if answer_json is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_json}]})
    return messages


def collate_train(
    processor: Any,
    batch: list[dict[str, Any]],
    system_msg: str,
    user_msg: str,
    reference_image_paths: list[str] | None = None,
) -> dict[str, Any]:
    full_msgs = [
        make_messages(
            x["image_path"], target_json_from_row(x), system_msg, user_msg, reference_image_paths
        )
        for x in batch
    ]
    prompt_msgs = [m[:-1] for m in full_msgs]  # drop assistant turn

    full_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in full_msgs]
    prompt_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompt_msgs]

    image_inputs: list[Any] = []
    for m in full_msgs:
        imgs, _ = process_vision_info(m)
        image_inputs.extend(imgs)

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


def make_prepare_inputs(user_msg: str, reference_image_paths: list[str] | None):
    """Build a per-sample generation inputs callback bound to user_msg + ref images."""
    def _prepare(processor, sample, system_msg, device, model_dtype):
        del model_dtype  # unused; Qwen processor handles dtype internally
        messages = make_messages(
            sample.image_path, None, system_msg, user_msg, reference_image_paths
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return _prepare


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-VL-4B on local BCS dataset")
    parser.add_argument("--dataset", default="datasets/cat_10k/train.csv")
    parser.add_argument("--eval-dataset", default="datasets/cat_10k/eval.csv",
                        help="Dedicated eval CSV. Overrides train/eval split when provided.")
    parser.add_argument("--no-held-out-eval", action="store_true",
                        help="Skip held-out eval (much faster smoke tests).")
    parser.add_argument("--prompt-mode", choices=["p3", "p8"], default="p3",
                        help="p3: single-image reasoning. p8: dual-species visual few-shot "
                             "(prepends cat+dog reference charts before target).")
    parser.add_argument("--reference-images", nargs="+",
                        default=["prompts/cat_bcs.jpg", "prompts/dog_bcs.jpg"],
                        help="Reference images prepended in p8 mode (ignored in p3 mode).")
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
    parser.add_argument("--train-size", type=int, default=95)
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Generation cap. Target JSON length p99=145, max=174 in cat_10k annotations.")
    parser.add_argument("--max-pixels", type=int, default=401408,
                        help="Processor max_pixels (512*28*28=401408). Caps visual tokens per image.")
    parser.add_argument("--resume-adapter", default="",
                        help="Path to existing LoRA adapter to continue training from (instead of fresh init).")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.prompt_mode == "p8":
        system_msg, user_msg = p8_vfewshot_prompts()
        reference_image_paths = [
            p if os.path.isabs(p) else os.path.join(base_dir, p)
            for p in args.reference_images
        ]
        for p in reference_image_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"reference image not found: {p}")
        print(f"Loaded P8 dual-species visual few-shot prompts ({len(system_msg)} sys chars, "
              f"{len(reference_image_paths)} reference images)")
    else:
        system_msg, user_msg = SYSTEM_MSG, USER_MSG
        reference_image_paths = None
        print(f"Loaded P3 reasoning prompts from scoring.prompts ({len(system_msg)} chars)")
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
    held_out_eval_set = (
        load_samples(base_dir, args.eval_dataset)
        if (args.eval_dataset and not args.no_held_out_eval)
        else []
    )
    print(f"samples: train={len(train_set)}, split_eval={len(split_eval_set)}, held_out_eval={len(held_out_eval_set)}")

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

    prepare_inputs = make_prepare_inputs(user_msg, reference_image_paths)

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
            r = run_eval(
                model, processor, eset, device, args.max_new_tokens, system_msg,
                prepare_inputs_fn=prepare_inputs, log_path=log_path,
            )
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
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    epoch_eval_cb = EpochEvalCallback(
        split_eval_set, held_out_eval_set, processor, args.max_new_tokens, system_msg,
        adapter_dir, prepare_inputs_fn=prepare_inputs,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        data_collator=lambda b: collate_train(
            processor, b, system_msg, user_msg, reference_image_paths
        ),
        processing_class=processor,
        callbacks=[epoch_eval_cb],
    )
    trainer.train()

    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    metrics: dict[str, Any] = {"train_size": len(train_set), "model_id": model_id}
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