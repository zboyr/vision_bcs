#!/usr/bin/env python3
"""Shared helpers for BCS LoRA fine-tuning scripts.

Used by:
- finetune_qwen3_vl_4b_lora.py
- finetune_gemma4_4b_lora.py

The pieces that differ between models (message format, collate, input
preparation for generation) stay in the per-model scripts. Everything else
— dataset loading, JSON target building, parsing, eval loop, epoch callback
— lives here.
"""
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch
from transformers import TrainerCallback

# Canonical prompt parser shared with the scoring pipelines.
# p3_prompts() returns (system_prompt_reasoning, user_prompt_reasoning) — the
# same pair used at inference time by scoring/pipelines.py, so training and
# eval stay in lockstep.
from scoring.prompts import p3_prompts


SYSTEM_MSG, USER_MSG = p3_prompts()


@dataclass
class Sample:
    image_path: str
    bcs_primary: int
    reasoning: str


def try_load_file(expected_filepath: str, dataset_csv_filepath: str) -> str:
    # The working directory of the caller sometimes differs from the dataset
    # root. Try the path as-is first; fall back to resolving relative to the
    # CSV's own directory. Raise if neither exists.
    if os.path.exists(expected_filepath):
        return expected_filepath
    recoverePath = os.path.join(
        os.path.dirname(dataset_csv_filepath), os.path.basename(expected_filepath)
    )
    if os.path.exists(recoverePath):
        return recoverePath
    raise ValueError(
        f"{expected_filepath} or {recoverePath} don't exist, check your dataset.csv"
    )


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


# Callback signature: (processor, sample, system_msg, device, model_dtype) -> inputs dict
# inputs must contain "input_ids" and be ready to pass to model.generate(**inputs).
PrepareInputsFn = Callable[..., dict]


def run_eval(
    model: Any,
    processor: Any,
    eval_set: list[Sample],
    device: torch.device,
    max_new_tokens: int,
    system_msg: str,
    prepare_inputs_fn: PrepareInputsFn,
    log_path: str | None = None,
) -> dict[str, Any]:
    abs_errors: list[float] = []
    parsed = 0
    records: list[dict[str, Any]] = []
    model_dtype = next(model.parameters()).dtype
    for sample in eval_set:
        inputs = prepare_inputs_fn(
            processor=processor,
            sample=sample,
            system_msg=system_msg,
            device=device,
            model_dtype=model_dtype,
        )
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        output = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
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
    return {
        "eval_count": len(eval_set),
        "parsed": parsed,
        "coverage": coverage,
        "mae": mae,
    }


class EpochEvalCallback(TrainerCallback):
    def __init__(
        self,
        split_eval_set,
        held_out_eval_set,
        processor,
        max_new_tokens,
        system_msg,
        adapter_dir,
        prepare_inputs_fn: PrepareInputsFn,
    ):
        self.split_eval_set = split_eval_set
        self.held_out_eval_set = held_out_eval_set
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.system_msg = system_msg
        self.adapter_dir = adapter_dir
        self.prepare_inputs_fn = prepare_inputs_fn
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
        for name, eset in [
            ("split", self.split_eval_set),
            ("held_out", self.held_out_eval_set),
        ]:
            if not eset:
                continue
            log_path = os.path.join(
                self.adapter_dir, f"eval_outputs_epoch{epoch}_{name}.json"
            )
            r = run_eval(
                model,
                self.processor,
                eset,
                device,
                self.max_new_tokens,
                self.system_msg,
                prepare_inputs_fn=self.prepare_inputs_fn,
                log_path=log_path,
            )
            print(
                f"[epoch {epoch}] {name}: mae={r['mae']:.3f} "
                f"coverage={r['coverage']:.2f} parsed={r['parsed']}/{r['eval_count']}"
            )
            m[name] = r
        self.history.append(m)
        if was_training:
            model.train()
        torch.cuda.empty_cache()
