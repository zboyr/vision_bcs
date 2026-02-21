#!/usr/bin/env python3
import argparse
import json
import os
import re

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


SYSTEM_MSG = (
    "You are a veterinary clinician. Estimate feline body condition score (BCS) from image. "
    "Output strict JSON with keys: bcs, confidence, second_score, reasoning."
)
USER_MSG = "Assess this cat and return JSON only."


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Test fine-tuned Qwen3-VL-4B LoRA adapter")
    parser.add_argument("--adapter-dir", default="outputs/qwen3_vl_4b_lora_bcs")
    parser.add_argument("--image", default="images/cat_01.jpeg")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(base_dir, args.adapter_dir)
    image_path = os.path.join(base_dir, args.image)

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_MSG},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    output = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    bcs = parse_bcs(output)

    print("image=", image_path)
    print("adapter=", adapter_dir)
    print("parsed_bcs=", bcs)
    print("raw_output=", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
