#!/usr/bin/env python3
"""
Minimal OpenAI-compatible API server for Qwen2-VL-2B-Instruct.
Usage: python serve_qwen2vl.py [--port 8000]
"""

import argparse
import base64
import io
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

app = FastAPI()

model = None
processor = None
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


def load_model():
    global model, processor
    print(f"Loading {MODEL_NAME}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("Model loaded.")


class ChatMessage(BaseModel):
    role: str
    content: object  # str or list


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.1


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Convert OpenAI format to Qwen format
    qwen_messages = []
    for msg in req.messages:
        if isinstance(msg.content, str):
            qwen_messages.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
        elif isinstance(msg.content, list):
            qwen_content = []
            for part in msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        qwen_content.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            # base64 encoded image
                            qwen_content.append({"type": "image", "image": url})
                        else:
                            qwen_content.append({"type": "image", "image": url})
            qwen_messages.append({"role": msg.role, "content": qwen_content})
        else:
            qwen_messages.append({"role": msg.role, "content": [{"type": "text", "text": str(msg.content)}]})

    text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(qwen_messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=max(req.temperature, 0.01),
            do_sample=req.temperature > 0,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    load_model()
    uvicorn.run(app, host=args.host, port=args.port)
