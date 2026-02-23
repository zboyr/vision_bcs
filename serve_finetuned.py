#!/usr/bin/env python3
"""
Minimal OpenAI-compatible server for the fine-tuned Qwen3-VL-4B model.

Usage:
    python serve_finetuned.py [--port 8000] [--merger-weights outputs/qwen3_vl_4b_projector_bcs/merger_weights.pt]
"""

import argparse
import base64
import io
import json
import os
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_ID = BASE_MODEL_ID  # updated after loading weights
model = None
processor = None
device = None
generate_lock = Lock()


def load_model(weights_path: str):
    global model, processor, device, MODEL_ID

    print(f"Loading processor from {BASE_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    print(f"Loading model {BASE_MODEL_ID} in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True,
        quantization_config=bnb, device_map="auto",
    )

    # Detect whether weights_path is a LoRA adapter dir or raw weights file
    is_lora = False
    if weights_path and os.path.isdir(weights_path):
        if os.path.exists(os.path.join(weights_path, "adapter_config.json")):
            is_lora = True

    if is_lora:
        print(f"Loading LoRA adapter from {weights_path}...", flush=True)
        model = PeftModel.from_pretrained(model, weights_path)
        model = model.merge_and_unload()

        # Load merger/projector weights if present (from projector+lora training)
        merger_path = os.path.join(weights_path, "merger_weights.pt")
        if os.path.exists(merger_path):
            print(f"Loading merger weights from {merger_path}...", flush=True)
            merger_state = torch.load(merger_path, map_location="cpu", weights_only=True)
            loaded = 0
            for name, module in model.named_modules():
                for pname, param in list(module.named_parameters(recurse=False)):
                    full_name = f"{name}.{pname}" if name else pname
                    if full_name in merger_state:
                        target_device = param.device
                        new_data = merger_state[full_name].to(dtype=param.dtype, device=target_device)
                        new_param = torch.nn.Parameter(new_data, requires_grad=False)
                        setattr(module, pname, new_param)
                        loaded += 1
            print(f"Loaded {loaded}/{len(merger_state)} merger tensors", flush=True)

        # Read model_id from metrics.json if available
        metrics_path = os.path.join(weights_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            MODEL_ID = metrics.get("model_id", MODEL_ID)
        print(f"LoRA adapter merged successfully, MODEL_ID={MODEL_ID}", flush=True)
    elif weights_path and os.path.exists(weights_path):
        print(f"Loading fine-tuned weights from {weights_path}...", flush=True)
        ft_state = torch.load(weights_path, map_location="cpu", weights_only=True)

        # Replace matching params entirely (handles bnb Params4bit correctly)
        ft_keys = set(ft_state.keys())
        loaded = 0
        for name, module in model.named_modules():
            for pname, param in list(module.named_parameters(recurse=False)):
                full_name = f"{name}.{pname}" if name else pname
                if full_name in ft_keys:
                    target_device = param.device
                    new_data = ft_state[full_name].to(dtype=torch.bfloat16, device=target_device)
                    new_param = torch.nn.Parameter(new_data, requires_grad=False)
                    setattr(module, pname, new_param)
                    loaded += 1
        print(f"Loaded {loaded}/{len(ft_state)} fine-tuned tensors", flush=True)
    else:
        print("No fine-tuned weights found, using base model", flush=True)

    model.eval()
    device = next(model.parameters()).device
    print(f"Model ready on {device}")


def decode_base64_image(data_uri: str) -> Image.Image:
    """Extract and decode base64 image from data URI."""
    # data:image/jpeg;base64,/9j/...
    if data_uri.startswith("data:"):
        _, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    img_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_bytes))


def run_inference(messages: list, max_tokens: int = 300, temperature: float = 0.1) -> str:
    """Run inference on the model with the given messages."""
    # Convert OpenAI format to Qwen format
    qwen_messages = []
    pil_images = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            qwen_content = []
            for part in content:
                if part["type"] == "text":
                    qwen_content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    img = decode_base64_image(url)
                    pil_images.append(img)
                    qwen_content.append({"type": "image", "image": img})
            qwen_messages.append({"role": role, "content": qwen_content})

    text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(qwen_messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with generate_lock:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )

    output = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    return output.strip()


class OpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json({
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json({"error": "not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        req = json.loads(body)

        messages = req.get("messages", [])
        max_tokens = req.get("max_completion_tokens") or req.get("max_tokens", 300)
        temperature = req.get("temperature", 0.1)

        try:
            output = run_inference(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
            return

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send_json(resp)


def main():
    parser = argparse.ArgumentParser(description="Serve fine-tuned Qwen3-VL-4B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights",
                        default="outputs/qwen3_vl_4b_lora_bcs",
                        help="Path to LoRA adapter dir or raw weights .pt file")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, args.weights)

    load_model(weights_path)

    server = HTTPServer(("0.0.0.0", args.port), OpenAIHandler)
    print(f"\nServer listening on http://0.0.0.0:{args.port}")
    print(f"Use: python llm_scoring.py --provider local --base-url http://127.0.0.1:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
