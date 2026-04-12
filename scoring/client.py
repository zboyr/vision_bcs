"""LLM client creation and API communication."""

import base64
import importlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def load_dotenv(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _get_openai_class():
    try:
        return getattr(importlib.import_module("openai"), "OpenAI")
    except ImportError:
        print("请先安装 openai: pip install openai")
        sys.exit(1)


def create_client(provider: str, base_url: Optional[str] = None,
                  api_key: Optional[str] = None,
                  request_timeout: float = 60.0) -> Any:
    """Create OpenAI-compatible client for the given provider."""
    cls = _get_openai_class()

    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("请设置 OPENAI_API_KEY")
        return cls(api_key=key, timeout=request_timeout)

    if provider == "openrouter":
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("请设置 OPENROUTER_API_KEY")
        url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return cls(
            api_key=key, base_url=url,
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_APP_URL",
                                               "http://localhost"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "vision_bcs"),
            },
            timeout=request_timeout,
        )

    if provider == "local":
        url = base_url or os.environ.get(
            "LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        key = (api_key
               or os.environ.get("LOCAL_OPENAI_API_KEY")
               or os.environ.get("OPENAI_API_KEY")
               or "EMPTY")
        return cls(api_key=key, base_url=url, timeout=request_timeout)

    if provider == "transformers":
        return TransformersClient(base_url=base_url, api_key=api_key)

    raise ValueError(f"不支持的 provider: {provider}")


# ── Transformers-based local client (no vllm required) ──────────────

class TransformersClient:
    """Drop-in replacement for OpenAI client using transformers directly.

    Config mapping in next_study.yaml::

        M4:
            provider: transformers
            base_url: google/gemma-4-E2B-it          # HF model id or local path
            api_key: outputs/.../epoch_1              # adapter path (or "none")
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        import torch
        self._model_id = base_url
        self._adapter = api_key if (api_key and api_key.lower() != "none") else None
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self.chat = self  # so client.chat.completions.create works
        self.completions = self

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        print(f"[TransformersClient] Loading {self._model_id}...")
        self._processor = AutoProcessor.from_pretrained(
            self._model_id, trust_remote_code=True)

        # Detect model type
        is_gemma = "gemma" in self._model_id.lower()
        is_qwen = "qwen" in self._model_id.lower()

        if is_gemma:
            from transformers import AutoModelForMultimodalLM
            self._processor.image_processor.image_seq_length = 140
            self._processor.image_processor.max_soft_tokens = 140
            model = AutoModelForMultimodalLM.from_pretrained(
                self._model_id, trust_remote_code=True,
                dtype=torch.bfloat16, device_map="auto",
            )
            # Drop audio tower for memory
            if hasattr(model.model, "audio_tower"):
                del model.model.audio_tower
                if hasattr(model.model, "embed_audio"):
                    del model.model.embed_audio
                torch.cuda.empty_cache()
        else:
            from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                self._model_id, trust_remote_code=True,
                quantization_config=bnb, device_map="auto",
            )

        if self._adapter:
            from peft import PeftModel
            print(f"[TransformersClient] Loading adapter {self._adapter}...")
            model = PeftModel.from_pretrained(model, self._adapter)

        model.eval()
        self._model = model
        self._device = next(model.parameters()).device
        self._dtype = next(model.parameters()).dtype
        self._is_gemma = is_gemma
        print(f"[TransformersClient] Ready ({type(model).__name__}, "
              f"device={self._device}, dtype={self._dtype})")

    def create(self, *, model: str, messages: list, max_tokens: int = 2048,
               temperature: float = 0.1, **kwargs) -> Any:
        """Mimic openai.chat.completions.create()."""
        import torch
        self._ensure_loaded()

        # Convert OpenAI messages → model-native format
        native_msgs = self._convert_messages(messages)

        if self._is_gemma:
            inputs = self._processor.apply_chat_template(
                [native_msgs], tokenize=True, return_dict=True,
                return_tensors="pt", padding=True, add_generation_prompt=True,
            )
            inputs = inputs.to(self._device, dtype=self._dtype)
        else:
            # Qwen style
            from qwen_vl_utils import process_vision_info
            text = self._processor.apply_chat_template(
                native_msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(native_msgs)
            inputs = self._processor(
                text=[text], images=imgs, videos=vids,
                return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

        do_sample = temperature > 0.01
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        content = self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        return _FakeResponse(content)

    def _convert_messages(self, messages: list) -> list:
        """Convert OpenAI-format messages to native transformers format."""
        from PIL import Image
        import io
        native = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if isinstance(content, str):
                native.append({"role": role, "content": [
                    {"type": "text", "text": content}]})
                continue

            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        # base64 data URL → PIL → temp path or direct
                        b64_data = url.split(",", 1)[1]
                        img = Image.open(io.BytesIO(
                            base64.b64decode(b64_data))).convert("RGB")
                        if self._is_gemma:
                            parts.append({"type": "image", "image": img})
                        else:
                            parts.append({"type": "image", "image": img})
                    else:
                        if self._is_gemma:
                            parts.append({"type": "image", "url": url})
                        else:
                            parts.append({"type": "image", "image": url})
            native.append({"role": role, "content": parts})
        return native


class _FakeResponse:
    """Minimal OpenAI-compatible response object."""
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.reasoning = None


def check_local_endpoint(base_url: str, timeout: float = 3.0) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True, "ok"
    except Exception as e:
        return False, f"无法连接: {url} ({e})"


def fetch_local_model_name(base_url: str) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("data", [])
            if models:
                return models[0].get("id")
    except Exception:
        pass
    return None


def _retry_backoff(attempt: int, error: Exception) -> None:
    """Sleep with exponential backoff; extra-long wait for 429 rate limits."""
    err_str = str(error)
    if "429" in err_str:
        # Rate limit: wait 30/60/120s
        wait = 30 * (2 ** attempt)
        print(f"  [429 rate limit] waiting {wait}s...")
    else:
        wait = 2 ** (attempt + 1)
    time.sleep(wait)


def _extract_content(resp) -> tuple[Optional[str], Optional[str]]:
    """Extract content from API response. Returns (content, error)."""
    if not resp.choices:
        return None, "模型返回空 choices"
    choice = resp.choices[0]
    content = choice.message.content
    if not content:
        reasoning = getattr(choice.message, "reasoning", None)
        if reasoning:
            return reasoning.strip(), None
        return None, "模型返回空内容"
    return content.strip(), None


def _call_with_retry(client, model, messages, max_retries, temperature,
                     max_tokens) -> tuple[Optional[str], Optional[str]]:
    """Shared retry loop for all LLM calls."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            content, err = _extract_content(resp)
            if content is not None:
                return content, None
            # Retryable empty response
            if attempt < max_retries - 1:
                continue
            return None, err

        except Exception as e:
            if attempt < max_retries - 1:
                _retry_backoff(attempt, e)
            else:
                return None, str(e)

    return None, "所有重试均失败"


def call_llm(client: Any, model: str, system_prompt: str, user_prompt: str,
             image_path: str, max_retries: int = 3, temperature: float = 0.1,
             max_tokens: int = 2048) -> tuple[Optional[str], Optional[str]]:
    """Call LLM with a single image.

    Returns (content_string, error_string). Exactly one is None.
    """
    abs_path = (os.path.join(BASE_DIR, image_path)
                if not os.path.isabs(image_path) else image_path)
    if not os.path.exists(abs_path):
        return None, f"图片不存在: {abs_path}"

    b64 = encode_image_to_base64(abs_path)
    media_type = get_image_media_type(abs_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:{media_type};base64,{b64}",
                "detail": "high",
            }},
        ]},
    ]

    return _call_with_retry(client, model, messages, max_retries,
                            temperature, max_tokens)


def call_llm_raw(client: Any, model: str, messages: list,
                 max_retries: int = 3, temperature: float = 0.1,
                 max_tokens: int = 2048) -> tuple[Optional[str], Optional[str]]:
    """Call LLM with a pre-built messages list (for multi-image etc.).

    Returns (content_string, error_string). Exactly one is None.
    """
    return _call_with_retry(client, model, messages, max_retries,
                            temperature, max_tokens)


def build_image_part(image_path: str) -> tuple[Optional[dict], Optional[str]]:
    """Build an image_url content part. Returns (part_dict, error)."""
    abs_path = (os.path.join(BASE_DIR, image_path)
                if not os.path.isabs(image_path) else image_path)
    if not os.path.exists(abs_path):
        return None, f"图片不存在: {abs_path}"
    b64 = encode_image_to_base64(abs_path)
    media_type = get_image_media_type(abs_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}",
                      "detail": "high"},
    }, None
