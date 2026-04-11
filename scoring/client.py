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

    raise ValueError(f"不支持的 provider: {provider}")


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
