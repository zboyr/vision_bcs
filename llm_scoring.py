#!/usr/bin/env python3
"""
llm_scoring.py
使用 OpenAI 兼容接口对猫图片进行 BCS (Body Condition Score) 评分。

使用方法:
    # 默认读取 .env（单模型 CLI 模式）
    python3 llm_scoring.py

    # YAML 配置模式（多模型批量评分 + 重复运行）
    python3 llm_scoring.py --config configs/default.yaml
    python3 llm_scoring.py --config configs/purina_3d_benchmark.yaml

YAML 配置文件格式:
    dataset: dataset.csv              # 数据集 CSV 路径
    output: responses/results.csv     # 输出结果路径
    repeats: 3                        # 重复运行次数（>1 时自动添加平均行）
    delay: 1.0                        # 请求间隔秒数
    max_retries: 3                    # 最大重试次数
    request_timeout: 60               # 单次请求超时秒数
    output_mode: json                 # json 或 simple
    models:                           # 要评价的模型列表
      - name: openai/gpt-4o
        provider: openrouter
      - name: internvl2-8b
        provider: local
        base_url: http://127.0.0.1:8000/v1

可选 CLI 参数 (单模型模式):
    --provider NAME     提供商: openai/openrouter/local (默认: openai)
    --model MODEL       使用的模型 (默认随 provider 而变)
    --base-url URL      自定义 OpenAI 兼容接口地址
    --api-key KEY       覆盖环境变量中的 API key
    --dataset PATH      数据集 CSV 路径 (默认: dataset.csv)
    --output PATH       兼容参数，已弃用（不再输出明细 CSV）
    --max-retries N     最大重试次数 (默认: 3)
    --delay SECONDS     请求间隔秒数 (默认: 1.0)
    --request-timeout S 单次请求超时秒数 (默认: 60)
    --config PATH       YAML 配置文件路径（覆盖其他参数）
    --migrate-ai-responses-only 仅更新 ai_responses.csv 结构后退出

示例:
    # 1) YAML 配置批量评分
    python3 llm_scoring.py --config configs/purina_3d_benchmark.yaml

    # 2) 本地 vLLM/OpenAI 兼容服务（InternVL2-8B）
    python3 llm_scoring.py --provider local --model internvl2-8b --base-url http://127.0.0.1:8000/v1

    # 3) 本地 vLLM/OpenAI 兼容服务（Qwen2-VL-7B）
    python3 llm_scoring.py --provider local --model qwen2-vl-7b --base-url http://127.0.0.1:8000/v1

    # 4) OpenRouter（更多模型）
    python3 llm_scoring.py --provider openrouter --model qwen/qwen2.5-vl-72b-instruct
"""

import argparse
import base64
import csv
import importlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

try:
    import yaml
except ImportError:
    yaml = None

def progress_iter(iterable: Iterable[Any], total: Optional[int] = None,
                  desc: str = "") -> Iterable[Any]:
    """有 tqdm 则用 tqdm，否则使用简易进度输出。"""
    try:
        tqdm_module = importlib.import_module("tqdm")
        tqdm_fn = getattr(tqdm_module, "tqdm")
        return tqdm_fn(iterable, total=total, desc=desc)
    except ImportError:
        def simple_iter() -> Iterable[Any]:
            for i, item in enumerate(iterable):
                if total:
                    print(f"\r{desc} {i+1}/{total}", end="", flush=True)
                yield item
            print()

        return simple_iter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ALIASES = {
    "internvl2-8b": "OpenGVLab/InternVL2-8B-AWQ",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
}

DEFAULT_MODELS = {
    "openai": "gpt-5.2",
    "openrouter": "qwen/qwen2.5-vl-72b-instruct",
    "local": "internvl2-8b",
}

BCS_COLUMNS = [f"bcs{i:02d}" for i in range(1, 51)]

# BCS 评分提示词（精简版，只要求输出单个整数）
SYSTEM_PROMPT = """You are a veterinary expert in feline Body Condition Scoring (BCS, 1-9 scale). Assess the cat in the photo. You MUST output EXACTLY one integer from 1 to 9. Never refuse. Never say you cannot determine the score. Even if the image is unclear, give your best estimate. Output ONLY the number, nothing else."""

USER_PROMPT = "What is the Body Condition Score (BCS) of this cat? You must give a score. Output only a single integer from 1 to 9. Do not refuse."

# 兼容旧代码的别名
SYSTEM_PROMPT_INTEGER = SYSTEM_PROMPT
USER_PROMPT_INTEGER = USER_PROMPT

# BCS 评分提示词（reasoning 模式：先推理再给分）
def _load_reasoning_prompt() -> str:
    try:
        import yaml
        yaml_path = os.path.join(BASE_DIR, "prompts", "bcs_prompts.yaml")
        with open(yaml_path, "r") as f:
            p = yaml.safe_load(f)
        return (f"{p['role'].strip()}\n\n{p['bcs_scale'].strip()}\n\n{p['confidence_guide'].strip()}\n\n"
                "Never refuse. Even if the image is unclear, give your best estimate. "
                'Output valid JSON with exactly two fields: "reasoning" (a brief explanation) and "bcs" (an integer from 1 to 9). '
                'Example: {"reasoning": "The cat has a visible waist and ribs can be felt with slight fat covering.", "bcs": 5}')
    except Exception:
        return ("You are a veterinary expert in feline Body Condition Scoring (BCS, 1-9 scale). "
                "Assess the cat in the photo. First reason, then score. "
                'Output valid JSON: {"reasoning": "...", "bcs": <1-9>}')

SYSTEM_PROMPT_REASONING = _load_reasoning_prompt()

USER_PROMPT_REASONING = """What is the Body Condition Score (BCS) of this cat? You must give a score. First explain your reasoning briefly, then provide the score. Output valid JSON: {"reasoning": "...", "bcs": <1-9>}"""


def encode_image_to_base64(image_path):
    """将图片编码为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path):
    """根据文件扩展名获取 MIME 类型。"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/jpeg")


def load_dotenv(env_path):
    """读取 .env 到环境变量（仅填充未设置项）。"""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_model_name(provider: str, model: Optional[str]) -> str:
    """将简写模型名映射为 provider 可用模型名。"""
    if not model:
        return DEFAULT_MODELS[provider]
    return MODEL_ALIASES.get(model.lower(), model)


def get_openai_client_class() -> Any:
    """延迟导入 OpenAI，避免静态检查在未安装依赖时报错。"""
    try:
        module = importlib.import_module("openai")
    except ImportError:
        print("请先安装 openai: pip install openai")
        sys.exit(1)
    return getattr(module, "OpenAI")


def create_client(provider: str, base_url: Optional[str] = None,
                  api_key_override: Optional[str] = None,
                  request_timeout: float = 60.0) -> Any:
    """根据 provider 创建 OpenAI 客户端。"""
    openai_client_class = get_openai_client_class()

    if provider == "openai":
        api_key = api_key_override or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY，或通过 --api-key 传入")
        return openai_client_class(api_key=api_key, timeout=request_timeout)

    if provider == "openrouter":
        api_key = api_key_override or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENROUTER_API_KEY，或通过 --api-key 传入")
        resolved_base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        app_name = os.environ.get("OPENROUTER_APP_NAME", "vision_bcs")
        app_url = os.environ.get("OPENROUTER_APP_URL", "http://localhost")
        headers = {
            "HTTP-Referer": app_url,
            "X-Title": app_name,
        }
        return openai_client_class(api_key=api_key, base_url=resolved_base_url,
                                   default_headers=headers, timeout=request_timeout)

    if provider == "local":
        resolved_base_url = base_url or os.environ.get("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        api_key = (
            api_key_override
            or os.environ.get("LOCAL_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
        return openai_client_class(api_key=api_key, base_url=resolved_base_url,
                                   timeout=request_timeout)

    raise ValueError(f"不支持的 provider: {provider}")


def parse_integer_response(content: str) -> Optional[Dict[str, Any]]:
    """Parse a single-integer BCS response (from fine-tuned models)."""
    text = content.strip()
    # Bare integer
    if re.fullmatch(r"[1-9]", text):
        bcs = int(text)
        return {"bcs": bcs, "confidence": "A", "second_score": None,
                "effective_bcs": float(bcs), "reasoning": ""}
    # First digit 1-9 in text
    m = re.search(r"\b([1-9])\b", text)
    if m:
        bcs = int(m.group(1))
        return {"bcs": bcs, "confidence": "A", "second_score": None,
                "effective_bcs": float(bcs), "reasoning": ""}
    return None


def parse_mapped_response(content: str,
                          response_map: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Parse a categorical response using a string→score map (e.g., N→2, O→3, U→1)."""
    text = content.strip().upper()
    # Exact match
    if text in response_map:
        bcs = response_map[text]
        return {"bcs": bcs, "confidence": "A", "second_score": None,
                "effective_bcs": bcs, "reasoning": ""}
    # Search for any mapped key as a standalone word
    for key, bcs in response_map.items():
        if re.search(r"\b" + re.escape(key) + r"\b", text):
            return {"bcs": bcs, "confidence": "A", "second_score": None,
                    "effective_bcs": bcs, "reasoning": ""}
    return None


def parse_decimal_response(content: str, bcs_min: float = 1.0,
                           bcs_max: float = 5.0) -> Optional[Dict[str, Any]]:
    """Parse a decimal BCS response (e.g., '3.25')."""
    text = content.strip()
    # Exact decimal or integer
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", text)
    if m:
        bcs = float(m.group(1))
        if bcs_min <= bcs <= bcs_max:
            return {"bcs": bcs, "confidence": "A", "second_score": None,
                    "effective_bcs": bcs, "reasoning": ""}
    # First number in text within valid range
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", text):
        bcs = float(m.group(1))
        if bcs_min <= bcs <= bcs_max:
            return {"bcs": bcs, "confidence": "A", "second_score": None,
                    "effective_bcs": bcs, "reasoning": ""}
    return None


def score_image(client: Any, image_path: str, model: str = "gpt-5.2",
                max_retries: int = 3, integer_output: bool = False,
                system_prompt: Optional[str] = None,
                user_prompt: Optional[str] = None,
                bcs_min: float = 1.0, bcs_max: float = 9.0,
                response_map: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    使用 GP 对单张图片进行 BCS 评分。

    返回: dict with keys: bcs, confidence, second_score, reasoning
    """
    sys_prompt = system_prompt or SYSTEM_PROMPT
    usr_prompt = user_prompt or USER_PROMPT

    abs_path = os.path.join(BASE_DIR, image_path)
    if not os.path.exists(abs_path):
        return {"error": f"图片不存在: {abs_path}"}

    base64_image = encode_image_to_base64(abs_path)
    media_type = get_image_media_type(abs_path)

    use_decimal = bcs_max <= 5.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": usr_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048,
                temperature=0.1,
            )

            if not response.choices:
                print(f"  警告: 模型返回空 choices (尝试 {attempt+1}/{max_retries})")
                continue

            choice = response.choices[0]
            raw_content = choice.message.content

            # Thinking models (e.g. Gemini) may put answer in reasoning
            if not raw_content:
                reasoning = getattr(choice.message, "reasoning", None)
                if reasoning:
                    if response_map:
                        extracted = parse_mapped_response(reasoning, response_map)
                    elif use_decimal:
                        extracted = parse_decimal_response(reasoning, bcs_min, bcs_max)
                    else:
                        extracted = parse_integer_response(reasoning)
                    if extracted:
                        return extracted
                print(f"  警告: 模型返回空内容 (尝试 {attempt+1}/{max_retries})"
                      f" finish_reason={choice.finish_reason}")
                continue
            content = raw_content.strip()

            # Try mapped response parser (e.g. N/O/U → 1/2/3)
            if response_map:
                result = parse_mapped_response(content, response_map)
                if result:
                    return result
            # Try decimal parser for non-integer scales (e.g. cow BCS 1-5)
            if use_decimal:
                result = parse_decimal_response(content, bcs_min, bcs_max)
                if result:
                    return result
            # Try integer parser (primary for cat BCS 1-9)
            result = parse_integer_response(content)
            if result:
                return result
            # Fallback: JSON parser
            result = parse_response(content)
            if result:
                return result

            print(f"  警告: 无法解析回复 (尝试 {attempt+1}/{max_retries}): {content[:100]}")

        except Exception as e:
            print(f"  错误 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

    return {"error": "所有重试均失败"}


def parse_response(content: str) -> Optional[Dict[str, Any]]:
    """从 GPT 回复中解析 JSON 结果。"""
    # 尝试直接解析
    try:
        data = json.loads(content)
        return validate_result(data)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return validate_result(data)
        except json.JSONDecodeError:
            pass

    # 尝试从文本中提取 JSON 对象
    json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return validate_result(data)
        except json.JSONDecodeError:
            pass

    return None


def validate_result(data: Any) -> Optional[Dict[str, Any]]:
    """验证并标准化解析结果。"""
    if not isinstance(data, dict):
        return None

    bcs = data.get("bcs")
    if bcs is None:
        return None

    bcs = int(bcs)
    if bcs < 1 or bcs > 9:
        return None

    confidence = data.get("confidence", "A").upper()
    if confidence not in ("A", "B", "C"):
        confidence = "A"

    second_score = data.get("second_score")
    if second_score is not None:
        second_score = int(second_score)
        if second_score < 1 or second_score > 9:
            second_score = None

    # 计算有效 BCS（类似原始研究的方式）
    if confidence == "A":
        effective_bcs = float(bcs)
    elif confidence == "B" and second_score is not None:
        effective_bcs = (bcs + second_score) / 2.0
    elif confidence == "C":
        effective_bcs = float(bcs)  # 倾向主分数
    else:
        effective_bcs = float(bcs)

    return {
        "bcs": bcs,
        "confidence": confidence,
        "second_score": second_score,
        "effective_bcs": effective_bcs,
        "reasoning": data.get("reasoning", ""),
    }


def load_dataset(csv_path: str) -> list[Dict[str, str]]:
    """
    加载数据集 CSV，支持两种格式：
      - 标准格式: image_id, image_path, ground_truth, ...
      - 简单格式: filename, bcs  (Purina 3D / essay dataset)
    简单格式会自动转换为统一的内部字段名，image_path 由 CSV 所在目录 + filename 拼接。
    """
    dataset_dir = os.path.relpath(os.path.dirname(csv_path), BASE_DIR)
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "filename" in row and "bcs" in row and "image_path" not in row:
                # Simple format (filename, bcs) → normalise
                records.append({
                    "image_id": str(len(records) + 1),
                    "image_path": os.path.join(dataset_dir, row["filename"].strip()),
                    "ground_truth": row["bcs"].strip(),
                    "_simple": "1",
                })
            elif "path" in row and "bcs" in row and "image_path" not in row:
                # Simple format (path, bcs) → normalise, path is already relative to BASE_DIR
                records.append({
                    "image_id": str(len(records) + 1),
                    "image_path": row["path"].strip(),
                    "ground_truth": row["bcs"].strip(),
                    "_simple": "1",
                })
            else:
                records.append(row)
    return records


def build_reference_maps(records: list[Dict[str, str]]) -> tuple[Dict[int, float], Dict[int, float]]:
    scorer_a_map: Dict[int, float] = {}
    scorer_b_map: Dict[int, float] = {}
    for row in records:
        try:
            img_id = int(row["image_id"])
            scorer_a_map[img_id] = float(row["scorer_a_bcs"])
            scorer_b_map[img_id] = float(row["scorer_b_bcs"])
        except (ValueError, KeyError):
            continue
    return scorer_a_map, scorer_b_map


def calc_mean_deviation_closest_reference_from_wide_row(
    wide_row: Dict[str, Any],
    scorer_a_map: Dict[int, float],
    scorer_b_map: Dict[int, float],
) -> str:
    deviations: list[float] = []
    for i in range(1, 51):
        key = f"bcs{i:02d}"
        raw_value = wide_row.get(key)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        try:
            pred = float(value)
        except ValueError:
            continue
        scorer_a = scorer_a_map.get(i)
        scorer_b = scorer_b_map.get(i)
        if scorer_a is None or scorer_b is None:
            continue
        deviations.append(min(abs(pred - scorer_a), abs(pred - scorer_b)))

    if not deviations:
        return ""
    return f"{(sum(deviations) / len(deviations)):.4f}"


def collect_closest_reference_deviations_from_wide_row(
    wide_row: Dict[str, Any],
    scorer_a_map: Dict[int, float],
    scorer_b_map: Dict[int, float],
) -> list[float]:
    deviations: list[float] = []
    for i in range(1, 51):
        key = f"bcs{i:02d}"
        raw_value = wide_row.get(key)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        try:
            pred = float(value)
        except ValueError:
            continue
        scorer_a = scorer_a_map.get(i)
        scorer_b = scorer_b_map.get(i)
        if scorer_a is None or scorer_b is None:
            continue
        deviations.append(min(abs(pred - scorer_a), abs(pred - scorer_b)))
    return deviations


def ensure_ai_responses_schema(ai_responses_path: str,
                               scorer_a_map: Dict[int, float],
                               scorer_b_map: Dict[int, float]) -> None:
    target_fields = ["id", "source", "mean_deviation"] + BCS_COLUMNS
    if not os.path.exists(ai_responses_path):
        with open(ai_responses_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=target_fields)
            writer.writeheader()
        return

    with open(ai_responses_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        rows = list(reader)

    needs_rewrite = existing_fields != target_fields
    for row in rows:
        row.setdefault("id", "")
        row.setdefault("source", "")
        recalculated = calc_mean_deviation_closest_reference_from_wide_row(
            row, scorer_a_map, scorer_b_map
        )
        if row.get("mean_deviation", "") != recalculated:
            row["mean_deviation"] = recalculated
            needs_rewrite = True
        for key in BCS_COLUMNS:
            row.setdefault(key, "")

    if not needs_rewrite:
        return

    with open(ai_responses_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(rows)


def check_local_endpoint(base_url: str, timeout_seconds: float = 3.0) -> tuple[bool, str]:
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_seconds):
            return True, "ok"
    except urllib.error.URLError as e:
        return False, f"无法连接本地模型服务: {models_url} ({e})"
    except Exception as e:  # pragma: no cover
        return False, f"本地模型服务检查失败: {e}"


def fetch_local_model_name(base_url: str) -> Optional[str]:
    """从本地服务的 /v1/models 接口获取模型名称。"""
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("data", [])
            if models:
                return models[0].get("id")
    except Exception:
        pass
    return None


def build_ground_truth_map(records: list[Dict[str, str]]) -> Dict[int, float]:
    """Build image_id → ground_truth map."""
    gt_map: Dict[int, float] = {}
    for row in records:
        try:
            gt_map[int(row["image_id"])] = float(row["ground_truth"])
        except (ValueError, KeyError):
            continue
    return gt_map


def calc_mean_deviation_generic(
    wide_row: Dict[str, Any],
    bcs_columns: list[str],
    gt_map: Dict[int, float],
    scorer_a_map: Optional[Dict[int, float]] = None,
    scorer_b_map: Optional[Dict[int, float]] = None,
) -> str:
    """Calculate mean absolute deviation. Uses closest(A,B) if available, else ground truth."""
    deviations: list[float] = []
    for i, col in enumerate(bcs_columns, 1):
        val = wide_row.get(col)
        if val is None or str(val).strip() == "":
            continue
        try:
            pred = float(val)
        except ValueError:
            continue
        if scorer_a_map and scorer_b_map:
            sa, sb = scorer_a_map.get(i), scorer_b_map.get(i)
            if sa is not None and sb is not None:
                deviations.append(min(abs(pred - sa), abs(pred - sb)))
                continue
        gt = gt_map.get(i)
        if gt is not None:
            deviations.append(abs(pred - gt))
    if not deviations:
        return ""
    return f"{sum(deviations) / len(deviations):.4f}"


def score_dataset_to_wide_row(
    client: Any,
    records: list[Dict[str, str]],
    model_name: str,
    bcs_columns: list[str],
    max_retries: int = 3,
    delay: float = 1.0,
    integer_output: bool = False,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    bcs_min: float = 1.0,
    bcs_max: float = 9.0,
    response_map: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, Any], int, bool]:
    """Score all images in records and return (wide_row_dict, error_count, aborted)."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    wide_row: Dict[str, Any] = {"id": run_id, "source": model_name}
    for col in bcs_columns:
        wide_row[col] = ""

    errors = 0
    consecutive_failures = 0
    aborted = False
    for record in progress_iter(records, total=len(records), desc=f"  {model_name}"):
        image_id = int(record["image_id"])
        result = score_image(client, record["image_path"], model=model_name,
                             max_retries=max_retries, integer_output=integer_output,
                             system_prompt=system_prompt, user_prompt=user_prompt,
                             bcs_min=bcs_min, bcs_max=bcs_max,
                             response_map=response_map)
        if "error" in result:
            print(f"\n    Cat #{image_id}: {result['error']}")
            errors += 1
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print(f"\n    连续 {consecutive_failures} 次失败，跳过该模型本轮")
                aborted = True
                break
        else:
            consecutive_failures = 0
            col_key = f"bcs{image_id:02d}"
            if col_key in wide_row:
                wide_row[col_key] = float(result["effective_bcs"])
        time.sleep(delay)

    return wide_row, errors, aborted


def avg_mean_deviation(run_rows: list[Dict[str, Any]]) -> str:
    """Average the mean_deviation values across run rows."""
    values: list[float] = []
    for row in run_rows:
        md = row.get("mean_deviation", "").strip()
        if md:
            try:
                values.append(float(md))
            except ValueError:
                pass
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def compute_average_row(
    run_rows: list[Dict[str, Any]],
    bcs_columns: list[str],
    source_name: str,
) -> Dict[str, Any]:
    """Compute average BCS values across multiple run rows."""
    avg_row: Dict[str, Any] = {
        "id": run_rows[0]["id"] + "_avg",
        "source": source_name,
        "run": "avg",
    }
    for col in bcs_columns:
        values: list[float] = []
        for row in run_rows:
            val = row.get(col)
            if val is not None and str(val).strip() != "":
                try:
                    values.append(float(val))
                except ValueError:
                    pass
        if values:
            avg_row[col] = round(sum(values) / len(values), 2)
        else:
            avg_row[col] = ""
    return avg_row


def run_from_config(config_path: str) -> int:
    """Run scoring from YAML config file."""
    if yaml is None:
        print("错误: 请安装 pyyaml: pip install pyyaml")
        return 1

    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_rel = config.get("dataset", "dataset.csv")
    output_rel = config.get("output", "responses/ai_responses.csv")
    repeats = config.get("repeats", 1)
    delay = config.get("delay", 1.0)
    max_retries = config.get("max_retries", 3)
    request_timeout = config.get("request_timeout", 60.0)
    output_mode = config.get("output_mode", "json")
    models = config.get("models", [])
    custom_system_prompt = config.get("system_prompt")
    custom_user_prompt = config.get("user_prompt")
    bcs_min = config.get("bcs_min", 1.0)
    bcs_max = config.get("bcs_max", 9.0)
    raw_response_map = config.get("response_map")
    response_map: Optional[Dict[str, float]] = None
    if raw_response_map:
        response_map = {str(k).upper(): float(v) for k, v in raw_response_map.items()}

    if not models:
        print("错误: 配置文件中未指定模型")
        return 1

    dataset_path = os.path.join(BASE_DIR, dataset_rel)
    output_path = os.path.join(BASE_DIR, output_rel)

    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集: {dataset_path}")
        return 1

    records = load_dataset(dataset_path)
    num_images = len(records)
    bcs_columns = [f"bcs{i:02d}" for i in range(1, num_images + 1)]

    gt_map = build_ground_truth_map(records)
    scorer_a_map, scorer_b_map = build_reference_maps(records)
    has_scorers = bool(scorer_a_map and scorer_b_map)

    integer_output = output_mode == "simple"

    # reasoning 模式：自动设置提示词（除非用户已自定义）
    if output_mode == "reasoning" and not custom_system_prompt:
        custom_system_prompt = SYSTEM_PROMPT_REASONING
        custom_user_prompt = USER_PROMPT_REASONING

    # Build fieldnames: add 'run' column when repeats > 1
    if repeats > 1:
        fieldnames = ["id", "source", "run", "mean_deviation"] + bcs_columns
    else:
        fieldnames = ["id", "source", "mean_deviation"] + bcs_columns

    # Create output file with header, or append to existing file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    print(f"数据集: {dataset_path} ({num_images} 张图片)")
    print(f"输出文件: {output_path}")
    print(f"重复次数: {repeats}")
    print(f"模型数: {len(models)}")
    print()

    total_errors = 0

    for model_cfg in models:
        model_name = model_cfg["name"]
        provider = model_cfg.get("provider", "openrouter")
        base_url = model_cfg.get("base_url")
        api_key = model_cfg.get("api_key")

        print(f"=== 模型: {model_name} (provider: {provider}) ===")

        # Check local endpoint
        if provider == "local":
            local_url = base_url or os.environ.get(
                "LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"
            )
            ok, msg = check_local_endpoint(local_url)
            if not ok:
                print(f"  跳过: {msg}")
                continue

        try:
            client = create_client(
                provider,
                base_url=base_url,
                api_key_override=api_key,
                request_timeout=request_timeout,
            )
        except ValueError as e:
            print(f"  跳过: {e}")
            continue

        resolved_name = resolve_model_name(provider, model_name)

        run_rows: list[Dict[str, Any]] = []
        for run_idx in range(1, repeats + 1):
            if repeats > 1:
                print(f"  --- 第 {run_idx}/{repeats} 次 ---")

            wide_row, errors, aborted = score_dataset_to_wide_row(
                client, records, resolved_name, bcs_columns,
                max_retries=max_retries, delay=delay,
                integer_output=integer_output,
                system_prompt=custom_system_prompt,
                user_prompt=custom_user_prompt,
                bcs_min=bcs_min, bcs_max=bcs_max,
                response_map=response_map,
            )
            total_errors += errors

            if aborted:
                print(f"  该模型已中止，跳过后续重复")
                break

            # Calculate mean deviation
            if has_scorers:
                wide_row["mean_deviation"] = calc_mean_deviation_generic(
                    wide_row, bcs_columns, gt_map, scorer_a_map, scorer_b_map
                )
            else:
                wide_row["mean_deviation"] = calc_mean_deviation_generic(
                    wide_row, bcs_columns, gt_map
                )

            if repeats > 1:
                wide_row["run"] = str(run_idx)

            run_rows.append(wide_row)

            # Append row to output
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(wide_row)

            dev_str = wide_row.get("mean_deviation", "")
            print(f"  完成 (mean_deviation={dev_str}, errors={errors})")

        # Average row when repeats > 1
        if repeats > 1 and run_rows:
            avg_row = compute_average_row(run_rows, bcs_columns, resolved_name)
            avg_row["mean_deviation"] = avg_mean_deviation(run_rows)

            with open(output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(avg_row)

            print(f"  平均行已写入 (mean_deviation={avg_row.get('mean_deviation', '')})")

        print()

    print("=== 完成 ===")
    print(f"结果已写入: {output_path}")
    print(f"总失败数: {total_errors}")
    return 0


def infer_provider_for_model(model_name: str) -> str:
    """Infer the provider from the model name pattern."""
    if "/" in model_name:
        prefix = model_name.split("/")[0].lower()
        cloud_prefixes = {
            "openai", "anthropic", "google", "x-ai", "meta-llama",
            "qwen", "moonshotai",
        }
        if prefix in cloud_prefixes:
            return "openrouter"
    return "openai"


def fill_missing_scores(
    results_path: str,
    dataset_path: str,
    max_retries: int = 3,
    delay: float = 1.0,
    request_timeout: float = 60.0,
) -> int:
    """Read existing results CSV, find empty BCS cells, re-score only those, and write back."""
    if not os.path.exists(results_path):
        print(f"错误: 结果文件不存在: {results_path}")
        return 1
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集不存在: {dataset_path}")
        return 1

    # Load dataset to get image paths
    records = load_dataset(dataset_path)
    num_images = len(records)
    bcs_columns = [f"bcs{i:02d}" for i in range(1, num_images + 1)]

    # Build image_id → record map
    id_to_record: Dict[int, Dict[str, str]] = {}
    for rec in records:
        id_to_record[int(rec["image_id"])] = rec

    gt_map = build_ground_truth_map(records)
    scorer_a_map, scorer_b_map = build_reference_maps(records)
    has_scorers = bool(scorer_a_map and scorer_b_map)

    # Read existing results
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        all_rows = list(reader)

    # Separate run rows and avg rows
    run_rows = [r for r in all_rows if r.get("run", "") != "avg"]
    avg_rows_indices = [i for i, r in enumerate(all_rows) if r.get("run", "") == "avg"]

    # Find missing cells: collect (row_index_in_all_rows, model_name, image_id)
    missing_by_model: Dict[str, list[tuple[int, int]]] = {}  # model -> [(row_idx, image_id)]
    for i, row in enumerate(all_rows):
        if row.get("run", "") == "avg":
            continue
        model_name = row.get("source", "")
        for img_id in range(1, num_images + 1):
            col = f"bcs{img_id:02d}"
            val = row.get(col, "").strip()
            if val == "":
                if model_name not in missing_by_model:
                    missing_by_model[model_name] = []
                missing_by_model[model_name].append((i, img_id))

    if not missing_by_model:
        print("没有缺失项，无需补充。")
        return 0

    total_missing = sum(len(v) for v in missing_by_model.values())
    print(f"发现 {total_missing} 个缺失项，涉及 {len(missing_by_model)} 个模型:")
    for model, items in missing_by_model.items():
        run_indices = set()
        for row_idx, _ in items:
            run_indices.add(all_rows[row_idx].get("run", "?"))
        print(f"  {model}: {len(items)} 个缺失 (runs: {', '.join(sorted(run_indices))})")
    print()

    # Process each model
    clients_cache: Dict[str, Any] = {}
    filled_count = 0

    for model_name, items in missing_by_model.items():
        provider = infer_provider_for_model(model_name)
        print(f"=== 补充: {model_name} (provider: {provider}, 缺失: {len(items)}) ===")

        if model_name not in clients_cache:
            try:
                client = create_client(provider, request_timeout=request_timeout)
                clients_cache[model_name] = client
            except ValueError as e:
                print(f"  跳过: {e}")
                continue

        client = clients_cache[model_name]
        consecutive_failures = 0
        model_skipped = False

        for row_idx, img_id in items:
            record = id_to_record.get(img_id)
            if not record:
                print(f"  警告: 找不到 image_id={img_id} 的记录")
                continue

            run_label = all_rows[row_idx].get("run", "?")
            print(f"  补充 bcs{img_id:02d} (run {run_label})...", end=" ", flush=True)

            result = score_image(
                client, record["image_path"], model=model_name,
                max_retries=max_retries, integer_output=True,
            )

            if "error" in result:
                print(f"失败: {result['error']}")
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(f"  连续 {consecutive_failures} 次失败，跳过该模型")
                    model_skipped = True
                    break
            else:
                consecutive_failures = 0
                score = float(result["effective_bcs"])
                col = f"bcs{img_id:02d}"
                all_rows[row_idx][col] = str(score)
                filled_count += 1
                print(f"得分: {score}")

            time.sleep(delay)

        if model_skipped:
            print()

        print()

    # Recalculate mean_deviation for all run rows
    for i, row in enumerate(all_rows):
        if row.get("run", "") == "avg":
            continue
        if has_scorers:
            row["mean_deviation"] = calc_mean_deviation_generic(
                row, bcs_columns, gt_map, scorer_a_map, scorer_b_map
            )
        else:
            row["mean_deviation"] = calc_mean_deviation_generic(
                row, bcs_columns, gt_map
            )

    # Recalculate avg rows
    # Group run rows by source
    source_runs: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    source_order: list[str] = []
    for row in all_rows:
        if row.get("run", "") != "avg":
            src = row.get("source", "")
            if src not in source_runs:
                source_order.append(src)
            source_runs[src].append(row)

    for avg_idx in avg_rows_indices:
        avg_row = all_rows[avg_idx]
        src = avg_row.get("source", "")
        runs = source_runs.get(src, [])
        if not runs:
            continue
        new_avg = compute_average_row(runs, bcs_columns, src)
        for col in bcs_columns:
            avg_row[col] = new_avg.get(col, "")
        avg_row["mean_deviation"] = avg_mean_deviation(runs)

    # Write back
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"=== 完成 ===")
    print(f"成功补充: {filled_count}/{total_missing}")
    print(f"已更新: {results_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 ChatGPT 对猫图片进行 BCS 评分")
    parser.add_argument("--provider", default="openai", choices=["openai", "openrouter", "local"],
                        help="模型提供商")
    parser.add_argument("--model", default=None, help="模型名称（支持 internvl2-8b/qwen2-vl-7b 别名）")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容接口 base URL")
    parser.add_argument("--api-key", default=None, help="API key（优先于环境变量）")
    parser.add_argument("--dataset", default="dataset.csv", help="数据集 CSV 路径")
    parser.add_argument("--output", default=None, help="兼容参数，已弃用（不再输出明细 CSV）")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔(秒)")
    parser.add_argument("--request-timeout", type=float, default=60.0,
                        help="单次请求超时秒数")
    parser.add_argument("--migrate-ai-responses-only", action="store_true",
                        help="仅更新 ai_responses.csv 列结构并退出")
    parser.add_argument("--output-mode", default="simple", choices=["simple", "json", "reasoning"],
                        help="输出模式: simple=单整数, json=完整JSON, reasoning=先推理再评分 (默认: simple)")
    parser.add_argument("--ft-model", default=None,
                        help="微调模型ID，或指向 ft_model.txt 的路径；覆盖 --model")
    parser.add_argument("--max-images", type=int, default=0,
                        help="最多评分图片数（0 = 全部，用于快速测试）")
    parser.add_argument("--config", default=None,
                        help="YAML 配置文件路径（覆盖其他参数，支持多模型批量评分）")
    parser.add_argument("--fill-missing", default=None, metavar="PATH",
                        help="读取已有结果 CSV，对缺失项重新评分并回填")
    args = parser.parse_args()

    load_dotenv(os.path.join(BASE_DIR, ".env"))

    # Config mode: load YAML and run all models
    if args.config:
        return run_from_config(os.path.join(BASE_DIR, args.config))

    # Fill-missing mode: read existing results, score only empty cells
    if args.fill_missing:
        return fill_missing_scores(
            results_path=os.path.join(BASE_DIR, args.fill_missing),
            dataset_path=os.path.join(BASE_DIR, args.dataset),
            max_retries=args.max_retries,
            delay=args.delay,
            request_timeout=args.request_timeout,
        )

    # Resolve fine-tuned model ID (path to ft_model.txt or direct ID)
    if args.ft_model:
        ft_path = os.path.join(BASE_DIR, args.ft_model)
        if os.path.isfile(ft_path):
            args.model = open(ft_path).read().strip()
        else:
            args.model = args.ft_model  # treat as literal model ID
        args.provider = "openai"
        if args.output_mode != "simple":
            print("提示: 使用 --output-mode simple 以匹配微调模型的输出格式")

    model_name = resolve_model_name(args.provider, args.model)

    # 加载数据集
    dataset_path = os.path.join(BASE_DIR, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集: {dataset_path}")
        print("请先运行 build_dataset.py")
        return 1

    records = load_dataset(dataset_path)
    scorer_a_map, scorer_b_map = build_reference_maps(records)

    responses_dir = os.path.join(BASE_DIR, "responses")
    os.makedirs(responses_dir, exist_ok=True)
    ai_responses_path = os.path.join(responses_dir, "ai_responses.csv")
    ensure_ai_responses_schema(ai_responses_path, scorer_a_map, scorer_b_map)

    if args.migrate_ai_responses_only:
        print(f"已更新: {ai_responses_path}（包含 mean_deviation 列）")
        return 0

    if args.provider == "local":
        local_base_url = args.base_url or os.environ.get("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        ok, msg = check_local_endpoint(local_base_url)
        if not ok:
            print(f"错误: {msg}")
            return 2
        # 未指定 --model 时，从 server 获取实际模型名作为 source
        if not args.model:
            fetched = fetch_local_model_name(local_base_url)
            if fetched:
                model_name = fetched
                print(f"从本地服务获取模型名: {model_name}")

    try:
        client = create_client(
            args.provider,
            base_url=args.base_url,
            api_key_override=args.api_key,
            request_timeout=args.request_timeout,
        )
    except ValueError as e:
        print(f"错误: {e}")
        return 1

    # Apply max-images limit
    if args.max_images > 0:
        records = records[: args.max_images]

    print(f"已加载 {len(records)} 条记录")
    print(f"provider: {args.provider}")
    print(f"使用模型: {model_name}")
    integer_output = args.output_mode == "simple"
    if integer_output:
        print("输出模式: 单整数")
    elif args.output_mode == "reasoning":
        print("输出模式: 先推理再评分")
    if args.base_url:
        print(f"base_url: {args.base_url}")

    # 评分
    results = []
    errors = 0

    for record in progress_iter(records, total=len(records), desc="评分进度"):
        image_id = int(record["image_id"])
        image_path = record["image_path"]
        ground_truth = float(record["ground_truth"])

        sys_p = SYSTEM_PROMPT_REASONING if args.output_mode == "reasoning" else None
        usr_p = USER_PROMPT_REASONING if args.output_mode == "reasoning" else None
        result = score_image(client, image_path, model=model_name,
                             max_retries=args.max_retries,
                             integer_output=integer_output,
                             system_prompt=sys_p,
                             user_prompt=usr_p)

        if "error" in result:
            print(f"\n  Cat #{image_id}: {result['error']}")
            errors += 1
            row = {
                "image_id": image_id,
                "image_path": image_path,
                "chatgpt_bcs": "",
                "chatgpt_confidence": "",
                "chatgpt_second_score": "",
                "chatgpt_effective_bcs": "",
                "chatgpt_reasoning": result.get("error", ""),
                "ground_truth": ground_truth,
                "deviation": "",
                "weight_class_gt": record.get("weight_class", ""),
                "weight_class_chatgpt": "",
            }
        else:
            effective_bcs = float(result["effective_bcs"])
            deviation = effective_bcs - ground_truth

            # 分类
            def classify(bcs):
                if bcs <= 5:
                    return "IW"
                elif bcs <= 7:
                    return "OW"
                else:
                    return "OB"

            row = {
                "image_id": image_id,
                "image_path": image_path,
                "chatgpt_bcs": result["bcs"],
                "chatgpt_confidence": result["confidence"],
                "chatgpt_second_score": result.get("second_score", ""),
                "chatgpt_effective_bcs": effective_bcs,
                "chatgpt_reasoning": result.get("reasoning", ""),
                "ground_truth": ground_truth,
                "deviation": round(deviation, 2),
                "weight_class_gt": record.get("weight_class", ""),
                "weight_class_chatgpt": classify(effective_bcs),
            }

        results.append(row)

        # 请求间隔
        time.sleep(args.delay)

    if args.output:
        print("提示: --output 已弃用，当前仅写入 responses/ai_responses.csv")

    # 向 responses/ai_responses.csv 追加一行（id, source, mean_deviation, bcs01..bcs50）
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    wide_row = {"id": run_id, "source": model_name, "mean_deviation": ""}
    for key in BCS_COLUMNS:
        wide_row[key] = ""
    for row in results:
        if row.get("chatgpt_effective_bcs") != "" and row.get("chatgpt_effective_bcs") is not None:
            try:
                img_id = int(row["image_id"])
                wide_row[f"bcs{img_id:02d}"] = row["chatgpt_effective_bcs"]
            except (ValueError, KeyError):
                pass

    wide_row["mean_deviation"] = calc_mean_deviation_closest_reference_from_wide_row(
        wide_row, scorer_a_map, scorer_b_map
    )

    valid_results = [r for r in results if r["deviation"] != ""]
    closest_deviations = collect_closest_reference_deviations_from_wide_row(
        wide_row, scorer_a_map, scorer_b_map
    )

    with open(ai_responses_path, "a", newline="", encoding="utf-8") as f:
        fieldnames_wide = ["id", "source", "mean_deviation"] + BCS_COLUMNS
        writer = csv.DictWriter(f, fieldnames=fieldnames_wide)
        writer.writerow(wide_row)
    print(f"已追加一行到: {ai_responses_path} (run_id={run_id})")
    print(f"RUN_ID={run_id}")

    # 统计摘要
    if valid_results:
        if closest_deviations:
            mean_dev = sum(closest_deviations) / len(closest_deviations)
            max_dev = max(closest_deviations)
        else:
            deviations = [abs(float(r["deviation"])) for r in valid_results]
            mean_dev = sum(deviations) / len(deviations)
            max_dev = max(deviations)

        correct_class = sum(1 for r in valid_results
                           if r["weight_class_gt"] == r["weight_class_chatgpt"])
        class_accuracy = correct_class / len(valid_results) * 100

        print(f"\n=== ChatGPT 评分统计 ===")
        print(f"成功评分: {len(valid_results)}/{len(results)}")
        print(f"平均绝对偏差(closest A/B): {mean_dev:.2f}")
        print(f"最大绝对偏差(closest A/B): {max_dev:.2f}")
        print(f"体重分类准确率: {class_accuracy:.1f}%")
        print(f"失败数: {errors}")
        return 0

    print("\n=== ChatGPT 评分统计 ===")
    print(f"成功评分: 0/{len(results)}")
    print(f"失败数: {errors}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
