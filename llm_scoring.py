#!/usr/bin/env python3
"""
llm_scoring.py
使用 OpenAI 兼容接口对猫图片进行 BCS (Body Condition Score) 评分。

使用方法:
    # 默认读取 .env
    python3 llm_scoring.py

可选参数:
    --provider NAME     提供商: openai/openrouter/local (默认: openai)
    --model MODEL       使用的模型 (默认随 provider 而变)
    --base-url URL      自定义 OpenAI 兼容接口地址
    --api-key KEY       覆盖环境变量中的 API key
    --dataset PATH      数据集 CSV 路径 (默认: dataset.csv)
    --output PATH       兼容参数，已弃用（不再输出明细 CSV）
    --max-retries N     最大重试次数 (默认: 3)
    --delay SECONDS     请求间隔秒数 (默认: 1.0)
    --request-timeout S 单次请求超时秒数 (默认: 60)
    --migrate-ai-responses-only 仅更新 ai_responses.csv 结构后退出

示例:
    # 1) 本地 vLLM/OpenAI 兼容服务（InternVL2-8B）
    python3 llm_scoring.py --provider local --model internvl2-8b --base-url http://127.0.0.1:8000/v1

    # 2) 本地 vLLM/OpenAI 兼容服务（Qwen2-VL-7B）
    python3 llm_scoring.py --provider local --model qwen2-vl-7b --base-url http://127.0.0.1:8000/v1

    # 3) OpenRouter（更多模型）
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
from typing import Any, Dict, Iterable, Optional

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

# BCS 评分的系统提示
SYSTEM_PROMPT = """You are an expert veterinary clinician specializing in feline medicine with extensive experience in Body Condition Scoring (BCS) of cats. You will be shown a photograph of a cat and asked to assess its Body Condition Score.

BCS uses a standardized 9-point scale (Association for Pet Obesity Prevention / veterinary consensus). Evaluate these key areas from the photo: ribs/chest, spine and back, hips, abdominal tuck (side view), and waist (top view). Use the following criteria:

Score 1 (Emaciated): Ribs project prominently with sharp edges; no fat layer; visible in short-haired cats. Spine and back project prominently. Hips clearly visible, sharp and protruding. Severe abdominal tuck, concave. Waist extremely narrow, hourglass.

Score 2 (Very Thin): Ribs easily felt with very minimal fat; visible on short-haired cats. Spine projects prominently with sharp edges. Hips visible and prominent. Very pronounced abdominal tuck, concave. Waist very narrow, extreme hourglass.

Score 3 (Thin): Ribs easily felt with slight fat covering; visible on short-haired cats. Spine projects prominently with sharp edges. Hips visible. Pronounced abdominal tuck, small abdominal fat. Visible waistline.

Score 4 (Moderately Thin): Ribs easily felt with light fat covering; not visible. Spine can be felt. Hips less bony, some fat; still visible in short-haired cats. Slight abdominal tuck, small fat pad. Noticeable waistline, not overly narrow.

Score 5 (Ideal): Ribs can be felt with slight fat covering. Spine smooth and easily felt, not sharp or bony. Hips rounded with fat layer. Slight abdominal tuck, small fat pad. Noticeable waistline behind ribs, gently curved hourglass from above.

Score 6 (Moderately Above Ideal): Ribs can still be felt but with more difficulty; not visible. Spine can still be felt. Hips can be felt, not visible; some fat. Minimal abdominal tuck, slight rounding. Waist not defined.

Score 7 (Overweight): Ribs difficult to feel under fat. Spine becoming difficult to feel. Hips felt with some pressure; not visible; fat deposits. No abdominal tuck, rounded abdomen. Waist barely visible or absent; body rounded from above.

Score 8 (Obese): Ribs very difficult to feel under fat. Spine very difficult to feel. Hips difficult to feel under fat, little definition. Pronounced rounding and distension of abdomen. No waist, broad and rounded from above. Notable fat deposits on body.

Score 9 (Severe Obesity): Ribs unable to feel. Spine difficult to feel. Hips heavily padded, significant fat; no definition. Large hanging abdominal fat pad. No waist, broad and rounded. Significant fat over lower spine, neck and chest.

Note: Long hair can obscure contours—infer fat coverage from visible landmarks (rib outline, waist, belly curve). Primordial pouch (loose skin under belly) is normal and not necessarily excess fat.

Please assess the cat's BCS based solely on visual cues from the photograph, using the above criteria.

You must respond in EXACTLY this JSON format:
{
    "bcs": <integer from 1-9>,
    "confidence": "<A, B, or C>",
    "second_score": <integer from 1-9 or null>,
    "reasoning": "<brief explanation>"
}

Confidence levels:
- A: You are >90% confident in your single BCS score
- B: Two adjacent scores are equally likely (provide second_score)
- C: Two scores are possible but you lean toward the primary one (provide second_score)

Important: Only provide whole numbers for BCS scores."""

USER_PROMPT = """Please assess the Body Condition Score (BCS) of this cat based on the photograph. Provide your assessment in the specified JSON format."""


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


def score_image(client: Any, image_path: str, model: str = "gpt-5.2",
                max_retries: int = 3) -> Dict[str, Any]:
    """
    使用 GP 对单张图片进行 BCS 评分。

    返回: dict with keys: bcs, confidence, second_score, reasoning
    """
    abs_path = os.path.join(BASE_DIR, image_path)
    if not os.path.exists(abs_path):
        return {"error": f"图片不存在: {abs_path}"}

    base64_image = encode_image_to_base64(abs_path)
    media_type = get_image_media_type(abs_path)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
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
                max_completion_tokens=300,
                temperature=0.1,  # 低温度以获得更一致的评分
            )

            content = response.choices[0].message.content.strip()

            # 尝试从回复中提取 JSON
            result = parse_response(content)
            if result:
                return result
            else:
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
    """加载数据集 CSV。"""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    args = parser.parse_args()

    load_dotenv(os.path.join(BASE_DIR, ".env"))

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

    print(f"已加载 {len(records)} 条记录")

    print(f"provider: {args.provider}")
    print(f"使用模型: {model_name}")
    if args.base_url:
        print(f"base_url: {args.base_url}")

    # 评分
    results = []
    errors = 0

    for record in progress_iter(records, total=len(records), desc="评分进度"):
        image_id = int(record["image_id"])
        image_path = record["image_path"]
        ground_truth = float(record["ground_truth"])

        result = score_image(client, image_path, model=model_name,
                             max_retries=args.max_retries)

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
