#!/usr/bin/env python3
"""
score_cats_gpt4o.py
使用 GPT-4o 对猫图片进行 BCS 评分 (1-9 整数)，保留 logprobs，
迭代过滤低置信度图片直到剩余目标数量 (默认 1000)。

与 llm_scoring.py 的区别:
    - 输出仅为 1-9 单个整数 (max_tokens=1)
    - 启用 logprobs 记录每个评分的置信度
    - 迭代过滤: 删除 logprob 相对低的图片，重复直到达到目标数量

使用方法:
    python score_cats_gpt4o.py
    python score_cats_gpt4o.py --input-dir cat_data/filtered --target 1000
    python score_cats_gpt4o.py --logprob-threshold -0.5 --delay 0.5
    python score_cats_gpt4o.py --api-key sk-xxx --model gpt-4o

输出:
    cat_data/final/              最终筛选出的图片
    cat_data/scores.csv          评分结果 (含 logprobs)
    cat_data/scores_all.csv      所有评分记录 (含被过滤的)
"""

import argparse
import base64
import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "cat_data", "filtered")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "cat_data", "final")
DEFAULT_SCORES_CSV = os.path.join(BASE_DIR, "cat_data", "scores.csv")

# 图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# BCS 评分系统提示 (简化版，只要求输出整数)
SYSTEM_PROMPT = """You are an expert veterinary clinician specializing in feline Body Condition Scoring (BCS).

BCS uses a 9-point scale:
1 = Emaciated (ribs/spine prominent, no fat)
2 = Very Thin (ribs easily felt, minimal fat)
3 = Thin (ribs easily felt, slight fat covering)
4 = Moderately Thin (ribs easily felt, light fat)
5 = Ideal (ribs felt with slight fat, smooth spine)
6 = Moderately Above Ideal (ribs felt with difficulty)
7 = Overweight (ribs difficult to feel, rounded abdomen)
8 = Obese (ribs very difficult to feel, distended abdomen)
9 = Severe Obesity (ribs unable to feel, large fat deposits)

Respond with ONLY a single integer from 1 to 9. No other text."""

USER_PROMPT = "Rate this cat's Body Condition Score."


def load_dotenv(env_path: str) -> None:
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


def get_openai_client(api_key: Optional[str] = None,
                      base_url: Optional[str] = None,
                      timeout: float = 60.0) -> Any:
    """创建 OpenAI 客户端。"""
    try:
        import openai
    except ImportError:
        print("错误: 请先安装 openai: pip install openai")
        sys.exit(1)

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量，或通过 --api-key 传入")
        sys.exit(1)

    kwargs: Dict[str, Any] = {"api_key": resolved_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url

    return openai.OpenAI(**kwargs)


def encode_image_base64(image_path: str) -> str:
    """将图片编码为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(image_path: str) -> str:
    """根据扩展名获取 MIME 类型。"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


def score_single_image(
    client: Any,
    image_path: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    对单张图片进行 BCS 评分，返回分数和 logprob 信息。

    Returns:
        {
            "score": int (1-9),
            "logprob": float (chosen token logprob),
            "probability": float (0-1, exp(logprob)),
            "top_logprobs": {digit_str: prob, ...},
            "error": str or None,
        }
    """
    b64_image = encode_image_base64(image_path)
    mime_type = get_mime_type(image_path)

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
                                    "url": f"data:{mime_type};base64,{b64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                max_completion_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=9,
            )

            choice = response.choices[0]

            # 提取 logprob 数据
            if choice.logprobs and choice.logprobs.content:
                first_token = choice.logprobs.content[0]
                token_str = first_token.token.strip()
                token_logprob = first_token.logprob

                # 提取 top_logprobs 中的数字 token
                top_scores = {}
                for alt in first_token.top_logprobs:
                    t = alt.token.strip()
                    if t.isdigit() and 1 <= int(t) <= 9:
                        top_scores[t] = math.exp(alt.logprob)
            else:
                # logprobs 不可用（某些模型可能不支持）
                token_str = choice.message.content.strip() if choice.message.content else ""
                token_logprob = 0.0
                top_scores = {}

            # 验证 token 是有效数字
            if token_str.isdigit() and 1 <= int(token_str) <= 9:
                return {
                    "score": int(token_str),
                    "logprob": token_logprob,
                    "probability": math.exp(token_logprob),
                    "top_logprobs": top_scores,
                    "error": None,
                }
            else:
                # token 不是有效数字，尝试从 message content 提取
                content = choice.message.content.strip() if choice.message.content else ""
                # 尝试提取第一个数字
                for ch in content:
                    if ch.isdigit() and 1 <= int(ch) <= 9:
                        return {
                            "score": int(ch),
                            "logprob": token_logprob,
                            "probability": math.exp(token_logprob),
                            "top_logprobs": top_scores,
                            "error": None,
                        }

                print(f"  警告: 无效回复 '{content}' (尝试 {attempt+1}/{max_retries})")

        except Exception as e:
            print(f"  错误 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"  等待 {wait_time}s 重试...")
                time.sleep(wait_time)

    return {
        "score": None,
        "logprob": None,
        "probability": None,
        "top_logprobs": {},
        "error": "所有重试均失败",
    }


def score_all_images(
    client: Any,
    image_paths: List[str],
    model: str = "gpt-4o",
    max_retries: int = 3,
    delay: float = 0.5,
    existing_scores: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    对所有图片进行评分。支持断点续传。

    Args:
        client: OpenAI 客户端
        image_paths: 图片路径列表
        model: 模型名称
        max_retries: 重试次数
        delay: 请求间隔
        existing_scores: 已有评分 {filename: {score, logprob, ...}}

    Returns:
        评分结果列表
    """
    results = []
    errors = 0

    if tqdm:
        iterator = tqdm(image_paths, desc="GPT-4o 评分")
    else:
        iterator = image_paths

    for idx, img_path in enumerate(iterator):
        filename = os.path.basename(img_path)

        # 断点续传: 跳过已评分的
        if existing_scores and filename in existing_scores:
            results.append(existing_scores[filename])
            continue

        result = score_single_image(
            client, img_path, model=model, max_retries=max_retries
        )

        result["file"] = filename
        result["path"] = img_path
        results.append(result)

        if result["error"]:
            errors += 1

        # 请求间隔
        if delay > 0:
            time.sleep(delay)

        # 非 tqdm 进度
        if not tqdm and (idx + 1) % 50 == 0:
            print(f"\r  评分进度: {idx+1}/{len(image_paths)} | 错误: {errors}",
                  end="", flush=True)

    if not tqdm:
        print()

    valid = sum(1 for r in results if r.get("score") is not None)
    print(f"评分完成: {valid}/{len(results)} 成功, {errors} 失败")

    return results


def save_scores_csv(scores: List[Dict], csv_path: str) -> None:
    """保存评分结果到 CSV。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "file", "score", "logprob", "probability",
        "top_1", "top_1_prob", "top_2", "top_2_prob", "top_3", "top_3_prob",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in scores:
            # 提取 top-3 scores
            top_sorted = sorted(
                s.get("top_logprobs", {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            row = {
                "file": s.get("file", ""),
                "score": s.get("score", ""),
                "logprob": f"{s['logprob']:.6f}" if s.get("logprob") is not None else "",
                "probability": f"{s['probability']:.6f}" if s.get("probability") is not None else "",
                "top_1": top_sorted[0][0] if len(top_sorted) > 0 else "",
                "top_1_prob": f"{top_sorted[0][1]:.6f}" if len(top_sorted) > 0 else "",
                "top_2": top_sorted[1][0] if len(top_sorted) > 1 else "",
                "top_2_prob": f"{top_sorted[1][1]:.6f}" if len(top_sorted) > 1 else "",
                "top_3": top_sorted[2][0] if len(top_sorted) > 2 else "",
                "top_3_prob": f"{top_sorted[2][1]:.6f}" if len(top_sorted) > 2 else "",
                "error": s.get("error", ""),
            }
            writer.writerow(row)


def load_existing_scores(csv_path: str) -> Dict[str, Dict]:
    """加载已有评分数据。"""
    if not os.path.exists(csv_path):
        return {}

    scores = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("file", "")
            if not filename:
                continue

            score = None
            logprob = None
            probability = None

            if row.get("score"):
                try:
                    score = int(row["score"])
                except ValueError:
                    pass
            if row.get("logprob"):
                try:
                    logprob = float(row["logprob"])
                except ValueError:
                    pass
            if row.get("probability"):
                try:
                    probability = float(row["probability"])
                except ValueError:
                    pass

            scores[filename] = {
                "file": filename,
                "score": score,
                "logprob": logprob,
                "probability": probability,
                "top_logprobs": {},
                "error": row.get("error", "") or None,
            }

    return scores


def iterative_filter(
    scores: List[Dict],
    target: int,
    min_logprob: Optional[float] = None,
) -> List[Dict]:
    """
    迭代过滤: 删除 logprob 最低的图片，直到剩余 target 张。

    策略:
      1. 先删除评分失败的
      2. 如果设了 min_logprob 阈值，删除低于阈值的
      3. 按 logprob 从高到低排序，保留 top-target

    Args:
        scores: 全部评分结果
        target: 目标保留数量
        min_logprob: 可选的最低 logprob 阈值

    Returns:
        过滤后的评分列表
    """
    # Step 1: 删除评分失败的
    valid = [s for s in scores if s.get("score") is not None and s.get("logprob") is not None]
    removed_errors = len(scores) - len(valid)
    if removed_errors > 0:
        print(f"  移除评分失败: {removed_errors} 张")

    # Step 2: 如果设了阈值，先过滤
    if min_logprob is not None:
        before = len(valid)
        valid = [s for s in valid if s["logprob"] >= min_logprob]
        removed_threshold = before - len(valid)
        if removed_threshold > 0:
            print(f"  移除低于阈值 (logprob < {min_logprob:.2f}): {removed_threshold} 张")

    # Step 3: 按 logprob 排序，保留 top-target
    if len(valid) > target:
        valid.sort(key=lambda x: x["logprob"], reverse=True)
        removed_ranking = len(valid) - target
        valid = valid[:target]
        print(f"  按 logprob 排序保留 top-{target}: 移除 {removed_ranking} 张")

    print(f"  最终保留: {len(valid)} 张")

    if valid:
        logprobs = [s["logprob"] for s in valid]
        probs = [s["probability"] for s in valid]
        print(f"  logprob 范围: [{min(logprobs):.4f}, {max(logprobs):.4f}]")
        print(f"  probability 范围: [{min(probs):.4f}, {max(probs):.4f}]")
        print(f"  probability 均值: {sum(probs)/len(probs):.4f}")

    return valid


def copy_final_images(
    scores: List[Dict],
    source_dir: str,
    output_dir: str,
) -> int:
    """将最终选中的图片复制到输出目录。"""
    os.makedirs(output_dir, exist_ok=True)

    copied = 0
    for s in scores:
        filename = s["file"]
        src = os.path.join(source_dir, filename)
        if not os.path.exists(src):
            # 尝试从 path 字段
            src = s.get("path", src)

        if os.path.exists(src):
            dest = os.path.join(output_dir, filename)
            shutil.copy2(src, dest)
            copied += 1
        else:
            print(f"  警告: 找不到源文件: {src}")

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 GPT-4o 对猫图片 BCS 评分 + logprob 迭代过滤"
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT,
                        help=f"输入图片目录 (默认: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"最终图片输出目录 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--scores-csv", default=DEFAULT_SCORES_CSV,
                        help=f"评分 CSV 路径 (默认: {DEFAULT_SCORES_CSV})")
    parser.add_argument("--model", default="gpt-4o",
                        help="模型名称 (默认: gpt-4o)")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (优先于环境变量)")
    parser.add_argument("--base-url", default=None,
                        help="自定义 API base URL")
    parser.add_argument("--target", type=int, default=1000,
                        help="目标保留图片数量 (默认: 1000)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="单张图片最大重试次数 (默认: 3)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="请求间隔秒数 (默认: 0.5)")
    parser.add_argument("--request-timeout", type=float, default=60.0,
                        help="单次请求超时秒数 (默认: 60)")
    parser.add_argument("--logprob-threshold", type=float, default=None,
                        help="最低 logprob 阈值 (例如 -0.5，低于此值直接过滤)")
    parser.add_argument("--resume", action="store_true",
                        help="从已有 scores.csv 断点续传")
    parser.add_argument("--filter-only", action="store_true",
                        help="仅执行过滤步骤 (不重新评分，使用已有 CSV)")
    args = parser.parse_args()

    # 加载 .env
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        print("请先运行 yolo_filter_cats.py 筛选图片")
        return 1

    # 收集图片
    input_path = Path(input_dir)
    image_files = sorted([
        str(f) for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"错误: 在 {input_dir} 中未找到图片")
        return 1

    print(f"找到 {len(image_files)} 张待评分图片")
    print(f"目标保留: {args.target} 张")
    print(f"模型: {args.model}")

    # 加载已有评分 (断点续传)
    existing_scores = {}
    if args.resume or args.filter_only:
        existing_scores = load_existing_scores(args.scores_csv)
        if existing_scores:
            print(f"加载已有评分: {len(existing_scores)} 条")

    if args.filter_only:
        # 仅过滤模式
        all_scores = list(existing_scores.values())
        if not all_scores:
            print("错误: --filter-only 需要已有评分数据")
            return 1
    else:
        # 评分模式
        if len(image_files) < args.target:
            print(f"警告: 输入图片 ({len(image_files)}) 少于目标 ({args.target})")
            print("将对所有图片评分，但可能无法达到目标数量")

        client = get_openai_client(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.request_timeout,
        )

        # 评分所有图片
        all_scores = score_all_images(
            client=client,
            image_paths=image_files,
            model=args.model,
            max_retries=args.max_retries,
            delay=args.delay,
            existing_scores=existing_scores if args.resume else None,
        )

        # 保存全部评分 (含失败的)
        all_csv = args.scores_csv.replace(".csv", "_all.csv")
        save_scores_csv(all_scores, all_csv)
        print(f"全部评分已保存: {all_csv}")

    # 迭代过滤
    print(f"\n=== 迭代过滤 (目标: {args.target} 张) ===")
    final_scores = iterative_filter(
        scores=all_scores,
        target=args.target,
        min_logprob=args.logprob_threshold,
    )

    if not final_scores:
        print("错误: 过滤后没有图片保留")
        return 2

    # 保存最终评分
    save_scores_csv(final_scores, args.scores_csv)
    print(f"最终评分已保存: {args.scores_csv}")

    # 复制最终图片
    copied = copy_final_images(final_scores, input_dir, args.output_dir)
    print(f"\n已复制 {copied} 张图片到: {args.output_dir}")

    # 评分分布统计
    score_counts = {}
    for s in final_scores:
        score = s["score"]
        score_counts[score] = score_counts.get(score, 0) + 1

    print(f"\n=== 最终数据集统计 ===")
    print(f"  总图片数: {len(final_scores)}")
    print(f"  BCS 评分分布:")
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        bar = "█" * (count // 5) if count > 0 else ""
        print(f"    BCS {score}: {count:4d} {bar}")

    avg_score = sum(s["score"] for s in final_scores) / len(final_scores)
    avg_prob = sum(s["probability"] for s in final_scores) / len(final_scores)
    print(f"  平均 BCS: {avg_score:.2f}")
    print(f"  平均置信度: {avg_prob:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
