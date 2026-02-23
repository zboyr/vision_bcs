#!/usr/bin/env python3
"""
score_extra_cats.py
对 normal_cats/, overweight_cats/, underweight_cats/ 中的图片用 GPT-4o 打分,
追加到 scores.csv, 并将图片复制到 final/.

用法:
    python score_extra_cats.py
    python score_extra_cats.py --resume          # 断点续传
    python score_extra_cats.py --dry-run         # 仅检查，不调用 API
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

# 复用 score_cats_gpt4o 中的函数
from score_cats_gpt4o import (
    IMAGE_EXTENSIONS,
    get_openai_client,
    load_dotenv,
    load_existing_scores,
    save_scores_csv,
    score_all_images,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAT_DATA = os.path.join(BASE_DIR, "cat_data")
SCORES_CSV = os.path.join(CAT_DATA, "scores.csv")
FINAL_DIR = os.path.join(CAT_DATA, "final")

EXTRA_DIRS = ["normal_cats", "overweight_cats", "underweight_cats"]


def collect_extra_images(existing_files: set) -> list[tuple[str, str]]:
    """收集额外文件夹中的图片, 跳过已存在于 final/ 或 scores.csv 的文件。

    Returns:
        [(full_path, filename), ...]
    """
    seen = set()
    images = []

    for dirname in EXTRA_DIRS:
        dirpath = os.path.join(CAT_DATA, dirname)
        if not os.path.isdir(dirpath):
            print(f"  跳过不存在的目录: {dirpath}")
            continue

        for f in sorted(os.listdir(dirpath)):
            if Path(f).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if f in existing_files:
                print(f"  跳过已有: {f} (已在 scores.csv)")
                continue
            if f in seen:
                print(f"  跳过重名: {f} (已从其他文件夹收集)")
                continue
            seen.add(f)
            images.append((os.path.join(dirpath, f), f))

    return images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对额外猫图片文件夹 GPT-4o 评分并合并到 final/"
    )
    parser.add_argument("--model", default="gpt-4o", help="模型名称")
    parser.add_argument("--api-key", default=None, help="OpenAI API key")
    parser.add_argument("--base-url", default=None, help="自定义 API base URL")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.5, help="请求间隔秒数")
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不调用 API")
    args = parser.parse_args()

    load_dotenv(os.path.join(BASE_DIR, ".env"))

    # 加载已有评分
    existing_scores = load_existing_scores(SCORES_CSV)
    existing_files = set(existing_scores.keys())
    print(f"已有评分: {len(existing_files)} 条")

    # 收集待评分图片
    extra_images = collect_extra_images(existing_files)
    print(f"\n待评分图片: {len(extra_images)} 张")

    if not extra_images:
        print("没有需要评分的新图片")
        return 0

    for path, fname in extra_images[:10]:
        print(f"  {fname}  ({os.path.dirname(path).split(os.sep)[-1]})")
    if len(extra_images) > 10:
        print(f"  ... 共 {len(extra_images)} 张")

    if args.dry_run:
        print("\n[dry-run] 不调用 API")
        return 0

    # 创建客户端并评分
    client = get_openai_client(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
    )

    image_paths = [path for path, _ in extra_images]

    # 断点续传: 加载临时 CSV
    temp_csv = os.path.join(CAT_DATA, "scores_extra_temp.csv")
    resume_scores = {}
    if args.resume:
        resume_scores = load_existing_scores(temp_csv)
        if resume_scores:
            print(f"断点续传: 已有 {len(resume_scores)} 条临时评分")

    new_scores = score_all_images(
        client=client,
        image_paths=image_paths,
        model=args.model,
        max_retries=args.max_retries,
        delay=args.delay,
        existing_scores=resume_scores if args.resume else None,
    )

    # 保存临时 CSV (便于断点续传)
    save_scores_csv(new_scores, temp_csv)

    # 合并到主 scores.csv
    print(f"\n=== 合并到 {SCORES_CSV} ===")

    # 读取原有记录
    original_rows = []
    if os.path.exists(SCORES_CSV):
        with open(SCORES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                original_rows.append(row)

    # 读取新评分记录
    new_rows = []
    with open(temp_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_rows.append(row)

    # 合并写入
    all_rows = original_rows + new_rows
    fieldnames = [
        "file", "score", "logprob", "probability",
        "top_1", "top_1_prob", "top_2", "top_2_prob", "top_3", "top_3_prob",
        "error",
    ]
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"合并完成: {len(original_rows)} + {len(new_rows)} = {len(all_rows)} 条")

    # 复制图片到 final/
    print(f"\n=== 复制图片到 {FINAL_DIR} ===")
    os.makedirs(FINAL_DIR, exist_ok=True)
    copied = 0
    for src_path, fname in extra_images:
        dest = os.path.join(FINAL_DIR, fname)
        if os.path.exists(dest):
            print(f"  跳过已存在: {fname}")
            continue
        shutil.copy(src_path, dest)
        copied += 1

    print(f"复制完成: {copied} 张")

    # 清理临时文件
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
        print(f"已删除临时文件: {temp_csv}")

    # 统计
    valid_new = sum(1 for s in new_scores if s.get("score") is not None)
    print(f"\n=== 完成 ===")
    print(f"新评分: {valid_new}/{len(new_scores)} 成功")
    print(f"scores.csv 总记录: {len(all_rows)}")
    print(f"final/ 新增图片: {copied}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
