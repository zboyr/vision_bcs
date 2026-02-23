#!/usr/bin/env python3
"""重试 scores.csv 中评分失败的图片，更新 scores.csv，然后合并文件夹。"""

import csv
import os
import shutil
import sys
from pathlib import Path

from score_cats_gpt4o import (
    IMAGE_EXTENSIONS,
    get_openai_client,
    load_dotenv,
    score_single_image,
    save_scores_csv,
)
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAT_DATA = os.path.join(BASE_DIR, "cat_data")
SCORES_CSV = os.path.join(CAT_DATA, "scores.csv")
FINAL_DIR = os.path.join(CAT_DATA, "final")
EXTRA_DIRS = ["normal_cats", "overweight_cats", "underweight_cats"]


def main():
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    # 读取所有记录，找出失败的
    all_rows = []
    failed_indices = []
    with open(SCORES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            all_rows.append(row)
            if not row.get("score") or row["score"] == "":
                failed_indices.append(i)

    print(f"总记录: {len(all_rows)}, 失败: {len(failed_indices)}")

    if not failed_indices:
        print("没有需要重试的记录")
        return

    # 建立文件名到路径的映射
    file_to_path = {}
    for dirname in EXTRA_DIRS:
        dirpath = os.path.join(CAT_DATA, dirname)
        if not os.path.isdir(dirpath):
            continue
        for f in os.listdir(dirpath):
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                if f not in file_to_path:
                    file_to_path[f] = os.path.join(dirpath, f)
    # Also check final/
    for f in os.listdir(FINAL_DIR):
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
            if f not in file_to_path:
                file_to_path[f] = os.path.join(FINAL_DIR, f)

    client = get_openai_client()

    retried = 0
    success = 0
    for idx in failed_indices:
        filename = all_rows[idx]["file"]
        img_path = file_to_path.get(filename)
        if not img_path:
            print(f"  找不到文件: {filename}")
            continue

        print(f"  重试: {filename} ... ", end="", flush=True)
        result = score_single_image(client, img_path, model="gpt-4o", max_retries=3)
        retried += 1

        if result["score"] is not None:
            # 更新行
            top_sorted = sorted(
                result.get("top_logprobs", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            all_rows[idx]["score"] = str(result["score"])
            all_rows[idx]["logprob"] = f"{result['logprob']:.6f}" if result.get("logprob") is not None else ""
            all_rows[idx]["probability"] = f"{result['probability']:.6f}" if result.get("probability") is not None else ""
            all_rows[idx]["top_1"] = top_sorted[0][0] if len(top_sorted) > 0 else ""
            all_rows[idx]["top_1_prob"] = f"{top_sorted[0][1]:.6f}" if len(top_sorted) > 0 else ""
            all_rows[idx]["top_2"] = top_sorted[1][0] if len(top_sorted) > 1 else ""
            all_rows[idx]["top_2_prob"] = f"{top_sorted[1][1]:.6f}" if len(top_sorted) > 1 else ""
            all_rows[idx]["top_3"] = top_sorted[2][0] if len(top_sorted) > 2 else ""
            all_rows[idx]["top_3_prob"] = f"{top_sorted[2][1]:.6f}" if len(top_sorted) > 2 else ""
            all_rows[idx]["error"] = ""
            success += 1
            print(f"BCS={result['score']}")
        else:
            print(f"仍然失败")

    # 写回 CSV
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

    print(f"\n重试完成: {success}/{retried} 成功")

    # 统计最终失败数
    still_failed = sum(1 for r in all_rows if not r.get("score") or r["score"] == "")
    print(f"仍然失败: {still_failed}")


if __name__ == "__main__":
    main()
