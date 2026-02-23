#!/usr/bin/env python3
"""
merge_and_filter.py
合并两个评分 CSV，按 logprob 排序保留 top-N，复制最终图片。

使用方法:
    python merge_and_filter.py
    python merge_and_filter.py --target 1000
"""

import argparse
import csv
import math
import os
import shutil
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_scores(csv_path: str, source_dir: str) -> list:
    """加载评分 CSV，返回 list of dicts。"""
    if not os.path.exists(csv_path):
        print(f"警告: {csv_path} 不存在")
        return []

    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("score") or not row.get("logprob"):
                continue
            try:
                score = int(row["score"])
                logprob = float(row["logprob"])
                probability = float(row.get("probability", 0))
            except (ValueError, TypeError):
                continue

            if not (1 <= score <= 9):
                continue

            scores.append({
                "file": row["file"],
                "source_dir": source_dir,
                "score": score,
                "logprob": logprob,
                "probability": probability,
                "top_1": row.get("top_1", ""),
                "top_1_prob": row.get("top_1_prob", ""),
                "top_2": row.get("top_2", ""),
                "top_2_prob": row.get("top_2_prob", ""),
                "top_3": row.get("top_3", ""),
                "top_3_prob": row.get("top_3_prob", ""),
            })

    return scores


def main():
    parser = argparse.ArgumentParser(description="合并评分数据并过滤到目标数量")
    parser.add_argument("--target", type=int, default=1000, help="目标保留数量")
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "cat_data", "final_merged"))
    parser.add_argument("--output-csv", default=os.path.join(BASE_DIR, "cat_data", "scores_merged.csv"))
    args = parser.parse_args()

    # 加载两个数据集
    coco_scores = load_scores(
        os.path.join(BASE_DIR, "cat_data", "scores.csv"),
        os.path.join(BASE_DIR, "cat_data", "cropped"),
    )
    extra_scores = load_scores(
        os.path.join(BASE_DIR, "cat_data", "scores_extra.csv"),
        os.path.join(BASE_DIR, "cat_data", "cropped_extra"),
    )

    print(f"COCO 评分: {len(coco_scores)} 条")
    print(f"Extra 评分: {len(extra_scores)} 条")

    # 合并 (检查重名)
    seen_files = set()
    all_scores = []
    for s in coco_scores:
        if s["file"] not in seen_files:
            seen_files.add(s["file"])
            all_scores.append(s)
    for s in extra_scores:
        if s["file"] not in seen_files:
            seen_files.add(s["file"])
            all_scores.append(s)
        else:
            # 如果有重名，加前缀
            new_name = "extra_" + s["file"]
            s["file"] = new_name
            seen_files.add(new_name)
            all_scores.append(s)

    print(f"合并后总计: {len(all_scores)} 条 (去重后)")

    # 按 logprob 从高到低排序
    all_scores.sort(key=lambda x: x["logprob"], reverse=True)

    # 保留 top-target
    if len(all_scores) > args.target:
        cutoff_logprob = all_scores[args.target - 1]["logprob"]
        final = all_scores[:args.target]
        print(f"按 logprob 排序保留 top-{args.target}")
        print(f"  截断 logprob: {cutoff_logprob:.4f}")
    else:
        final = all_scores
        print(f"总数 ({len(all_scores)}) 不足 {args.target}，保留全部")

    # 复制最终图片
    os.makedirs(args.output_dir, exist_ok=True)
    copied = 0
    for s in final:
        src = os.path.join(s["source_dir"], s["file"].replace("extra_", "", 1) if s["file"].startswith("extra_") else s["file"])
        if not os.path.exists(src):
            # 尝试原始文件名
            src = os.path.join(s["source_dir"], s["file"])
        dest = os.path.join(args.output_dir, s["file"])
        if os.path.exists(src):
            shutil.copy2(src, dest)
            copied += 1
        else:
            print(f"  警告: 找不到 {src}")

    print(f"\n已复制 {copied} 张图片到: {args.output_dir}")

    # 保存合并评分 CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = [
        "file", "score", "logprob", "probability",
        "top_1", "top_1_prob", "top_2", "top_2_prob", "top_3", "top_3_prob",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in final:
            writer.writerow({
                "file": s["file"],
                "score": s["score"],
                "logprob": f"{s['logprob']:.6f}",
                "probability": f"{s['probability']:.6f}",
                "top_1": s.get("top_1", ""),
                "top_1_prob": s.get("top_1_prob", ""),
                "top_2": s.get("top_2", ""),
                "top_2_prob": s.get("top_2_prob", ""),
                "top_3": s.get("top_3", ""),
                "top_3_prob": s.get("top_3_prob", ""),
            })
    print(f"评分 CSV: {args.output_csv}")

    # 统计
    score_counts = Counter(s["score"] for s in final)
    logprobs = [s["logprob"] for s in final]
    probs = [s["probability"] for s in final]

    print(f"\n=== 最终数据集统计 ===")
    print(f"  总图片数: {len(final)}")
    print(f"  BCS 评分分布:")
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        bar = "█" * (count // 5) if count > 0 else ""
        print(f"    BCS {score}: {count:4d} {bar}")

    print(f"\n  logprob 范围: [{min(logprobs):.4f}, {max(logprobs):.4f}]")
    print(f"  probability 均值: {sum(probs)/len(probs):.4f}")

    avg_score = sum(s["score"] for s in final) / len(final)
    print(f"  平均 BCS: {avg_score:.2f}")

    # 来源统计
    coco_count = sum(1 for s in final if "cropped_extra" not in s["source_dir"])
    extra_count = sum(1 for s in final if "cropped_extra" in s["source_dir"])
    print(f"\n  来源: COCO={coco_count}, Extra(HF+Oxford)={extra_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
