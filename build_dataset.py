#!/usr/bin/env python3
"""
build_dataset.py
解析 DataS5 中的 BCS 评分数据，构建标注数据集。

评分格式说明：
  - "5=5"  -> BCS主分=5, 置信度=A（100%确信单一分数）
  - "5=6"  -> BCS主分=5.5, 置信度=B（两个相邻分数同等可能）
  - "5>6"  -> BCS主分=5, 置信度=C（倾向前者，约75:25）

Ground Truth = Scorer A 和 Scorer B 的平均 BCS

重复对: ID 2~50, 5~47, 42~49, 25~48
"""

import csv
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# DataS5 原始数据（从 PDF 中手动转录）
RAW_DATA = """1 4=4 4=4 5=5 5=5 4=4 4=4 5>4 4=5 5=5
2 5=5 5=6 6>5 7>6 4=5 5=5 6=6 5=5 7=7
3 8>7 8=8 8>9 8=9 8>9 8=9 8=8 9=9 8=8
4 6=7 7>8 7>6 6>7 6=6 6=7 7=7 7=7 6=7
5 4>3 4=4 5=5 4=5 3=4 4=4 4=4 4=4 5=5
6 4=5 5=5 5=6 5=5 4=4 5=5 5=5 5=5 4=5
7 5=5 5=5 5=6 5=5 3=3 5=5 4>5 5=5 5=5
8 1>2 1>2 2=3 1=1 1=1 1=1 2=3 2=2 2=2
9 5=5 5=5 5>6 6=6 4=5 5=5 6>5 6=6 5=5
10 7=7 7=7 6>7 6=6 6>7 6=7 7=7 6=7 7=7
11 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9
12 8>7 8=8 8=9 8=8 7=8 8=9 9=9 8=8 8=8
13 9=9 9=9 9=9 9=9 8=8 9=9 9=9 9=9 9=9
14 9=9 9=9 9=9 9=9 8>9 9=9 9=9 9=9 9=9
15 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9
16 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9
17 5>4 4=5 4>5 4=5 5=5 5=5 5=5 4=4 4=5
18 3>2 3>4 4>5 3=3 3=3 3=3 3=3 3=3 3=3
19 5=5 4>5 6>5 6=6 5>4 5=5 5=5 3=4 6=6
20 5=5 5=6 5=6 4=5 4=4 5=6 5=6 4=5 6=6
21 7=8 7>8 7=8 7>6 7=7 8>9 7=7 6=7 7>8
22 1=2 1=1 2>3 1=1 1=1 1=1 1=1 1=1 1=2
23 6>7 6=7 7>8 6>7 6=7 8=8 6=6 7=7 7=8
24 7=8 7=7 8>7 8>7 7=7 9=9 8>9 7=7 7=7
25 4>5 3>4 4=5 3=4 4=5 3=4 5=5 4=4 5=5
26 6=6 5=6 7>6 6>7 5=5 5=6 5>6 6=6 6=6
27 5>6 5=6 5>6 5=5 5=5 6=6 6=6 5=5 5=5
28 5=5 4=4 5>6 3=4 3>2 6>5 4=5 3=4 4=5
29 6=6 6=6 7=7 6=6 4=4 5=5 5=5 4=5 6=6
30 5>6 4>5 5=5 4>3 3=4 4=4 5=5 4=4 4=4
31 5>4 4=4 5=5 4=5 3>2 4=4 5=5 4=4 5=6
32 4>3 3=3 4=5 4>3 3=3 4=4 3=3 3=3 4=4
33 8=9 9=9 9=9 9>8 7=8 9=9 9=9 9=9 8=9
34 8=8 6=6 8=8 7=7 7=7 9=9 8>7 7=8 7=7
35 7>8 6>7 7>6 7=7 6=7 8=8 6=7 7=8 6=7
36 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9
37 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9 9=9
38 9>8 7=8 7>8 6=7 7=7 9=9 9=9 8=8 7=8
39 6=7 8=8 6>7 7=7 6=7 7=8 7=7 7=7 7=7
40 8>7 7=8 8>7 6=7 8=8 9=9 8=8 7=8 8=8
41 7=7 5=6 6>7 6=6 5>6 6=7 6=6 6=6 5=5
42 8>7 6=7 7>6 5=6 6=6 7=8 7=7 5=6 6=7
43 4>5 4=4 5=5 5>4 3=3 4=4 3=3 4=4 4=4
44 9=9 9=9 9=9 9=9 8>9 9=9 9=9 9=9 9=9
45 6>5 4=4 7>6 5=5 5=5 5=5 5=6 6=6 5=5
46 8=8 7>8 8>7 7=7 7=7 7=7 8=8 7=7 8=8
47 4>5 4>5 5=5 4=5 4=5 5=5 5=5 5=5 4=5
48 4>5 3=3 5=5 3=4 3=3 5=5 4=5 4=4 5=5
49 8>7 8>7 7>6 5=6 5=6 7=8 7=7 6=6 6=7
50 5=5 5=5 7>6 7>6 4=5 5=6 6=6 5=5 7=7"""

SCORER_NAMES = ["Scorer_A", "Scorer_B", "Scorer_C", "Scorer_D", "Scorer_E",
                "Scorer_F", "Scorer_G", "Scorer_H", "Scorer_I"]

# 重复对
DUPLICATE_PAIRS = {2: 50, 5: 47, 42: 49, 25: 48}


def parse_score(score_str):
    """
    解析单个评分字符串，返回 (bcs, confidence, raw_scores)。

    格式:
      "5=5"  -> bcs=5.0, confidence="A", raw=[5,5]
      "5=6"  -> bcs=5.5, confidence="B", raw=[5,6]
      "5>6"  -> bcs=5.0, confidence="C", raw=[5,6]
    """
    score_str = score_str.strip()

    # 匹配 "数字=数字" 或 "数字>数字"
    match = re.match(r"(\d+)([=>])(\d+)", score_str)
    if not match:
        raise ValueError(f"无法解析评分: '{score_str}'")

    num1 = int(match.group(1))
    operator = match.group(2)
    num2 = int(match.group(3))

    if operator == "=" and num1 == num2:
        # 置信度 A: 100% 确信
        bcs = float(num1)
        confidence = "A"
    elif operator == "=":
        # 置信度 B: 两个分数同等可能
        bcs = (num1 + num2) / 2.0
        confidence = "B"
    elif operator == ">":
        # 置信度 C: 倾向前者 (75:25)
        bcs = float(num1)
        confidence = "C"
    else:
        raise ValueError(f"未知操作符: '{operator}' in '{score_str}'")

    return bcs, confidence, [num1, num2]


def find_image_path(cat_id):
    """查找猫图片的路径。"""
    for ext in ["jpeg", "jpg", "png"]:
        path = os.path.join("images", f"cat_{cat_id:02d}.{ext}")
        if os.path.exists(os.path.join(BASE_DIR, path)):
            return path
    return f"images/cat_{cat_id:02d}.jpeg"  # 默认


def classify_bcs(bcs):
    """根据 BCS 分类体重状态。"""
    if bcs <= 5:
        return "IW"   # Ideal Weight (正常)
    elif bcs <= 7:
        return "OW"   # Overweight (超重)
    else:
        return "OB"   # Obese (肥胖)


def main():
    print("解析 DataS5 评分数据...")

    all_records = []
    all_records_detailed = []

    for line in RAW_DATA.strip().split("\n"):
        parts = line.strip().split()
        image_id = int(parts[0])
        scores_raw = parts[1:]

        if len(scores_raw) != 9:
            print(f"  警告: Image ID {image_id} 有 {len(scores_raw)} 个评分，期望 9 个")
            continue

        # 解析每个评分者的分数
        parsed_scores = {}
        for i, scorer_name in enumerate(SCORER_NAMES):
            bcs, confidence, raw = parse_score(scores_raw[i])
            parsed_scores[scorer_name] = {
                "bcs": bcs,
                "confidence": confidence,
                "raw_scores": raw,
                "raw_string": scores_raw[i]
            }

        # Ground Truth = Scorer A 和 Scorer B 的平均 BCS
        scorer_a_bcs = parsed_scores["Scorer_A"]["bcs"]
        scorer_b_bcs = parsed_scores["Scorer_B"]["bcs"]
        ground_truth = (scorer_a_bcs + scorer_b_bcs) / 2.0

        # 判断是否为重复图片
        is_duplicate = image_id in DUPLICATE_PAIRS or image_id in DUPLICATE_PAIRS.values()
        duplicate_of = None
        if image_id in DUPLICATE_PAIRS:
            duplicate_of = DUPLICATE_PAIRS[image_id]
        else:
            for k, v in DUPLICATE_PAIRS.items():
                if v == image_id:
                    duplicate_of = k
                    break

        image_path = find_image_path(image_id)
        weight_class = classify_bcs(ground_truth)

        # CSV 记录
        record = {
            "image_id": image_id,
            "image_path": image_path,
            "scorer_a_bcs": scorer_a_bcs,
            "scorer_b_bcs": scorer_b_bcs,
            "ground_truth": ground_truth,
            "weight_class": weight_class,
            "is_duplicate": is_duplicate,
            "duplicate_of": duplicate_of if duplicate_of else "",
        }
        # 添加其他评分者的 BCS
        for scorer_name in SCORER_NAMES[2:]:  # C ~ I
            record[f"{scorer_name.lower()}_bcs"] = parsed_scores[scorer_name]["bcs"]

        all_records.append(record)

        # 详细 JSON 记录
        detailed = {
            "image_id": image_id,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "weight_class": weight_class,
            "is_duplicate": is_duplicate,
            "duplicate_of": duplicate_of,
            "scorers": parsed_scores
        }
        all_records_detailed.append(detailed)

    # 保存 CSV
    csv_path = os.path.join(BASE_DIR, "dataset.csv")
    csv_fields = ["image_id", "image_path", "scorer_a_bcs", "scorer_b_bcs",
                  "ground_truth", "weight_class", "is_duplicate", "duplicate_of"]
    csv_fields += [f"{name.lower()}_bcs" for name in SCORER_NAMES[2:]]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"已保存 CSV: {csv_path} ({len(all_records)} 条记录)")

    # 保存详细 JSON
    json_path = os.path.join(BASE_DIR, "dataset_annotated.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records_detailed, f, indent=2, ensure_ascii=False)

    print(f"已保存 JSON: {json_path}")

    # 打印统计摘要
    print("\n=== 数据集统计 ===")
    gt_values = [r["ground_truth"] for r in all_records]
    print(f"图片总数: {len(all_records)}")
    print(f"Ground Truth 范围: {min(gt_values):.1f} ~ {max(gt_values):.1f}")
    print(f"Ground Truth 均值: {sum(gt_values)/len(gt_values):.2f}")

    # 体重分类统计
    from collections import Counter
    class_counts = Counter(r["weight_class"] for r in all_records)
    print(f"体重分类: IW(正常)={class_counts.get('IW',0)}, "
          f"OW(超重)={class_counts.get('OW',0)}, "
          f"OB(肥胖)={class_counts.get('OB',0)}")

    # 重复对统计
    dup_count = sum(1 for r in all_records if r["is_duplicate"])
    print(f"重复图片: {dup_count} 张 (4对)")


if __name__ == "__main__":
    main()
