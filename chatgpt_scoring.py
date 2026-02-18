#!/usr/bin/env python3
"""
chatgpt_scoring.py
使用 OpenAI GPT-4o API 对猫图片进行 BCS (Body Condition Score) 评分。

使用方法:
    export OPENAI_API_KEY="your-api-key"
    python3 chatgpt_scoring.py

可选参数:
    --model MODEL       使用的模型 (默认: gpt-5.2)
    --dataset PATH      数据集 CSV 路径 (默认: dataset.csv)
    --output PATH       输出 CSV 路径 (默认: chatgpt_results.csv)
    --max-retries N     最大重试次数 (默认: 3)
    --delay SECONDS     请求间隔秒数 (默认: 1.0)
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai: pip install openai")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，使用简单的替代
    def tqdm(iterable, **kwargs):
        total = kwargs.get("total", None)
        desc = kwargs.get("desc", "")
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc} {i+1}/{total}", end="", flush=True)
            yield item
        print()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


def score_image(client, image_path, model="gpt-5.2", max_retries=3):
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


def parse_response(content):
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


def validate_result(data):
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


def load_dataset(csv_path):
    """加载数据集 CSV。"""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description="使用 ChatGPT 对猫图片进行 BCS 评分")
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI 模型名称")
    parser.add_argument("--dataset", default="dataset.csv", help="数据集 CSV 路径")
    parser.add_argument("--output", default="chatgpt_results.csv", help="输出 CSV 路径")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔(秒)")
    args = parser.parse_args()

    # 检查 API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 OPENAI_API_KEY")
        print("  export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # 加载数据集
    dataset_path = os.path.join(BASE_DIR, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集: {dataset_path}")
        print("请先运行 build_dataset.py")
        sys.exit(1)

    records = load_dataset(dataset_path)
    print(f"已加载 {len(records)} 条记录")

    # 初始化 OpenAI 客户端
    client = OpenAI(api_key=api_key)
    print(f"使用模型: {args.model}")

    # 评分
    results = []
    errors = 0

    for record in tqdm(records, total=len(records), desc="评分进度"):
        image_id = int(record["image_id"])
        image_path = record["image_path"]
        ground_truth = float(record["ground_truth"])

        result = score_image(client, image_path, model=args.model,
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
            effective_bcs = result["effective_bcs"]
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

    # 保存结果（原有 per-image 明细）
    output_path = os.path.join(BASE_DIR, args.output)
    fieldnames = ["image_id", "image_path", "chatgpt_bcs", "chatgpt_confidence",
                  "chatgpt_second_score", "chatgpt_effective_bcs", "chatgpt_reasoning",
                  "ground_truth", "deviation", "weight_class_gt", "weight_class_chatgpt"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存: {output_path}")

    # 向 responses/ai_responses.csv 追加一行（与 human_responses 格式一致：id, source, bcs01..bcs50）
    responses_dir = os.path.join(BASE_DIR, "responses")
    os.makedirs(responses_dir, exist_ok=True)
    ai_responses_path = os.path.join(responses_dir, "ai_responses.csv")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    wide_row = {"id": run_id, "source": "ai"}
    for i in range(1, 51):
        wide_row[f"bcs{i:02d}"] = ""
    for row in results:
        if row.get("chatgpt_effective_bcs") != "" and row.get("chatgpt_effective_bcs") is not None:
            try:
                img_id = int(row["image_id"])
                wide_row[f"bcs{img_id:02d}"] = row["chatgpt_effective_bcs"]
            except (ValueError, KeyError):
                pass
    write_header = not os.path.exists(ai_responses_path)
    with open(ai_responses_path, "a", newline="", encoding="utf-8") as f:
        fieldnames_wide = ["id", "source"] + [f"bcs{i:02d}" for i in range(1, 51)]
        writer = csv.DictWriter(f, fieldnames=fieldnames_wide)
        if write_header:
            writer.writeheader()
        writer.writerow(wide_row)
    print(f"已追加一行到: {ai_responses_path} (run_id={run_id})")

    # 统计摘要
    valid_results = [r for r in results if r["deviation"] != ""]
    if valid_results:
        deviations = [abs(float(r["deviation"])) for r in valid_results]
        mean_dev = sum(deviations) / len(deviations)
        max_dev = max(deviations)

        correct_class = sum(1 for r in valid_results
                           if r["weight_class_gt"] == r["weight_class_chatgpt"])
        class_accuracy = correct_class / len(valid_results) * 100

        print(f"\n=== ChatGPT 评分统计 ===")
        print(f"成功评分: {len(valid_results)}/{len(results)}")
        print(f"平均绝对偏差: {mean_dev:.2f}")
        print(f"最大绝对偏差: {max_dev:.2f}")
        print(f"体重分类准确率: {class_accuracy:.1f}%")
        print(f"失败数: {errors}")


if __name__ == "__main__":
    main()
