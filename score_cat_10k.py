#!/usr/bin/env python3
"""
score_cat_10k.py
用 Gemini 3.1 Pro (via OpenRouter) 对 cat_10k 数据集打 BCS 分。

使用方法:
    python3 score_cat_10k.py                      # 跑前 10 张
    python3 score_cat_10k.py --num 100             # 跑 100 张
    python3 score_cat_10k.py --num 0               # 跑全部
    python3 score_cat_10k.py --concurrency 20      # 20 并发

输出:
    datasets/cat_10k/bcs_scores.csv
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import openai
except ImportError:
    print("错误: pip install openai")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cat breed ID list for the prompt
CAT_BREEDS = {
    1: "Domestic Shorthair (DSH)",
    2: "Domestic Longhair (DLH)",
    3: "Siamese",
    4: "Persian",
    5: "Maine Coon",
    6: "Ragdoll",
    7: "Bengal",
    8: "Abyssinian",
    9: "British Shorthair",
    10: "Sphynx",
    11: "Russian Blue",
    12: "Bombay",
    13: "Birman",
    14: "Egyptian Mau",
    15: "Munchkin",
    16: "Scottish Fold",
    17: "Norwegian Forest Cat",
    18: "Turkish Angora",
    19: "American Shorthair",
    20: "Other / Unknown",
}

BREED_LIST_STR = "\n".join(f"  {k}: {v}" for k, v in CAT_BREEDS.items())

SYSTEM_PROMPT = f"""You are an expert veterinarian specializing in feline nutrition and body condition assessment. You have extensive experience with the 9-point Body Condition Score (BCS) system.

BCS Scale Reference:
  1 - Emaciated: Ribs, spine, and pelvic bones are easily visible. No palpable fat. Severe muscle wasting.
  2 - Very Thin: Ribs, spine, and pelvic bones easily visible. Minimal fat. Obvious waist and abdominal tuck.
  3 - Thin: Ribs easily palpable with minimal fat covering. Lumbar vertebrae obvious. Obvious waist.
  4 - Underweight: Ribs palpable with minimal fat covering. Noticeable waist when viewed from above. Slight abdominal tuck.
  5 - Ideal: Well-proportioned. Ribs palpable without excess fat. Waist observed behind ribs when viewed from above. Abdominal tuck present.
  6 - Slightly Overweight: Ribs palpable with slight excess fat. Waist discernible but not prominent. Slight abdominal fat pad.
  7 - Overweight: Ribs palpable with difficulty; heavy fat covering. Fat deposits over lumbar area and face. Waist barely visible. Obvious rounding of abdomen.
  8 - Obese: Ribs not palpable under heavy fat. Heavy fat deposits over lumbar area, face, and limbs. No waist. Obvious abdominal distension.
  9 - Severely Obese: Massive fat deposits over thorax, spine, and limbs. No waist, no abdominal tuck. Obvious abdominal distension.

Cat Breed IDs:
{BREED_LIST_STR}

You must respond ONLY with valid JSON matching this exact schema:
{{
  "bcs_primary": <int 1-9>,
  "bcs_secondary": <int 1-9>,
  "reasoning": "<string: key visual evidence for your primary BCS assessment>",
  "confidence": <int 1-9>,
  "confidence_detractors": "<string: factors that reduce your confidence>",
  "breed_id": <int 1-20>
}}

Field definitions:
- bcs_primary: Most likely BCS value (1-9)
- bcs_secondary: Second most likely BCS value (1-9), must differ from bcs_primary
- reasoning: Key visual evidence (rib visibility, waist definition, abdominal tuck, fat deposits, muscle mass). Be concise but specific.
- confidence: 1 = pure guess, 9 = absolutely certain
- confidence_detractors: Factors lowering confidence (e.g., fur obscuring body shape, unusual pose, partial occlusion, low image quality, breed-specific body shape)
- breed_id: Best matching breed from the list above"""

USER_PROMPT = """Assess this cat's Body Condition Score. Examine the visible body shape, waist definition, abdominal profile, rib coverage, and overall fat/muscle distribution. Respond with JSON only."""


def load_env():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


def create_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "cat_bcs_scorer",
        },
        timeout=90,
    )


def encode_image(path: str) -> str:
    abs_path = os.path.join(BASE_DIR, path)
    with open(abs_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def score_one(client, row: dict, model: str, max_retries: int = 3) -> dict:
    """Score a single image. Returns result dict."""
    img_id = row["id"]
    img_path = row["path"]

    for attempt in range(max_retries):
        try:
            b64 = encode_image(img_path)
            ext = os.path.splitext(img_path)[1].lower()
            media = "image/png" if ext == ".png" else "image/jpeg"

            resp = client.chat.completions.create(
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
                                    "url": f"data:{media};base64,{b64}",
                                },
                            },
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=2048,
            )

            content = resp.choices[0].message.content.strip()
            # Extract JSON from possible markdown code block
            if "```" in content:
                import re
                m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
                if m:
                    content = m.group(1)

            data = json.loads(content)

            return {
                "id": img_id,
                "bcs_primary": data.get("bcs_primary"),
                "bcs_secondary": data.get("bcs_secondary"),
                "reasoning": data.get("reasoning", ""),
                "confidence": data.get("confidence"),
                "confidence_detractors": data.get("confidence_detractors", ""),
                "breed_id": data.get("breed_id"),
                "raw_response": content,
                "error": "",
            }

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return {
                    "id": img_id, "bcs_primary": None, "bcs_secondary": None,
                    "reasoning": "", "confidence": None,
                    "confidence_detractors": "", "breed_id": None,
                    "raw_response": content if 'content' in dir() else "",
                    "error": f"JSON parse error: {e}",
                }
            time.sleep(2)

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "id": img_id, "bcs_primary": None, "bcs_secondary": None,
                    "reasoning": "", "confidence": None,
                    "confidence_detractors": "", "breed_id": None,
                    "raw_response": "",
                    "error": str(e),
                }
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="Cat BCS scoring via Gemini 3.1 Pro")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of images to score (0=all, default=10)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrent API calls (default=10)")
    parser.add_argument("--model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--output", default="datasets/cat_10k/bcs_scores.csv")
    args = parser.parse_args()

    load_env()
    client = create_client()

    # Load dataset
    dataset_path = os.path.join(BASE_DIR, "datasets/cat_10k/dataset.csv")
    rows = []
    with open(dataset_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Load existing results to skip
    output_path = os.path.join(BASE_DIR, args.output)
    scored_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for r in csv.DictReader(f):
                if r.get("bcs_primary") and not r.get("error"):
                    scored_ids.add(r["id"])

    # Filter to unscored
    todo = [r for r in rows if r["id"] not in scored_ids]
    if args.num > 0:
        todo = todo[:args.num]

    if not todo:
        print("没有待评分的图片")
        return

    print(f"模型: {args.model}")
    print(f"待评分: {len(todo)} 张 (已跳过 {len(scored_ids)} 张)")
    print(f"并发: {args.concurrency}")

    # Score with concurrency
    results = []
    fieldnames = ["id", "bcs_primary", "bcs_secondary", "reasoning",
                   "confidence", "confidence_detractors", "breed_id",
                   "raw_response", "error"]

    # Write header if file doesn't exist
    write_header = not os.path.exists(output_path)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(score_one, client, row, args.model): row
            for row in todo
        }

        if tqdm:
            pbar = tqdm(total=len(todo), desc="Scoring")
        else:
            pbar = None

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # Append to CSV immediately
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerow(result)

            if pbar:
                pbar.update(1)
            else:
                done = len(results)
                if done % 5 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}")

        if pbar:
            pbar.close()

    # Summary
    ok = [r for r in results if r["bcs_primary"] is not None]
    err = [r for r in results if r["error"]]
    print(f"\n=== 完成 ===")
    print(f"  成功: {len(ok)}/{len(results)}")
    print(f"  错误: {len(err)}")
    if ok:
        from collections import Counter
        bcs_dist = Counter(r["bcs_primary"] for r in ok)
        print(f"  BCS 分布: {dict(sorted(bcs_dist.items()))}")
        breed_dist = Counter(r["breed_id"] for r in ok)
        print(f"  品种 Top5: {breed_dist.most_common(5)}")
    print(f"  输出: {args.output}")


if __name__ == "__main__":
    main()
