#!/usr/bin/env python3
"""
score_cat_10k.py
用 Gemini 3.1 Pro (via OpenRouter) 对猫/狗数据集打 BCS 分。

使用方法:
    python3 score_cat_10k.py --dataset datasets/cat_10k/dataset.csv
    python3 score_cat_10k.py --dataset datasets/dog_dataset/dataset.csv --num 0
    python3 score_cat_10k.py --num 10 --concurrency 10

输出:
    与 --dataset 同目录的 bcs_scores.csv
"""

import argparse
import base64
import csv
import json
import os
import re
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


def load_prompts():
    """Load BCS prompts from YAML."""
    import yaml
    yaml_path = os.path.join(BASE_DIR, "prompts", "bcs_prompts.yaml")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def build_system_prompt():
    p = load_prompts()
    return f"""{p['role'].strip()}

{p['bcs_scale'].strip()}

{p['confidence_guide'].strip()}

You must respond ONLY with valid JSON matching this exact schema:
{{
  "reasoning": "<string: key visual evidence for your BCS assessment>",
  "bcs": <int 1-9>,
  "confidence": <int 1-9>
}}

Field definitions:
- reasoning: Key visual evidence (rib visibility, waist definition, abdominal tuck, fat deposits, muscle mass). Be concise but specific. This MUST come first so you reason before scoring.
- bcs: Most likely BCS value (1-9)
- confidence: Your confidence per the Confidence Scale above (1-9)"""


SYSTEM_PROMPT = None  # loaded lazily


def build_user_prompt(row: dict) -> str:
    """Build per-image user prompt. Supports species-specific hints."""
    species = row.get("species", "cat")
    hint = row.get("hint", "")

    base = f"Assess this {species}'s Body Condition Score. Examine the visible body shape, waist definition, abdominal profile, rib coverage, and overall fat/muscle distribution."

    if hint:
        base += f"\n\nAdditional context: {hint}"

    base += "\n\nRespond with JSON only."
    return base


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
            "X-Title": "bcs_scorer",
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
    user_prompt = build_user_prompt(row)

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
                            {"type": "text", "text": user_prompt},
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
            if "```" in content:
                m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
                if m:
                    content = m.group(1)

            data = json.loads(content)

            return {
                "id": img_id,
                "reasoning": data.get("reasoning", ""),
                "bcs": data.get("bcs"),
                "confidence": data.get("confidence"),
                "raw_response": content,
                "error": "",
            }

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return {
                    "id": img_id, "reasoning": "",
                    "bcs": None, "confidence": None,
                    "raw_response": content if 'content' in dir() else "",
                    "error": f"JSON parse error: {e}",
                }
            time.sleep(2)

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "id": img_id, "reasoning": "",
                    "bcs": None, "confidence": None,
                    "raw_response": "",
                    "error": str(e),
                }
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="BCS scoring via Gemini 3.1 Pro")
    parser.add_argument("--dataset", default="datasets/cat_10k/dataset.csv",
                        help="Dataset CSV (must have id, path columns; optional: species, hint)")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of images to score (0=all, default=10)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrent API calls (default=10)")
    parser.add_argument("--model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: same dir as dataset)")
    args = parser.parse_args()

    load_env()
    client = create_client()

    global SYSTEM_PROMPT
    SYSTEM_PROMPT = build_system_prompt()

    # Load dataset
    dataset_path = os.path.join(BASE_DIR, args.dataset)
    rows = []
    with open(dataset_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Output path
    if args.output:
        output_path = os.path.join(BASE_DIR, args.output)
    else:
        output_path = os.path.join(os.path.dirname(dataset_path), "bcs_scores.csv")

    # Load existing results to skip
    scored_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for r in csv.DictReader(f):
                if r.get("bcs") and not r.get("error"):
                    scored_ids.add(r["id"])

    # Filter to unscored
    todo = [r for r in rows if r["id"] not in scored_ids]
    if args.num > 0:
        todo = todo[:args.num]

    if not todo:
        print("没有待评分的图片")
        return

    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"待评分: {len(todo)} 张 (已跳过 {len(scored_ids)} 张)")
    print(f"并发: {args.concurrency}")

    # Score with concurrency
    results = []
    fieldnames = ["id", "reasoning", "bcs", "confidence",
                   "raw_response", "error"]

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
    ok = [r for r in results if r.get("bcs") is not None]
    err = [r for r in results if r.get("error")]
    print(f"\n=== 完成 ===")
    print(f"  成功: {len(ok)}/{len(results)}")
    print(f"  错误: {len(err)}")
    if ok:
        from collections import Counter
        bcs_dist = Counter(r["bcs"] for r in ok)
        print(f"  BCS 分布: {dict(sorted(bcs_dist.items()))}")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
