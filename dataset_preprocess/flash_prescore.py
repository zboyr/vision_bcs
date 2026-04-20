#!/usr/bin/env python3
"""
flash_prescore.py
用 Gemini Flash 对新图片快速预打分 BCS，筛选非 BCS=5 的图片。

使用方法:
    # 对 cat_db.json 中没有 pre_bcs 的 active 图片预打分
    python3 flash_prescore.py

    # 指定目标数量
    python3 flash_prescore.py --need-below5 2000 --need-above5 1000

    # 从 staging 目录导入新图片 (YOLO+ViTPose) 并预打分
    python3 flash_prescore.py --input-dir datasets/hf_new_cats
"""

import argparse
import base64
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import openai
except ImportError:
    print("错误: pip install openai")
    sys.exit(1)

try:
    from tinydb import TinyDB, Query
except ImportError:
    print("错误: pip install tinydb")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_flash_prompt():
    """Load BCS scale + confidence guide from YAML, build Flash prompt."""
    import yaml
    yaml_path = os.path.join(BASE_DIR, "prompts", "bcs_prompts.yaml")
    with open(yaml_path, "r") as f:
        p = yaml.safe_load(f)
    return f"""{p['role'].strip()}

{p['bcs_scale'].strip()}

{p['confidence_guide'].strip()}

Output ONLY two integers separated by comma: bcs,confidence
Example: 4,7"""


FLASH_PROMPT = None  # loaded lazily


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
            "X-Title": "cat_bcs_flash",
        },
        timeout=30,
    )


def flash_score_one(client, md5: str, img_path: str, model: str) -> dict:
    """Flash pre-score a single image. Returns {pre_bcs, pre_confidence} or {error}."""
    abs_path = os.path.join(BASE_DIR, img_path)
    for attempt in range(3):
        try:
            b64 = base64.b64encode(open(abs_path, "rb").read()).decode()
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FLASH_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                temperature=0.1,
                max_tokens=16,
            )
            content = resp.choices[0].message.content.strip()
            # Parse "bcs,confidence"
            m = re.match(r"(\d),\s*(\d)", content)
            if m:
                return {
                    "md5": md5,
                    "pre_bcs": int(m.group(1)),
                    "pre_confidence": int(m.group(2)),
                }
            # Fallback: find two single digits
            digits = re.findall(r"\b([1-9])\b", content)
            if len(digits) >= 2:
                return {
                    "md5": md5,
                    "pre_bcs": int(digits[0]),
                    "pre_confidence": int(digits[1]),
                }
            if len(digits) == 1:
                return {
                    "md5": md5,
                    "pre_bcs": int(digits[0]),
                    "pre_confidence": 5,
                }
            return {"md5": md5, "error": f"parse fail: {content}"}
        except Exception as e:
            if attempt == 2:
                return {"md5": md5, "error": str(e)[:100]}
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="Flash BCS pre-scoring")
    parser.add_argument("--model", default="google/gemini-3.1-flash-image-preview")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--need-below5", type=int, default=2000)
    parser.add_argument("--need-above5", type=int, default=1000)
    parser.add_argument("--species", default=None, help="Filter by species (cat/dog)")
    parser.add_argument("--conf-threshold-eq5", type=int, default=5,
                        help="Min Flash confidence to keep BCS=5 (default 5)")
    parser.add_argument("--conf-threshold-neq5", type=int, default=4,
                        help="Min Flash confidence to keep BCS≠5 (default 4)")
    parser.add_argument("--input-dir", help="Optional: import new images first (YOLO+ViTPose)")
    args = parser.parse_args()

    load_env()
    client = create_client()

    global FLASH_PROMPT
    FLASH_PROMPT = load_flash_prompt()

    # Optional: import new images first
    if args.input_dir:
        print(f"Importing from {args.input_dir}...")
        os.system(f"python3 build_cat_10k.py --input-dir {args.input_dir} --target 999999")

    db = TinyDB(os.path.join(BASE_DIR, "datasets/cat_10k/cat_db.json"))
    q = Query()

    # Find active images without pro bcs (new images only)
    all_active = db.search(q.status == "active")
    todo = []
    have_below5 = 0
    have_above5 = 0

    for r in all_active:
        # Species filter
        if args.species and r.get("species") != args.species:
            continue
        tag = r.get("llm_tag") or {}
        bcs = tag.get("bcs")
        pre = tag.get("pre_bcs")
        if bcs is not None:
            continue  # already pro-scored, skip
        # Count NEW images only
        if pre is not None:
            pc = tag.get("pre_confidence", 0) or 0
            # Apply confidence threshold
            if pre == 5 and pc < args.conf_threshold_eq5:
                continue
            if pre != 5 and pc < args.conf_threshold_neq5:
                continue
            if pre < 5:
                have_below5 += 1
            elif pre > 5:
                have_above5 += 1
        else:
            todo.append(r)

    species_str = args.species or "all"
    print(f"Active total: {len(all_active)} (species={species_str})")
    print(f"New flash-scored (passing conf threshold): below5={have_below5}, above5={have_above5}")
    print(f"Need NEW: below5={max(0, args.need_below5 - have_below5)}, above5={max(0, args.need_above5 - have_above5)}")
    print(f"To pre-score: {len(todo)}")
    print(f"Model: {args.model}")
    print(f"Conf thresholds: eq5>={args.conf_threshold_eq5}, neq5>={args.conf_threshold_neq5}")

    if not todo:
        print("没有待预打分的图片")
        return

    # Pre-score with concurrency
    db_lock = Lock()
    scored = 0
    errors = 0

    def process_one(rec):
        nonlocal scored, errors, have_below5, have_above5
        md5 = rec["md5"]
        img_path = f"datasets/cat_10k/images/{md5}.jpg"
        result = flash_score_one(client, md5, img_path, args.model)

        if "error" in result:
            errors += 1
            return result

        pre_bcs = result["pre_bcs"]
        pre_conf = result["pre_confidence"]

        with db_lock:
            tag = rec.get("llm_tag") or {}
            tag["pre_bcs"] = pre_bcs
            tag["pre_confidence"] = pre_conf
            db.update({"llm_tag": tag}, doc_ids=[rec.doc_id])

            # Apply confidence threshold
            passes = True
            if pre_bcs == 5 and pre_conf < args.conf_threshold_eq5:
                passes = False
            if pre_bcs != 5 and pre_conf < args.conf_threshold_neq5:
                passes = False

            if passes:
                if pre_bcs < 5:
                    have_below5 += 1
                elif pre_bcs > 5:
                    have_above5 += 1
            scored += 1

        return result

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_one, r): r for r in todo}
        if tqdm:
            pbar = tqdm(total=len(todo), desc="Flash pre-score")
        else:
            pbar = None

        for future in as_completed(futures):
            result = future.result()
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"<5": have_below5, ">5": have_above5, "err": errors})

            # Check if targets met
            if have_below5 >= args.need_below5 and have_above5 >= args.need_above5:
                # Cancel remaining
                for f in futures:
                    f.cancel()
                break

        if pbar:
            pbar.close()

    # Summary
    from collections import Counter
    all_active = db.search(q.status == "active")
    pre_dist = Counter()
    for r in all_active:
        tag = r.get("llm_tag") or {}
        pb = tag.get("pre_bcs") or tag.get("bcs")
        if pb:
            pre_dist[pb] += 1

    print(f"\n=== Pre-score 完成 ===")
    print(f"  已打分: {scored}, 错误: {errors}")
    print(f"  below5: {have_below5}, above5: {have_above5}")
    print(f"  Pre-BCS distribution:")
    for b in sorted(pre_dist):
        print(f"    BCS {b}: {pre_dist[b]}")


if __name__ == "__main__":
    main()
