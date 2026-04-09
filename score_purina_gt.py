#!/usr/bin/env python3
"""
score_purina_gt.py
用 Gemini 3.1 Pro 为 Purina 3D ground truth 图片生成 reasoning。
提示词包含 ground truth BCS，模型只需输出视觉依据和置信度。
"""

import base64, csv, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import openai
except ImportError:
    print("pip install openai"); sys.exit(1)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_prompts():
    import yaml
    with open(os.path.join(BASE_DIR, "prompts", "bcs_prompts.yaml")) as f:
        return yaml.safe_load(f)


def build_system_prompt():
    p = load_prompts()
    return f"""{p['role'].strip()}

{p['bcs_scale'].strip()}

{p['confidence_guide'].strip()}

You must respond ONLY with valid JSON matching this exact schema:
{{
  "reasoning": "<string: detailed visual evidence supporting the given BCS>",
  "confidence": <int 1-9>
}}

Field definitions:
- reasoning: Describe the specific visual evidence you observe that supports the given BCS score. Include details about rib visibility, waist definition, abdominal tuck, fat deposits, and muscle mass. Be thorough and specific. This MUST come first.
- confidence: Your confidence in the visual assessment per the Confidence Scale above (1-9). This reflects how clearly the body condition landmarks are visible in the image, NOT whether you agree with the BCS."""


SYSTEM_PROMPT = None


def build_user_prompt(row: dict) -> str:
    species = row.get("species", "cat")
    gt_bcs = row["gt_bcs"]
    return f"""This {species} has been assigned a Body Condition Score of {gt_bcs}/9 by veterinary experts.

Examine the image carefully and describe the visual evidence that supports this BCS {gt_bcs} assessment. Focus on:
- Rib visibility and palpability clues
- Waist definition when viewed from above (if visible)
- Abdominal tuck from the side (if visible)
- Fat deposits (lumbar area, limbs, face)
- Overall body shape and muscle mass

Respond with JSON only."""


def load_env():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
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
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "bcs_scorer"},
        timeout=90,
    )


def encode_image(path: str) -> str:
    with open(os.path.join(BASE_DIR, path), "rb") as f:
        return base64.b64encode(f.read()).decode()


def score_one(client, row, model, max_retries=3):
    img_id = row["id"]
    user_prompt = build_user_prompt(row)

    for attempt in range(max_retries):
        try:
            b64 = encode_image(row["path"])
            ext = os.path.splitext(row["path"])[1].lower()
            media = "image/png" if ext == ".png" else "image/jpeg"

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
                    ]},
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
                "confidence": data.get("confidence"),
                "error": "",
            }

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return {"id": img_id, "reasoning": "", "confidence": None,
                        "error": f"JSON parse error: {e}"}
            time.sleep(2)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"id": img_id, "reasoning": "", "confidence": None,
                        "error": str(e)}
            time.sleep(3)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cat_10k/dataset.csv")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--model", default="google/gemini-3.1-pro-preview")
    args = parser.parse_args()

    load_env()
    client = create_client()
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = build_system_prompt()

    dataset_path = os.path.join(BASE_DIR, args.dataset)
    rows = list(csv.DictReader(open(dataset_path)))

    output_path = os.path.join(os.path.dirname(dataset_path), "purina_scores.csv")

    # Skip already scored
    scored_ids = set()
    if os.path.exists(output_path):
        for r in csv.DictReader(open(output_path)):
            if r.get("reasoning") and not r.get("error"):
                scored_ids.add(r["id"])

    todo = [r for r in rows if r["id"] not in scored_ids]
    if not todo:
        print("没有待评分的图片"); return

    print(f"待评分: {len(todo)} 张 (已跳过 {len(scored_ids)} 张)")
    print(f"模型: {args.model}, 并发: {args.concurrency}")

    fieldnames = ["id", "reasoning", "confidence", "error"]
    write_header = not os.path.exists(output_path)

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(score_one, client, row, args.model): row for row in todo}
        pbar = tqdm(total=len(todo), desc="Scoring") if tqdm else None

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerow(result)
            if pbar: pbar.update(1)

        if pbar: pbar.close()

    ok = [r for r in results if r.get("reasoning")]
    err = [r for r in results if r.get("error")]
    print(f"\n成功: {len(ok)}/{len(results)}, 错误: {len(err)}")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    main()
