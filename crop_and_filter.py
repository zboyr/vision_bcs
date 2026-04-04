#!/usr/bin/env python3
"""
crop_and_filter.py
对 YOLO 筛选后的猫图片进行裁切（裁切到猫的 bounding box）并筛去分辨率过低的。

使用方法:
    python crop_and_filter.py
    python crop_and_filter.py --min-dim 256 --padding 0.1
    python crop_and_filter.py --input-dir cat_data/filtered --output-dir cat_data/cropped
"""

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "cat_data", "filtered")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "cat_data", "cropped")

COCO_CAT_CLASS_ID = 15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def crop_and_filter(
    input_dir: str,
    output_dir: str,
    model_name: str = "yolo11s.pt",
    conf_threshold: float = 0.5,
    padding: float = 0.08,
    min_dim: int = 224,
    device: str = "",
) -> dict:
    """
    对每张图片:
      1. YOLO 检测猫 bbox
      2. 按 bbox + padding 裁切
      3. 如果裁切后最短边 < min_dim，丢弃

    Returns: stats dict
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: pip install ultralytics")
        sys.exit(1)

    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"错误: {input_dir} 中无图片")
        return {"kept": 0, "rejected": 0}

    print(f"输入: {len(image_files)} 张图片")
    print(f"模型: {model_name} | padding: {padding:.0%} | 最低分辨率: {min_dim}px")
    model = YOLO(model_name)

    kept = 0
    rejected_no_cat = 0
    rejected_low_res = 0
    log_rows = []

    iterator = tqdm(image_files, desc="裁切筛选") if tqdm else image_files

    for idx, img_path in enumerate(iterator):
        try:
            results = model(
                str(img_path),
                conf=conf_threshold,
                classes=[COCO_CAT_CLASS_ID],
                device=device if device else None,
                verbose=False,
            )
        except Exception as e:
            rejected_no_cat += 1
            log_rows.append({"file": img_path.name, "kept": False, "reason": f"推理错误: {e}"})
            continue

        result = results[0]
        img_h, img_w = result.orig_shape

        if len(result.boxes) == 0:
            rejected_no_cat += 1
            log_rows.append({"file": img_path.name, "kept": False, "reason": "未检测到猫"})
            continue

        # 取最高置信度的猫
        best_idx = int(result.boxes.conf.argmax())
        x1, y1, x2, y2 = result.boxes.xyxy[best_idx].cpu().numpy()

        # 加 padding
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * padding
        pad_y = bh * padding
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(img_w, int(x2 + pad_x))
        cy2 = min(img_h, int(y2 + pad_y))

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        if min(crop_w, crop_h) < min_dim:
            rejected_low_res += 1
            log_rows.append({
                "file": img_path.name, "kept": False,
                "reason": f"裁切后分辨率过低 ({crop_w}x{crop_h})",
                "crop_w": crop_w, "crop_h": crop_h,
            })
            continue

        # 裁切并保存
        im = Image.open(img_path)
        cropped = im.crop((cx1, cy1, cx2, cy2))
        # 确保 RGB 模式 (处理 P/RGBA/L 等模式的图片)
        if cropped.mode != 'RGB':
            cropped = cropped.convert('RGB')
        out_path = os.path.join(output_dir, img_path.stem + '.jpg')
        cropped.save(out_path, 'JPEG', quality=95)
        kept += 1
        log_rows.append({
            "file": img_path.name, "kept": True,
            "crop_w": crop_w, "crop_h": crop_h,
        })

        if not tqdm and (idx + 1) % 200 == 0:
            print(f"\r  进度: {idx+1}/{len(image_files)} | 保留: {kept}", end="", flush=True)

    if not tqdm:
        print()

    total_rejected = rejected_no_cat + rejected_low_res
    stats = {
        "total_input": len(image_files),
        "kept": kept,
        "rejected_no_cat": rejected_no_cat,
        "rejected_low_res": rejected_low_res,
        "rejected_total": total_rejected,
    }

    print(f"\n=== 裁切筛选结果 ===")
    print(f"  输入: {stats['total_input']}")
    print(f"  保留: {stats['kept']}")
    print(f"  拒绝 - 未检测到猫: {stats['rejected_no_cat']}")
    print(f"  拒绝 - 分辨率过低 (<{min_dim}px): {stats['rejected_low_res']}")
    print(f"  输出目录: {output_dir}")

    # 保存日志
    log_path = os.path.join(os.path.dirname(output_dir), "crop_filter_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "kept", "reason", "crop_w", "crop_h"])
        writer.writeheader()
        for row in log_rows:
            writer.writerow({
                "file": row.get("file", ""),
                "kept": row.get("kept", ""),
                "reason": row.get("reason", ""),
                "crop_w": row.get("crop_w", ""),
                "crop_h": row.get("crop_h", ""),
            })
    print(f"  日志: {log_path}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="裁切猫图片并筛去低分辨率")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.08,
                        help="bbox 外扩比例 (默认 0.08 即 8%%)")
    parser.add_argument("--min-dim", type=int, default=224,
                        help="裁切后最短边最低像素 (默认 224)")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"错误: {args.input_dir} 不存在")
        return 1

    stats = crop_and_filter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        conf_threshold=args.conf,
        padding=args.padding,
        min_dim=args.min_dim,
        device=args.device,
    )

    if stats["kept"] == 0:
        print("警告: 无图片通过，尝试降低 --min-dim")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
