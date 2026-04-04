#!/usr/bin/env python3
"""
build_cat_10k.py
构建 cat_10k 数据集：用 YOLO 检测猫并裁剪身体和四肢区域。

使用方法:
    # 先处理 datasets/final 中的现有图片
    python build_cat_10k.py --input-dir datasets/final --ignore scores.csv

    # 再处理 COCO 下载的图片，直到凑够 10000
    python build_cat_10k.py --input-dir cat_data/raw --target 10000

输出结构:
    datasets/cat_10k/
    ├── images/          # YOLO 裁剪的猫身体+四肢图片 (md5 命名)
    ├── raw_images/      # 原始图片 (md5 命名)
    └── cat_db.json      # TinyDB 记录

每条 TinyDB 记录:
    md5, original_name, source_dir,
    body_confidence, limb_confidence, limb_score,
    all_limbs_visible, area_ratio,
    bbox, crop_bbox, image_size, crop_size

依赖:
    pip install ultralytics tinydb pillow
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("错误: pip install Pillow")
    sys.exit(1)

try:
    from tinydb import TinyDB, Query
except ImportError:
    print("错误: pip install tinydb")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: pip install ultralytics")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "datasets", "cat_10k")

# COCO 类别
COCO_CAT_CLASS_ID = 15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def compute_file_md5(file_path: str) -> str:
    """计算文件 MD5."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def estimate_limb_visibility(
    box_xyxy, img_w: int, img_h: int, edge_margin_pct: float = 0.02
) -> dict:
    """
    估算猫四肢可见性。

    bbox 不触碰图片边缘 → 对应方向的肢体完整。
    每个方向贡献 0.25，满分 1.0。
    """
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    margin_x = img_w * edge_margin_pct
    margin_y = img_h * edge_margin_pct

    touches = {
        "left": x1 < margin_x,
        "top": y1 < margin_y,
        "right": x2 > (img_w - margin_x),
        "bottom": y2 > (img_h - margin_y),
    }

    limb_score = sum(0.25 for v in touches.values() if not v)

    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    area_ratio = box_area / img_area if img_area > 0 else 0

    return {
        "limb_score": round(limb_score, 2),
        "touches_edge": touches,
        "area_ratio": round(area_ratio, 4),
        "all_limbs_visible": limb_score == 1.0,
    }


def process_directory(
    input_dir: str,
    output_dir: str,
    model_name: str = "yolo11m.pt",
    conf_threshold: float = 0.4,
    device: str = "",
    target_count: int = 0,
    padding_pct: float = 0.05,
    ignore_files: set | None = None,
) -> dict:
    """处理输入目录，检测并裁剪猫，写入 TinyDB。"""

    images_dir = os.path.join(output_dir, "images")
    raw_dir = os.path.join(output_dir, "raw_images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    db_path = os.path.join(output_dir, "cat_db.json")
    db = TinyDB(db_path)
    CatQ = Query()

    existing_count = len(db)
    print(f"TinyDB 已有 {existing_count} 条记录")

    if target_count > 0 and existing_count >= target_count:
        print(f"已达到目标 {target_count}，跳过。")
        return {"kept": 0, "skipped": 0, "rejected": 0, "total": existing_count}

    print(f"加载 YOLO 模型: {model_name} ...")
    model = YOLO(model_name)

    input_path = Path(input_dir)
    image_files = sorted(
        f
        for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
        and (ignore_files is None or f.name not in ignore_files)
    )

    if not image_files:
        print(f"在 {input_dir} 中未找到图片")
        return {"kept": 0, "skipped": 0, "rejected": 0, "total": existing_count}

    print(f"找到 {len(image_files)} 张图片待处理")

    kept = 0
    skipped = 0
    rejected = 0

    iterator = tqdm(image_files, desc="YOLO 检测") if tqdm else image_files

    for idx, img_path in enumerate(iterator):
        if target_count > 0 and (existing_count + kept) >= target_count:
            break

        md5 = compute_file_md5(str(img_path))

        if db.search(CatQ.md5 == md5):
            skipped += 1
            continue

        try:
            results = model(
                str(img_path),
                conf=conf_threshold,
                classes=[COCO_CAT_CLASS_ID],
                device=device if device else None,
                verbose=False,
            )
        except Exception as e:
            print(f"\nYOLO 失败: {img_path.name} -> {e}")
            rejected += 1
            continue

        result = results[0]
        img_h, img_w = result.orig_shape

        if len(result.boxes) == 0:
            rejected += 1
            continue

        # 取置信度最高的猫
        boxes = result.boxes
        cat_indices = [
            i
            for i in range(len(boxes))
            if int(boxes.cls[i]) == COCO_CAT_CLASS_ID
        ]
        if not cat_indices:
            rejected += 1
            continue

        confs = [float(boxes.conf[i]) for i in cat_indices]
        best_local = cat_indices[confs.index(max(confs))]
        best_conf = float(boxes.conf[best_local])
        best_box = boxes.xyxy[best_local].cpu().numpy()

        x1, y1, x2, y2 = best_box

        limb_info = estimate_limb_visibility(best_box, img_w, img_h)

        # 加 padding
        pad_x = (x2 - x1) * padding_pct
        pad_y = (y2 - y1) * padding_pct
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(img_w, int(x2 + pad_x))
        cy2 = min(img_h, int(y2 + pad_y))

        try:
            img = Image.open(str(img_path))
            if img.mode != "RGB":
                img = img.convert("RGB")
            crop = img.crop((cx1, cy1, cx2, cy2))
        except Exception as e:
            print(f"\n图片读取失败: {img_path.name} -> {e}")
            rejected += 1
            continue

        # 保存原始图和裁剪图（统一 .jpg）
        fname = f"{md5}.jpg"

        raw_dest = os.path.join(raw_dir, fname)
        if not os.path.exists(raw_dest):
            if img_path.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copy2(str(img_path), raw_dest)
            else:
                img.save(raw_dest, "JPEG", quality=95)

        crop_dest = os.path.join(images_dir, fname)
        crop.save(crop_dest, "JPEG", quality=95)

        record = {
            "md5": md5,
            "original_name": img_path.name,
            "source_dir": str(input_dir),
            "body_confidence": round(best_conf, 4),
            "limb_confidence": round(best_conf * limb_info["limb_score"], 4),
            "limb_score": limb_info["limb_score"],
            "all_limbs_visible": limb_info["all_limbs_visible"],
            "area_ratio": limb_info["area_ratio"],
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "crop_bbox": [cx1, cy1, cx2, cy2],
            "image_size": [img_w, img_h],
            "crop_size": [cx2 - cx1, cy2 - cy1],
        }
        db.insert(record)
        kept += 1

        if not tqdm and (idx + 1) % 50 == 0:
            print(
                f"\r  进度: {idx + 1}/{len(image_files)} | 保留: {kept}",
                end="",
                flush=True,
            )

    if not tqdm:
        print()

    total_in_db = len(db)
    print(f"\n=== 处理完成 ===")
    print(f"  新增: {kept}")
    print(f"  跳过(已存在): {skipped}")
    print(f"  拒绝(无猫/失败): {rejected}")
    print(f"  数据库总记录: {total_in_db}")
    print(f"  输出目录: {output_dir}")

    return {"kept": kept, "skipped": skipped, "rejected": rejected, "total": total_in_db}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="构建 cat_10k 数据集: YOLO 检测并裁剪猫"
    )
    parser.add_argument(
        "--input-dir", required=True, help="输入图片目录"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model", default="yolo11m.pt", help="YOLO 模型 (默认: yolo11m.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="最低置信度 (默认: 0.4)"
    )
    parser.add_argument(
        "--target", type=int, default=0, help="目标数量 (0=不限)"
    )
    parser.add_argument("--device", default="", help="推理设备")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="裁剪边距比例 (默认: 0.05)",
    )
    parser.add_argument(
        "--ignore", nargs="*", default=[], help="忽略的文件名"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误: 目录不存在: {args.input_dir}")
        return 1

    ignore_files = set(args.ignore) if args.ignore else None

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device,
        target_count=args.target,
        padding_pct=args.padding,
        ignore_files=ignore_files,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
