#!/usr/bin/env python3
"""
build_cat_10k.py
构建 cat_10k 数据集：YOLO 检测猫 → 裁剪 → ViTPose 验证身体+3肢体 → 保存。

使用方法:
    python3 build_cat_10k.py --input-dir datasets/final --ignore scores.csv
    python3 build_cat_10k.py --input-dir datasets/raw --target 10000
    python3 build_cat_10k.py --input-dir some_dir --no-verify   # 跳过 ViTPose

输出结构:
    datasets/cat_10k/
    ├── images/          # YOLO 裁剪的猫图片 (md5.jpg)
    ├── raw_images/      # 原始图片 (md5 + 原扩展名)
    ├── cat_db.json      # TinyDB 记录 (通过的)
    └── rejected_md5.txt # 已拒绝的 MD5 列表 (避免重复处理)

依赖:
    pip install ultralytics tinydb pillow rtmlib
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

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

COCO_CAT_CLASS_ID = 15
COCO_DOG_CLASS_ID = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ViTPose AP-10K
POSE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx"
LIMB_GROUPS = {
    "left_front":  [5, 6, 7],
    "right_front": [8, 9, 10],
    "left_back":   [11, 12, 13],
    "right_back":  [14, 15, 16],
}
BODY_INDICES = [3, 4]  # Neck, Root_of_Tail


def compute_file_md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rejected_md5s(output_dir: str) -> set:
    path = os.path.join(output_dir, "rejected_md5.txt")
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def save_rejected_md5(output_dir: str, md5: str):
    path = os.path.join(output_dir, "rejected_md5.txt")
    with open(path, "a") as f:
        f.write(md5 + "\n")


def vitpose_check(pose_model, crop_bgr: np.ndarray,
                  kpt_thr: float = 0.5, min_limbs: int = 3,
                  min_spread: float = 0.3) -> bool:
    """用 ViTPose 检查裁剪图是否能看见身体 + >=3 条肢体。"""
    h, w = crop_bgr.shape[:2]
    keypoints, scores = pose_model(crop_bgr, bboxes=np.array([[0, 0, w, h]]))
    if scores is None or len(scores) == 0:
        return False
    s = scores[0]
    kpts = keypoints[0]

    body_visible = any(s[i] >= kpt_thr for i in BODY_INDICES)
    visible_limbs = sum(
        1 for indices in LIMB_GROUPS.values()
        if any(s[i] >= kpt_thr for i in indices)
    )

    confident_mask = s >= kpt_thr
    if confident_mask.sum() >= 2:
        pts = kpts[confident_mask]
        spread = max(
            (pts[:, 0].max() - pts[:, 0].min()) / w if w > 0 else 0,
            (pts[:, 1].max() - pts[:, 1].min()) / h if h > 0 else 0,
        )
    else:
        spread = 0

    return body_visible and visible_limbs >= min_limbs and spread >= min_spread


def estimate_limb_visibility(box_xyxy, img_w, img_h, edge_margin_pct=0.02):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    mx, my = img_w * edge_margin_pct, img_h * edge_margin_pct
    touches = {
        "left": x1 < mx, "top": y1 < my,
        "right": x2 > (img_w - mx), "bottom": y2 > (img_h - my),
    }
    limb_score = sum(0.25 for v in touches.values() if not v)
    box_area = (x2 - x1) * (y2 - y1)
    area_ratio = box_area / (img_w * img_h) if img_w * img_h > 0 else 0
    return {
        "limb_score": round(limb_score, 2),
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
    ignore_files: set | None = None,
    verify: bool = True,
    species: str = "cat",
) -> dict:
    images_dir = os.path.join(output_dir, "images")
    raw_dir = os.path.join(output_dir, "raw_images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    db_path = os.path.join(output_dir, "cat_db.json")
    db = TinyDB(db_path)
    CatQ = Query()

    existing_count = len(db)
    rejected_md5s = load_rejected_md5s(output_dir)
    print(f"TinyDB 已有 {existing_count} 条, 拒绝列表 {len(rejected_md5s)} 条")

    if target_count > 0 and existing_count >= target_count:
        print(f"已达到目标 {target_count}，跳过。")
        return {"kept": 0, "skipped": 0, "rejected": 0, "total": existing_count}

    print(f"加载 YOLO: {model_name} ...")
    yolo_model = YOLO(model_name)

    pose_model = None
    if verify:
        from rtmlib import ViTPose
        print("加载 ViTPose (AP-10K) ...")
        pose_model = ViTPose(POSE_URL, model_input_size=(192, 256),
                             backend="onnxruntime", device="cpu")

    input_path = Path(input_dir)
    image_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
        and (ignore_files is None or f.name not in ignore_files)
    )
    if not image_files:
        print(f"在 {input_dir} 中未找到图片")
        return {"kept": 0, "skipped": 0, "rejected": 0, "total": existing_count}

    yolo_class_id = COCO_DOG_CLASS_ID if species == "dog" else COCO_CAT_CLASS_ID
    print(f"找到 {len(image_files)} 张图片, verify={verify}, species={species}, yolo_class={yolo_class_id}")

    kept = 0
    skipped = 0
    rejected_yolo = 0
    rejected_pose = 0

    iterator = tqdm(image_files, desc="处理中") if tqdm else image_files

    for idx, img_path in enumerate(iterator):
        if target_count > 0 and (existing_count + kept) >= target_count:
            break

        md5 = compute_file_md5(str(img_path))

        if db.search(CatQ.md5 == md5) or md5 in rejected_md5s:
            skipped += 1
            continue

        # YOLO
        try:
            results = yolo_model(
                str(img_path), conf=conf_threshold,
                classes=[yolo_class_id],
                device=device if device else None, verbose=False,
            )
        except Exception as e:
            rejected_yolo += 1
            rejected_md5s.add(md5)
            save_rejected_md5(output_dir, md5)
            continue

        result = results[0]
        img_h, img_w = result.orig_shape

        if len(result.boxes) == 0:
            rejected_yolo += 1
            rejected_md5s.add(md5)
            save_rejected_md5(output_dir, md5)
            continue

        boxes = result.boxes
        cat_indices = [i for i in range(len(boxes))
                       if int(boxes.cls[i]) == yolo_class_id]
        if not cat_indices:
            rejected_yolo += 1
            rejected_md5s.add(md5)
            save_rejected_md5(output_dir, md5)
            continue

        confs = [float(boxes.conf[i]) for i in cat_indices]
        best_local = cat_indices[confs.index(max(confs))]
        best_conf = float(boxes.conf[best_local])
        best_box = boxes.xyxy[best_local].cpu().numpy()
        x1, y1, x2, y2 = best_box

        cx1, cy1, cx2, cy2 = int(x1), int(y1), int(x2), int(y2)

        try:
            img = Image.open(str(img_path))
            if img.mode != "RGB":
                img = img.convert("RGB")
            crop = img.crop((cx1, cy1, cx2, cy2))
        except Exception:
            rejected_yolo += 1
            rejected_md5s.add(md5)
            save_rejected_md5(output_dir, md5)
            continue

        # ViTPose 验证
        if verify and pose_model is not None:
            crop_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            if not vitpose_check(pose_model, crop_bgr):
                rejected_pose += 1
                rejected_md5s.add(md5)
                save_rejected_md5(output_dir, md5)
                continue

        # 保存
        raw_ext = img_path.suffix.lower()
        raw_dest = os.path.join(raw_dir, f"{md5}{raw_ext}")
        if not os.path.exists(raw_dest):
            shutil.copy2(str(img_path), raw_dest)

        crop_dest = os.path.join(images_dir, f"{md5}.jpg")
        crop.save(crop_dest, "JPEG", quality=95)

        record = {
            "original_md5": md5,
            "status": "active",
            "md5": md5,
            "species": species,
            "llm_tag": {
                "bcs": None,
                "reasoning": "",
                "confidence": None,
                "pre_bcs": None,
            },
        }
        db.insert(record)
        kept += 1

        if not tqdm and (idx + 1) % 50 == 0:
            print(f"\r  {idx+1}/{len(image_files)} kept={kept}", end="", flush=True)

    if not tqdm:
        print()

    total_in_db = len(db)
    print(f"\n=== 处理完成 ===")
    print(f"  新增: {kept}")
    print(f"  跳过(已存在/已拒绝): {skipped}")
    print(f"  YOLO拒绝: {rejected_yolo}")
    if verify:
        print(f"  ViTPose拒绝: {rejected_pose}")
    print(f"  数据库总记录: {total_in_db}")
    print(f"  输出目录: {output_dir}")

    return {"kept": kept, "skipped": skipped,
            "rejected_yolo": rejected_yolo, "rejected_pose": rejected_pose,
            "total": total_in_db}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="构建 cat_10k 数据集: YOLO + ViTPose"
    )
    parser.add_argument("--input-dir", required=True, help="输入图片目录")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--target", type=int, default=0, help="目标数量 (0=不限)")
    parser.add_argument("--device", default="")
    parser.add_argument("--ignore", nargs="*", default=[])
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 ViTPose 验证")
    parser.add_argument("--species", default="cat", choices=["cat", "dog"],
                        help="物种 (默认: cat)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误: 目录不存在: {args.input_dir}")
        return 1

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device,
        target_count=args.target,
        ignore_files=set(args.ignore) if args.ignore else None,
        verify=not args.no_verify,
        species=args.species,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
