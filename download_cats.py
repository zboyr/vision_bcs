#!/usr/bin/env python3
"""
download_cats.py
从 COCO 2017 数据集下载猫类别图片。

使用方法:
    python download_cats.py
    python download_cats.py --max-images 6000 --workers 16
    python download_cats.py --output-dir ./my_cats

说明:
    1. 下载 COCO 2017 annotations (~241MB zip)
    2. 解析 instances_train2017.json + instances_val2017.json 获取猫图片 ID
    3. 并发下载猫图片到 cat_data/raw/ 目录
    4. 支持断点续传（已存在的文件自动跳过）
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "cat_data", "raw")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "cat_data", "_annotations")

# COCO 2017 URLs
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_BASE_URLS = {
    "train": "http://images.cocodataset.org/train2017",
    "val": "http://images.cocodataset.org/val2017",
}
COCO_ANNOTATION_FILES = {
    "train": "annotations/instances_train2017.json",
    "val": "annotations/instances_val2017.json",
}

# COCO 类别 ID: cat = 17
COCO_CAT_CATEGORY_ID = 17


def download_with_progress(url: str, dest: str, desc: str = "") -> bool:
    """下载文件，带进度显示。返回是否成功。"""
    if os.path.exists(dest):
        print(f"  已存在，跳过: {dest}")
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_dest = dest + ".tmp"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 8192
            downloaded = 0

            with open(tmp_dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  {desc} {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)",
                              end="", flush=True)

            print()
        os.rename(tmp_dest, dest)
        return True

    except Exception as e:
        print(f"\n  下载失败: {url} -> {e}")
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)
        return False


def download_single_image(args: tuple) -> tuple:
    """下载单张图片。返回 (file_name, success, error_msg)。"""
    url, dest = args
    if os.path.exists(dest):
        return (os.path.basename(dest), True, "skipped")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return (os.path.basename(dest), True, "")
    except Exception as e:
        return (os.path.basename(dest), False, str(e))


def parse_coco_cat_images(annotation_file: str) -> list:
    """
    从 COCO annotation JSON 中提取猫图片信息。
    返回: [{id, file_name, width, height, cat_annotations: [...]}]
    """
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 找到 cat 类别 ID (应该是 17)
    cat_category_ids = set()
    for cat in coco["categories"]:
        if cat["name"] == "cat":
            cat_category_ids.add(cat["id"])

    if not cat_category_ids:
        print(f"  警告: 在 {annotation_file} 中未找到 cat 类别")
        return []

    # 收集所有猫标注
    cat_annotations = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_category_ids:
            img_id = ann["image_id"]
            if img_id not in cat_annotations:
                cat_annotations[img_id] = []
            cat_annotations[img_id].append({
                "bbox": ann["bbox"],  # [x, y, width, height]
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })

    # 关联图片信息
    images_map = {img["id"]: img for img in coco["images"]}
    result = []
    for img_id, anns in cat_annotations.items():
        if img_id in images_map:
            img_info = images_map[img_id]
            result.append({
                "id": img_id,
                "file_name": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
                "cat_annotations": anns,
            })

    return result


def download_coco_annotations() -> str:
    """下载并解压 COCO 2017 annotations，返回解压目录。"""
    zip_path = os.path.join(ANNOTATIONS_DIR, "annotations_trainval2017.zip")

    # 检查是否已解压
    train_json = os.path.join(ANNOTATIONS_DIR, "annotations", "instances_train2017.json")
    val_json = os.path.join(ANNOTATIONS_DIR, "annotations", "instances_val2017.json")
    if os.path.exists(train_json) and os.path.exists(val_json):
        print("COCO annotations 已存在，跳过下载。")
        return ANNOTATIONS_DIR

    # 下载
    print("正在下载 COCO 2017 annotations (~241MB)...")
    success = download_with_progress(COCO_ANNOTATIONS_URL, zip_path,
                                     desc="annotations")
    if not success:
        print("错误: annotations 下载失败")
        sys.exit(1)

    # 解压
    print("正在解压 annotations...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # 只解压 instances 文件
        for member in z.namelist():
            if "instances_" in member and member.endswith(".json"):
                z.extract(member, ANNOTATIONS_DIR)

    print("annotations 解压完成。")

    # 清理 zip
    try:
        os.remove(zip_path)
    except OSError:
        pass

    return ANNOTATIONS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 COCO 2017 数据集下载猫图片"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"图片保存目录 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--max-images", type=int, default=None,
                        help="最大下载数量 (默认: 全部)")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发下载线程数 (默认: 8)")
    parser.add_argument("--min-bbox-ratio", type=float, default=0.02,
                        help="猫 bbox 占图片面积最小比例 (默认: 0.02，即 2%%)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 下载 annotations
    ann_dir = download_coco_annotations()

    # Step 2: 解析猫图片信息
    all_cat_images = []
    for split, ann_file_name in COCO_ANNOTATION_FILES.items():
        ann_path = os.path.join(ann_dir, ann_file_name)
        if not os.path.exists(ann_path):
            print(f"  警告: 找不到 {ann_path}，跳过 {split}")
            continue

        print(f"正在解析 {split} 集...")
        images = parse_coco_cat_images(ann_path)

        # 预筛选: 去掉 bbox 过小的（猫在图中占比太小的不要）
        filtered = []
        for img in images:
            img_area = img["width"] * img["height"]
            max_cat_ratio = max(
                ann["area"] / img_area
                for ann in img["cat_annotations"]
                if not ann["iscrowd"]
            ) if img["cat_annotations"] else 0

            if max_cat_ratio >= args.min_bbox_ratio:
                img["split"] = split
                img["max_cat_ratio"] = max_cat_ratio
                filtered.append(img)

        print(f"  {split}: 找到 {len(images)} 张猫图片，"
              f"预筛选后 {len(filtered)} 张 (bbox>{args.min_bbox_ratio:.0%})")
        all_cat_images.extend(filtered)

    # 去重 (跨 train/val 不太可能重复，但以防万一)
    seen_filenames = set()
    unique_images = []
    for img in all_cat_images:
        if img["file_name"] not in seen_filenames:
            seen_filenames.add(img["file_name"])
            unique_images.append(img)

    # 按猫占比降序排序 (优先下载猫更大的图片)
    unique_images.sort(key=lambda x: x["max_cat_ratio"], reverse=True)

    if args.max_images and len(unique_images) > args.max_images:
        unique_images = unique_images[:args.max_images]

    print(f"\n总计: 准备下载 {len(unique_images)} 张猫图片")

    # Step 3: 并发下载
    download_tasks = []
    for img in unique_images:
        split = img["split"]
        base_url = COCO_IMAGE_BASE_URLS[split]
        url = f"{base_url}/{img['file_name']}"
        dest = os.path.join(output_dir, img["file_name"])
        download_tasks.append((url, dest))

    success_count = 0
    fail_count = 0
    skip_count = 0

    print(f"开始下载 (线程数: {args.workers})...")

    if tqdm:
        pbar = tqdm(total=len(download_tasks), desc="下载进度")
    else:
        pbar = None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single_image, task): task
                   for task in download_tasks}

        for future in as_completed(futures):
            file_name, ok, msg = future.result()
            if ok:
                if msg == "skipped":
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1

            if pbar:
                pbar.update(1)
            elif (success_count + fail_count + skip_count) % 100 == 0:
                total = success_count + fail_count + skip_count
                print(f"\r  进度: {total}/{len(download_tasks)}", end="", flush=True)

    if pbar:
        pbar.close()
    else:
        print()

    # 统计
    total_files = len(list(Path(output_dir).glob("*.jpg")))
    print(f"\n=== 下载完成 ===")
    print(f"  新下载: {success_count}")
    print(f"  已存在跳过: {skip_count}")
    print(f"  失败: {fail_count}")
    print(f"  目录内总图片数: {total_files}")
    print(f"  保存目录: {output_dir}")

    # 保存图片元信息 (供后续 YOLO 筛选参考)
    meta_path = os.path.join(os.path.dirname(output_dir), "coco_cat_meta.json")
    meta = {}
    for img in unique_images:
        meta[img["file_name"]] = {
            "coco_id": img["id"],
            "width": img["width"],
            "height": img["height"],
            "max_cat_ratio": round(img["max_cat_ratio"], 4),
            "num_cats": len(img["cat_annotations"]),
            "cat_bboxes": [ann["bbox"] for ann in img["cat_annotations"]],
        }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  元信息: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
