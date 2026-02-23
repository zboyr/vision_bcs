#!/usr/bin/env python3
"""
download_cats_extra.py
从多个来源下载更多猫图片，补充 COCO 数据集不足。

来源:
    1. HuggingFace microsoft/cats_vs_dogs (~12,500 猫)
    2. Oxford-IIIT Pet Dataset (~2,371 猫, 12 品种)

使用方法:
    python download_cats_extra.py
    python download_cats_extra.py --source all
    python download_cats_extra.py --source huggingface --max-images 5000
    python download_cats_extra.py --source oxford
"""

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "cat_data", "raw_extra")

# Oxford-IIIT Pet 猫品种 (首字母大写的是猫)
CAT_BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
]

OXFORD_IMAGES_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"


def download_huggingface_cats(output_dir: str, max_images: int = 0) -> int:
    """从 HuggingFace microsoft/cats_vs_dogs 下载猫图片。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("错误: pip install datasets")
        return 0

    print("正在从 HuggingFace 加载 cats_vs_dogs 数据集...")
    ds = load_dataset("microsoft/cats_vs_dogs", split="train", trust_remote_code=True)

    # label 0 = cat
    print("正在筛选猫图片...")
    cat_indices = [i for i in range(len(ds)) if ds[i]["labels"] == 0]
    print(f"找到 {len(cat_indices)} 张猫图片")

    if max_images > 0:
        cat_indices = cat_indices[:max_images]

    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    skipped = 0
    iterator = tqdm(cat_indices, desc="保存 HF 猫图片") if tqdm else cat_indices

    for idx in iterator:
        filename = f"hf_cat_{idx:06d}.jpg"
        dest = os.path.join(output_dir, filename)
        if os.path.exists(dest):
            skipped += 1
            continue

        try:
            img = ds[idx]["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dest, "JPEG", quality=95)
            saved += 1
        except Exception as e:
            print(f"\n  保存失败 #{idx}: {e}")

    print(f"HuggingFace: 新保存 {saved}, 跳过 {skipped}")
    return saved + skipped


def download_oxford_cats(output_dir: str) -> int:
    """从 Oxford-IIIT Pet Dataset 下载猫图片。"""
    tar_path = os.path.join(BASE_DIR, "cat_data", "_oxford_images.tar.gz")
    extract_dir = os.path.join(BASE_DIR, "cat_data", "_oxford_tmp")

    # 下载
    if not os.path.exists(tar_path):
        print("正在下载 Oxford-IIIT Pet 图片 (~800MB)...")
        try:
            req = urllib.request.Request(OXFORD_IMAGES_URL,
                                        headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(tar_path + ".tmp", "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            mb = downloaded / (1024 * 1024)
                            print(f"\r  {mb:.0f}/{total/(1024*1024):.0f} MB ({pct:.0f}%)",
                                  end="", flush=True)
                print()
            os.rename(tar_path + ".tmp", tar_path)
        except Exception as e:
            print(f"Oxford 下载失败: {e}")
            if os.path.exists(tar_path + ".tmp"):
                os.remove(tar_path + ".tmp")
            return 0
    else:
        print("Oxford tar.gz 已存在，跳过下载")

    # 解压
    if not os.path.exists(extract_dir):
        print("正在解压 Oxford 数据集...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        print("解压完成")

    # 复制猫图片
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(extract_dir, "images")
    if not os.path.exists(images_dir):
        # 可能在子目录
        for root, dirs, files in os.walk(extract_dir):
            if any(f.endswith(".jpg") for f in files):
                images_dir = root
                break

    saved = 0
    skipped = 0

    for f in sorted(os.listdir(images_dir)):
        if not f.endswith(".jpg"):
            continue
        # 猫品种文件名以品种名开头
        is_cat = any(f.startswith(breed) for breed in CAT_BREEDS)
        if not is_cat:
            continue

        dest = os.path.join(output_dir, f"ox_{f}")
        if os.path.exists(dest):
            skipped += 1
            continue

        src = os.path.join(images_dir, f)
        shutil.copy2(src, dest)
        saved += 1

    print(f"Oxford: 新保存 {saved}, 跳过 {skipped}")

    # 清理临时目录
    try:
        shutil.rmtree(extract_dir)
        os.remove(tar_path)
    except OSError:
        pass

    return saved + skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="下载更多猫图片数据")
    parser.add_argument("--source", default="all",
                        choices=["all", "huggingface", "oxford"],
                        help="数据来源 (默认: all)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-images", type=int, default=0,
                        help="HuggingFace 最大下载数 (默认: 全部)")
    args = parser.parse_args()

    total = 0

    if args.source in ("all", "huggingface"):
        n = download_huggingface_cats(args.output_dir, args.max_images)
        total += n

    if args.source in ("all", "oxford"):
        n = download_oxford_cats(args.output_dir)
        total += n

    total_files = len([
        f for f in os.listdir(args.output_dir)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]) if os.path.exists(args.output_dir) else 0

    print(f"\n=== 下载完成 ===")
    print(f"  总图片数: {total_files}")
    print(f"  保存目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
