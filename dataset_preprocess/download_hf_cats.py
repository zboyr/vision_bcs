#!/usr/bin/env python3
"""从 HuggingFace 数据集下载猫图片到 staging 目录。"""

import hashlib
import os
import sys

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES = [
    {
        "name": "AIOmarRehan/Cats_and_Dogs",
        "output_dir": os.path.join(BASE_DIR, "datasets", "hf_aio_cats"),
        "split": "train",
        "filter_fn": lambda x: x["label"] == 0,
        "image_key": "image",
        "max_images": 8000,
    },
    {
        "name": "yashikota/cat-image-dataset",
        "output_dir": os.path.join(BASE_DIR, "datasets", "hf_yashikota_cats"),
        "split": "train",
        "filter_fn": None,  # all cat
        "image_key": "image",
        "max_images": 5000,
    },
    {
        "name": "lmassaron/dogs-cats-openimages",
        "output_dir": os.path.join(BASE_DIR, "datasets", "hf_openimages_cats"),
        "split": "train",
        "filter_fn": lambda x: x["label"] == 1,
        "image_key": "image",
        "max_images": 3000,
    },
]


def save_dataset(source: dict):
    name = source["name"]
    out_dir = source["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith((".jpg", ".png"))])
    if existing >= source["max_images"]:
        print(f"[{name}] 已有 {existing} 张，跳过")
        return existing

    print(f"[{name}] 加载数据集...")
    try:
        ds = load_dataset(name, split=source["split"])
    except Exception as e:
        print(f"[{name}] 加载失败: {e}")
        return 0

    if source["filter_fn"]:
        print(f"[{name}] 筛选猫图片...")
        ds = ds.filter(source["filter_fn"])

    print(f"[{name}] 共 {len(ds)} 张猫图片，目标 {source['max_images']} 张")

    saved = 0
    for i in tqdm(range(min(len(ds), source["max_images"])), desc=name):
        try:
            img = ds[i][source["image_key"]]
            if not isinstance(img, Image.Image):
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 用内容 MD5 命名
            import io
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=95)
            data = buf.getvalue()
            md5 = hashlib.md5(data).hexdigest()

            dest = os.path.join(out_dir, f"{md5}.jpg")
            if not os.path.exists(dest):
                with open(dest, "wb") as f:
                    f.write(data)
            saved += 1
        except Exception as e:
            continue

    total = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    print(f"[{name}] 完成: 新增 {saved}, 目录总计 {total}")
    return total


def main():
    total = 0
    for src in SOURCES:
        total += save_dataset(src)
    print(f"\n所有数据集下载完成，总计 {total} 张")


if __name__ == "__main__":
    main()
