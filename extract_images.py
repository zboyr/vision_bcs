#!/usr/bin/env python3
"""
extract_images.py
从 DataS3_BTS1-Quiz_images.pdf 中提取50张猫图片。

PDF 结构（共54页，0-indexed）：
  - 页0-1: 说明页（跳过）
  - 页2-52: Cat #1 ~ Cat #50 的图片（其中页37是狗的trick question，跳过）
  - 页53: 感谢页（跳过）

输出到 images/ 目录，命名为 cat_01.jpg ~ cat_50.jpg
"""

import os
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    print("请先安装 PyMuPDF: pip install PyMuPDF")
    sys.exit(1)


PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "DataS3_BTS1-Quiz_images.pdf")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# 页面映射：PDF页索引(0-based) -> Cat ID
# 页0,1 = 说明页; 页2 = Cat#1; ... 页37 = 狗(跳过); ... 页52 = Cat#50; 页53 = 感谢页
SKIP_PAGES = {0, 1, 37, 53}  # 说明页、狗、感谢页


def build_page_to_cat_mapping():
    """构建 PDF 页索引 -> Cat ID 的映射。"""
    mapping = {}
    cat_id = 1
    for page_idx in range(2, 53):  # 页2到页52
        if page_idx in SKIP_PAGES:
            continue
        mapping[page_idx] = cat_id
        cat_id += 1
    return mapping


def extract_largest_image_from_page(page):
    """从页面中提取最大的图片（按像素面积）。"""
    images = page.get_images(full=True)
    if not images:
        return None, None

    best_image = None
    best_area = 0
    best_ext = "png"

    for img_info in images:
        xref = img_info[0]
        try:
            base_image = page.parent.extract_image(xref)
            if base_image is None:
                continue
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            area = width * height
            if area > best_area:
                best_area = area
                best_image = base_image["image"]
                best_ext = base_image.get("ext", "png")
        except Exception as e:
            print(f"  警告: 提取 xref={xref} 时出错: {e}")
            continue

    return best_image, best_ext


def render_page_as_image(page, dpi=200):
    """将整个页面渲染为图片（备用方案）。"""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png"), "png"


def main():
    if not os.path.exists(PDF_PATH):
        print(f"错误: 找不到 PDF 文件: {PDF_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mapping = build_page_to_cat_mapping()
    print(f"将从 PDF 中提取 {len(mapping)} 张猫图片...")

    doc = fitz.open(PDF_PATH)
    print(f"PDF 共 {len(doc)} 页")

    success_count = 0
    fail_count = 0

    for page_idx, cat_id in sorted(mapping.items()):
        page = doc[page_idx]
        page_label = f"页{page_idx} -> Cat #{cat_id}"

        # 尝试提取嵌入图片
        img_data, ext = extract_largest_image_from_page(page)

        if img_data is None:
            # 备用方案：渲染整个页面
            print(f"  {page_label}: 无嵌入图片，渲染整页...")
            img_data, ext = render_page_as_image(page)

        if img_data:
            # 统一保存为 jpg（如果原始是 png 也转换）
            filename = f"cat_{cat_id:02d}.{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(img_data)
            size_kb = len(img_data) / 1024
            print(f"  {page_label}: 已保存 {filename} ({size_kb:.1f} KB)")
            success_count += 1
        else:
            print(f"  {page_label}: 提取失败!")
            fail_count += 1

    doc.close()

    print(f"\n完成! 成功: {success_count}, 失败: {fail_count}")
    print(f"图片保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
