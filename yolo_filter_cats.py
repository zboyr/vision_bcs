#!/usr/bin/env python3
"""
yolo_filter_cats.py
使用 YOLO 检测模型筛选出毛发和四肢完全可见的猫图片。

筛选条件:
    1. 图中检测到猫 (COCO class 15, confidence >= 阈值)
    2. 猫的 bounding box 不触碰图片边缘 (四肢未被裁切)
    3. 猫在图中占比足够大 (能看清全身)
    4. 仅有一只主要猫 (避免遮挡)
    5. 检测置信度高 (猫轮廓清晰，说明毛发四肢可见)

使用方法:
    python yolo_filter_cats.py
    python yolo_filter_cats.py --input-dir cat_data/raw --output-dir cat_data/filtered
    python yolo_filter_cats.py --target 2000 --conf 0.7 --edge-margin 0.03
    python yolo_filter_cats.py --model yolo11m.pt --device cpu

依赖:
    pip install ultralytics
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "cat_data", "raw")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "cat_data", "filtered")

# COCO 类别索引: cat = 15
COCO_CAT_CLASS_ID = 15

# 图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_cat_fully_visible(
    box_xyxy,
    img_w: int,
    img_h: int,
    edge_margin_pct: float = 0.025,
    min_area_pct: float = 0.05,
    max_area_pct: float = 0.95,
) -> tuple:
    """
    判断猫是否全身可见（毛发四肢完整展示）。

    逻辑:
      - bounding box 不能触碰图片边缘 (否则说明四肢被裁切)
      - 猫的面积在合理范围内 (太小看不清, 太大可能只拍了局部)
      - 宽高比在合理范围 (过于极端的比例说明姿态不自然或被遮挡)

    Args:
        box_xyxy: [x1, y1, x2, y2] 像素坐标
        img_w, img_h: 图片尺寸
        edge_margin_pct: 边缘区域占图片的比例 (默认 2.5%)
        min_area_pct: 猫最小面积占比 (默认 5%)
        max_area_pct: 猫最大面积占比 (默认 95%)

    Returns:
        (is_visible: bool, details: dict)
    """
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    margin_x = img_w * edge_margin_pct
    margin_y = img_h * edge_margin_pct

    # 1. 检查 bbox 是否触碰边缘
    touches_left = x1 < margin_x
    touches_top = y1 < margin_y
    touches_right = x2 > (img_w - margin_x)
    touches_bottom = y2 > (img_h - margin_y)
    touches_any_edge = any([touches_left, touches_top, touches_right, touches_bottom])

    # 2. 检查猫的面积占比
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    area_ratio = box_area / img_area if img_area > 0 else 0

    area_ok = min_area_pct <= area_ratio <= max_area_pct

    # 3. 检查宽高比 (全身猫通常在 0.3:1 ~ 3:1 之间)
    box_w = x2 - x1
    box_h = y2 - y1
    aspect_ratio = box_w / box_h if box_h > 0 else 0
    aspect_ok = 0.3 <= aspect_ratio <= 3.0

    is_visible = not touches_any_edge and area_ok and aspect_ok

    details = {
        "touches_edge": touches_any_edge,
        "edge_sides": {
            "left": touches_left,
            "top": touches_top,
            "right": touches_right,
            "bottom": touches_bottom,
        },
        "area_ratio": round(area_ratio, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "area_ok": area_ok,
        "aspect_ok": aspect_ok,
    }

    return is_visible, details


def get_rejection_reason(details: dict) -> str:
    """根据 details 生成拒绝原因。"""
    reasons = []
    if details["touches_edge"]:
        sides = [k for k, v in details["edge_sides"].items() if v]
        reasons.append(f"触碰边缘({','.join(sides)})")
    if not details["area_ok"]:
        reasons.append(f"面积占比异常({details['area_ratio']:.1%})")
    if not details["aspect_ok"]:
        reasons.append(f"宽高比异常({details['aspect_ratio']:.2f})")
    return "; ".join(reasons) if reasons else "unknown"


def filter_images(
    input_dir: str,
    output_dir: str,
    model_name: str = "yolo11s.pt",
    conf_threshold: float = 0.65,
    edge_margin_pct: float = 0.025,
    min_area_pct: float = 0.05,
    max_area_pct: float = 0.95,
    target_count: int = 0,
    device: str = "",
    batch_size: int = 16,
) -> dict:
    """
    使用 YOLO 筛选猫图片。

    Args:
        input_dir: 输入图片目录
        output_dir: 输出图片目录
        model_name: YOLO 模型名称
        conf_threshold: 最低检测置信度
        edge_margin_pct: 边缘判定的比例
        min_area_pct: 最小面积占比
        max_area_pct: 最大面积占比
        target_count: 目标数量 (0=不限制)
        device: 推理设备 ('', 'cpu', 'cuda', '0', etc.)
        batch_size: 批处理大小

    Returns:
        统计字典
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有图片
    input_path = Path(input_dir)
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"错误: 在 {input_dir} 中未找到图片")
        return {"kept": 0, "rejected": 0}

    print(f"找到 {len(image_files)} 张图片")
    print(f"加载 YOLO 模型: {model_name}...")

    model = YOLO(model_name)

    # 记录结果
    kept = 0
    rejected = 0
    no_cat = 0
    results_log = []

    # 逐张处理 (YOLO batch 推理需要图片尺寸一致，逐张更灵活)
    if tqdm:
        iterator = tqdm(image_files, desc="YOLO 筛选")
    else:
        iterator = image_files

    for idx, img_path in enumerate(iterator):
        # 如果达到目标数量就停止
        if target_count > 0 and kept >= target_count:
            break

        try:
            results = model(
                str(img_path),
                conf=conf_threshold,
                classes=[COCO_CAT_CLASS_ID],
                device=device if device else None,
                verbose=False,
            )
        except Exception as e:
            print(f"\n  YOLO 推理失败: {img_path.name} -> {e}")
            rejected += 1
            results_log.append({
                "file": img_path.name,
                "kept": False,
                "reason": f"推理错误: {e}",
            })
            continue

        result = results[0]
        img_h, img_w = result.orig_shape

        # 没检测到猫
        if len(result.boxes) == 0:
            no_cat += 1
            rejected += 1
            results_log.append({
                "file": img_path.name,
                "kept": False,
                "reason": "未检测到猫",
            })
            continue

        # 取置信度最高的猫
        boxes = result.boxes
        best_idx = int(boxes.conf.argmax())
        best_conf = float(boxes.conf[best_idx])
        best_box = boxes.xyxy[best_idx].cpu().numpy()

        # 检查是否只有一只显著的猫 (避免多猫遮挡)
        num_cats = len(boxes)
        if num_cats > 1:
            # 如果有多只猫，检查最大猫是否显著大于其他猫
            all_areas = []
            for i in range(num_cats):
                bx = boxes.xyxy[i].cpu().numpy()
                all_areas.append((bx[2] - bx[0]) * (bx[3] - bx[1]))
            best_area = all_areas[best_idx]
            second_largest = max(a for i, a in enumerate(all_areas) if i != best_idx)
            # 如果第二大猫面积超过最大猫的 40%，认为多猫遮挡风险高
            if second_largest > best_area * 0.4:
                rejected += 1
                results_log.append({
                    "file": img_path.name,
                    "kept": False,
                    "reason": f"多猫({num_cats}只)，可能遮挡",
                    "conf": best_conf,
                })
                continue

        # 检查全身可见性
        is_visible, details = is_cat_fully_visible(
            best_box, img_w, img_h,
            edge_margin_pct=edge_margin_pct,
            min_area_pct=min_area_pct,
            max_area_pct=max_area_pct,
        )

        if is_visible:
            # 复制到输出目录
            dest = os.path.join(output_dir, img_path.name)
            shutil.copy2(str(img_path), dest)
            kept += 1
            results_log.append({
                "file": img_path.name,
                "kept": True,
                "conf": best_conf,
                "area_ratio": details["area_ratio"],
                "aspect_ratio": details["aspect_ratio"],
            })
        else:
            rejected += 1
            reason = get_rejection_reason(details)
            results_log.append({
                "file": img_path.name,
                "kept": False,
                "reason": reason,
                "conf": best_conf,
            })

        # 非 tqdm 进度显示
        if not tqdm and (idx + 1) % 100 == 0:
            print(f"\r  进度: {idx+1}/{len(image_files)} | 保留: {kept} | 拒绝: {rejected}",
                  end="", flush=True)

    if not tqdm:
        print()

    # 统计
    total_processed = kept + rejected
    stats = {
        "total_input": len(image_files),
        "total_processed": total_processed,
        "kept": kept,
        "rejected": rejected,
        "no_cat_detected": no_cat,
        "acceptance_rate": kept / total_processed * 100 if total_processed > 0 else 0,
    }

    print(f"\n=== YOLO 筛选结果 ===")
    print(f"  总输入: {stats['total_input']}")
    print(f"  已处理: {stats['total_processed']}")
    print(f"  保留: {stats['kept']}")
    print(f"  拒绝: {stats['rejected']} (未检测到猫: {stats['no_cat_detected']})")
    print(f"  通过率: {stats['acceptance_rate']:.1f}%")
    print(f"  输出目录: {output_dir}")

    # 保存筛选日志
    log_path = os.path.join(os.path.dirname(output_dir), "yolo_filter_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "kept", "reason", "conf", "area_ratio", "aspect_ratio"
        ])
        writer.writeheader()
        for entry in results_log:
            row = {
                "file": entry.get("file", ""),
                "kept": entry.get("kept", False),
                "reason": entry.get("reason", ""),
                "conf": entry.get("conf", ""),
                "area_ratio": entry.get("area_ratio", ""),
                "aspect_ratio": entry.get("aspect_ratio", ""),
            }
            writer.writerow(row)
    print(f"  筛选日志: {log_path}")

    # 拒绝原因统计
    from collections import Counter
    reject_reasons = Counter()
    for entry in results_log:
        if not entry.get("kept"):
            reason = entry.get("reason", "unknown")
            # 简化原因
            if "未检测到猫" in reason:
                reject_reasons["未检测到猫"] += 1
            elif "触碰边缘" in reason:
                reject_reasons["触碰图片边缘(四肢裁切)"] += 1
            elif "面积" in reason:
                reject_reasons["面积占比不合适"] += 1
            elif "宽高比" in reason:
                reject_reasons["宽高比异常"] += 1
            elif "多猫" in reason:
                reject_reasons["多猫遮挡风险"] += 1
            else:
                reject_reasons["其他"] += 1

    if reject_reasons:
        print(f"\n  拒绝原因分布:")
        for reason, count in reject_reasons.most_common():
            print(f"    {reason}: {count}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 YOLO 筛选全身可见的猫图片"
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT,
                        help=f"输入图片目录 (默认: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help=f"输出图片目录 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", default="yolo11s.pt",
                        help="YOLO 模型名称 (默认: yolo11s.pt)")
    parser.add_argument("--conf", type=float, default=0.65,
                        help="最低检测置信度 (默认: 0.65)")
    parser.add_argument("--edge-margin", type=float, default=0.025,
                        help="边缘判定比例 (默认: 0.025，即 2.5%%)")
    parser.add_argument("--min-area", type=float, default=0.05,
                        help="猫最小面积占比 (默认: 0.05，即 5%%)")
    parser.add_argument("--max-area", type=float, default=0.95,
                        help="猫最大面积占比 (默认: 0.95，即 95%%)")
    parser.add_argument("--target", type=int, default=0,
                        help="目标保留数量 (默认: 0=不限制)")
    parser.add_argument("--device", default="",
                        help="推理设备 (默认: 自动, 可选: cpu, cuda, 0)")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        print("请先运行 download_cats.py 下载图片")
        return 1

    stats = filter_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        conf_threshold=args.conf,
        edge_margin_pct=args.edge_margin,
        min_area_pct=args.min_area,
        max_area_pct=args.max_area,
        target_count=args.target,
        device=args.device,
    )

    if stats["kept"] == 0:
        print("\n警告: 没有图片通过筛选。尝试降低 --conf 或 --edge-margin 参数。")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
