#!/usr/bin/env python3
"""
verify_cat_limbs.py
用 ViTPose (AP-10K 动物关键点) 检验 cat_10k 数据集，
去掉看不见身体和四肢的图片。

AP-10K 17 关键点:
  0  L_Eye        5  L_Shoulder   8  R_Shoulder  11 L_Hip       14 R_Hip
  1  R_Eye        6  L_Elbow      9  R_Elbow     12 L_Knee      15 R_Knee
  2  Nose         7  L_F_Paw     10  R_F_Paw     13 L_B_Paw     16 R_B_Paw
  3  Neck
  4  Root_of_Tail

判定标准:
  - 身体可见: Neck(3) 或 Root_of_Tail(4) 置信度 >= 阈值
  - 肢体可见: 4 条腿中至少 2 条有 >=1 个关键点置信度 >= 阈值
    左前腿: L_Shoulder(5), L_Elbow(6), L_F_Paw(7)
    右前腿: R_Shoulder(8), R_Elbow(9), R_F_Paw(10)
    左后腿: L_Hip(11), L_Knee(12), L_B_Paw(13)
    右后腿: R_Hip(14), R_Knee(15), R_B_Paw(16)

使用:
    python3 verify_cat_limbs.py                          # 检验并报告
    python3 verify_cat_limbs.py --remove                 # 检验并删除不合格图片
    python3 verify_cat_limbs.py --kpt-thr 0.3 --min-limbs 2
"""

import argparse
import os
import sys

import cv2
import numpy as np

try:
    from rtmlib import ViTPose
except ImportError:
    print("错误: pip install rtmlib")
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
DEFAULT_DATASET = os.path.join(BASE_DIR, "datasets", "cat_10k")

POSE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx"

AP10K_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_Tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]

# 四条腿的关键点索引
LIMB_GROUPS = {
    "left_front":  [5, 6, 7],    # L_Shoulder, L_Elbow, L_F_Paw
    "right_front": [8, 9, 10],   # R_Shoulder, R_Elbow, R_F_Paw
    "left_back":   [11, 12, 13], # L_Hip, L_Knee, L_B_Paw
    "right_back":  [14, 15, 16], # R_Hip, R_Knee, R_B_Paw
}

# 身体关键点索引
BODY_INDICES = [3, 4]  # Neck, Root_of_Tail


def check_body_and_limbs(
    keypoints: np.ndarray,
    scores: np.ndarray,
    img_w: int,
    img_h: int,
    kpt_thr: float = 0.5,
    min_limbs: int = 3,
    min_spread: float = 0.3,
) -> dict:
    """
    检查关键点是否满足身体+肢体可见条件。

    Args:
        keypoints: shape (17, 2) 关键点坐标
        scores: shape (17,) 每个关键点的置信度
        img_w, img_h: 图片尺寸
        kpt_thr: 关键点置信度阈值
        min_limbs: 最少可见肢体数
        min_spread: 关键点最小空间跨度（占图片高或宽的比例）

    Returns:
        dict with pass/fail info
    """
    # 身体: Neck 或 Root_of_Tail
    body_visible = any(scores[i] >= kpt_thr for i in BODY_INDICES)

    # 四肢
    visible_limbs = []
    for name, indices in LIMB_GROUPS.items():
        if any(scores[i] >= kpt_thr for i in indices):
            visible_limbs.append(name)

    # 空间分布检查: 防止模型在脸部特写上幻觉出肢体关键点
    # 所有高置信度关键点应跨越图片的一定比例
    confident_mask = scores >= kpt_thr
    spread_ok = False
    spread_ratio = 0.0
    if confident_mask.sum() >= 2:
        pts = keypoints[confident_mask]
        x_spread = pts[:, 0].max() - pts[:, 0].min()
        y_spread = pts[:, 1].max() - pts[:, 1].min()
        spread_ratio = max(
            x_spread / img_w if img_w > 0 else 0,
            y_spread / img_h if img_h > 0 else 0,
        )
        spread_ok = spread_ratio >= min_spread
    elif confident_mask.sum() == 1:
        spread_ok = False  # 只有一个点，不可能是完整猫

    passed = body_visible and len(visible_limbs) >= min_limbs and spread_ok

    return {
        "passed": passed,
        "body_visible": body_visible,
        "visible_limbs": visible_limbs,
        "num_visible_limbs": len(visible_limbs),
        "spread_ratio": round(float(spread_ratio), 3),
        "spread_ok": spread_ok,
        "body_scores": {AP10K_NAMES[i]: round(float(scores[i]), 4) for i in BODY_INDICES},
        "limb_max_scores": {
            name: round(float(max(scores[i] for i in indices)), 4)
            for name, indices in LIMB_GROUPS.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="用 ViTPose 检验猫身体和四肢可见性")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    parser.add_argument("--kpt-thr", type=float, default=0.5,
                        help="关键点置信度阈值 (默认 0.5)")
    parser.add_argument("--min-limbs", type=int, default=3,
                        help="最少可见肢体数 (默认 3)")
    parser.add_argument("--min-spread", type=float, default=0.3,
                        help="关键点最小空间跨度比例 (默认 0.3)")
    parser.add_argument("--remove", action="store_true",
                        help="删除不合格图片及其 DB 记录")
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset_dir, "images")
    raw_dir = os.path.join(args.dataset_dir, "raw_images")
    db_path = os.path.join(args.dataset_dir, "cat_db.json")

    if not os.path.isdir(images_dir):
        print(f"错误: {images_dir} 不存在")
        return 1

    db = TinyDB(db_path)
    CatQ = Query()

    # 加载 ViTPose
    print(f"加载 ViTPose (AP-10K animal keypoints)...")
    pose_model = ViTPose(
        POSE_URL,
        model_input_size=(192, 256),
        backend="onnxruntime",
        device="cpu",
    )

    # 收集图片
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"共 {len(image_files)} 张图片待检验")
    print(f"阈值: kpt_thr={args.kpt_thr}, min_limbs={args.min_limbs}")

    passed_count = 0
    failed_count = 0
    error_count = 0
    failed_files = []

    iterator = tqdm(image_files, desc="ViTPose 检验") if tqdm else image_files

    for fname in iterator:
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            error_count += 1
            failed_files.append((fname, "读取失败"))
            continue

        try:
            h, w = img.shape[:2]
            # 对裁剪图直接推理，bbox 设为全图
            keypoints, scores = pose_model(img, bboxes=np.array([[0, 0, w, h]]))
        except Exception as e:
            error_count += 1
            failed_files.append((fname, f"推理失败: {e}"))
            continue

        if scores is None or len(scores) == 0:
            failed_count += 1
            failed_files.append((fname, "无关键点"))
            continue

        result = check_body_and_limbs(
            keypoints[0], scores[0], w, h,
            kpt_thr=args.kpt_thr,
            min_limbs=args.min_limbs,
            min_spread=args.min_spread,
        )

        if result["passed"]:
            passed_count += 1
        else:
            failed_count += 1
            reason_parts = []
            if not result["body_visible"]:
                reason_parts.append("身体不可见")
            if result["num_visible_limbs"] < args.min_limbs:
                reason_parts.append(
                    f"肢体仅{result['num_visible_limbs']}条可见(<{args.min_limbs})"
                )
            if not result["spread_ok"]:
                reason_parts.append(
                    f"关键点过于集中(spread={result['spread_ratio']:.2f}<{args.min_spread})"
                )
            reason = "; ".join(reason_parts)
            failed_files.append((fname, reason))

    total = passed_count + failed_count + error_count
    print(f"\n=== 检验结果 ===")
    print(f"  通过: {passed_count}/{total}")
    print(f"  不合格: {failed_count}")
    print(f"  错误: {error_count}")

    if failed_files:
        print(f"\n不合格图片 (前 20 个):")
        for fname, reason in failed_files[:20]:
            print(f"  {fname}: {reason}")
        if len(failed_files) > 20:
            print(f"  ... 共 {len(failed_files)} 个")

    # 删除不合格图片
    if args.remove and failed_files:
        print(f"\n正在删除 {len(failed_files)} 张不合格图片...")
        removed = 0
        for fname, _ in failed_files:
            md5 = os.path.splitext(fname)[0]

            # 删 images/
            img_path = os.path.join(images_dir, fname)
            if os.path.exists(img_path):
                os.remove(img_path)

            # 删 raw_images/ (可能有不同扩展名)
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                raw_path = os.path.join(raw_dir, md5 + ext)
                if os.path.exists(raw_path):
                    os.remove(raw_path)

            # 删 DB 记录
            db.remove(CatQ.md5 == md5)
            removed += 1

        remaining = len(db)
        print(f"  已删除: {removed}")
        print(f"  数据库剩余: {remaining}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
