#!/usr/bin/env python3
"""
analyze_results.py
从 LimeSurvey 收集问卷结果，结合 ChatGPT 评分数据，生成统计图表。

使用方法:
    python3 analyze_results.py

可选参数:
    --survey-id ID       LimeSurvey 问卷 ID
    --chatgpt-csv PATH   ChatGPT 评分结果 CSV (默认: chatgpt_results.csv)
    --dataset-csv PATH   标注数据集 CSV (默认: dataset.csv)
    --output-dir PATH    图表输出目录 (默认: results/)
    --skip-download      跳过从 LimeSurvey 下载，使用本地 human_responses.csv
"""

import argparse
import base64
import csv
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"请先安装依赖: pip install numpy pandas matplotlib seaborn")
    print(f"缺少: {e}")
    sys.exit(1)

try:
    from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
except ImportError:
    print("请安装 scikit-learn: pip install scikit-learn")
    sys.exit(1)

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("警告: pingouin 未安装，将跳过 ICC 计算。pip install pingouin")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 统一问卷与 AI 结果存放目录（与收集来的、AI 生成的两文件格式一致）
RESPONSES_DIR = os.path.join(BASE_DIR, "responses")
WIDE_BCS_COLS = [f"bcs{i:02d}" for i in range(1, 51)]  # bcs01..bcs50

# LimeSurvey 配置
LS_URL = "https://shawarmai.com/bcs/index.php/admin/remotecontrol"
LS_USER = "admin"
LS_PASS = "fh4829hd0h."

# 图表样式
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})
sns.set_theme(style="whitegrid", palette="Set2")


def _long_to_wide_response_df(long_df, source="human"):
    """将 long 格式 (respondent_id, image_id, effective_bcs) 转为 wide：一行一个被试，列 id, source, bcs01..bcs50。"""
    if long_df is None or len(long_df) == 0:
        return pd.DataFrame(columns=["id", "source"] + WIDE_BCS_COLS)
    wide_rows = []
    for rid, grp in long_df.groupby("respondent_id"):
        row = {"id": rid, "source": source}
        for _, r in grp.iterrows():
            col = f"bcs{int(r['image_id']):02d}"
            row[col] = r["effective_bcs"]
        for c in WIDE_BCS_COLS:
            if c not in row:
                row[c] = np.nan
        wide_rows.append(row)
    df = pd.DataFrame(wide_rows)
    return df[["id", "source"] + WIDE_BCS_COLS]


def _wide_to_long_response_df(wide_df):
    """将 wide 格式 (id, source, bcs01..bcs50) 转为 long：respondent_id, image_id, effective_bcs。"""
    if wide_df is None or len(wide_df) == 0:
        return pd.DataFrame(columns=["respondent_id", "image_id", "effective_bcs"])
    id_col = "id" if "id" in wide_df.columns else wide_df.columns[0]
    long_rows = []
    for _, row in wide_df.iterrows():
        rid = row[id_col]
        for i in range(1, 51):
            col = f"bcs{i:02d}"
            if col not in row:
                continue
            val = row[col]
            if pd.isna(val) or val == "":
                continue
            try:
                long_rows.append({
                    "respondent_id": rid,
                    "image_id": i,
                    "effective_bcs": float(val),
                })
            except (ValueError, TypeError):
                pass
    return pd.DataFrame(long_rows)


def download_limesurvey_responses(survey_id):
    """从 LimeSurvey 下载问卷回复；保存到 responses/human_responses.csv（与 AI 文件格式一致）。"""
    try:
        import requests
    except ImportError:
        print("需要 requests 库来下载问卷数据")
        return None

    print(f"从 LimeSurvey 下载问卷 {survey_id} 的回复...")

    # JSON-RPC 调用
    def rpc_call(method, params):
        payload = {"method": method, "params": params, "id": 1}
        headers = {"content-type": "application/json"}
        try:
            resp = requests.post(LS_URL, json=payload, headers=headers,
                                 timeout=60, verify=True)
        except Exception:
            resp = requests.post(LS_URL, json=payload, headers=headers,
                                 timeout=60, verify=False)
        return resp.json().get("result")

    # 获取 session key
    session_key = rpc_call("get_session_key", [LS_USER, LS_PASS])
    if isinstance(session_key, dict):
        print(f"登录失败: {session_key}")
        return None

    try:
        # 导出回复 (CSV 格式, base64 编码)
        result = rpc_call("export_responses", [
            session_key, str(survey_id), "csv", "en", "all", "code", "long"
        ])

        if isinstance(result, dict) and "status" in result:
            print(f"导出失败: {result['status']}")
            return None

        # 解码 base64（LimeSurvey 导出常用分号分隔）
        csv_data = base64.b64decode(result).decode("utf-8")
        raw_df = pd.read_csv(io.StringIO(csv_data), sep=";")

        # 解析为 long 格式
        long_df = parse_human_responses(raw_df)
        if long_df is not None and len(long_df) > 0:
            # 统一保存到 responses/，格式与 ai_responses 一致（一行一个被试，bcs01..bcs50）
            os.makedirs(RESPONSES_DIR, exist_ok=True)
            wide_df = _long_to_wide_response_df(long_df, source="human")
            human_path = os.path.join(RESPONSES_DIR, "human_responses.csv")
            wide_df.to_csv(human_path, index=False, encoding="utf-8")
            print(f"已保存到: {human_path} (共 {len(wide_df)} 条被试)")
        # 兼容：仍保存一份原始导出到项目根目录
        output_path = os.path.join(BASE_DIR, "human_responses.csv")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(csv_data)
        print(f"原始导出已保存到: {output_path}")

        return raw_df

    finally:
        rpc_call("release_session_key", [session_key])


def load_responses_wide(csv_path):
    """加载 wide 格式的 responses CSV（human 或 ai），列 id, source, bcs01..bcs50。"""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for c in WIDE_BCS_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def mean_deviation_closest_reference(wide_row, dataset_df):
    """
    按论文：每张图与 Scorer A、B 中较近者比较，取偏差；50 张图取平均。
    wide_row: 一行 wide 格式（Series 或 dict），含 bcs01..bcs50
    dataset_df: 含 image_id, scorer_a_bcs, scorer_b_bcs
    """
    a = dataset_df.set_index("image_id")["scorer_a_bcs"].to_dict()
    b = dataset_df.set_index("image_id")["scorer_b_bcs"].to_dict()
    devs = []
    for i in range(1, 51):
        col = f"bcs{i:02d}"
        s = wide_row.get(col, np.nan)
        if pd.isna(s) or s == "":
            continue
        try:
            score = float(s)
        except (ValueError, TypeError):
            continue
        ai, bi = a.get(i), b.get(i)
        if pd.isna(ai) or pd.isna(bi):
            continue
        dev_a = abs(score - float(ai))
        dev_b = abs(score - float(bi))
        devs.append(min(dev_a, dev_b))
    return np.mean(devs) if devs else np.nan


def parse_human_responses(df):
    """
    解析人类问卷回复，提取每个参与者对每张猫的 BCS 评分。

    返回: DataFrame，列为 [respondent_id, image_id, bcs, confidence, second_bcs]
    """
    results = []

    # 识别 BCS 评分列 (格式: bcsXX)
    bcs_cols = [c for c in df.columns if c.startswith("bcs") and not c.startswith("bcs2_")]
    conf_cols = [c for c in df.columns if c.startswith("conf")]
    bcs2_cols = [c for c in df.columns if c.startswith("bcs2_")]

    for resp_idx, row in df.iterrows():
        resp_id = row.get("id", resp_idx)

        for bcs_col in bcs_cols:
            # 提取 cat ID
            cat_id_str = bcs_col.replace("bcs", "")
            try:
                cat_id = int(cat_id_str)
            except ValueError:
                continue

            bcs_val = row.get(bcs_col)
            if pd.isna(bcs_val) or bcs_val == "":
                continue

            try:
                bcs = int(float(bcs_val))
            except (ValueError, TypeError):
                continue

            # 获取置信度
            conf_col = f"conf{cat_id:02d}"
            confidence = row.get(conf_col, "A")
            if pd.isna(confidence):
                confidence = "A"

            # 获取第二评分（LimeSurvey 导出列名为 bcstwo01 或 bcs2_01）
            bcs2_col = f"bcs2_{cat_id:02d}"
            if bcs2_col not in row.index and f"bcstwo{cat_id:02d}" in row.index:
                bcs2_col = f"bcstwo{cat_id:02d}"
            second_bcs = row.get(bcs2_col)
            if pd.isna(second_bcs) or second_bcs == "":
                second_bcs = None
            else:
                try:
                    second_bcs = int(float(second_bcs))
                except (ValueError, TypeError):
                    second_bcs = None

            # 计算有效 BCS
            if confidence == "B" and second_bcs is not None:
                effective_bcs = (bcs + second_bcs) / 2.0
            elif confidence == "C":
                effective_bcs = float(bcs)
            else:
                effective_bcs = float(bcs)

            results.append({
                "respondent_id": resp_id,
                "image_id": cat_id,
                "bcs": bcs,
                "confidence": confidence,
                "second_bcs": second_bcs,
                "effective_bcs": effective_bcs,
            })

    return pd.DataFrame(results)


def classify_bcs(bcs):
    """BCS 分类。"""
    if bcs <= 5:
        return "IW"
    elif bcs <= 7:
        return "OW"
    else:
        return "OB"


def load_dataset(csv_path):
    """加载标注数据集。"""
    return pd.read_csv(csv_path)


def load_chatgpt_results(csv_path):
    """加载 ChatGPT 评分结果。"""
    if not os.path.exists(csv_path):
        print(f"ChatGPT 结果文件不存在: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def generate_plots(dataset_df, human_df=None, chatgpt_df=None, output_dir="results"):
    """生成所有统计图表。"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 生成统计图表 ===")

    # ---- 1. Ground Truth 分布 ----
    plot_ground_truth_distribution(dataset_df, output_dir)

    # ---- 2. 专家评分者间一致性（原始9位评分者） ----
    plot_expert_agreement(dataset_df, output_dir)

    # ---- 3. 体重分类分布 ----
    plot_weight_class_distribution(dataset_df, output_dir)

    # ---- 4. 如果有人类问卷数据 ----
    if human_df is not None and len(human_df) > 0:
        plot_human_scores_boxplot(human_df, dataset_df, output_dir)
        plot_human_vs_ground_truth(human_df, dataset_df, output_dir)
        if "confidence" in human_df.columns:
            plot_human_confidence_distribution(human_df, output_dir)

    # ---- 5. 如果有 ChatGPT 数据 ----
    if chatgpt_df is not None and len(chatgpt_df) > 0:
        plot_chatgpt_vs_ground_truth(chatgpt_df, output_dir)
        plot_chatgpt_confusion_matrix(chatgpt_df, output_dir)

    # ---- 6. 三方对比（如果都有数据且人类数据非空） ----
    if (human_df is not None and len(human_df) > 0 and chatgpt_df is not None
            and "image_id" in human_df.columns):
        plot_three_way_comparison(dataset_df, human_df, chatgpt_df, output_dir)

    # ---- 7. ICC 分析 ----
    if HAS_PINGOUIN:
        compute_icc(dataset_df, human_df, output_dir)

    # ---- 8. 每条数据的平均偏差（最重要）：与 Scorer A/B 中较近者比较，逐条输出并绘图 ----
    human_wide_path = os.path.join(RESPONSES_DIR, "human_responses.csv")
    ai_wide_path = os.path.join(RESPONSES_DIR, "ai_responses.csv")
    human_wide = load_responses_wide(human_wide_path)
    ai_wide = load_responses_wide(ai_wide_path)
    if human_wide is not None or ai_wide is not None:
        save_mean_deviation_per_record(dataset_df, human_wide, ai_wide, output_dir)
        plot_mean_deviation_closest_reference(
            dataset_df, human_wide, ai_wide, output_dir
        )

    print(f"\n所有图表已保存到: {output_dir}/")


def save_mean_deviation_per_record(dataset_df, human_wide, ai_wide, output_dir):
    """
    为每条数据（每位人类被试、每次 AI 运行）计算平均偏差（与 Scorer A/B 中较近者比较），
    保存到 CSV。这是最重要的输出。
    """
    rows = []
    if human_wide is not None and len(human_wide) > 0:
        id_col = "id" if "id" in human_wide.columns else human_wide.columns[0]
        for _, row in human_wide.iterrows():
            md = mean_deviation_closest_reference(row, dataset_df)
            if not np.isnan(md):
                rows.append({"id": row[id_col], "source": "human", "mean_deviation": round(md, 4)})
    if ai_wide is not None and len(ai_wide) > 0:
        id_col = "id" if "id" in ai_wide.columns else ai_wide.columns[0]
        for _, row in ai_wide.iterrows():
            md = mean_deviation_closest_reference(row, dataset_df)
            if not np.isnan(md):
                rows.append({"id": row[id_col], "source": "ai", "mean_deviation": round(md, 4)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "mean_deviation_per_record.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  已保存（每条数据的平均偏差）: {path}")
    # 打印汇总
    for src in ["human", "ai"]:
        sub = df[df["source"] == src]
        if len(sub) > 0:
            print(f"    {src}: n={len(sub)}, mean={sub['mean_deviation'].mean():.3f}, "
                  f"min={sub['mean_deviation'].min():.3f}, max={sub['mean_deviation'].max():.3f}")


def plot_mean_deviation_closest_reference(dataset_df, human_wide, ai_wide, output_dir):
    """
    按论文 Figure 1D/1E：与 Scorer A、B 中较近者比较的偏差，计算人与 AI 的平均偏差并绘图。
    """
    human_devs = []
    if human_wide is not None and len(human_wide) > 0:
        for _, row in human_wide.iterrows():
            md = mean_deviation_closest_reference(row, dataset_df)
            if not np.isnan(md):
                human_devs.append(md)
    ai_devs = []
    if ai_wide is not None and len(ai_wide) > 0:
        for _, row in ai_wide.iterrows():
            md = mean_deviation_closest_reference(row, dataset_df)
            if not np.isnan(md):
                ai_devs.append(md)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels, data, colors = [], [], []
    if human_devs:
        labels.append("Human")
        data.append(human_devs)
        colors.append("#4C72B0")
    if ai_devs:
        labels.append("AI")
        data.append(ai_devs)
        colors.append("#DD8452")
    if not data:
        plt.close(fig)
        return

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean deviation (closest of Scorer A or B)")
    ax.set_title("BCS Mean Deviation vs Answer Key (Closest Reference)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # 标注均值
    for i, (label, devs) in enumerate(zip(labels, data)):
        mean_d = np.mean(devs)
        n = len(devs)
        se = np.std(devs, ddof=1) / np.sqrt(n) if n > 1 else 0
        ax.text(i + 1, mean_d + 0.02, f"mean={mean_d:.2f}\nn={n}", ha="center", fontsize=9)

    path = os.path.join(output_dir, "11_mean_deviation_human_vs_ai.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_ground_truth_distribution(dataset_df, output_dir):
    """Ground Truth BCS 分布直方图。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    gt = dataset_df["ground_truth"]
    bins = np.arange(0.5, 10.5, 1)
    ax.hist(gt, bins=bins, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.set_xlabel("Ground Truth BCS")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ground Truth BCS (Scorer A & B Average)")
    ax.set_xticks(range(1, 10))
    ax.axvline(gt.mean(), color="red", linestyle="--", label=f"Mean = {gt.mean():.2f}")
    ax.legend()

    path = os.path.join(output_dir, "01_ground_truth_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_expert_agreement(dataset_df, output_dir):
    """原始9位专家评分者的评分热力图。"""
    scorer_cols = ["scorer_a_bcs", "scorer_b_bcs"] + \
                  [f"scorer_{c}_bcs" for c in "cdefghi"]
    available_cols = [c for c in scorer_cols if c in dataset_df.columns]

    if len(available_cols) < 2:
        return

    scorer_data = dataset_df[available_cols].astype(float)
    scorer_labels = [c.replace("_bcs", "").replace("scorer_", "").upper()
                     for c in available_cols]

    # 评分者间相关矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = scorer_data.corr()
    corr.index = scorer_labels
    corr.columns = scorer_labels
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0.5, vmax=1.0,
                ax=ax, square=True)
    ax.set_title("Inter-Evaluator Correlation Matrix (9 Expert Scorers)")

    path = os.path.join(output_dir, "02_expert_correlation_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")

    # 评分者偏差箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    deviations = pd.DataFrame()
    gt = dataset_df["ground_truth"].astype(float)
    for col, label in zip(available_cols, scorer_labels):
        dev = scorer_data[col] - gt
        deviations[label] = dev

    deviations.boxplot(ax=ax)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Scorer")
    ax.set_ylabel("Deviation from Ground Truth")
    ax.set_title("Expert Scorer Deviations from Ground Truth (A&B Average)")

    path = os.path.join(output_dir, "03_expert_deviation_boxplot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_weight_class_distribution(dataset_df, output_dir):
    """体重分类饼图。"""
    fig, ax = plt.subplots(figsize=(8, 8))
    counts = dataset_df["weight_class"].value_counts()
    labels = {"IW": "Ideal Weight (BCS 1-5)", "OW": "Overweight (BCS 6-7)",
              "OB": "Obese (BCS 8-9)"}
    colors = {"IW": "#55a868", "OW": "#c44e52", "OB": "#dd8452"}

    plot_labels = [labels.get(c, c) for c in counts.index]
    plot_colors = [colors.get(c, "#4c72b0") for c in counts.index]

    wedges, texts, autotexts = ax.pie(
        counts.values, labels=plot_labels, colors=plot_colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11}
    )
    ax.set_title("Weight Classification Distribution (Based on Ground Truth)")

    path = os.path.join(output_dir, "04_weight_class_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_human_scores_boxplot(human_df, dataset_df, output_dir):
    """每张图片的人类评分分布箱线图。"""
    fig, ax = plt.subplots(figsize=(16, 8))

    # 按 ground truth 排序
    gt_map = dataset_df.set_index("image_id")["ground_truth"].to_dict()
    human_df["ground_truth"] = human_df["image_id"].map(gt_map)

    # 按 ground truth 排序的 image_id
    sorted_ids = sorted(human_df["image_id"].unique(),
                        key=lambda x: gt_map.get(x, 0))

    plot_data = []
    for img_id in sorted_ids:
        scores = human_df[human_df["image_id"] == img_id]["effective_bcs"]
        for s in scores:
            plot_data.append({"Cat ID": str(img_id), "BCS": s})

    plot_df = pd.DataFrame(plot_data)
    if len(plot_df) > 0:
        sns.boxplot(x="Cat ID", y="BCS", data=plot_df, ax=ax, order=[str(i) for i in sorted_ids])

        # 标注 ground truth
        for i, img_id in enumerate(sorted_ids):
            gt = gt_map.get(img_id, None)
            if gt:
                ax.scatter(i, gt, color="red", marker="*", s=100, zorder=5)

        ax.set_xlabel("Cat ID (sorted by Ground Truth)")
        ax.set_ylabel("BCS Score")
        ax.set_title("Human BCS Scores Distribution per Cat (Red star = Ground Truth)")
        ax.tick_params(axis="x", rotation=90, labelsize=8)

    path = os.path.join(output_dir, "05_human_scores_boxplot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_human_vs_ground_truth(human_df, dataset_df, output_dir):
    """人类评分 vs Ground Truth 散点图。"""
    fig, ax = plt.subplots(figsize=(10, 10))

    gt_map = dataset_df.set_index("image_id")["ground_truth"].to_dict()

    # 计算每张图片的人类平均评分
    avg_human = human_df.groupby("image_id")["effective_bcs"].mean()
    gt_vals = [gt_map.get(img_id, None) for img_id in avg_human.index]

    valid = [(g, h) for g, h in zip(gt_vals, avg_human.values) if g is not None]
    if valid:
        gt_arr, human_arr = zip(*valid)
        ax.scatter(gt_arr, human_arr, alpha=0.7, s=80, edgecolors="black", linewidth=0.5)

        # 完美一致线
        ax.plot([0, 10], [0, 10], "r--", alpha=0.5, label="Perfect Agreement")

        # 回归线
        z = np.polyfit(gt_arr, human_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(1, 9, 100)
        ax.plot(x_line, p(x_line), "b-", alpha=0.7,
                label=f"Regression (y={z[0]:.2f}x+{z[1]:.2f})")

        # 相关系数
        corr = np.corrcoef(gt_arr, human_arr)[0, 1]
        mae = np.mean(np.abs(np.array(human_arr) - np.array(gt_arr)))
        ax.text(0.05, 0.95, f"r = {corr:.3f}\nMAE = {mae:.2f}",
                transform=ax.transAxes, fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Ground Truth BCS")
    ax.set_ylabel("Mean Human BCS")
    ax.set_title("Human Assessment vs Ground Truth")
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.legend()
    ax.set_aspect("equal")

    path = os.path.join(output_dir, "06_human_vs_ground_truth.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_human_confidence_distribution(human_df, output_dir):
    """人类评分置信度分布。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 整体置信度分布
    conf_counts = human_df["confidence"].value_counts()
    colors = {"A": "#55a868", "B": "#c44e52", "C": "#dd8452"}
    conf_colors = [colors.get(c, "#4c72b0") for c in conf_counts.index]

    axes[0].pie(conf_counts.values, labels=conf_counts.index, colors=conf_colors,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
    axes[0].set_title("Overall Confidence Level Distribution")

    # 按 BCS 分组的置信度
    conf_by_bcs = human_df.groupby(["bcs", "confidence"]).size().unstack(fill_value=0)
    conf_by_bcs.plot(kind="bar", stacked=True, ax=axes[1],
                     color=[colors.get(c, "#4c72b0") for c in conf_by_bcs.columns])
    axes[1].set_xlabel("BCS Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Level by BCS Score")
    axes[1].legend(title="Confidence")
    axes[1].tick_params(axis="x", rotation=0)

    path = os.path.join(output_dir, "07_human_confidence_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_chatgpt_vs_ground_truth(chatgpt_df, output_dir):
    """ChatGPT 评分 vs Ground Truth。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    valid = chatgpt_df.dropna(subset=["chatgpt_effective_bcs", "ground_truth"])
    if len(valid) == 0:
        plt.close(fig)
        return

    gt = valid["ground_truth"].astype(float)
    pred = valid["chatgpt_effective_bcs"].astype(float)

    # 散点图
    ax = axes[0]
    ax.scatter(gt, pred, alpha=0.7, s=80, edgecolors="black", linewidth=0.5)
    ax.plot([0, 10], [0, 10], "r--", alpha=0.5, label="Perfect Agreement")

    z = np.polyfit(gt, pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, 9, 100)
    ax.plot(x_line, p(x_line), "b-", alpha=0.7,
            label=f"Regression (y={z[0]:.2f}x+{z[1]:.2f})")

    corr = np.corrcoef(gt, pred)[0, 1]
    mae = np.mean(np.abs(pred - gt))
    ax.text(0.05, 0.95, f"r = {corr:.3f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Ground Truth BCS")
    ax.set_ylabel("ChatGPT BCS")
    ax.set_title("ChatGPT vs Ground Truth")
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.legend()
    ax.set_aspect("equal")

    # 偏差分布
    ax = axes[1]
    deviation = pred - gt
    ax.hist(deviation, bins=np.arange(-4.5, 5.5, 1), edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.axvline(deviation.mean(), color="blue", linestyle="--",
               label=f"Mean = {deviation.mean():.2f}")
    ax.set_xlabel("Deviation (ChatGPT - Ground Truth)")
    ax.set_ylabel("Count")
    ax.set_title("ChatGPT Scoring Deviation Distribution")
    ax.legend()

    path = os.path.join(output_dir, "08_chatgpt_vs_ground_truth.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_chatgpt_confusion_matrix(chatgpt_df, output_dir):
    """ChatGPT 体重分类混淆矩阵。"""
    valid = chatgpt_df.dropna(subset=["weight_class_gt", "weight_class_chatgpt"])
    if len(valid) == 0:
        return

    labels = ["IW", "OW", "OB"]
    label_names = ["Ideal Weight", "Overweight", "Obese"]

    y_true = valid["weight_class_gt"]
    y_pred = valid["weight_class_chatgpt"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("ChatGPT Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("ChatGPT Weight Classification Confusion Matrix")

    # 添加准确率信息
    accuracy = np.trace(cm) / np.sum(cm) * 100
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    ax.text(0.5, -0.12, f"Accuracy: {accuracy:.1f}% | Cohen's Kappa: {kappa:.3f}",
            transform=ax.transAxes, ha="center", fontsize=12)

    path = os.path.join(output_dir, "09_chatgpt_confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")

    # 打印分类报告
    print("\n  ChatGPT 分类报告:")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=label_names))


def plot_three_way_comparison(dataset_df, human_df, chatgpt_df, output_dir):
    """Ground Truth vs Human vs ChatGPT 三方对比。"""
    fig, ax = plt.subplots(figsize=(16, 8))

    gt_map = dataset_df.set_index("image_id")["ground_truth"].to_dict()

    # 人类平均分
    human_avg = human_df.groupby("image_id")["effective_bcs"].mean().to_dict()

    # ChatGPT 分数
    chatgpt_map = {}
    for _, row in chatgpt_df.iterrows():
        img_id = int(row["image_id"])
        if pd.notna(row.get("chatgpt_effective_bcs")):
            chatgpt_map[img_id] = float(row["chatgpt_effective_bcs"])

    # 合并数据
    all_ids = sorted(set(gt_map.keys()) & (set(human_avg.keys()) | set(chatgpt_map.keys())))
    if not all_ids:
        all_ids = sorted(gt_map.keys())

    x = np.arange(len(all_ids))
    width = 0.25

    gt_vals = [gt_map.get(i, 0) for i in all_ids]
    human_vals = [human_avg.get(i, 0) for i in all_ids]
    chatgpt_vals = [chatgpt_map.get(i, 0) for i in all_ids]

    bars1 = ax.bar(x - width, gt_vals, width, label="Ground Truth", color="#4C72B0", alpha=0.8)
    if any(v > 0 for v in human_vals):
        bars2 = ax.bar(x, human_vals, width, label="Human Average", color="#55A868", alpha=0.8)
    if any(v > 0 for v in chatgpt_vals):
        bars3 = ax.bar(x + width, chatgpt_vals, width, label="ChatGPT", color="#C44E52", alpha=0.8)

    ax.set_xlabel("Cat ID")
    ax.set_ylabel("BCS Score")
    ax.set_title("Three-Way BCS Comparison: Ground Truth vs Human vs ChatGPT")
    ax.set_xticks(x)
    ax.set_xticklabels(all_ids, rotation=90, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 10)

    path = os.path.join(output_dir, "10_three_way_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def compute_icc(dataset_df, human_df, output_dir):
    """计算组内相关系数 (ICC)。"""
    if not HAS_PINGOUIN:
        return

    print("\n  计算 ICC (Intraclass Correlation Coefficient)...")

    # 原始专家评分者的 ICC
    scorer_cols = ["scorer_a_bcs", "scorer_b_bcs"] + \
                  [f"scorer_{c}_bcs" for c in "cdefghi"]
    available_cols = [c for c in scorer_cols if c in dataset_df.columns]

    if len(available_cols) >= 2:
        # 转换为长格式
        long_data = []
        for _, row in dataset_df.iterrows():
            img_id = row["image_id"]
            for col in available_cols:
                scorer = col.replace("_bcs", "")
                long_data.append({
                    "image_id": img_id,
                    "scorer": scorer,
                    "bcs": float(row[col])
                })

        long_df = pd.DataFrame(long_data)

        try:
            icc_result = pg.intraclass_corr(
                data=long_df, targets="image_id", raters="scorer", ratings="bcs"
            )
            icc_path = os.path.join(output_dir, "icc_expert_results.csv")
            icc_result.to_csv(icc_path, index=False)
            print(f"  专家 ICC 结果已保存: {icc_path}")
            print(icc_result[["Type", "ICC", "CI95%"]].to_string(index=False))
        except Exception as e:
            print(f"  ICC 计算失败: {e}")

    # 人类问卷评分者的 ICC
    if human_df is not None and len(human_df) > 0:
        try:
            icc_human = pg.intraclass_corr(
                data=human_df, targets="image_id", raters="respondent_id",
                ratings="effective_bcs"
            )
            icc_path = os.path.join(output_dir, "icc_human_results.csv")
            icc_human.to_csv(icc_path, index=False)
            print(f"  人类问卷 ICC 结果已保存: {icc_path}")
        except Exception as e:
            print(f"  人类问卷 ICC 计算失败: {e}")


def generate_summary_report(dataset_df, human_df, chatgpt_df, output_dir):
    """生成文字摘要报告。"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("BCS Visual Assessment Study - Summary Report")
    report_lines.append("=" * 60)

    # 数据集统计
    report_lines.append(f"\n--- Dataset Overview ---")
    report_lines.append(f"Total images: {len(dataset_df)}")
    gt = dataset_df["ground_truth"]
    report_lines.append(f"Ground Truth range: {gt.min():.1f} - {gt.max():.1f}")
    report_lines.append(f"Ground Truth mean: {gt.mean():.2f} (SD: {gt.std():.2f})")

    wc = dataset_df["weight_class"].value_counts()
    report_lines.append(f"Weight classes: IW={wc.get('IW',0)}, "
                        f"OW={wc.get('OW',0)}, OB={wc.get('OB',0)}")

    # 专家评分者统计
    scorer_cols = [c for c in dataset_df.columns if c.endswith("_bcs")]
    if scorer_cols:
        report_lines.append(f"\n--- Expert Scorers (n={len(scorer_cols)}) ---")
        gt_vals = dataset_df["ground_truth"].values
        for col in scorer_cols:
            scorer_vals = dataset_df[col].astype(float).values
            mae = np.mean(np.abs(scorer_vals - gt_vals))
            report_lines.append(f"  {col}: MAE = {mae:.2f}")

    # ChatGPT 统计
    if chatgpt_df is not None and len(chatgpt_df) > 0:
        valid = chatgpt_df.dropna(subset=["chatgpt_effective_bcs"])
        if len(valid) > 0:
            report_lines.append(f"\n--- ChatGPT Assessment ---")
            report_lines.append(f"Scored images: {len(valid)}/{len(chatgpt_df)}")
            dev = valid["chatgpt_effective_bcs"].astype(float) - valid["ground_truth"].astype(float)
            report_lines.append(f"MAE: {np.mean(np.abs(dev)):.2f}")
            report_lines.append(f"Mean deviation: {dev.mean():.2f} (SD: {dev.std():.2f})")

    # 人类问卷统计
    if human_df is not None and len(human_df) > 0:
        report_lines.append(f"\n--- Human Survey Assessment ---")
        n_respondents = human_df["respondent_id"].nunique()
        report_lines.append(f"Respondents: {n_respondents}")
        report_lines.append(f"Total ratings: {len(human_df)}")

        gt_map = dataset_df.set_index("image_id")["ground_truth"].to_dict()
        human_df_temp = human_df.copy()
        human_df_temp["gt"] = human_df_temp["image_id"].map(gt_map)
        valid_h = human_df_temp.dropna(subset=["gt"])
        if len(valid_h) > 0:
            dev = valid_h["effective_bcs"] - valid_h["gt"]
            report_lines.append(f"MAE: {np.mean(np.abs(dev)):.2f}")
            report_lines.append(f"Mean deviation: {dev.mean():.2f} (SD: {dev.std():.2f})")

    # 与 Scorer A/B 中较近者比较的平均偏差（论文 Figure 1D/1E 口径）
    human_wide_path = os.path.join(RESPONSES_DIR, "human_responses.csv")
    ai_wide_path = os.path.join(RESPONSES_DIR, "ai_responses.csv")
    human_wide = load_responses_wide(human_wide_path)
    ai_wide = load_responses_wide(ai_wide_path)
    if human_wide is not None or ai_wide is not None:
        report_lines.append(f"\n--- Mean Deviation vs Closest Reference (Scorer A or B) ---")
        if human_wide is not None and len(human_wide) > 0:
            human_devs = [mean_deviation_closest_reference(row, dataset_df) for _, row in human_wide.iterrows()]
            human_devs = [d for d in human_devs if not np.isnan(d)]
            if human_devs:
                report_lines.append(f"Human: n={len(human_devs)}, mean deviation = {np.mean(human_devs):.3f} (SD {np.std(human_devs):.3f})")
        if ai_wide is not None and len(ai_wide) > 0:
            ai_devs = [mean_deviation_closest_reference(row, dataset_df) for _, row in ai_wide.iterrows()]
            ai_devs = [d for d in ai_devs if not np.isnan(d)]
            if ai_devs:
                report_lines.append(f"AI: n={len(ai_devs)} runs, mean deviation = {np.mean(ai_devs):.3f} (SD {np.std(ai_devs):.3f})")

    report = "\n".join(report_lines)
    print(report)

    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BCS 评分结果分析与统计图表生成")
    parser.add_argument("--survey-id", type=int, default=None,
                        help="LimeSurvey 问卷 ID")
    parser.add_argument("--chatgpt-csv", default="chatgpt_results.csv",
                        help="ChatGPT 评分结果 CSV")
    parser.add_argument("--dataset-csv", default="dataset.csv",
                        help="标注数据集 CSV")
    parser.add_argument("--output-dir", default="results",
                        help="图表输出目录")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过从 LimeSurvey 下载")
    args = parser.parse_args()

    output_dir = os.path.join(BASE_DIR, args.output_dir)

    # 加载标注数据集
    dataset_path = os.path.join(BASE_DIR, args.dataset_csv)
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集: {dataset_path}")
        print("请先运行 build_dataset.py")
        sys.exit(1)

    dataset_df = load_dataset(dataset_path)
    print(f"已加载标注数据集: {len(dataset_df)} 条记录")

    # 加载 ChatGPT 结果
    chatgpt_path = os.path.join(BASE_DIR, args.chatgpt_csv)
    chatgpt_df = load_chatgpt_results(chatgpt_path)
    if chatgpt_df is not None:
        print(f"已加载 ChatGPT 结果: {len(chatgpt_df)} 条记录")

    # 获取人类问卷数据（优先使用 responses/ 内与 AI 格式一致的 wide 文件）
    human_df = None
    human_wide_path = os.path.join(RESPONSES_DIR, "human_responses.csv")
    human_csv_path = os.path.join(BASE_DIR, "human_responses.csv")

    if not args.skip_download and args.survey_id:
        # 从 LimeSurvey 下载（会保存到 responses/human_responses.csv）
        raw_df = download_limesurvey_responses(args.survey_id)
        if raw_df is not None:
            human_df = parse_human_responses(raw_df)
            print(f"已解析人类问卷: {len(human_df)} 条评分记录")
    elif os.path.exists(human_wide_path):
        # 使用 responses/ 内统一格式
        print(f"使用人类问卷数据: {human_wide_path}")
        wide_df = load_responses_wide(human_wide_path)
        if wide_df is not None and len(wide_df) > 0:
            human_df = _wide_to_long_response_df(wide_df)
            print(f"已解析人类问卷: {len(human_df)} 条评分记录")
    elif os.path.exists(human_csv_path):
        # 兼容旧路径（LimeSurvey 导出多为分号分隔）
        print(f"使用本地人类问卷数据: {human_csv_path}")
        try:
            raw_df = pd.read_csv(human_csv_path, sep=";")
        except Exception:
            raw_df = pd.read_csv(human_csv_path)
        if "effective_bcs" in raw_df.columns:
            human_df = raw_df
        else:
            human_df = parse_human_responses(raw_df)
        if human_df is not None:
            print(f"已解析人类问卷: {len(human_df)} 条评分记录")
    else:
        print("未找到人类问卷数据（可通过 --survey-id 指定问卷 ID 下载）")

    # 生成图表
    generate_plots(dataset_df, human_df, chatgpt_df, output_dir)

    # 生成摘要报告
    generate_summary_report(dataset_df, human_df, chatgpt_df, output_dir)


if __name__ == "__main__":
    main()
