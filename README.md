# BCS (Body Condition Score) 猫体况评分数据集与评估系统

本项目基于论文 *"Inter-evaluator bias and applicability of feline body condition score from visual assessment"* (Graff et al., 2025) 的数据，构建猫体况评分 (BCS) 的标注数据集，并通过 ChatGPT 和人类问卷两种方式进行评分评估。

## 项目结构

```
bcs/
├── images/                     # 从 PDF 提取的 50 张猫图片
│   ├── cat_01.jpeg ~ cat_50.jpeg
├── results/                    # 统计分析图表输出
├── extract_images.py           # 从 PDF 提取猫图片
├── build_dataset.py            # 构建标注数据集
├── llm_scoring.py              # 多模型自动评分（OpenAI/OpenRouter/本地）
├── setup_limesurvey.py         # LimeSurvey 问卷设置
├── analyze_results.py          # 统计分析与图表生成
├── dataset.csv                 # 标注数据集 (CSV)
├── dataset_annotated.json      # 详细标注数据 (JSON)
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

## 数据说明

### BCS 评分量表 (1-9 分)
- **1-3**: 偏瘦 (Underweight)
- **4-5**: 理想体重 (Ideal Weight, IW)
- **6-7**: 超重 (Overweight, OW)
- **8-9**: 肥胖 (Obese, OB)

### 评分置信度
- **A**: >90% 确信单一分数
- **B**: 两个相邻分数同等可能
- **C**: 倾向其中一个分数 (~75:25)

### Ground Truth
由两位资深临床医生 (Scorer A 和 Scorer B) 的平均 BCS 评分作为标准答案。

### 重复图片对
用于验证评分一致性：ID 2~50, 5~47, 42~49, 25~48

## 安装

```bash
pip install -r requirements.txt
```

## 使用流程

### 1. 提取图片

从 `DataS3_BTS1-Quiz_images.pdf` 中提取 50 张猫图片：

```bash
python3 extract_images.py
```

输出：`images/cat_01.jpeg` ~ `images/cat_50.jpeg`

### 2. 构建数据集

解析 DataS5 评分数据，生成标注数据集：

```bash
python3 build_dataset.py
```

输出：`dataset.csv`, `dataset_annotated.json`

### 3. 多模型自动评分

使用 OpenAI/OpenRouter/本地 OpenAI 兼容接口对每张图片进行 BCS 评分：

```bash
python3 llm_scoring.py --provider openai --model gpt-5.2

# 或 OpenRouter
python3 llm_scoring.py --provider openrouter --model qwen/qwen2.5-vl-72b-instruct

# 或本地 vLLM / OpenAI 兼容服务
python3 llm_scoring.py --provider local --model internvl2-8b --base-url http://127.0.0.1:8000/v1
```

可选参数：
- `--provider NAME`: 提供商，`openai/openrouter/local`
- `--model MODEL`: 模型名称（支持别名 `internvl2-8b`、`qwen2-vl-7b`）
- `--base-url URL`: OpenAI 兼容接口地址（本地部署常用）
- `--api-key KEY`: 覆盖环境变量中的 API key
- `--request-timeout S`: 单次请求超时秒数
- `--delay SECONDS`: 请求间隔 (默认: 1.0)

输出：仅追加到 `responses/ai_responses.csv`（包含 `mean_deviation` 列）

本地双模型部署（vLLM）可用：

```bash
python3 deploy_local_models_vllm.py --dry-run --target internvl2-8b
# 或
python3 deploy_local_models_vllm.py --dry-run --target qwen2-vl-7b
```

说明：默认不建议同时启动 7B+8B（容易显存不足导致系统卡死）。

批量实验（每模型 3 次）可用：

```bash
python3 run_model_experiments.py --runs 3 --models openrouter
# 本地模型建议分开跑：
python3 run_model_experiments.py --runs 3 --models internvl_local
python3 run_model_experiments.py --runs 3 --models qwen_local
```

### 4. 设置 LimeSurvey 问卷

通过 LimeSurvey API 创建人类评分问卷：

```bash
python3 setup_limesurvey.py
```

问卷服务器：`shawarmai.com/bcs`

### 5. 统计分析

收集问卷结果并生成统计图表：

```bash
# 仅基于现有数据集和 ChatGPT 结果
python3 analyze_results.py --skip-download

# 从 LimeSurvey 下载问卷数据
python3 analyze_results.py --survey-id YOUR_SURVEY_ID

# 使用本地问卷数据
python3 analyze_results.py --skip-download
```

输出图表（保存在 `results/` 目录）：
1. Ground Truth BCS 分布直方图
2. 专家评分者间相关矩阵热力图
3. 专家评分偏差箱线图
4. 体重分类分布饼图
5. 人类评分分布箱线图（需问卷数据）
6. 人类评分 vs Ground Truth 散点图（需问卷数据）
7. 置信度分布图（需问卷数据）
8. ChatGPT vs Ground Truth 散点图（需 ChatGPT 结果）
9. ChatGPT 体重分类混淆矩阵（需 ChatGPT 结果）
10. 三方对比柱状图（需全部数据）

## 数据来源

- **DataS3_BTS1-Quiz_images.pdf**: BTS1 测试集猫图片
- **DataS5_BTS1_VA-BCS.pdf**: 9 位评分者的 VA-BCS 评分数据
- **fvets-12-1604557.pdf**: 原始论文全文

## 参考文献

Graff EC, Lea CR, Delmain D, et al. (2025) Inter-evaluator bias and applicability of feline body condition score from visual assessment. *Front. Vet. Sci.* 12:1604557. doi: 10.3389/fvets.2025.1604557
