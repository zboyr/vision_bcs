# Next Study: 7 Models x 7 Prompts BCS Scoring Experiment

## Overview

在 Essay 猫 BCS 数据集上，使用 **7 个模型** 与 **7 种 Prompt 策略** 的全组合（7 x 7 = 49 组实验），每组重复 3 次，系统比较不同模型能力与 Prompt 工程对 BCS 评分准确度的影响。

---

## Models (7)

| # | Model | Type | Provider | Notes |
|---|-------|------|----------|-------|
| M1 | Qwen2.5-VL-3B-Instruct | Open-source (3B) | Local (vLLM) | Baseline 小模型 |
| M2 | Qwen2.5-VL-3B-Instruct (fine-tuned) | Open-source (3B, LoRA) | Local (vLLM) | BCS fine-tuned |
| M3 | Gemma 4 | Open-source | Local (vLLM) | Google 开源模型 |
| M4 | Gemma 4 (fine-tuned) | Open-source (LoRA) | Local (vLLM) | BCS fine-tuned |
| M5 | Gemini-3.1-Pro-Preview | Proprietary | OpenRouter | Google 旗舰 |
| M6 | Gemini-3.1-Flash-Image-Preview | Proprietary | OpenRouter | Google 轻量多模态 |
| M7 | GPT-5.4 | Proprietary | OpenAI / OpenRouter | OpenAI 最新 |

---

## Prompts (7)

### P1: Direct Integer (直接输出整数)

- `output_mode: simple`
- 最简 prompt，仅要求模型输出一个 1-9 整数
- System prompt: 简洁 BCS 量表描述
- User prompt: "What is the BCS of this cat? Reply with a single integer (1-9)."

### P2: JSON Mode (结构化 JSON 输出)

- `output_mode: json`
- 完整 prompt，包含 BCS 量表详细描述 + 品种列表
- 要求输出结构化 JSON: `{bcs_primary, bcs_secondary, reasoning, confidence, confidence_detractors, breed_id}`
- 对应现有 `score_cat_10k.py` 中的 `SYSTEM_PROMPT`

### P3: Reasoning Mode (先推理再评分)

- `output_mode: reasoning`
- System prompt 引导模型先描述观察到的身体特征（肋骨、腰线、腹部等），再给出 BCS 评分
- 输出 JSON: `{reasoning, bcs}`
- 对应现有 `SYSTEM_PROMPT_REASONING` + `USER_PROMPT_REASONING`

### P4: Best-of-5 (BO5)

- 对同一张图片独立调用 5 次（temperature > 0），取 **众数 (majority vote)** 作为最终 BCS
- 底层 prompt 使用 P1 (Direct Integer) 以降低成本
- 目的：通过多次采样减少随机性，提升稳定性

### P5: Agent-as-a-Verifier v1 (AAV1)

- 两阶段流程：
  1. **Scorer Agent**: 使用 P2 (JSON Mode) 评分，输出 BCS + reasoning
  2. **Verifier Agent**: 接收图片 + Scorer 的输出，独立判断是否同意，若不同意则给出修正分数
- 最终取 Verifier 的分数（若 Verifier 同意则保留 Scorer 分数）

### P6: Visual Few-Shot (视觉参考图 Prompt)

- 在 prompt 中附带 **BCS 参考图集**（从已标注数据中选取 BCS 1/3/5/7/9 各一张典型猫图）
- 多图输入：参考图（带 BCS 标签）+ 待评图，共 6 张图片
- System prompt: 简洁 BCS 量表描述 + "以下是不同 BCS 等级的参考图片"
- User prompt: "参考以上示例图片的体型特征，评估最后一张图片的 BCS。输出 JSON: `{bcs, reasoning}`"
- 研究问题：**视觉参考（而非纯文字描述）能否提升 VLM 的 BCS 评分准确度？**

### P7: Debate v1

- 多 Agent 辩论：
  1. **Agent A**: 独立评分（P3 Reasoning Mode），输出 BCS + reasoning
  2. **Agent B**: 独立评分（P3 Reasoning Mode），输出 BCS + reasoning
  3. 若 A、B 分数不一致 → 进入 **辩论轮**：双方看到对方 reasoning，各自给出修正分数
  4. 最终取辩论后的平均值（四舍五入）
- 目的：通过对抗性讨论提升评分可靠性

---

## Experiment Matrix

|  | P1 Direct | P2 JSON | P3 Reasoning | P4 BO5 | P5 AAV1 | P6 VFewShot | P7 Debate |
|--|-----------|---------|-------------|--------|---------|---------|-----------|
| **M1** Qwen2.5-3B | x3 | x3 | x3 | x3 | x3 | x3 | x3 |
| **M2** Qwen2.5-3B-FT | x3 | x3 | x3 | x3 | x3 | x3 | x3 |
| **M3** Gemma 4 | x3 | x3 | x3 | x3 | x3 | x3 | x3 |
| **M4** Gemma 4-FT | x3 | x3 | x3 | x3 | x3 | x3 | x3 |
| **M5** Gemini-3.1-Pro | x3 | x3 | x3 | x3 | x3 | x3 | x3 |
| **M6** Gemini-3.1-Flash | x3 | x3 | x3 | x3 | x3 | x3 | x3 |
| **M7** GPT-5.4 | x3 | x3 | x3 | x3 | x3 | x3 | x3 |

- 每格 x3 = 3 次重复运行
- 共 7 x 7 x 3 = **147 轮评分**
- P4 (BO5) 每轮内部 5 次调用 → 实际 API 调用更多
- P5-P7 每轮内部多次调用 → 实际 API 调用量倍增

---

## Dataset

- **Essay Cat BCS Dataset** (`cat_data/essay/dataset_full.csv`)
- 49 张猫图片，每张有 ground truth BCS (由专业兽医标注)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Mean Deviation** | `mean(abs(predicted - ground_truth))`，越低越好 |
| **Exact Match Rate** | 预测 == ground truth 的比例 |
| **Within-1 Accuracy** | `abs(predicted - ground_truth) <= 1` 的比例 |
| **Run Variance** | 3 次重复间的标准差，衡量稳定性 |
| **Cohen's Kappa** | 与 ground truth 的一致性系数 |

---

## Implementation Plan

### Phase 1: Infrastructure

1. 扩展 `llm_scoring.py` 支持新 output_mode:
   - `bo5`: Best-of-5 majority vote
   - `aav1`: Agent-as-a-Verifier v1
   - `vfewshot`: Visual Few-Shot (多图参考)
   - `debate`: Debate v1
2. 为每种 prompt 策略编写对应的 system/user prompt
3. 创建 7 x 7 = 49 个 YAML 配置文件（或支持 batch config）

### Phase 2: Fine-tuning

1. Fine-tune Qwen2.5-VL-3B on BCS 数据 → M2
2. Fine-tune Gemma 4 on BCS 数据 → M4
3. 验证 fine-tuned 模型可通过 vLLM 部署

### Phase 3: Execution

1. 部署本地模型 (M1-M4) via vLLM
2. 按模型依次执行所有 prompt 策略
3. API 模型 (M5-M7) 通过 OpenRouter/OpenAI 执行
4. 每步操作前 git commit

### Phase 4: Analysis

1. 汇总所有 49 组实验结果
2. 按模型维度分析：哪些模型整体最优
3. 按 Prompt 维度分析：哪种策略整体最优
4. 交互效应分析：特定 model-prompt 组合是否有惊喜
5. Fine-tuning 增益分析：M2 vs M1, M4 vs M3
6. 成本效益分析：API 调用次数 vs 准确度提升

---

## Expected Outputs

```
responses/
  essay_results.csv          # 所有结果汇总（追加写入）

results/
  report/
    next_study_results.tex   # LaTeX 报告
    figures/                 # 生成图表
      heatmap_model_prompt.png
      bar_mean_deviation.png
      box_run_variance.png
```

---

## Research Questions

1. **Prompt 策略对评分准确度的影响有多大？** P1 (最简) vs P2 (JSON) vs P3 (Reasoning) 的差距是否显著？
2. **Multi-agent 策略 (P5/P7) 是否优于 single-call 策略 (P1-P3)?** 额外的 API 成本是否值得？
3. **BO5 majority vote 能否有效降低 variance?**
4. **Visual Few-Shot (P6) 是否优于纯文字描述 (P2/P3)?** 视觉参考能否帮助模型建立更准确的 BCS 体型映射？
5. **Fine-tuning 小模型能否追上甚至超过大型 proprietary 模型？** M2 vs M5/M6/M7
6. **模型大小与 Prompt 复杂度是否有交互效应？** 小模型是否更受益于简单 prompt？大模型是否更能利用复杂 prompt？
