# CLAUDE.md

## Environment

- **项目位置**：`/home/zby/vision_bcs/`（WSL ext4，已从 `/mnt/c/...` 迁移以避免 9p 文件系统的 I/O 性能损失）。
- **Python venv**：`/home/zby/vision_bcs/.venv/`（Python 3.12）。这是**完整的训练环境**，不是轻量 venv，包含 torch 2.10.0+cu128、transformers 5.2.0、peft、trl、datasets、qwen_vl_utils、bitsandbytes 0.49.2 等。
- **激活方式**：`source .venv/bin/activate`，或直接用 `.venv/bin/python script.py`。
- **GPU**：RTX 5080 Laptop 17GB，bf16 支持，通过 WSL CUDA 直通。

## Prompt 管理

- 所有 prompt 统一放在 `prompts/` 目录下，YAML 格式，按组件拆分（role、bcs_scale、confidence_guide 等），方便复用。
- 核心文件：`prompts/bcs_prompts.yaml`，包含 `role`、`bcs_scale`、`confidence_guide`、`breed_ids` 四个组件。
- 训练脚本和评分脚本应从 `prompts/` 加载，不要在代码中硬编码 system prompt 内容。

## 训练格式（LoRA 微调）

训练脚本 `finetune_qwen3_vl_4b_lora.py` 的对话格式：

- **System**：`role` + `bcs_scale` + `confidence_guide`（从 `prompts/bcs_prompts.yaml` 加载拼接）
- **User**：图片 + 固定指令 `"Assess this cat's Body Condition Score. Examine the visible body shape, waist definition, abdominal profile, rib coverage, and overall fat/muscle distribution. Respond with JSON only."`
- **Assistant**（训练目标）：`{"bcs_primary": int, "bcs_secondary": int, "reasoning": str, "confidence": int, "confidence_detractors": str, "breed_id": int}`

评分脚本（`llm_scoring.py` 等）推理时使用相同的 system + user 格式，只是不提供 assistant 回复，让模型生成。

## Rules

- **每一步操作之前都要 commit**：在执行任何可能修改数据文件（CSV、结果文件等）的操作之前，先 git commit 当前所有改动，防止中断导致数据丢失。
- 对已有结果文件做写回操作时，先备份再写入。
