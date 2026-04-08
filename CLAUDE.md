# CLAUDE.md

## Environment

- **项目位置**：`/home/zby/vision_bcs/`（WSL ext4，已从 `/mnt/c/...` 迁移以避免 9p 文件系统的 I/O 性能损失）。
- **Python venv**：`/home/zby/vision_bcs/.venv/`（Python 3.12）。这是**完整的训练环境**，不是轻量 venv，包含 torch 2.10.0+cu128、transformers 5.2.0、peft、trl、datasets、qwen_vl_utils、bitsandbytes 0.49.2 等。
- **激活方式**：`source .venv/bin/activate`，或直接用 `.venv/bin/python script.py`。
- **GPU**：RTX 5080 Laptop 17GB，bf16 支持，通过 WSL CUDA 直通。

## Rules

- **每一步操作之前都要 commit**：在执行任何可能修改数据文件（CSV、结果文件等）的操作之前，先 git commit 当前所有改动，防止中断导致数据丢失。
- 对已有结果文件做写回操作时，先备份再写入。
