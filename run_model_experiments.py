#!/usr/bin/env python3
"""
run_model_experiments.py
批量运行多模型实验：每个模型运行多次（默认 3 次）。

注意：不再生成任何新的结果 CSV。
所有结果由 llm_scoring.py 直接追加到 responses/ai_responses.csv。
"""

import argparse
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def check_local_endpoint(base_url: str, timeout_seconds: float = 3.0) -> tuple[bool, str]:
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_seconds):
            return True, "ok"
    except urllib.error.URLError as e:
        return False, f"{models_url} 不可用: {e}"
    except Exception as e:  # pragma: no cover
        return False, f"{models_url} 检查失败: {e}"


def parse_run_id(output_text: str) -> str:
    match = re.search(r"RUN_ID=([0-9_]+)", output_text)
    if match:
        return match.group(1)
    return ""


def pick_message(stderr_text: str, stdout_text: str) -> str:
    merged = (stderr_text or "") + "\n" + (stdout_text or "")
    lines = [line.strip() for line in merged.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1][:300]


def run_once(provider: str, model: str, run_idx: int, base_url: Optional[str],
             dataset: str, delay: float, max_retries: int, request_timeout: float,
             per_run_timeout: int) -> dict[str, str]:
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "llm_scoring.py"),
        "--provider", provider,
        "--model", model,
        "--dataset", dataset,
        "--delay", str(delay),
        "--max-retries", str(max_retries),
        "--request-timeout", str(request_timeout),
    ]
    if base_url:
        cmd.extend(["--base-url", base_url])

    print(f"\n[RUN] provider={provider} model={model} run={run_idx}", flush=True)
    print("[CMD]", " ".join(cmd), flush=True)

    started = datetime.now().isoformat(timespec="seconds")
    try:
        proc = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=per_run_timeout,
        )
        ended = datetime.now().isoformat(timespec="seconds")

        run_id = parse_run_id(proc.stdout)
        if proc.returncode == 0:
            status = "success"
            message = "ok"
        elif proc.returncode == 2:
            status = "failed"
            message = pick_message(proc.stderr, proc.stdout) or "0 successful scores"
        else:
            status = "failed"
            message = pick_message(proc.stderr, proc.stdout) or f"exit code {proc.returncode}"

        if proc.stdout:
            print(proc.stdout.strip()[-1200:])
        if proc.stderr:
            print(proc.stderr.strip()[-1200:])

        return {
            "provider": provider,
            "model": model,
            "run": str(run_idx),
            "status": status,
            "return_code": str(proc.returncode),
            "run_id": run_id,
            "started_at": started,
            "ended_at": ended,
            "message": message,
        }
    except subprocess.TimeoutExpired:
        ended = datetime.now().isoformat(timespec="seconds")
        return {
            "provider": provider,
            "model": model,
            "run": str(run_idx),
            "status": "timeout",
            "return_code": "124",
            "run_id": "",
            "started_at": started,
            "ended_at": ended,
            "message": f"exceeded {per_run_timeout}s",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="批量运行多模型 BCS 评分实验")
    parser.add_argument("--runs", type=int, default=3, help="每个模型运行次数")
    parser.add_argument("--dataset", default="dataset.csv", help="实验数据集 CSV")
    parser.add_argument("--delay", type=float, default=1.0, help="llm_scoring 请求间隔")
    parser.add_argument("--max-retries", type=int, default=3, help="llm_scoring 最大重试")
    parser.add_argument("--request-timeout", type=float, default=60.0,
                        help="llm_scoring 单次请求超时")
    parser.add_argument("--per-run-timeout", type=int, default=3600,
                        help="单次模型运行总超时秒数")
    parser.add_argument("--internvl-base-url",
                        default=os.environ.get("INTERNVL_BASE_URL", "http://127.0.0.1:11434/v1"),
                        help="InternVL 本地服务地址（Ollama 默认 11434）")
    parser.add_argument("--qwen-base-url",
                        default=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:11434/v1"),
                        help="Qwen-VL 本地服务地址（Ollama 默认 11434）")
    parser.add_argument("--internvl-model",
                        default=os.environ.get("INTERNVL_MODEL", "blaifa/InternVL3_5:8b"),
                        help="InternVL 本地模型名（Ollama）")
    parser.add_argument("--qwen-model",
                        default=os.environ.get("QWEN_MODEL", "qwen2.5vl:7b"),
                        help="Qwen-VL 本地模型名（Ollama）")
    parser.add_argument("--openrouter-model", default="qwen/qwen2.5-vl-72b-instruct",
                        help="OpenRouter 模型名")
    parser.add_argument(
        "--models",
        default="internvl_local,qwen_local",
        help="要运行的模型集合: internvl_local,qwen_local,openrouter（逗号分隔）",
    )
    args = parser.parse_args()

    dataset_path = os.path.join(BASE_DIR, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集不存在: {dataset_path}")
        return 1

    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    valid_keys = {"internvl_local", "qwen_local", "openrouter"}
    unknown = sorted(selected - valid_keys)
    if unknown:
        print(f"错误: 不支持的 models: {', '.join(unknown)}")
        print("可选: internvl_local,qwen_local,openrouter")
        return 1

    checks: dict[str, tuple[bool, str]] = {}
    if "internvl_local" in selected:
        checks["internvl_local"] = check_local_endpoint(args.internvl_base_url)
    if "qwen_local" in selected:
        checks["qwen_local"] = check_local_endpoint(args.qwen_base_url)
    for key, (ok, msg) in checks.items():
        model_name = args.internvl_model if key == "internvl_local" else args.qwen_model
        state = "OK" if ok else "NOT_READY"
        print(f"[LOCAL CHECK] {key} ({model_name}) -> {state} | {msg}")

    model_matrix: list[tuple[str, str, Optional[str], str]] = []
    if "internvl_local" in selected:
        model_matrix.append(("local", args.internvl_model, args.internvl_base_url, "internvl_local"))
    if "qwen_local" in selected:
        model_matrix.append(("local", args.qwen_model, args.qwen_base_url, "qwen_local"))
    if "openrouter" in selected:
        model_matrix.append(("openrouter", args.openrouter_model, None, "openrouter"))

    rows: list[dict[str, str]] = []
    for provider, model, base_url, local_key in model_matrix:
        if provider == "local" and not checks[local_key][0]:
            for run_idx in range(1, args.runs + 1):
                rows.append({
                    "provider": provider,
                    "model": model,
                    "run": str(run_idx),
                    "status": "deploy_required",
                    "return_code": "125",
                    "run_id": "",
                    "started_at": "",
                    "ended_at": "",
                    "message": checks[local_key][1],
                })
            continue

        for run_idx in range(1, args.runs + 1):
            rows.append(run_once(
                provider=provider,
                model=model,
                run_idx=run_idx,
                base_url=base_url,
                dataset=args.dataset,
                delay=args.delay,
                max_retries=args.max_retries,
                request_timeout=args.request_timeout,
                per_run_timeout=args.per_run_timeout,
            ))

    success_count = sum(1 for r in rows if r["status"] == "success")
    print("\n=== 实验完成 ===")
    print(f"总运行数: {len(rows)}")
    print(f"成功: {success_count}")
    print(f"非成功: {len(rows) - success_count}")
    print("明细:")
    for r in rows:
        print(f"- {r['provider']} {r['model']} run={r['run']} status={r['status']} run_id={r['run_id']} msg={r['message']}")

    return 0 if success_count == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
