"""Checkpoint / resume support.

Each image result is appended to a JSONL file immediately after scoring,
so progress is never lost on interruption.

Directory layout::

    {log_dir}/{model_id}_{prompt_id}/run{N}.jsonl
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def get_log_dir(base_log_dir: str, model_id: str, prompt_id: str) -> str:
    path = os.path.join(base_log_dir, f"{model_id}_{prompt_id}")
    os.makedirs(path, exist_ok=True)
    return path


def get_run_path(log_dir: str, run_idx: int) -> str:
    return os.path.join(log_dir, f"run{run_idx}.jsonl")


def load_completed(run_path: str) -> Dict[int, Dict[str, Any]]:
    """Load already-scored image results from a checkpoint file.

    Returns ``{image_id: result_dict}``.
    """
    completed: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(run_path):
        return completed
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                img_id = data.get("image_id")
                if img_id is not None:
                    completed[int(img_id)] = data
            except (json.JSONDecodeError, ValueError):
                continue
    return completed


def save_result(run_path: str, image_id: int, bcs: Optional[float],
                ground_truth: float, pipeline_result: Dict[str, Any]) -> None:
    """Append one image result to the JSONL checkpoint."""
    entry: Dict[str, Any] = {
        "image_id": image_id,
        "bcs": bcs,
        "ground_truth": ground_truth,
        "timestamp": datetime.now().isoformat(),
        "error": pipeline_result.get("error"),
    }
    # Persist all pipeline-specific fields (reasoning, votes, …)
    for k, v in pipeline_result.items():
        if k not in ("bcs", "error"):
            entry[k] = v
    with open(run_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_failed_ids(run_path: str) -> list[int]:
    """Return image_ids that failed (bcs is None) in the checkpoint."""
    completed = load_completed(run_path)
    return [img_id for img_id, data in completed.items()
            if data.get("bcs") is None]


def compute_run_mae(run_path: str) -> Optional[float]:
    """Compute MAE for a single run from its checkpoint file."""
    completed = load_completed(run_path)
    devs = []
    for data in completed.values():
        bcs, gt = data.get("bcs"), data.get("ground_truth")
        if bcs is not None and gt is not None:
            devs.append(abs(float(bcs) - float(gt)))
    return sum(devs) / len(devs) if devs else None
