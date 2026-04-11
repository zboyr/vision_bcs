"""Evaluation metrics and 7x7 matrix I/O.

The matrix CSV is always **regenerated** from checkpoint JSONL files,
so it is safe to rerun after partial experiments.
"""

import csv
import os
import re
from typing import Dict, List, Optional, Tuple

from .checkpoint import compute_run_mae, get_log_dir, get_run_path


def discover_log_cells(log_dir: str) -> Tuple[List[str], List[str]]:
    """Scan ``log_dir`` for ``{model}_{prompt}`` directories.

    Returns ``(model_ids, prompt_ids)`` sorted naturally — preserves any
    historical experiments so the matrix never loses rows/columns when
    a new config only mentions a subset of models or prompts.
    """
    if not os.path.isdir(log_dir):
        return [], []
    pattern = re.compile(r"^(M\d+)_(P\d+)$")
    models, prompts = set(), set()
    for name in os.listdir(log_dir):
        m = pattern.match(name)
        if m:
            models.add(m.group(1))
            prompts.add(m.group(2))
    return (
        sorted(models, key=lambda s: int(s[1:])),
        sorted(prompts, key=lambda s: int(s[1:])),
    )


def compute_cell_mae(log_dir: str, model_id: str, prompt_id: str,
                     repeats: int) -> Optional[float]:
    """Average MAE across all completed runs for one (model, prompt) cell."""
    cell_dir = os.path.join(log_dir, f"{model_id}_{prompt_id}")
    maes: list[float] = []
    for run_idx in range(1, repeats + 1):
        mae = compute_run_mae(get_run_path(cell_dir, run_idx))
        if mae is not None:
            maes.append(mae)
    return sum(maes) / len(maes) if maes else None


def build_matrix(log_dir: str, model_ids: List[str],
                 prompt_ids: List[str], repeats: int,
                 model_labels: Optional[Dict[str, str]] = None,
                 prompt_labels: Optional[Dict[str, str]] = None,
                 ) -> tuple[List[str], List[Dict[str, str]]]:
    """Build the experiment result matrix from checkpoint data.

    Returns ``(column_names, rows)`` where each row is a dict.
    """
    col_names = [prompt_labels.get(pid, pid) if prompt_labels else pid
                 for pid in prompt_ids]
    rows: List[Dict[str, str]] = []
    for mid in model_ids:
        label = model_labels[mid] if model_labels and mid in model_labels else mid
        row: Dict[str, str] = {"model": label}
        for pid, col in zip(prompt_ids, col_names):
            mae = compute_cell_mae(log_dir, mid, pid, repeats)
            row[col] = f"{mae:.4f}" if mae is not None else ""
        rows.append(row)
    return col_names, rows


def save_matrix(csv_path: str, col_names: List[str],
                rows: List[Dict[str, str]]) -> None:
    fieldnames = ["model"] + col_names
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_matrix(csv_path: str, log_dir: str,
                  model_ids: List[str], prompt_ids: List[str],
                  repeats: int,
                  model_labels: Optional[Dict[str, str]] = None,
                  prompt_labels: Optional[Dict[str, str]] = None) -> None:
    """Regenerate the matrix CSV from checkpoint data."""
    col_names, rows = build_matrix(
        log_dir, model_ids, prompt_ids, repeats,
        model_labels=model_labels, prompt_labels=prompt_labels,
    )
    save_matrix(csv_path, col_names, rows)
