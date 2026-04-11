"""Dataset loading and ground-truth utilities."""

import csv
import os
from typing import Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(csv_path: str) -> list[Dict[str, str]]:
    """Load dataset CSV.

    Supports:
      - Standard: image_id, image_path, ground_truth
      - Simple:   filename, bcs  (path relative to CSV directory)
      - Simple:   path, bcs      (path relative to project root)
    """
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(BASE_DIR, csv_path)
    dataset_dir = os.path.relpath(os.path.dirname(csv_path), BASE_DIR)

    records: list[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "filename" in row and "bcs" in row and "image_path" not in row:
                records.append({
                    "image_id": str(len(records) + 1),
                    "image_path": os.path.join(dataset_dir,
                                               row["filename"].strip()),
                    "ground_truth": row["bcs"].strip(),
                })
            elif "path" in row and "bcs" in row and "image_path" not in row:
                records.append({
                    "image_id": str(len(records) + 1),
                    "image_path": row["path"].strip(),
                    "ground_truth": row["bcs"].strip(),
                })
            else:
                records.append(dict(row))
    return records


def build_ground_truth_map(records: list[Dict[str, str]]) -> Dict[int, float]:
    gt: Dict[int, float] = {}
    for row in records:
        try:
            gt[int(row["image_id"])] = float(row["ground_truth"])
        except (ValueError, KeyError):
            continue
    return gt
