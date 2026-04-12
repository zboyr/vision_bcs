#!/usr/bin/env python3
"""
7 x 7  BCS Scoring Experiment Runner
=====================================

Usage::

    # Full matrix (all models x all prompts)
    python run_experiment.py --config configs/next_study.yaml

    # Single row (one model, all prompts)
    python run_experiment.py --config configs/next_study.yaml --model M1

    # Single column (one prompt, all models)
    python run_experiment.py --config configs/next_study.yaml --prompt P1

    # Single cell
    python run_experiment.py --config configs/next_study.yaml --model M1 --prompt P1

Interruption-safe: every scored image is written to a JSONL checkpoint
immediately.  Rerunning the same command skips already-scored images.

Concurrency: images within a run are scored in parallel (default 10 workers).
Set ``concurrency`` in config YAML to adjust.
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from scoring.client import create_client, check_local_endpoint, load_dotenv
from scoring.dataset import load_dataset, build_ground_truth_map
from scoring.pipelines import get_pipeline, PIPELINES, PIPELINE_LABELS, set_reuse_context
from scoring.checkpoint import (
    get_log_dir, get_run_path, load_completed, save_result,
    compute_run_mae, get_failed_ids,
)
from scoring.evaluate import compute_cell_mae, update_matrix, discover_log_cells

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Core: score one (model, prompt, run) ─────────────────────────────

def run_cell(client, model_name, pipeline_fn, records, gt_map,
             log_dir, model_id, prompt_id, run_idx, *,
             max_retries=3, delay=1.0, concurrency=10,
             pipeline_kwargs=None, retry_ids=None):
    """Score all images for one run of one (model, prompt) cell.

    If *retry_ids* is given, only re-score those image IDs (overwriting
    previous failed results in the checkpoint).

    Returns ``(n_errors, aborted)``.
    """
    cell_dir = get_log_dir(log_dir, model_id, prompt_id)
    run_path = get_run_path(cell_dir, run_idx)
    completed = load_completed(run_path)
    extra_kw = pipeline_kwargs or {}

    total = len(records)
    if retry_ids is not None:
        retry_set = set(retry_ids)
        pending = [r for r in records
                   if int(r["image_id"]) in retry_set
                   and int(r["image_id"]) in gt_map]
    else:
        pending = [r for r in records
                   if int(r["image_id"]) not in completed
                   and int(r["image_id"]) in gt_map]

    if not pending:
        return 0, False

    lock = threading.Lock()
    abort = threading.Event()
    errors = [0]
    done = [len(completed)]

    def _score(rec):
        if abort.is_set():
            return
        img_id = int(rec["image_id"])
        gt = gt_map[img_id]

        result = pipeline_fn(client, model_name, rec["image_path"],
                             max_retries=max_retries, delay=delay,
                             _image_id=img_id, **extra_kw)
        bcs = result.get("bcs")

        with lock:
            save_result(run_path, img_id, bcs, gt, result)
            done[0] += 1
            n = done[0]
            if bcs is None:
                errors[0] += 1
                print(f"    [{n}/{total}] #{img_id} FAIL "
                      f"{result.get('error', '')[:60]}")
                if errors[0] >= max(5, int(total * 0.3)):
                    print(f"    !! Too many errors ({errors[0]}), aborting")
                    abort.set()
            else:
                print(f"    [{n}/{total}] #{img_id} -> {bcs}  "
                      f"(gt={gt:.0f} dev={abs(bcs - gt):.0f})")

    if concurrency <= 1:
        # Sequential mode
        for rec in pending:
            _score(rec)
            if abort.is_set():
                break
            time.sleep(delay)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as exe:
            futs = [exe.submit(_score, r) for r in pending]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    with lock:
                        print(f"    !! Exception: {e}")

    return errors[0], abort.is_set()


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="7x7 BCS Scoring Experiment")
    parser.add_argument("--config", required=True,
                        help="Experiment config YAML")
    parser.add_argument("--model", default=None,
                        help="Run single model row (e.g. M1)")
    parser.add_argument("--prompt", default=None,
                        help="Run single prompt column (e.g. P1)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Override concurrency (default from config or 10)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Only re-score images that previously failed")
    args = parser.parse_args()

    load_dotenv(os.path.join(BASE_DIR, ".env"))

    # ── Load config ──────────────────────────────────────────────────

    cfg_path = (args.config if os.path.isabs(args.config)
                else os.path.join(BASE_DIR, args.config))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg["dataset"]
    output_dir   = cfg.get("output_dir", "responses/next_study")
    repeats      = cfg.get("repeats", 3)
    delay        = cfg.get("delay", 1.0)
    max_retries  = cfg.get("max_retries", 3)
    req_timeout  = cfg.get("request_timeout", 60.0)
    concurrency  = args.concurrency or cfg.get("concurrency", 10)
    models_cfg   = cfg["models"]
    prompt_ids   = cfg.get("prompts", list(PIPELINE_LABELS.keys()))
    p6_ref_imgs  = cfg.get("p6_reference_images") or [
        cfg.get("p6_reference_image", "prompts/cat_bcs.jpg")
    ]

    log_dir     = os.path.join(BASE_DIR, output_dir, "logs")
    matrix_path = os.path.join(BASE_DIR, output_dir, "experiment_matrix.csv")
    os.makedirs(log_dir, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────

    records = load_dataset(dataset_path)
    gt_map  = build_ground_truth_map(records)
    n_imgs  = len(records)
    print(f"Dataset: {dataset_path} ({n_imgs} images)")

    # ── Filter by --model / --prompt ─────────────────────────────────

    all_model_ids  = list(models_cfg.keys())
    all_prompt_ids = list(prompt_ids)

    run_model_ids  = all_model_ids
    run_prompt_ids = all_prompt_ids

    if args.model:
        if args.model not in all_model_ids:
            print(f"Error: unknown model '{args.model}'. "
                  f"Available: {all_model_ids}")
            return 1
        run_model_ids = [args.model]

    if args.prompt:
        if args.prompt not in all_prompt_ids:
            print(f"Error: unknown prompt '{args.prompt}'. "
                  f"Available: {all_prompt_ids}")
            return 1
        run_prompt_ids = [args.prompt]

    print(f"Models:      {run_model_ids}")
    print(f"Prompts:     {run_prompt_ids}")
    print(f"Repeats:     {repeats}")
    print(f"Concurrency: {concurrency}")
    print(f"Logs:        {log_dir}")
    print()

    # ── Run experiments ──────────────────────────────────────────────

    for mid in run_model_ids:
        mcfg       = models_cfg[mid]
        model_name = mcfg["name"]
        provider   = mcfg.get("provider", "openrouter")
        base_url   = mcfg.get("base_url")
        api_key    = mcfg.get("api_key")

        print(f"=== {mid}: {model_name} ({provider}) ===")

        # Transformers inference is single-threaded on GPU
        model_concurrency = 1 if provider == "transformers" else concurrency

        if provider == "local":
            url = base_url or "http://127.0.0.1:8000/v1"
            ok, msg = check_local_endpoint(url)
            if not ok:
                print(f"  SKIP: {msg}\n")
                continue

        try:
            client = create_client(provider, base_url=base_url,
                                   api_key=api_key,
                                   request_timeout=req_timeout)
        except ValueError as e:
            print(f"  SKIP: {e}\n")
            continue

        # Allow P5/P7 to reuse P2/P3 results from this model's logs
        set_reuse_context(log_dir, mid)

        for pid in run_prompt_ids:
            pipeline_fn = get_pipeline(pid)
            label = PIPELINE_LABELS.get(pid, pid)

            # Per-pipeline extra kwargs
            p_kwargs = {}
            if pid == "P6":
                p_kwargs["reference_images"] = p6_ref_imgs

            for run_idx in range(1, repeats + 1):
                cell_dir = get_log_dir(log_dir, mid, pid)
                run_path = get_run_path(cell_dir, run_idx)
                done     = load_completed(run_path)

                # --retry-errors: only re-score failed images
                r_ids = None
                if args.retry_errors:
                    r_ids = get_failed_ids(run_path)
                    if not r_ids:
                        mae = compute_run_mae(run_path)
                        mae_s = f" (mae={mae:.4f})" if mae else ""
                        print(f"  {label} run{run_idx}/{repeats}: "
                              f"no errors to retry{mae_s}")
                        continue
                    print(f"  {label} run{run_idx}/{repeats}: "
                          f"retrying {len(r_ids)} failed images")
                elif len(done) >= n_imgs:
                    mae = compute_run_mae(run_path)
                    print(f"  {label} run{run_idx}/{repeats}: "
                          f"complete (mae={mae:.4f})" if mae else
                          f"  {label} run{run_idx}/{repeats}: complete")
                    continue
                else:
                    print(f"  {label} run{run_idx}/{repeats} "
                          f"({len(done)}/{n_imgs} cached)")

                errs, aborted = run_cell(
                    client, model_name, pipeline_fn,
                    records, gt_map,
                    log_dir, mid, pid, run_idx,
                    max_retries=max_retries, delay=delay,
                    concurrency=model_concurrency,
                    pipeline_kwargs=p_kwargs,
                    retry_ids=r_ids,
                )

                mae = compute_run_mae(run_path)
                mae_str = f"{mae:.4f}" if mae is not None else "?"
                print(f"  {label} run{run_idx}: "
                      f"mae={mae_str} errors={errs}")

                if aborted:
                    print(f"  !! Aborted — skipping remaining runs "
                          f"for {label}")
                    break

            # Cell-level MAE after all runs
            cell_mae = compute_cell_mae(log_dir, mid, pid, repeats)
            if cell_mae is not None:
                print(f"  {label} avg_mae={cell_mae:.4f}")
            print()

    # ── Update 7x7 matrix ───────────────────────────────────────────
    # Discover historical models/prompts from log dir so the matrix
    # never loses rows/columns from earlier experiments.
    disc_models, disc_prompts = discover_log_cells(log_dir)
    matrix_models  = sorted(set(all_model_ids)  | set(disc_models),
                            key=lambda s: int(s[1:]))
    matrix_prompts = sorted(set(all_prompt_ids) | set(disc_prompts),
                            key=lambda s: int(s[1:]))

    model_labels  = {mid: mid for mid in matrix_models}
    prompt_labels = {pid: PIPELINE_LABELS.get(pid, pid)
                     for pid in matrix_prompts}

    update_matrix(matrix_path, log_dir,
                  matrix_models, matrix_prompts, repeats,
                  model_labels=model_labels,
                  prompt_labels=prompt_labels)

    print(f"=== Matrix saved: {matrix_path} ===")
    _print_matrix(matrix_path)
    return 0


def _print_matrix(path: str) -> None:
    """Pretty-print the matrix CSV to stdout."""
    if not os.path.exists(path):
        return
    import csv
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    widths = [max(len(r[j]) if j < len(r) else 0 for r in rows)
              for j in range(len(rows[0]))]

    print()
    for r in rows:
        cells = [r[j].ljust(widths[j]) if j < len(r) else " " * widths[j]
                 for j in range(len(widths))]
        print("  " + " | ".join(cells))
        if r is rows[0]:
            print("  " + "-+-".join("-" * w for w in widths))


if __name__ == "__main__":
    sys.exit(main())
