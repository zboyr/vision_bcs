#!/usr/bin/env python3
"""Fine-tune ConvNeXt-Tiny (ImageNet pretrained) with CORAL ordinal loss for BCS prediction.

CORAL (Consistent Rank Logits) treats ordinal regression as K-1 binary tasks:
each task k predicts P(Y > k). A single shared weight vector feeds into K-1
bias terms, enforcing rank consistency. At inference the predicted label is
the number of thresholds exceeded (i.e. sum of P(Y > k) > 0.5).

Reference: Cao, Mirjalili & Raschka, "Rank consistent ordinal regression for
neural networks with application to age estimation", Pattern Recognition Letters, 2020.
"""
import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_small,
    convnext_tiny,
)

CONVNEXT_VARIANTS = {
    "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "small": (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
}


# ---------------------------------------------------------------------------
# CORAL components
# ---------------------------------------------------------------------------

class CoralLayer(nn.Module):
    """CORAL output layer: one shared linear projection + K-1 independent biases."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        # Shared weight vector (no bias in Linear; biases are separate).
        self.fc = nn.Linear(in_features, 1, bias=False)
        # K-1 independent bias terms (one per threshold).
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (batch, num_classes - 1)."""
        return self.fc(x) + self.biases  # broadcast: (B,1) + (K-1,) -> (B, K-1)


def coral_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute CORAL loss (sum of K-1 binary cross-entropies).

    Args:
        logits: (B, K-1) raw logits from CoralLayer.
        labels: (B,) integer labels in [0, K-1].
        num_classes: K.
    """
    # Build binary targets: for label y, tasks 0..y-1 are positive (Y > k).
    levels = torch.arange(num_classes - 1, device=logits.device)
    targets = (labels.unsqueeze(1) > levels).float()  # (B, K-1)
    return F.binary_cross_entropy_with_logits(logits, targets)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """Predict ordinal label from CORAL logits.

    Returns integer labels in [0, K-1].
    """
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1).long()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvNeXtCoral(nn.Module):
    """ConvNeXt backbone + CORAL head for ordinal BCS regression."""

    def __init__(
        self,
        num_classes: int = 9,
        pretrained: bool = True,
        dropout: float = 0.2,
        variant: str = "tiny",
    ):
        super().__init__()
        if variant not in CONVNEXT_VARIANTS:
            raise ValueError(f"Unknown variant {variant}; expected one of {list(CONVNEXT_VARIANTS)}")
        builder, weights_enum = CONVNEXT_VARIANTS[variant]
        weights = weights_enum if pretrained else None
        backbone = builder(weights=weights)
        # ConvNeXt classifier is Sequential(LayerNorm, Flatten, Linear).
        # Keep the LayerNorm + Flatten, replace the Linear.
        in_features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.coral = CoralLayer(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns CORAL logits of shape (B, num_classes - 1)."""
        features = self.backbone(x)
        features = self.dropout(features)
        return self.coral(features)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    image_path: str
    bcs: int  # 1-9


def try_load_file(expected_filepath: str, dataset_csv_filepath: str) -> str:
    if os.path.exists(expected_filepath):
        return expected_filepath
    recover_path = os.path.join(os.path.dirname(dataset_csv_filepath), os.path.basename(expected_filepath))
    if os.path.exists(recover_path):
        return recover_path
    raise ValueError(f"{expected_filepath} or {recover_path} don't exist, check your dataset.csv")


def load_samples(base_dir: str, dataset_csv: str) -> list[Sample]:
    path = os.path.join(base_dir, dataset_csv)
    rows: list[Sample] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("error"):
                continue
            img_path = os.path.join(base_dir, r["path"])
            corrected = try_load_file(img_path, path)
            bcs = int(r.get("bcs") or r.get("bcs_primary"))
            rows.append(Sample(image_path=corrected, bcs=bcs))
    return rows


class BCSDataset(Dataset):
    """PyTorch dataset for BCS images."""

    def __init__(self, samples: list[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        img = self.transform(img)
        # CORAL uses 0-indexed labels internally; BCS 1-9 -> label 0-8.
        return {"pixel_values": img, "label": s.bcs - 1}


def build_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    """ImageNet-normalised transforms with augmentation for training."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total = 0
    abs_errors: list[float] = []
    correct = 0
    within_1 = 0
    all_preds: list[int] = []
    all_gts: list[int] = []

    for batch in loader:
        imgs = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        logits = model(imgs)
        preds = coral_predict(logits)

        for p, g in zip(preds.tolist(), labels.tolist()):
            pred_bcs = p + 1  # back to 1-9
            gt_bcs = g + 1
            all_preds.append(pred_bcs)
            all_gts.append(gt_bcs)
            abs_errors.append(abs(pred_bcs - gt_bcs))
            if pred_bcs == gt_bcs:
                correct += 1
            if abs(pred_bcs - gt_bcs) <= 1:
                within_1 += 1
            total += 1

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    accuracy = correct / total if total else 0.0
    within_1_acc = within_1 / total if total else 0.0
    return {
        "count": total,
        "mae": mae,
        "accuracy": accuracy,
        "within_1": within_1_acc,
        "preds": all_preds,
        "gts": all_gts,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_sampler(samples: list[Sample], num_classes: int) -> WeightedRandomSampler:
    """Inverse-frequency weighted sampler to handle class imbalance."""
    counts = [0] * num_classes
    for s in samples:
        counts[s.bcs - 1] += 1
    weights_per_class = [1.0 / max(c, 1) for c in counts]
    sample_weights = [weights_per_class[s.bcs - 1] for s in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(samples), replacement=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny + CORAL for BCS ordinal regression")
    parser.add_argument("--dataset", default="datasets/cat_10k/train.csv")
    parser.add_argument("--eval-dataset", default="datasets/cat_10k/eval.csv",
                        help="Held-out eval CSV.")
    parser.add_argument("--no-held-out-eval", action="store_true")
    parser.add_argument("--output-dir", default="outputs/convnext_coral_bcs")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap total samples loaded (0=all).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Separate (lower) LR for pretrained backbone layers.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--backbone", choices=list(CONVNEXT_VARIANTS.keys()), default="tiny",
                        help="ConvNeXt variant: tiny (~28M), small (~50M), base (~89M).")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=80,
                        help="Percentage of dataset for training (rest is split eval).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable inverse-frequency weighted sampling.")
    parser.add_argument("--baseline-eval", action="store_true",
                        help="Run eval before training (ImageNet baseline).")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print training loss every N batches.")
    parser.add_argument("--resume", default="",
                        help="Path to checkpoint .pt file to resume from.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    NUM_CLASSES = 9  # BCS 1-9

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir + "_" + datetime.now().strftime("%m%d_%H%M"))
    os.makedirs(output_dir, exist_ok=True)

    # Save args for reproducibility.
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    samples = load_samples(base_dir, args.dataset)
    if len(samples) < 10:
        raise RuntimeError("dataset too small")
    random.shuffle(samples)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    train_size = int(len(samples) * (args.train_size / 100))
    train_samples = samples[:train_size]
    split_eval_samples = samples[train_size:]
    held_out_samples = (
        load_samples(base_dir, args.eval_dataset)
        if (args.eval_dataset and not args.no_held_out_eval)
        else []
    )
    print(f"samples: train={len(train_samples)}, split_eval={len(split_eval_samples)}, "
          f"held_out={len(held_out_samples)}")

    # Print class distribution.
    dist = [0] * NUM_CLASSES
    for s in train_samples:
        dist[s.bcs - 1] += 1
    print(f"train class distribution (BCS 1-9): {dist}")

    train_tf = build_transforms(args.image_size, is_train=True)
    eval_tf = build_transforms(args.image_size, is_train=False)

    train_ds = BCSDataset(train_samples, train_tf)
    split_eval_ds = BCSDataset(split_eval_samples, eval_tf) if split_eval_samples else None
    held_out_ds = BCSDataset(held_out_samples, eval_tf) if held_out_samples else None

    sampler = None if args.no_weighted_sampler else make_sampler(train_samples, NUM_CLASSES)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    eval_loaders: list[tuple[str, DataLoader]] = []
    if split_eval_ds:
        eval_loaders.append(("split", DataLoader(
            split_eval_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )))
    if held_out_ds:
        eval_loaders.append(("held_out", DataLoader(
            held_out_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtCoral(
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=args.dropout,
        variant=args.backbone,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: ConvNeXt-{args.backbone} ({n_params:.1f}M params)")

    if args.resume:
        resume_path = os.path.join(base_dir, args.resume) if not os.path.isabs(args.resume) else args.resume
        print(f"Resuming from checkpoint: {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])

    model = model.to(device)

    # Differential LR: lower for backbone, higher for new CORAL head.
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.coral.parameters()) + list(model.dropout.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # Cosine annealing with warmup.
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(len(train_loader), total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Baseline eval
    # ------------------------------------------------------------------
    if args.baseline_eval and eval_loaders:
        print("Running baseline eval (ImageNet weights, no fine-tuning)...")
        for name, loader in eval_loaders:
            r = run_eval(model, loader, device, NUM_CLASSES)
            print(f"  baseline {name}: mae={r['mae']:.3f} acc={r['accuracy']:.3f} within_1={r['within_1']:.3f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_mae = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    total_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        interval_loss = 0.0
        interval_count = 0

        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(imgs)
                    loss = coral_loss(logits, labels, NUM_CLASSES)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = coral_loss(logits, labels, NUM_CLASSES)
                loss.backward()
                optimizer.step()

            scheduler.step()
            cur_loss = loss.item()
            running_loss += cur_loss
            n_batches += 1
            interval_loss += cur_loss
            interval_count += 1

            if n_batches % args.log_interval == 0 or n_batches == total_batches:
                avg_interval = interval_loss / interval_count
                lr_now = optimizer.param_groups[1]["lr"]
                print(f"  [epoch {epoch}/{args.epochs}] batch {n_batches}/{total_batches}  "
                      f"loss={avg_interval:.4f}  lr={lr_now:.2e}")
                interval_loss = 0.0
                interval_count = 0

        avg_loss = running_loss / max(n_batches, 1)
        print(f"[epoch {epoch}/{args.epochs}] avg_loss={avg_loss:.4f}")

        # Eval.
        entry: dict[str, Any] = {"epoch": epoch, "train_loss": avg_loss}
        for name, loader in eval_loaders:
            r = run_eval(model, loader, device, NUM_CLASSES)
            print(f"  {name}: mae={r['mae']:.3f} acc={r['accuracy']:.3f} within_1={r['within_1']:.3f}")
            entry[name] = {"mae": r["mae"], "accuracy": r["accuracy"], "within_1": r["within_1"], "count": r["count"]}

        history.append(entry)

        # Save epoch checkpoint.
        ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, ckpt_path)

        # Track best by held_out MAE (fallback to split).
        eval_key = "held_out" if "held_out" in entry else "split"
        if eval_key in entry and entry[eval_key]["mae"] < best_mae:
            best_mae = entry[eval_key]["mae"]
            best_epoch = epoch
            best_path = os.path.join(output_dir, "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, best_path)
            print(f"  -> new best ({eval_key} mae={best_mae:.3f}), saved to {best_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    metrics: dict[str, Any] = {
        "train_size": len(train_samples),
        "best_epoch": best_epoch,
        "best_mae": best_mae,
        "history": history,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nsaved_output_dir= {output_dir}")
    print(f"best_epoch= {best_epoch}, best_mae= {best_mae:.4f}")
    print(f"metrics= {json.dumps(metrics, ensure_ascii=False, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
