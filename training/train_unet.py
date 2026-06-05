"""
train_unet.py
-------------
Full training loop for the U-Net acne segmentation model.

Features:
  - BCEWithLogitsLoss + Dice loss combined
  - Adam optimiser with ReduceLROnPlateau scheduler
  - Mixed precision (fp16) via torch.cuda.amp
  - Checkpoint saving (best + latest)
  - Validation loop with IoU metric
  - Prediction visualisation after each epoch
  - Designed for RTX 3050 4GB GPU (batch size 4, fp16)

Usage:
    python training/train_unet.py \
        --acne_dir   dataset/acne/ \
        --mask_dir   dataset/masks/ \
        --epochs     50 \
        --batch_size 4 \
        --lr         1e-4 \
        --checkpoint_dir checkpoints/
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet import build_unet
from utils.dataset import get_unet_loaders


# ─── Loss Functions ──────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    Measures overlap between prediction and ground truth.
    Dice = 2*|P∩G| / (|P|+|G|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits)
        # Flatten spatial dims
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    BCE + Dice combined loss.
    BCE handles pixel-level accuracy; Dice handles region overlap.
    alpha controls the blend (default: 0.5 each).
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha    = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.alpha * bce + (1 - self.alpha) * dice


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute mean Intersection-over-Union for a batch."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union        = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ─── Visualisation ───────────────────────────────────────────────────────────

def save_prediction_grid(
    model:      nn.Module,
    loader,
    device:     str,
    save_path:  str,
    n_samples:  int = 4,
) -> None:
    """
    Save a side-by-side grid: [Input | Ground Truth Mask | Predicted Mask]
    Writes a PNG to save_path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (works in headless env)
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed — skipping visualisation.")
        return

    model.eval()
    images, masks = next(iter(loader))
    images = images[:n_samples].to(device)
    masks  = masks[:n_samples].to(device)

    with torch.no_grad():
        logits = model(images)
        preds  = (torch.sigmoid(logits) > 0.5).float()

    # Denormalise images: [-1,1] → [0,1]
    images_vis = (images.cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    masks_vis  = masks.cpu().squeeze(1).numpy()
    preds_vis  = preds.cpu().squeeze(1).numpy()

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis]  # ensure 2D indexing works

    col_titles = ["Input Image", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for i in range(n_samples):
        axes[i, 0].imshow(images_vis[i])
        axes[i, 1].imshow(masks_vis[i],  cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(preds_vis[i],  cmap="gray", vmin=0, vmax=1)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"  [VIS] Saved grid -> {save_path}")


# ─── Training Epoch ──────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader,
    criterion:  nn.Module,
    optimizer:  optim.Optimizer,
    scaler:     GradScaler,
    device:     str,
) -> dict:
    """Run one full training epoch. Returns dict of mean metrics."""
    model.train()
    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Mixed precision forward pass ──
        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        # ── Scaled backward pass ──
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics (detach from graph)
        total_loss += loss.item()
        total_iou  += compute_iou(logits.detach(), masks.detach())

    return {
        "loss": total_loss / n_batches,
        "iou":  total_iou  / n_batches,
    }


# ─── Validation Epoch ────────────────────────────────────────────────────────

def validate(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    str,
) -> dict:
    """Run validation. Returns dict of mean metrics."""
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = len(loader)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with autocast():
                logits = model(images)
                loss   = criterion(logits, masks)

            total_loss += loss.item()
            total_iou  += compute_iou(logits, masks)

    return {
        "loss": total_loss / n_batches,
        "iou":  total_iou  / n_batches,
    }


# ─── Checkpoint Helpers ──────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device: str = "cpu") -> int:
    """Load checkpoint and return the epoch number."""
    ckpt   = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    print(f"[INFO] Resumed from checkpoint: {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0)


# ─── Main Training Loop ──────────────────────────────────────────────────────

def train(args) -> None:
    # ── Setup ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir,        exist_ok=True)

    # ── Data ──
    train_loader, val_loader = get_unet_loaders(
        acne_dir=args.acne_dir,
        mask_dir=args.mask_dir,
        image_size=256,
        batch_size=args.batch_size,
        val_split=0.15,
        num_workers=args.num_workers,
    )

    # ── Model ──
    # Use base_features=32 if 4GB VRAM is too tight with batch=4
    model = build_unet(base_features=64, device=device)
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ── Loss, optimiser, scheduler ──
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    scaler = GradScaler(enabled=(device == "cuda"))

    # ── Resume from checkpoint if specified ──
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)

    # ── Training ──
    best_val_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    print(f"\n{'='*55}")
    print(f"  Starting U-Net training for {args.epochs} epochs")
    print(f"  Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"  Mixed precision: {device == 'cuda'}")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        # Validate
        val_metrics   = validate(model, val_loader, criterion, device)
        # Scheduler step on validation IoU
        scheduler.step(val_metrics["iou"])

        elapsed = time.time() - t0

        # ── Log ──
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])

        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"| Train  Loss: {train_metrics['loss']:.4f}  IoU: {train_metrics['iou']:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f}  IoU: {val_metrics['iou']:.4f} "
            f"| LR: {optimizer.param_groups[0]['lr']:.2e} "
            f"| {elapsed:.1f}s"
        )

        # ── Save latest checkpoint ──
        latest_path = os.path.join(args.checkpoint_dir, "unet_latest.pth")
        save_checkpoint({
            "epoch":       epoch + 1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_iou":     val_metrics["iou"],
        }, latest_path)

        # ── Save best checkpoint ──
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_path = os.path.join(args.checkpoint_dir, "unet.pth")
            save_checkpoint({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "val_iou":     best_val_iou,
            }, best_path)
            print(f"  [!] New best IoU: {best_val_iou:.4f} -> saved {best_path}")

        # --- Visualise every N epochs ---
        if (epoch + 1) % args.vis_every == 0 or epoch == 0:
            vis_path = os.path.join(args.vis_dir, f"epoch_{epoch+1:03d}.png")
            save_prediction_grid(model, val_loader, device, vis_path)

    # ── Final summary ──
    print(f"\n{'='*55}")
    print(f"  Training complete.")
    print(f"  Best Val IoU : {best_val_iou:.4f}")
    print(f"  Best model   : {os.path.join(args.checkpoint_dir, 'unet.pth')}")
    print(f"{'='*55}\n")

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "unet_history.npy")
    np.save(history_path, history)
    print(f"[INFO] Training history saved -> {history_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for acne segmentation")
    p.add_argument("--acne_dir",      type=str,   default="dataset/acne/")
    p.add_argument("--mask_dir",      type=str,   default="dataset/masks/")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=4,   help="4 for 4GB GPU")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--checkpoint_dir",type=str,   default="checkpoints/")
    p.add_argument("--vis_dir",       type=str,   default="outputs/unet_vis/")
    p.add_argument("--vis_every",     type=int,   default=5,   help="Visualise every N epochs")
    p.add_argument("--resume",        type=str,   default=None, help="Path to checkpoint to resume")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)