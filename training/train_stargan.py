"""
train_stargan.py
----------------
Full StarGAN training loop for acne-to-clear-skin image translation.

DESIGNED FOR: Google Colab T4 GPU (16GB VRAM)
  - Mixed precision (fp16) via torch.cuda.amp
  - Batch size 2 (safe for 16GB; use 1 if needed)
  - Image size: 256×256 (use 128 if OOM)

Training strategy:
  1. Train Discriminator for n_critic steps (WGAN-GP)
  2. Train Generator every n_critic steps
  3. Losses:
       G: adversarial + lambda_cls * domain_cls + lambda_rec * reconstruction
       D: adversarial + lambda_cls * domain_cls + lambda_gp * gradient_penalty

Usage on Google Colab:
    !python training/train_stargan.py \
        --acne_dir   dataset/acne/ \
        --celeba_dir dataset/celeba/ \
        --epochs     100 \
        --batch_size 2

After training, download:
    checkpoints/stargan_G.pth
    checkpoints/stargan_D.pth
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stargan import (
    build_stargan,
    compute_gradient_penalty,
    adversarial_loss,
    domain_classification_loss,
    reconstruction_loss,
)
from utils.dataset import get_stargan_loader


# ─── Visualisation ───────────────────────────────────────────────────────────

def save_sample_images(
    G:         nn.Module,
    sample_batch: dict,
    device:    str,
    save_path: str,
    n_samples: int = 4,
) -> None:
    """Save a grid showing: [acne input | translated to clean | back-translated]."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib unavailable — skipping visualisation.")
        return

    G.eval()
    with torch.no_grad():
        n_samples = min(n_samples, len(sample_batch["acne"]))
        acne   = sample_batch["acne"][:n_samples].to(device)
        celeba = sample_batch["celeba"][:n_samples].to(device)
        lbl_acne   = sample_batch["label_acne"][:n_samples].to(device)
        lbl_celeba = sample_batch["label_celeba"][:n_samples].to(device)

        # Acne → Clean
        fake_clean = G(acne, lbl_celeba)
        # Reconstruct: Clean → Acne (cycle)
        rec_acne   = G(fake_clean, lbl_acne)

    def to_vis(t):
        """Convert tensor (B,3,H,W) in [-1,1] to numpy (B,H,W,3) in [0,1]."""
        return (t.cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    acne_vis       = to_vis(acne)
    fake_clean_vis = to_vis(fake_clean)
    rec_acne_vis   = to_vis(rec_acne)

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis]

    titles = ["Acne Input", "Translated (Clean)", "Reconstructed (Acne)"]
    for col, t in enumerate(titles):
        axes[0, col].set_title(t, fontsize=9, fontweight="bold")

    for i in range(n_samples):
        axes[i, 0].imshow(acne_vis[i])
        axes[i, 1].imshow(fake_clean_vis[i])
        axes[i, 2].imshow(rec_acne_vis[i])
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  [VIS] Saved -> {save_path}")
    G.train()


# ─── Training Loop ───────────────────────────────────────────────────────────

def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mb   = torch.cuda.get_device_properties(0).total_memory // 1024**2
        print(f"[INFO] GPU: {gpu_name} ({gpu_mb} MB)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir,        exist_ok=True)

    # ── Data ──
    loader = get_stargan_loader(
        acne_dir=args.acne_dir,
        celeba_dir=args.celeba_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_celeba=args.max_celeba,
    )

    # ── Models ──
    G, D = build_stargan(
        n_domains=2,
        conv_dim=64,
        n_res_blocks=6,
        image_size=args.image_size,
        device=device,
    )

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"[INFO] G params: {g_params:.2f}M | D params: {d_params:.2f}M")

    # ── Optimisers ──
    opt_G = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    # Linear LR decay in second half of training
    def lr_lambda(epoch):
        if epoch < args.epochs // 2:
            return 1.0
        return 1.0 - (epoch - args.epochs // 2) / max(1, args.epochs // 2)

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    scaler_G = GradScaler(enabled=(device == "cuda"))
    scaler_D = GradScaler(enabled=(device == "cuda"))

    # ── Resume ──
    start_epoch = 0
    state_path = os.path.join(args.checkpoint_dir, "stargan_training_state_latest.pt")
    if args.resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        start_epoch = state['epoch']
        G.load_state_dict(state['G'])
        D.load_state_dict(state['D'])
        opt_G.load_state_dict(state['opt_G'])
        opt_D.load_state_dict(state['opt_D'])
        scaler_G.load_state_dict(state['scaler_G'])
        scaler_D.load_state_dict(state['scaler_D'])
        sched_G.load_state_dict(state['sched_G'])
        sched_D.load_state_dict(state['sched_D'])
        print(f"[INFO] Resumed full training state from epoch {start_epoch}")
    else:
        if args.resume_G and os.path.exists(args.resume_G):
            G.load_state_dict(torch.load(args.resume_G, map_location=device))
            print(f"[INFO] Loaded G from: {args.resume_G}")
        if args.resume_D and os.path.exists(args.resume_D):
            D.load_state_dict(torch.load(args.resume_D, map_location=device))
            print(f"[INFO] Loaded D from: {args.resume_D}")

    # Grab a fixed sample batch for consistent visualisation
    sample_batch = next(iter(loader))

    # Loss weights
    lambda_cls = args.lambda_cls   # domain classification
    lambda_rec = args.lambda_rec   # cycle reconstruction
    lambda_gp  = args.lambda_gp    # gradient penalty

    print(f"\n{'='*60}")
    print(f"  Starting StarGAN training for {args.epochs} epochs")
    print(f"  Image size: {args.image_size}x{args.image_size} | Batch: {args.batch_size}")
    print(f"  lambda_cls={lambda_cls} | lambda_rec={lambda_rec} | lambda_gp={lambda_gp} | n_critic={args.n_critic}")
    print(f"{'='*60}\n")

    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        d_losses, g_losses = [], []
        d_adv_list, d_cls_list, d_gp_list = [], [], []
        g_adv_list, g_cls_list, g_rec_list = [], [], []

        for batch in loader:
            x_acne   = batch["acne"].to(device,   non_blocking=True)
            x_celeba = batch["celeba"].to(device,  non_blocking=True)
            lbl_acne   = batch["label_acne"].to(device,   non_blocking=True)
            lbl_celeba = batch["label_celeba"].to(device, non_blocking=True)

            # ────────────────────────────────────────────────────────────────
            # DISCRIMINATOR UPDATE (n_critic times per G update)
            # ────────────────────────────────────────────────────────────────
            for _ in range(args.n_critic):
                opt_D.zero_grad(set_to_none=True)

                with autocast():
                    # Real acne images → D
                    src_real, cls_real = D(x_acne)
                    d_loss_real        = adversarial_loss(src_real, is_real=True)
                    d_loss_cls_real    = domain_classification_loss(cls_real, lbl_acne)

                    # Fake clean images (acne → clean) → D
                    with torch.no_grad():
                        fake_clean = G(x_acne, lbl_celeba)
                    src_fake, _       = D(fake_clean.detach())
                    d_loss_fake        = adversarial_loss(src_fake, is_real=False)

                    # Gradient penalty (only WGAN-GP part; no autocast for autograd)
                gp = compute_gradient_penalty(D, x_acne, fake_clean.detach(), device)

                with autocast():
                    d_loss = (d_loss_real + d_loss_fake
                              + lambda_cls * d_loss_cls_real
                              + lambda_gp  * gp)

                scaler_D.scale(d_loss).backward()
                scaler_D.step(opt_D)
                scaler_D.update()

            d_losses.append(d_loss.item())
            d_adv_list.append((d_loss_real + d_loss_fake).item())
            d_cls_list.append(d_loss_cls_real.item())
            d_gp_list.append(gp.item())

            # ────────────────────────────────────────────────────────────────
            # GENERATOR UPDATE (once per n_critic D steps)
            # ────────────────────────────────────────────────────────────────
            opt_G.zero_grad(set_to_none=True)

            with autocast():
                # 1. Translate acne → clean
                fake_clean       = G(x_acne, lbl_celeba)
                src_fake, cls_fake = D(fake_clean)
                g_loss_adv       = adversarial_loss(src_fake, is_real=True)   # fool D
                g_loss_cls       = domain_classification_loss(cls_fake, lbl_celeba)

                # 2. Cycle: clean → acne (reconstruction)
                x_rec            = G(fake_clean, lbl_acne)
                g_loss_rec       = reconstruction_loss(x_rec, x_acne)

                # 3. Identity: translating to own domain should preserve image
                x_id             = G(x_celeba, lbl_celeba)
                g_loss_id        = reconstruction_loss(x_id, x_celeba) * 0.5

                g_loss = (g_loss_adv
                          + lambda_cls * g_loss_cls
                          + lambda_rec * (g_loss_rec + g_loss_id))

            scaler_G.scale(g_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            g_losses.append(g_loss.item())
            g_adv_list.append(g_loss_adv.item())
            g_cls_list.append(g_loss_cls.item())
            g_rec_list.append(g_loss_rec.item())

            global_step += 1

        # ── Scheduler ──
        sched_G.step()
        sched_D.step()

        elapsed = time.time() - t0

        # ── Log ──
        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"D_loss: {np.mean(d_losses):.4f} "
            f"(adv={np.mean(d_adv_list):.3f} cls={np.mean(d_cls_list):.3f} gp={np.mean(d_gp_list):.3f}) | "
            f"G_loss: {np.mean(g_losses):.4f} "
            f"(adv={np.mean(g_adv_list):.3f} cls={np.mean(g_cls_list):.3f} rec={np.mean(g_rec_list):.3f}) | "
            f"{elapsed:.1f}s"
        )

        # ── Save checkpoints ──
        torch.save(G.state_dict(), os.path.join(args.checkpoint_dir, "stargan_G_latest.pth"))
        torch.save(D.state_dict(), os.path.join(args.checkpoint_dir, "stargan_D_latest.pth"))

        # Save training state for resumption
        training_state = {
            'epoch': epoch + 1,
            'G': G.state_dict(),
            'D': D.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'scaler_G': scaler_G.state_dict(),
            'scaler_D': scaler_D.state_dict(),
            'sched_G': sched_G.state_dict(),
            'sched_D': sched_D.state_dict(),
        }
        torch.save(training_state, os.path.join(args.checkpoint_dir, "stargan_training_state_latest.pt"))

        # Full checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save(G.state_dict(), os.path.join(args.checkpoint_dir, f"stargan_G_ep{epoch+1}.pth"))
            torch.save(D.state_dict(), os.path.join(args.checkpoint_dir, f"stargan_D_ep{epoch+1}.pth"))
            torch.save(training_state, os.path.join(args.checkpoint_dir, f"stargan_training_state_ep{epoch+1}.pt"))
            print(f"  [CKPT] Saved epoch {epoch+1} checkpoints and training state.")

        # ── Visualise ──
        if (epoch + 1) % args.vis_every == 0 or epoch == 0:
            vis_path = os.path.join(args.vis_dir, f"epoch_{epoch+1:03d}.png")
            save_sample_images(G, sample_batch, device, vis_path)

    # ── Final save ──
    final_G = os.path.join(args.checkpoint_dir, "stargan_G.pth")
    final_D = os.path.join(args.checkpoint_dir, "stargan_D.pth")
    torch.save(G.state_dict(), final_G)
    torch.save(D.state_dict(), final_D)

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Generator     -> {final_G}")
    print(f"  Discriminator -> {final_D}")
    print(f"  Download these files and place in checkpoints/ locally.")
    print(f"{'='*60}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train StarGAN on Colab")
    p.add_argument("--acne_dir",      type=str,   default="dataset/Acne/")
    p.add_argument("--celeba_dir",    type=str,   default="dataset/celeba_hq/")
    p.add_argument("--max_celeba",    type=int,   default=500, help="Maximum number of CelebA images to load")
    p.add_argument("--image_size",    type=int,   default=256)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--g_lr",          type=float, default=1e-4)
    p.add_argument("--d_lr",          type=float, default=1e-4)
    p.add_argument("--n_critic",      type=int,   default=5,   help="D steps per G step")
    p.add_argument("--lambda_cls",    type=float, default=1.0)
    p.add_argument("--lambda_rec",    type=float, default=10.0)
    p.add_argument("--lambda_gp",     type=float, default=10.0)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--checkpoint_dir",type=str,   default="checkpoints/")
    p.add_argument("--vis_dir",       type=str,   default="outputs/stargan_vis/")
    p.add_argument("--vis_every",     type=int,   default=5)
    p.add_argument("--save_every",    type=int,   default=10)
    p.add_argument("--resume",        action="store_true", help="Resume training from latest training state")
    p.add_argument("--resume_G",      type=str,   default=None)
    p.add_argument("--resume_D",      type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)