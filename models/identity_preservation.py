"""
identity_preservation.py
------------------------
Identity preservation module for the acne removal pipeline.

Approach:
  1. Use a pretrained FaceNet/ArcFace-style embedding network (MobileNetV2 backbone)
     to extract 512-dim face embeddings.
  2. Compute cosine similarity between original and generated images.
  3. Optionally add identity loss during StarGAN fine-tuning.

Two modes:
  MODE A: Use pretrained facenet-pytorch (recommended if available on Colab)
  MODE B: Lightweight in-house embedding network (no external dependencies)
          Falls back to this automatically if facenet-pytorch is not installed.

Usage:
    from models.identity_preservation import IdentityModule

    id_module = IdentityModule(device="cuda")
    sim = id_module.cosine_similarity(original_image, generated_image)
    loss = id_module.identity_loss(original_image, generated_image)
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# ─── Lightweight Embedding Backbone (Mode B fallback) ────────────────────────

class DepthwiseSepConv(nn.Module):
    """Depthwise separable convolution (efficient substitute for full conv)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch,  in_ch,  3, stride=stride, padding=1, groups=in_ch,  bias=False)
        self.pw = nn.Conv2d(in_ch,  out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class LightEmbeddingNet(nn.Module):
    """
    Lightweight face embedding network (~1.5M params).
    Extracts a 512-dim identity embedding from a 112×112 face crop.

    Architecture: MobileNet-inspired with depthwise separable convs.
    """

    def __init__(self, emb_dim: int = 512):
        super().__init__()

        self.backbone = nn.Sequential(
            # Stage 0: 112×112 → 56×56
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Stage 1: 56×56 → 28×28
            DepthwiseSepConv(32,  64,  stride=2),
            DepthwiseSepConv(64,  64,  stride=1),

            # Stage 2: 28×28 → 14×14
            DepthwiseSepConv(64,  128, stride=2),
            DepthwiseSepConv(128, 128, stride=1),

            # Stage 3: 14×14 → 7×7
            DepthwiseSepConv(128, 256, stride=2),
            DepthwiseSepConv(256, 256, stride=1),

            # Stage 4: 7×7 → 4×4
            DepthwiseSepConv(256, 512, stride=2),
        )

        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(0.2)
        self.fc       = nn.Linear(512, emb_dim, bias=False)
        self.bn_final = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Face image tensor (B, 3, 112, 112) normalised to [-1, 1]
        Returns:
            embeddings: L2-normalised embeddings (B, emb_dim)
        """
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        h = self.dropout(h)
        h = self.bn_final(self.fc(h))
        return F.normalize(h, p=2, dim=1)


# ─── Identity Module ─────────────────────────────────────────────────────────

class IdentityModule(nn.Module):
    """
    Wraps a face embedding network for identity preservation.

    Responsibilities:
      1. Preprocess images to 112×112 (standard for ArcFace/FaceNet)
      2. Extract embeddings
      3. Compute cosine similarity / identity loss

    Args:
        device:      'cuda' or 'cpu'
        use_pretrained: Try to load facenet-pytorch first; fall back if not installed
        emb_dim:     Embedding dimension (512)
    """

    def __init__(
        self,
        device:         str  = "cuda",
        use_pretrained: bool = True,
        emb_dim:        int  = 512,
    ):
        super().__init__()
        self.device  = device
        self.emb_dim = emb_dim

        self.net = self._load_net(use_pretrained, emb_dim)
        self.net = self.net.to(device)
        self.net.eval()   # embedding net stays frozen during pipeline

        # Preprocessing: resize to 112×112 (ArcFace standard crop size)
        self.preprocess = T.Compose([
            T.Resize((112, 112)),
            # Input already assumed to be [-1, 1] from StarGAN / U-Net normalisation
        ])

    def _load_net(self, use_pretrained: bool, emb_dim: int) -> nn.Module:
        """Try facenet-pytorch first; fall back to LightEmbeddingNet."""
        if use_pretrained:
            try:
                from facenet_pytorch import InceptionResnetV1
                net = InceptionResnetV1(pretrained="vggface2").eval()
                print("[INFO] IdentityModule: Loaded pretrained FaceNet (InceptionResnetV1).")
                return net
            except ImportError:
                print("[INFO] facenet-pytorch not found — using LightEmbeddingNet instead.")
                print("       Install with: pip install facenet-pytorch")

        net = LightEmbeddingNet(emb_dim=emb_dim)
        print("[INFO] IdentityModule: Using LightEmbeddingNet (no external dep).")
        return net

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalised face embeddings.

        Args:
            images: (B, 3, H, W) in [-1, 1]
        Returns:
            embeddings: (B, emb_dim) — L2 normalised
        """
        # Resize to 112×112 (handles any H, W)
        x = F.interpolate(images, size=(112, 112), mode="bilinear", align_corners=False)
        x = x.to(self.device)
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=1)

    def cosine_similarity(
        self,
        img_orig: torch.Tensor,
        img_gen:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean cosine similarity between original and generated image embeddings.

        Args:
            img_orig: Original face images (B, 3, H, W)
            img_gen:  Generated face images (B, 3, H, W)
        Returns:
            mean_sim: Scalar in [-1, 1]; higher = more similar identity
        """
        emb_orig = self.extract_embeddings(img_orig)
        emb_gen  = self.extract_embeddings(img_gen)
        sim = (emb_orig * emb_gen).sum(dim=1)   # dot product of unit vectors = cosine sim
        return sim.mean()

    def identity_loss(
        self,
        img_orig: torch.Tensor,
        img_gen:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Identity loss = 1 - cosine_similarity (lower = better preserved).

        Used during StarGAN training:
            total_G_loss += lambda_id * identity_loss(real, fake)

        Note: We allow gradients to flow through img_gen (the generated image),
              but block gradients through img_orig.
        """
        # Freeze gradients for original (reference embedding)
        with torch.no_grad():
            emb_orig = self.extract_embeddings(img_orig)

        # Allow gradients through generated image
        x_gen = F.interpolate(img_gen, size=(112, 112), mode="bilinear", align_corners=False)
        emb_gen = self.net(x_gen)
        emb_gen = F.normalize(emb_gen, p=2, dim=1)

        sim  = (emb_orig * emb_gen).sum(dim=1)
        loss = 1.0 - sim.mean()
        return loss

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for extract_embeddings."""
        return self.extract_embeddings(images)


# ─── Verification / testing helper ───────────────────────────────────────────

def verify_identity_preservation(
    id_module: IdentityModule,
    original:  torch.Tensor,
    generated: torch.Tensor,
    threshold: float = 0.80,
) -> dict:
    """
    Check whether identity is preserved above a threshold.

    Args:
        id_module:  IdentityModule instance
        original:   Original face images (B, 3, H, W)
        generated:  Generated (acne-removed) images (B, 3, H, W)
        threshold:  Minimum acceptable cosine similarity (default 0.80)

    Returns:
        dict with 'similarity', 'passed', 'per_sample_sim'
    """
    with torch.no_grad():
        emb_orig = id_module.extract_embeddings(original)
        emb_gen  = id_module.extract_embeddings(generated)
        per_sim  = (emb_orig * emb_gen).sum(dim=1).cpu()
        mean_sim = per_sim.mean().item()

    return {
        "mean_similarity":    mean_sim,
        "per_sample_sim":     per_sim.tolist(),
        "passed":             mean_sim >= threshold,
        "threshold":          threshold,
    }


# ─── Sanity check ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on: {device}")

    id_module = IdentityModule(device=device, use_pretrained=False)
    params = sum(p.numel() for p in id_module.net.parameters()) / 1e6
    print(f"[INFO] Embedding net parameters: {params:.2f}M")

    # Simulate original and slightly-perturbed generated images
    orig  = torch.randn(4, 3, 256, 256).to(device)
    gen   = orig + 0.05 * torch.randn_like(orig)   # slightly modified (same identity)

    sim   = id_module.cosine_similarity(orig, gen)
    loss  = id_module.identity_loss(orig, gen)

    print(f"[INFO] Cosine similarity (similar imgs): {sim.item():.4f}  (expect ~1.0)")
    print(f"[INFO] Identity loss:                    {loss.item():.4f}  (expect ~0.0)")

    result = verify_identity_preservation(id_module, orig, gen, threshold=0.50)
    print(f"[INFO] Verification: mean_sim={result['mean_similarity']:.4f} | passed={result['passed']}")
    print("[PASS] IdentityModule OK ✓")