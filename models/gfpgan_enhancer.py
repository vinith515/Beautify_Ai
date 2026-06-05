"""
gfpgan_enhancer.py
------------------
GFPGAN face enhancement wrapper for the acne removal pipeline.

GFPGAN (GAN Prior-based Face Enhancement) restores and enhances facial details
after the StarGAN translation step, producing photorealistic, high-quality output.

Installation (run once):
    pip install gfpgan basicsr facexlib realesrgan

Model weights are downloaded automatically on first use to:
    checkpoints/GFPGANv1.4.pth

Pipeline position:
    StarGAN output → GFPGANEnhancer → Final output

If GFPGAN is not installed, the module falls back to a lightweight
sharpening + CLAHE enhancement (no external dependency).

Usage:
    enhancer = GFPGANEnhancer(device="cuda")
    enhanced = enhancer.enhance(image_tensor)   # (1, 3, H, W) in [-1,1]
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# ─── GFPGAN wrapper ──────────────────────────────────────────────────────────

class GFPGANEnhancer(nn.Module):
    """
    GFPGAN-based face enhancement module.

    Two modes:
      Mode A (preferred): Use real GFPGAN if `gfpgan` package is installed.
      Mode B (fallback):  Lightweight OpenCV-based enhancement (no extra deps).

    Args:
        model_path:  Path to GFPGANv1.4.pth weights (downloaded if not present)
        device:      'cuda' or 'cpu'
        upscale:     Output upscale factor (1 = same resolution, 2 = 2×)
        use_gfpgan:  Force GFPGAN mode (True) or fallback mode (False)
    """

    GFPGAN_URL = (
        "https://github.com/TencentARC/GFPGAN/releases/download/"
        "v1.3.0/GFPGANv1.4.pth"
    )

    def __init__(
        self,
        model_path: str  = "checkpoints/GFPGANv1.4.pth",
        device:     str  = "cuda",
        upscale:    int  = 1,
        use_gfpgan: bool = True,
    ):
        super().__init__()
        self.device    = device
        self.upscale   = upscale
        self.model_path = model_path

        self._gfpgan_model = None
        self._mode         = "fallback"

        if use_gfpgan:
            self._try_load_gfpgan(model_path, device, upscale)

        if self._mode == "fallback":
            print("[INFO] GFPGANEnhancer: Running in fallback (OpenCV sharpening) mode.")
            print("       Install GFPGAN with: pip install gfpgan basicsr facexlib realesrgan")

    def _try_load_gfpgan(self, model_path: str, device: str, upscale: int) -> None:
        """Attempt to load GFPGAN; silently fall back if unavailable."""
        try:
            from gfpgan import GFPGANer
        except ImportError:
            return

        # Download weights if not present
        if not os.path.exists(model_path):
            print(f"[INFO] GFPGAN weights not found. Downloading to {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                import urllib.request
                urllib.request.urlretrieve(self.GFPGAN_URL, model_path)
                print(f"[INFO] Downloaded GFPGAN weights to {model_path}")
            except Exception as e:
                print(f"[WARNING] Could not download GFPGAN weights: {e}")
                return

        try:
            # GFPGANer uses CPU/GPU internally via basicsr
            use_gpu = (device == "cuda") and torch.cuda.is_available()
            self._gfpgan_model = GFPGANer(
                model_path=model_path,
                upscale=upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,   # No background upsampling (faster)
            )
            self._mode = "gfpgan"
            print(f"[INFO] GFPGANEnhancer: Loaded GFPGANv1.4 ({'GPU' if use_gpu else 'CPU'} mode).")
        except Exception as e:
            print(f"[WARNING] Failed to initialise GFPGAN: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Enhance a face image tensor.

        Args:
            image: (1, 3, H, W) or (3, H, W) tensor in [-1, 1]
        Returns:
            enhanced: Same shape tensor in [-1, 1]
        """
        # Handle both 3D and 4D input
        single = (image.dim() == 3)
        if single:
            image = image.unsqueeze(0)

        if self._mode == "gfpgan":
            result = self._enhance_gfpgan_batch(image)
        else:
            result = self._enhance_fallback_batch(image)

        return result.squeeze(0) if single else result

    def enhance_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Enhance a batch of face images.

        Args:
            images: (B, 3, H, W) in [-1, 1]
        Returns:
            enhanced: (B, 3, H, W) in [-1, 1]
        """
        if self._mode == "gfpgan":
            return self._enhance_gfpgan_batch(images)
        return self._enhance_fallback_batch(images)

    # ── GFPGAN mode ───────────────────────────────────────────────────────────

    def _enhance_gfpgan_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Process each image in the batch through GFPGAN."""
        B = images.size(0)
        results = []

        for i in range(B):
            enhanced = self._enhance_single_gfpgan(images[i])
            results.append(enhanced)

        return torch.stack(results, dim=0)

    def _enhance_single_gfpgan(self, image: torch.Tensor) -> torch.Tensor:
        """
        Enhance a single image (3, H, W) tensor using GFPGAN.
        Converts to uint8 BGR numpy, runs GFPGAN, converts back.
        """
        # [-1,1] → [0,255] uint8 BGR
        img_np = self._tensor_to_bgr(image)

        try:
            _, _, restored = self._gfpgan_model.enhance(
                img_np,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            result_np = restored
        except Exception as e:
            print(f"[WARNING] GFPGAN enhancement failed: {e} — using fallback for this image.")
            result_np = self._enhance_fallback_np(img_np)

        # Resize back to original size if upscale != 1
        H, W = image.shape[1], image.shape[2]
        if result_np.shape[:2] != (H, W):
            result_np = cv2.resize(result_np, (W, H), interpolation=cv2.INTER_LANCZOS4)

        return self._bgr_to_tensor(result_np).to(self.device)

    # ── Fallback (OpenCV) mode ────────────────────────────────────────────────

    def _enhance_fallback_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Process each image in the batch with OpenCV enhancement."""
        results = []
        for i in range(images.size(0)):
            results.append(self._enhance_fallback_tensor(images[i]))
        return torch.stack(results, dim=0)

    def _enhance_fallback_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """OpenCV-based sharpening + CLAHE (no GFPGAN)."""
        img_np = self._tensor_to_bgr(image)
        result_np = self._enhance_fallback_np(img_np)
        return self._bgr_to_tensor(result_np).to(self.device)

    @staticmethod
    def _enhance_fallback_np(img_bgr: np.ndarray) -> np.ndarray:
        """
        Lightweight image enhancement using OpenCV:
          1. CLAHE on L channel (contrast enhancement)
          2. Bilateral filter (edge-preserving smoothing)
          3. Unsharp masking (detail sharpening)
        """
        # Convert to LAB colour space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # CLAHE on luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_enh = clahe.apply(L)

        lab_enh = cv2.merge([L_enh, A, B])
        img_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

        # Bilateral filter for edge-preserving smoothing
        img_smooth = cv2.bilateralFilter(img_enh, d=9, sigmaColor=75, sigmaSpace=75)

        # Unsharp masking for subtle sharpening
        blur   = cv2.GaussianBlur(img_smooth, (0, 0), sigmaX=2.0)
        sharp  = cv2.addWeighted(img_smooth, 1.5, blur, -0.5, 0)

        return sharp.clip(0, 255).astype(np.uint8)

    # ── Tensor ↔ numpy helpers ────────────────────────────────────────────────

    @staticmethod
    def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
        """(3, H, W) in [-1,1] → (H, W, 3) uint8 BGR."""
        img = (t.detach().cpu() * 0.5 + 0.5).clamp(0, 1)   # [-1,1] → [0,1]
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)   # HWC
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def _bgr_to_tensor(img: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 BGR → (3, H, W) float in [-1,1]."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)  # [0,1]
        t = t.permute(2, 0, 1)          # (3, H, W)
        t = t * 2.0 - 1.0              # [0,1] → [-1,1]
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """nn.Module-compatible forward."""
        return self.enhance(x)


# ─── Sanity check ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on: {device}")

    # Force fallback mode for testing (no GFPGAN required)
    enhancer = GFPGANEnhancer(device=device, use_gfpgan=False)

    dummy = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(device)
    with torch.no_grad():
        out = enhancer.enhance_batch(dummy)

    print(f"[INFO] Input  shape: {dummy.shape}")
    print(f"[INFO] Output shape: {out.shape}")
    assert out.shape == dummy.shape, "Shape mismatch!"
    print(f"[INFO] Output range: [{out.min():.3f}, {out.max():.3f}]")
    print("[PASS] GFPGANEnhancer OK ✓")