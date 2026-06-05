"""
unet.py
-------
U-Net architecture for acne segmentation.

Input:  (B, 3, 256, 256)  — RGB facial image
Output: (B, 1, 256, 256)  — Binary acne mask logits

Architecture:
  Encoder: 4 downsampling blocks (Conv → BN → ReLU × 2, then MaxPool)
  Bottleneck: deepest feature map
  Decoder: 4 upsampling blocks (TransposeConv + skip connection → Conv → BN → ReLU × 2)
  Head: 1×1 Conv → single-channel logit map

Memory estimate on 4GB GPU with batch=4, fp16: ~1.2 GB ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm → ReLU layers.
    This is the standard U-Net 'feature extraction' block.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    Encoder block: MaxPool2d (halves spatial dims) → DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Decoder block: Upsample (or TransposeConv) → concat skip → DoubleConv.

    bilinear=True  → lighter (fewer params), slightly smoother output
    bilinear=False → learnable TransposeConv upsampling
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            # Upsample then halve channels with a 1×1 conv
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Learnable transpose convolution
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle odd-sized feature maps: pad x to match skip dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h > 0 or diff_w > 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along channel dimension (skip connection)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1×1 convolution mapping to num_classes channels.
    For binary segmentation: num_classes=1 (sigmoid applied in loss / inference).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─── U-Net ───────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Full U-Net for binary acne segmentation.

    Args:
        in_channels:  Number of input channels (3 for RGB)
        out_channels: Number of output channels (1 for binary mask)
        base_features: Number of feature maps in first encoder block (default 64)
                       Reduce to 32 if VRAM is tight.
        bilinear:     Use bilinear upsampling instead of transposed convolutions
    """

    def __init__(
        self,
        in_channels:   int  = 3,
        out_channels:  int  = 1,
        base_features: int  = 64,
        bilinear:      bool = True,
    ):
        super().__init__()
        f = base_features   # shorthand
        factor = 2 if bilinear else 1

        # ── Encoder ──────────────────────────────────────────────────────────
        self.inc   = DoubleConv(in_channels, f)          # 256 → 256, ch: 3→64
        self.down1 = Down(f,          f * 2)             # 256 → 128, ch: 64→128
        self.down2 = Down(f * 2,      f * 4)             # 128 → 64,  ch: 128→256
        self.down3 = Down(f * 4,      f * 8)             # 64  → 32,  ch: 256→512
        self.down4 = Down(f * 8,      f * 16 // factor)  # 32  → 16,  ch: 512→1024(or 512)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up1   = Up(f * 16,       f * 8  // factor, bilinear)
        self.up2   = Up(f * 8,        f * 4  // factor, bilinear)
        self.up3   = Up(f * 4,        f * 2  // factor, bilinear)
        self.up4   = Up(f * 2,        f,                bilinear)

        # ── Output head ──────────────────────────────────────────────────────
        self.outc  = OutConv(f, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            logits: Raw logit map (B, 1, H, W)
                    Apply sigmoid for probabilities, then threshold at 0.5 for mask.
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder (with skip connections)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)

        # Output logits
        logits = self.outc(x)
        return logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Convenience method: returns binary mask (0/1).
        Args:
            x:         Input tensor (B, 3, H, W)
            threshold: Sigmoid threshold for positive class
        Returns:
            mask: Binary tensor (B, 1, H, W), dtype=torch.uint8
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            mask   = (probs > threshold).to(torch.uint8)
        return mask


# ─── Model factory ───────────────────────────────────────────────────────────

def build_unet(base_features: int = 64, device: str = "cuda") -> UNet:
    """
    Instantiate U-Net and move to device.
    Use base_features=32 on very low VRAM.
    """
    model = UNet(in_channels=3, out_channels=1, base_features=base_features)
    model = model.to(device)
    return model


# ─── Quick sanity check ──────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on: {device}")

    model = build_unet(base_features=64, device=device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] U-Net parameters: {total_params:.2f}M")

    # Test forward pass with batch size 4
    dummy = torch.randn(4, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(dummy)

    print(f"[INFO] Input shape:  {dummy.shape}")
    print(f"[INFO] Output shape: {out.shape}")   # expect (4, 1, 256, 256)
    assert out.shape == (4, 1, 256, 256), "Shape mismatch!"
    print("[PASS] U-Net forward pass OK ✓")