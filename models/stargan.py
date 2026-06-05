"""
stargan.py
----------
StarGAN implementation for acne-to-clear-skin image translation.

Architecture:
  Generator:
    - Input:  (B, 3+n_domains, H, W)  (image + target domain one-hot)
    - Output: (B, 3, H, W)
    - Structure: Encoder → Residual Blocks → Decoder
    - Uses Instance Normalisation (standard for image-translation tasks)

  Discriminator:
    - PatchGAN discriminator with auxiliary domain classifier
    - Input:  (B, 3, H, W)
    - Output: (B, 1, H', W'), (B, n_domains)  [real/fake patch + domain logits]

Domain setup:
  n_domains = 2
  Domain 0: Acne
  Domain 1: Clear skin

Training runs on Google Colab (T4 GPU) with mixed precision.

Reference: StarGAN v1 (Choi et al. 2018), adapted for 2-domain acne removal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Residual Block ──────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual block with Instance Normalisation (used inside the generator bottleneck).
    Preserves spatial resolution and allows deep networks without vanishing gradients.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ─── Generator ───────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    StarGAN Generator.

    Takes a source image and a target domain one-hot label as input,
    produces a translated image in the target domain.

    Architecture:
      Encoder:       3 downsampling conv layers (stride 2)
      Bottleneck:    n_res_blocks residual blocks
      Decoder:       3 upsampling transpose conv layers
      Output:        Tanh activation → pixel values in [-1, 1]

    Args:
        n_domains:     Number of domains (default 2: acne, clean)
        conv_dim:      Base number of filters (default 64)
        n_res_blocks:  Number of residual blocks (default 6)
        image_size:    Target image size (256 or 128)
    """

    def __init__(
        self,
        n_domains:    int = 2,
        conv_dim:     int = 64,
        n_res_blocks: int = 6,
        image_size:   int = 256,
    ):
        super().__init__()
        self.n_domains = n_domains

        layers = []

        # ── Encoder (3 → conv_dim*4, spatial: 256 → 32) ──
        # Initial conv: full resolution feature extraction
        layers += [
            nn.Conv2d(3 + n_domains, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
        ]
        # Two downsampling steps
        curr_dim = conv_dim
        for _ in range(2):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # ── Bottleneck (residual blocks, no spatial change) ──
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock(curr_dim))

        # ── Decoder (conv_dim*4 → 3, spatial: 32 → 256) ──
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim //= 2

        # Output: project to 3-channel RGB
        layers += [
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Source image (B, 3, H, W)
            c: Target domain one-hot (B, n_domains)

        Returns:
            Translated image (B, 3, H, W) in [-1, 1]
        """
        # Broadcast domain label c to spatial size and concatenate with image
        c = c.view(c.size(0), c.size(1), 1, 1)          # (B, n_domains, 1, 1)
        c = c.expand(-1, -1, x.size(2), x.size(3))       # (B, n_domains, H, W)
        x = torch.cat([x, c], dim=1)                     # (B, 3+n_domains, H, W)
        return self.net(x)


class GeneratorCompat(nn.Module):
    """
    Compatibility Generator matching the architecture used during the Colab
    training run (epoch 59 checkpoint).

    Differences from the standard Generator:
      - No InstanceNorm in encoder/decoder (only ReLU after convs)
      - bias=True on all Conv2d / ConvTranspose2d layers
      - Residual blocks still use InstanceNorm (unchanged)

    Use this when loading weights from `checkpoint (1).pth` or
    `generator (1).pth`.
    """

    def __init__(
        self,
        n_domains:    int = 2,
        conv_dim:     int = 64,
        n_res_blocks: int = 6,
        image_size:   int = 256,
    ):
        super().__init__()
        self.n_domains = n_domains

        layers = []

        # ── Encoder ──
        # net.0: Conv(5→64, 7×7, bias=True)
        layers.append(nn.Conv2d(3 + n_domains, conv_dim, kernel_size=7,
                                stride=1, padding=3, bias=True))
        # net.1: ReLU
        layers.append(nn.ReLU(inplace=True))

        curr_dim = conv_dim
        for _ in range(2):
            # Conv(dim→dim*2, 4×4, s=2, bias=True) → ReLU
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4,
                                    stride=2, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim *= 2

        # ── Bottleneck (residual blocks — same as standard) ──
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock(curr_dim))

        # ── Decoder ──
        for _ in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim //= 2

        # Output
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1,
                                padding=3, bias=True))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Same interface as Generator."""
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.net(x)


# ─── Discriminator ───────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    StarGAN PatchGAN Discriminator with auxiliary domain classifier.

    Outputs:
      - src: (B, 1, H', W')  — real/fake patch prediction (no sigmoid, used with WGAN-GP)
      - cls: (B, n_domains)  — domain classification logits

    Architecture:
      6 strided Conv layers (no BatchNorm, uses spectral norm for stability)
      Two output heads: PatchGAN + domain classifier

    Args:
        n_domains:  Number of domains
        conv_dim:   Base filter count
        n_layers:   Number of downsampling layers
        image_size: Input image size
    """

    def __init__(
        self,
        n_domains:  int = 2,
        conv_dim:   int = 64,
        n_layers:   int = 6,
        image_size: int = 256,
    ):
        super().__init__()
        self.n_domains  = n_domains
        self.image_size = image_size

        layers = []
        # Input: (B, 3, H, W)
        # Use LeakyReLU (no BN in discriminator — standard GAN practice)
        layers += [
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        curr_dim = conv_dim
        for _ in range(1, n_layers):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.01, inplace=True),
            ]
            curr_dim *= 2

        self.shared = nn.Sequential(*layers)

        # Compute spatial size after n_layers of stride-2 conv
        k = image_size // (2 ** n_layers)   # e.g. 256/(2^6) = 4
        k = max(k, 1)

        # PatchGAN output: real/fake score per spatial location
        self.src_head = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Domain classifier: global average pool → FC → domain logits
        self.cls_head = nn.Sequential(
            nn.Conv2d(curr_dim, n_domains, kernel_size=k, bias=False),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Image tensor (B, 3, H, W)
        Returns:
            src: Real/fake patch scores (B, 1, H', W')
            cls: Domain logits (B, n_domains)
        """
        h = self.shared(x)
        src = self.src_head(h)
        cls = self.cls_head(h).view(h.size(0), self.n_domains)
        return src, cls


# ─── WGAN-GP Gradient Penalty ────────────────────────────────────────────────

def compute_gradient_penalty(
    D:      nn.Module,
    real:   torch.Tensor,
    fake:   torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Compute WGAN gradient penalty: E[(||∇D(x̂)||₂ - 1)²]
    where x̂ is a random interpolation between real and fake samples.

    Improves training stability over vanilla GAN loss.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real)

    interpolated = (alpha * real + (1 - alpha) * fake).detach().requires_grad_(True)

    src, _ = D(interpolated)
    grad_out = torch.ones(src.size(), device=device)

    grads = torch.autograd.grad(
        outputs=src,
        inputs=interpolated,
        grad_outputs=grad_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ─── Loss Helpers ────────────────────────────────────────────────────────────

def adversarial_loss(src: torch.Tensor, is_real: bool) -> torch.Tensor:
    """WGAN loss: -E[D(real)] or +E[D(fake)]."""
    if is_real:
        return -src.mean()
    else:
        return  src.mean()


def domain_classification_loss(
    cls_pred: torch.Tensor,
    cls_target: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross entropy for domain classification.
    cls_pred:   (B, n_domains) raw logits
    cls_target: (B, n_domains) one-hot float
    """
    return F.binary_cross_entropy_with_logits(cls_pred, cls_target)


def reconstruction_loss(x_rec: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss (cycle consistency)."""
    return F.l1_loss(x_rec, x_real)


# ─── Model factory ───────────────────────────────────────────────────────────

def build_stargan(
    n_domains:    int  = 2,
    conv_dim:     int  = 64,
    n_res_blocks: int  = 6,
    image_size:   int  = 256,
    device:       str  = "cuda",
    compat_mode:  bool = False,
):
    """
    Instantiate Generator and Discriminator and move to device.

    Args:
        compat_mode: If True, use GeneratorCompat (matches the architecture
                     from the Colab training run — no InstanceNorm in
                     encoder/decoder, bias=True).  Required when loading
                     weights from ``checkpoint (1).pth`` or ``generator (1).pth``.

    Returns: (generator, discriminator)
    """
    GenClass = GeneratorCompat if compat_mode else Generator
    G = GenClass(n_domains=n_domains, conv_dim=conv_dim,
                 n_res_blocks=n_res_blocks, image_size=image_size).to(device)
    D = Discriminator(n_domains=n_domains, conv_dim=conv_dim,
                      image_size=image_size).to(device)

    # Weight initialisation: N(0, 0.02) for conv/linear, 1/0 for norm
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    G.apply(init_weights)
    D.apply(init_weights)

    return G, D


# ─── Sanity check ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on: {device}")

    G, D = build_stargan(n_domains=2, image_size=256, device=device)

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"[INFO] Generator    params: {g_params:.2f}M")
    print(f"[INFO] Discriminator params: {d_params:.2f}M")

    # Test forward pass
    B = 2
    x = torch.randn(B, 3, 256, 256).to(device)
    c = torch.zeros(B, 2).to(device)
    c[:, 1] = 1.0   # target domain: clear skin

    with torch.no_grad():
        fake     = G(x, c)
        src, cls = D(x)

    print(f"[INFO] G input:  {x.shape} | G output (fake): {fake.shape}")
    print(f"[INFO] D src:    {src.shape} | D cls:          {cls.shape}")
    assert fake.shape == (B, 3, 256, 256), "Generator output shape mismatch!"
    assert cls.shape  == (B, 2),           "Discriminator cls shape mismatch!"
    print("[PASS] StarGAN forward pass OK ✓")