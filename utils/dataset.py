"""
dataset.py
----------
PyTorch Dataset classes for the acne removal pipeline.

AcneSegmentationDataset:
    Pairs acne images with their binary masks for U-Net training.
    - Images resized to 256×256
    - Masks binarised at threshold 127
    - Augmentation: horizontal flip, colour jitter, random rotation

StarGANDataset:
    Unpaired image dataset with domain labels.
    - Domain 0: acne images
    - Domain 1: clean (CelebA) images
    - Used for StarGAN training on Colab
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─── U-Net Dataset ───────────────────────────────────────────────────────────

class AcneSegmentationDataset(Dataset):
    """
    Dataset for U-Net acne segmentation training.

    Expected structure:
        dataset/
          acne/        ← RGB acne face images  (.jpg / .png)
          masks/       ← matching binary masks  (*_mask.png)

    Mask naming convention:
        If image is `img001.jpg`, mask must be `img001_mask.png`
        (or any file whose stem contains the image stem — see _find_mask).

    Args:
        acne_dir:   Path to acne images directory
        mask_dir:   Path to mask images directory
        image_size: Resize target (default 256)
        augment:    Whether to apply data augmentation
    """

    def __init__(
        self,
        acne_dir:   str,
        mask_dir:   str,
        image_size: int  = 256,
        augment:    bool = True,
    ):
        self.acne_dir   = Path(acne_dir)
        self.mask_dir   = Path(mask_dir)
        self.image_size = image_size
        self.augment    = augment

        # Collect all image files
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = sorted([
            f for f in self.acne_dir.iterdir()
            if f.suffix.lower() in exts
        ])

        # Pair each image with its mask
        self.pairs = []
        for img_path in self.image_files:
            mask_path = self._find_mask(img_path)
            if mask_path is not None:
                self.pairs.append((img_path, mask_path))
            else:
                print(f"[WARNING] No mask found for {img_path.name}, skipping.")

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image-mask pairs found.\n"
                f"  Images in: {acne_dir}\n"
                f"  Masks  in: {mask_dir}\n"
                "Run utils/mask_generator.py first to generate masks."
            )
        print(f"[INFO] AcneSegmentationDataset: {len(self.pairs)} pairs loaded.")

    def _find_mask(self, img_path: Path) -> Optional[Path]:
        """Search mask_dir for a mask matching the image stem."""
        candidates = [
            self.mask_dir / f"{img_path.stem}_mask.png",
            self.mask_dir / f"{img_path.stem}.png",
            self.mask_dir / f"{img_path.stem}_mask.jpg",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Fuzzy search: find any mask containing the image stem
        for f in self.mask_dir.iterdir():
            if img_path.stem in f.stem:
                return f
        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # ── Load image (RGB) ──
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # ── Load mask (grayscale) ──
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # ── Augmentation (same transform applied to both) ──
        if self.augment:
            image, mask = self._augment(image, mask)

        # ── To tensor ──
        image = TF.to_tensor(image)                           # (3, H, W), float [0,1]
        image = TF.normalize(image, [0.5]*3, [0.5]*3)        # normalise to [-1, 1]

        mask  = torch.from_numpy(np.array(mask))              # (H, W), uint8
        mask  = (mask > 127).float().unsqueeze(0)             # (1, H, W), binary float

        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image):
        """Apply identical spatial augmentation to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random rotation ±15°
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=T.InterpolationMode.NEAREST)

        # Random crop and resize (zoom effect)
        if random.random() > 0.5:
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.85, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, h, w,
                                    (self.image_size, self.image_size), T.InterpolationMode.BILINEAR)
            mask  = TF.resized_crop(mask,  i, j, h, w,
                                    (self.image_size, self.image_size), T.InterpolationMode.NEAREST)

        # Colour jitter on image only (NOT mask)
        if random.random() > 0.5:
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            image  = jitter(image)

        return image, mask


# ─── StarGAN Dataset ─────────────────────────────────────────────────────────

class StarGANDataset(Dataset):
    """
    Unpaired image dataset with domain labels for StarGAN training.

    Domain 0 → Acne images  (dataset/acne/)
    Domain 1 → Clean images (dataset/celeba/)

    Each __getitem__ returns a random image from each domain,
    so the batch always has both domains represented.

    Args:
        acne_dir:   Path to acne images
        celeba_dir: Path to clean CelebA images
        image_size: Resize target (128 or 256)
        augment:    Whether to apply augmentation
    """

    def __init__(
        self,
        acne_dir:   str,
        celeba_dir: str,
        image_size: int  = 256,
        augment:    bool = True,
        max_celeba: Optional[int] = None,
    ):
        self.image_size = image_size
        self.augment    = augment

        exts = {".jpg", ".jpeg", ".png"}

        self.acne_files   = sorted([
            f for f in Path(acne_dir).rglob('*')
            if f.is_file() and f.suffix.lower() in exts
        ])
        self.celeba_files = sorted([
            f for f in Path(celeba_dir).rglob('*')
            if f.is_file() and f.suffix.lower() in exts
        ])
        
        if max_celeba is not None:
            self.celeba_files = self.celeba_files[:max_celeba]

        if not self.acne_files:
            raise RuntimeError(f"No images found in {acne_dir}")
        if not self.celeba_files:
            raise RuntimeError(f"No images found in {celeba_dir}")

        # Dataset length = longer of the two splits
        self._len = max(len(self.acne_files), len(self.celeba_files))

        self.transform = self._build_transform()
        print(f"[INFO] StarGANDataset: {len(self.acne_files)} acne | "
              f"{len(self.celeba_files)} clean | effective len={self._len}")

    def _build_transform(self) -> T.Compose:
        ops = [T.Resize((self.image_size, self.image_size), T.InterpolationMode.BILINEAR)]
        if self.augment:
            ops += [T.RandomHorizontalFlip(p=0.5)]
            ops += [T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)]
        ops += [
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ]
        return T.Compose(ops)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        # Cycle through lists (handle different lengths)
        acne_img   = Image.open(self.acne_files[idx % len(self.acne_files)]).convert("RGB")
        celeba_img = Image.open(self.celeba_files[idx % len(self.celeba_files)]).convert("RGB")

        acne_tensor   = self.transform(acne_img)    # domain 0
        celeba_tensor = self.transform(celeba_img)  # domain 1

        # Domain labels as 1-D float tensors (used in classifier head)
        label_acne   = torch.tensor([1, 0], dtype=torch.float32)  # one-hot domain 0
        label_celeba = torch.tensor([0, 1], dtype=torch.float32)  # one-hot domain 1

        return {
            "acne":          acne_tensor,
            "celeba":        celeba_tensor,
            "label_acne":    label_acne,
            "label_celeba":  label_celeba,
        }


# ─── DataLoader factories ────────────────────────────────────────────────────

def get_unet_loaders(
    acne_dir:   str,
    mask_dir:   str,
    image_size: int = 256,
    batch_size: int = 4,
    val_split:  float = 0.15,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for U-Net training.

    Splits dataset randomly into train / val.
    Uses pin_memory=True for faster GPU transfer.
    """
    full_dataset = AcneSegmentationDataset(acne_dir, mask_dir, image_size, augment=True)

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Disable augmentation for validation
    val_set.dataset.augment = False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Train samples: {n_train} | Val samples: {n_val}")
    return train_loader, val_loader


def get_stargan_loader(
    acne_dir:   str,
    celeba_dir: str,
    image_size: int = 256,
    batch_size: int = 2,
    num_workers: int = 2,
    max_celeba: Optional[int] = None,
) -> DataLoader:
    """Returns DataLoader for StarGAN training."""
    dataset = StarGANDataset(acne_dir, celeba_dir, image_size, augment=True, max_celeba=max_celeba)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("[TEST] Creating synthetic data for dataset tests...")

    # Create tiny dummy dataset for testing
    os.makedirs("dataset/acne",   exist_ok=True)
    os.makedirs("dataset/masks",  exist_ok=True)
    os.makedirs("dataset/celeba", exist_ok=True)

    # Write 10 dummy images and masks
    for i in range(10):
        img   = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        mask  = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (100 + i*5, 100), 20, 255, -1)

        cv2.imwrite(f"dataset/acne/img_{i:03d}.jpg",         img)
        cv2.imwrite(f"dataset/masks/img_{i:03d}_mask.png",   mask)
        cv2.imwrite(f"dataset/celeba/celeba_{i:03d}.jpg",    img)

    train_loader, val_loader = get_unet_loaders(
        acne_dir="dataset/acne",
        mask_dir="dataset/masks",
        batch_size=2,
        num_workers=0,
    )

    imgs, masks = next(iter(train_loader))
    print(f"[TEST] Image batch shape: {imgs.shape}")    # (2, 3, 256, 256)
    print(f"[TEST] Mask  batch shape: {masks.shape}")   # (2, 1, 256, 256)
    print(f"[TEST] Image range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"[TEST] Mask  unique values: {masks.unique().tolist()}")
    print("[PASS] Dataset loader OK ✓")