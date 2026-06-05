"""
test_pipeline.py
----------------
Test script that validates every component of the acne removal pipeline.

Tests run entirely with synthetic/random data — no real images needed.
Run this BEFORE training to confirm all code is correct.

Usage:
    python test_pipeline.py

Expected output:
    All tests PASS.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

import torch
import cv2


# ─── Colour codes ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def pass_msg(name):    print(f"  {GREEN}[PASS]{RESET} {name}")
def fail_msg(name, e): print(f"  {RED}[FAIL]{RESET} {name}: {e}")
def skip_msg(name, r): print(f"  {YELLOW}[SKIP]{RESET} {name}: {r}")


# ─── Test 1: Mask Generator ──────────────────────────────────────────────────

def test_mask_generator():
    print("\n[1] Mask Generator")
    try:
        from utils.mask_generator import generate_acne_mask, generate_synthetic_mask_for_testing

        # Create a dummy face image with a reddish circle (simulated acne)
        dummy = np.ones((256, 256, 3), dtype=np.uint8) * 150  # beige base
        cv2.circle(dummy, (128, 128), 30, (80, 80, 200), -1)  # reddish spot (BGR)

        # Save to temp file and run mask generator
        os.makedirs("temp_test_acne", exist_ok=True)
        os.makedirs("temp_test_masks", exist_ok=True)
        cv2.imwrite("temp_test_acne/face.jpg", dummy)

        mask = generate_acne_mask("temp_test_acne/face.jpg", "temp_test_masks/face_mask.png")
        assert mask.shape == (256, 256), f"Wrong mask shape: {mask.shape}"
        assert mask.dtype == np.uint8,   "Wrong mask dtype"
        assert os.path.exists("temp_test_masks/face_mask.png"), "Mask file not saved"
        pass_msg("generate_acne_mask - shape, dtype, file saved")

        synth = generate_synthetic_mask_for_testing("temp_test_masks/synth.png")
        assert synth.shape == (256, 256)
        assert synth.max()  == 255
        pass_msg("generate_synthetic_mask_for_testing")

    except Exception as e:
        fail_msg("Mask Generator", e)
        raise


# ─── Test 2: U-Net ───────────────────────────────────────────────────────────

def test_unet():
    print("\n[2] U-Net Model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from models.unet import build_unet, UNet

        # Standard build
        model = build_unet(base_features=64, device=device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"       Parameters: {params:.2f}M")
        pass_msg(f"build_unet (base_features=64)")

        # Forward pass
        x     = torch.randn(4, 3, 256, 256).to(device)
        logits = model(x)
        assert logits.shape == (4, 1, 256, 256), f"Wrong output shape: {logits.shape}"
        pass_msg("forward pass (B=4, 3x256x256 -> 1x256x256)")

        # predict_mask
        masks = model.predict_mask(x)
        assert masks.shape == (4, 1, 256, 256)
        unique_vals = masks.unique().tolist()
        assert all(v in [0, 1] for v in unique_vals), "Mask should be binary"
        pass_msg("predict_mask (binary output)")

        # Small model for 4GB GPU
        model_sm = build_unet(base_features=32, device=device)
        params_sm = sum(p.numel() for p in model_sm.parameters()) / 1e6
        print(f"       Small model params: {params_sm:.2f}M")
        out_sm = model_sm(x)
        assert out_sm.shape == (4, 1, 256, 256)
        pass_msg("build_unet (base_features=32, low-VRAM variant)")

    except Exception as e:
        fail_msg("U-Net", e)
        raise


# ─── Test 3: Dataset Loaders ─────────────────────────────────────────────────

def test_datasets():
    print("\n[3] Dataset Loaders")
    try:
        from utils.dataset import AcneSegmentationDataset, get_unet_loaders

        # Create minimal synthetic dataset
        acne_dir  = "temp_test_acne"
        mask_dir  = "temp_test_masks"
        os.makedirs(acne_dir,  exist_ok=True)
        os.makedirs(mask_dir,  exist_ok=True)

        for i in range(12):
            img  = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(mask, (100+i*5, 100), 20, 255, -1)
            cv2.imwrite(f"{acne_dir}/img_{i:03d}.jpg",        img)
            cv2.imwrite(f"{mask_dir}/img_{i:03d}_mask.png",   mask)

        dataset = AcneSegmentationDataset(acne_dir, mask_dir, image_size=256, augment=True)
        assert len(dataset) >= 12
        img_t, mask_t = dataset[0]
        assert img_t.shape  == (3, 256, 256)
        assert mask_t.shape == (1, 256, 256)
        assert img_t.min() >= -1.1 and img_t.max() <= 1.1
        unique = mask_t.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique)
        pass_msg("AcneSegmentationDataset - shape and value ranges")

        train_loader, val_loader = get_unet_loaders(
            acne_dir, mask_dir, batch_size=2, num_workers=0
        )
        imgs, masks = next(iter(train_loader))
        assert imgs.shape  == (2, 3, 256, 256)
        assert masks.shape == (2, 1, 256, 256)
        pass_msg("get_unet_loaders - DataLoader batches")

    except Exception as e:
        fail_msg("Dataset Loaders", e)
        raise


# ─── Test 4: StarGAN ─────────────────────────────────────────────────────────

def test_stargan():
    print("\n[4] StarGAN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from models.stargan import (
            build_stargan, compute_gradient_penalty,
            adversarial_loss, domain_classification_loss, reconstruction_loss,
        )

        G, D = build_stargan(n_domains=2, conv_dim=64, image_size=256, device=device)
        g_params = sum(p.numel() for p in G.parameters()) / 1e6
        d_params = sum(p.numel() for p in D.parameters()) / 1e6
        print(f"       G params: {g_params:.2f}M | D params: {d_params:.2f}M")
        pass_msg("build_stargan")

        B  = 2
        x  = torch.randn(B, 3, 256, 256).to(device)
        c  = torch.zeros(B, 2).to(device)
        c[:, 1] = 1.0

        with torch.no_grad():
            fake     = G(x, c)
            src, cls = D(x)

        assert fake.shape == (B, 3, 256, 256), f"G output shape wrong: {fake.shape}"
        assert src.shape[0]  == B,             f"D src shape wrong:    {src.shape}"
        assert cls.shape     == (B, 2),        f"D cls shape wrong:    {cls.shape}"
        pass_msg("Generator & Discriminator forward pass")

        # Loss functions
        adv = adversarial_loss(src, is_real=True)
        assert adv.item() != 0.0
        pass_msg("adversarial_loss")

        cls_loss = domain_classification_loss(cls, c)
        assert cls_loss.item() >= 0.0
        pass_msg("domain_classification_loss")

        rec_loss = reconstruction_loss(fake, x)
        assert rec_loss.item() >= 0.0
        pass_msg("reconstruction_loss")

        gp = compute_gradient_penalty(D, x, fake.detach(), device)
        assert gp.item() >= 0.0
        pass_msg("compute_gradient_penalty")

    except Exception as e:
        fail_msg("StarGAN", e)
        raise


# ─── Test 5: Identity Module ─────────────────────────────────────────────────

def test_identity():
    print("\n[5] Identity Preservation")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from models.identity_preservation import IdentityModule, verify_identity_preservation

        # Force lightweight fallback (no facenet-pytorch needed)
        id_mod = IdentityModule(device=device, use_pretrained=False)
        params = sum(p.numel() for p in id_mod.net.parameters()) / 1e6
        print(f"       Embedding net params: {params:.2f}M")
        pass_msg("IdentityModule initialised (LightEmbeddingNet)")

        orig = torch.randn(4, 3, 256, 256).to(device)
        gen  = orig + 0.05 * torch.randn_like(orig)   # near-identical

        emb = id_mod.extract_embeddings(orig)
        assert emb.shape == (4, 512)
        # Check L2 normalised
        norms = emb.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
        pass_msg("extract_embeddings - shape & L2 norm")

        sim = id_mod.cosine_similarity(orig, gen)
        assert 0.5 < sim.item() <= 1.01, f"Unexpected sim: {sim.item()}"
        pass_msg(f"cosine_similarity (similar imgs): {sim.item():.4f}")

        loss = id_mod.identity_loss(orig, gen)
        assert 0.0 <= loss.item() < 0.5
        pass_msg(f"identity_loss: {loss.item():.4f}")

        result = verify_identity_preservation(id_mod, orig, gen, threshold=0.50)
        assert result["passed"]
        pass_msg("verify_identity_preservation")

    except Exception as e:
        fail_msg("Identity Module", e)
        raise


# ─── Test 6: GFPGAN Enhancer ─────────────────────────────────────────────────

def test_gfpgan():
    print("\n[6] GFPGAN Enhancer")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from models.gfpgan_enhancer import GFPGANEnhancer

        # Force fallback mode (OpenCV only)
        enh = GFPGANEnhancer(device=device, use_gfpgan=False)

        dummy = torch.randn(2, 3, 256, 256).clamp(-1, 1).to(device)

        out_batch = enh.enhance_batch(dummy)
        assert out_batch.shape == (2, 3, 256, 256)
        pass_msg("enhance_batch - output shape")

        out_single = enh.enhance(dummy[0])
        assert out_single.shape == (3, 256, 256)
        pass_msg("enhance (single image)")

        assert out_batch.min() >= -1.05 and out_batch.max() <= 1.05
        pass_msg("enhance_batch - value range [-1, 1]")

    except Exception as e:
        fail_msg("GFPGAN Enhancer", e)
        raise


# ─── Test 7: Training losses ─────────────────────────────────────────────────

def test_training_losses():
    print("\n[7] Training Losses")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from training.train_unet import DiceLoss, CombinedLoss, compute_iou

        B, H, W = 4, 256, 256
        logits  = torch.randn(B, 1, H, W).to(device)
        targets = (torch.rand(B, 1, H, W) > 0.5).float().to(device)

        dice_loss = DiceLoss()(logits, targets)
        assert 0.0 <= dice_loss.item() <= 2.0
        pass_msg(f"DiceLoss: {dice_loss.item():.4f}")

        combined = CombinedLoss()(logits, targets)
        assert combined.item() > 0.0
        pass_msg(f"CombinedLoss: {combined.item():.4f}")

        iou = compute_iou(logits, targets)
        assert 0.0 <= iou <= 1.0
        pass_msg(f"IoU metric: {iou:.4f}")

    except Exception as e:
        fail_msg("Training Losses", e)
        raise


# ─── Test 8: Full Pipeline (mock weights) ────────────────────────────────────

def test_pipeline_integration():
    print("\n[8] Full Pipeline Integration (untrained weights)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from inference.pipeline import AcneRemovalPipeline, FaceDetector, load_image

        # Test face detector on dummy image
        detector = FaceDetector(device=device)
        dummy_bgr = np.ones((256, 256, 3), dtype=np.uint8) * 150
        face_crop, bbox = detector.detect_and_align(dummy_bgr, output_size=256)
        assert face_crop.shape == (256, 256, 3)
        pass_msg("FaceDetector.detect_and_align")

        # Test full pipeline with missing checkpoints (uses random weights)
        pipe = AcneRemovalPipeline(
            unet_checkpoint    = "MISSING_FOR_TEST.pth",
            stargan_checkpoint = "MISSING_FOR_TEST.pth",
            device             = device,
            use_gfpgan         = False,
            use_identity_check = True,
        )
        pass_msg("AcneRemovalPipeline initialised")

        # Process dummy image
        dummy_input = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result = pipe.process(dummy_input)

        assert "output" in result
        assert result["output"].shape == (256, 256, 3)
        assert "mask" in result
        assert "identity_similarity" in result
        assert "timing" in result
        pass_msg("pipeline.process() - returned correct keys")
        pass_msg(f"  timing total: {result['timing']['total']:.2f}s")

    except Exception as e:
        fail_msg("Pipeline Integration", e)
        raise


# ─── Run all tests ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Acne Removal Pipeline - Component Tests")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    tests = [
        ("Mask Generator",        test_mask_generator),
        ("U-Net Model",           test_unet),
        ("Dataset Loaders",       test_datasets),
        ("StarGAN",               test_stargan),
        ("Identity Preservation", test_identity),
        ("GFPGAN Enhancer",       test_gfpgan),
        ("Training Losses",       test_training_losses),
        ("Pipeline Integration",  test_pipeline_integration),
    ]

    passed, failed = 0, 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {RED}[FAILED]{RESET} {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET} / {len(tests)} total")
    print("=" * 60)

    if failed == 0:
        print(f"\n{GREEN}[v] All tests passed! Pipeline is ready.{RESET}\n")
    else:
        print(f"\n{RED}[x] {failed} test(s) failed. Fix errors before training.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()