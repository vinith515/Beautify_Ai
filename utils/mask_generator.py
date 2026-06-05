"""
mask_generator.py
-----------------
Generates binary acne masks from facial images using HSV color thresholding.
Detects reddish/inflamed skin regions (acne) and saves binary masks.

Usage:
    python utils/mask_generator.py --input dataset/acne/ --output dataset/masks/
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def generate_acne_mask(image_path: str, output_path: str, visualize: bool = False) -> np.ndarray:
    """
    Generate a binary acne mask from a facial image using HSV thresholding.

    Strategy:
      1. Convert BGR → HSV
      2. Threshold for reddish hues (acne tends to be red/pink)
      3. Apply morphological ops to clean up noise
      4. Save binary mask (255 = acne region, 0 = clear skin)

    Args:
        image_path:  Path to input acne image
        output_path: Path to save the binary mask
        visualize:   If True, saves a side-by-side debug image

    Returns:
        mask: Binary mask as numpy array (H, W), uint8
    """
    # --- Load image ---
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Resize to 256×256 for consistency
    img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    # --- Step 1: Slight blur to reduce noise before thresholding ---
    img_blur = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.5)

    # --- Step 2: Convert to HSV ---
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # --- Step 3: HSV range for reddish/inflamed acne ---
    # Red hues wrap around in HSV (0-10 and 160-180)
    # Acne: medium-high saturation, medium value (not pure white/black)
    lower_red1 = np.array([0,   40,  60],  dtype=np.uint8)
    upper_red1 = np.array([15,  255, 220], dtype=np.uint8)

    lower_red2 = np.array([155, 40,  60],  dtype=np.uint8)
    upper_red2 = np.array([180, 255, 220], dtype=np.uint8)

    # Pink/light-red range (early-stage acne)
    lower_pink = np.array([140, 20, 100], dtype=np.uint8)
    upper_pink = np.array([175, 120, 240], dtype=np.uint8)

    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(img_hsv, lower_pink,  upper_pink)

    # Combine all masks
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask,  mask3)

    # --- Step 4: Morphological cleanup ---
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Remove tiny noise spots
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
    # Fill small holes inside acne regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # --- Step 5: Keep only meaningful blobs (filter out tiny artifacts) ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = 30  # pixels — ignore anything smaller
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    # --- Save mask ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, clean_mask)

    # --- Optional: Save debug visualization ---
    if visualize:
        vis_path = output_path.replace(".png", "_debug.png")
        overlay = img_bgr.copy()
        overlay[clean_mask == 255] = [0, 0, 255]  # highlight acne in red
        side_by_side = np.hstack([img_bgr, cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR), overlay])
        cv2.imwrite(vis_path, side_by_side)

    return clean_mask


def batch_generate_masks(input_dir: str, output_dir: str, visualize: bool = False) -> None:
    """
    Process all images in input_dir and save masks to output_dir.

    Args:
        input_dir:  Folder containing acne images (.jpg/.jpeg/.png)
        output_dir: Folder to save binary masks
        visualize:  If True, also saves debug images
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"[WARNING] No images found in {input_dir}")
        return

    print(f"[INFO] Processing {len(image_files)} images...")
    success, failed = 0, 0

    for img_file in image_files:
        try:
            out_file = output_path / (img_file.stem + "_mask.png")
            generate_acne_mask(str(img_file), str(out_file), visualize=visualize)
            success += 1
            if success % 50 == 0:
                print(f"  [{success}/{len(image_files)}] processed...")
        except Exception as e:
            print(f"  [ERROR] {img_file.name}: {e}")
            failed += 1

    print(f"[DONE] Success: {success} | Failed: {failed} | Masks saved to: {output_dir}")


def generate_synthetic_mask_for_testing(output_path: str = "dataset/masks/test_mask.png") -> np.ndarray:
    """
    Generate a synthetic test mask (circles simulating acne spots).
    Useful for testing the pipeline without real data.

    Returns:
        mask: Synthetic binary mask (256, 256)
    """
    mask = np.zeros((256, 256), dtype=np.uint8)

    # Simulate acne spots at realistic face positions
    spots = [
        (80,  90,  12),   # left cheek, upper
        (100, 110, 8),
        (175, 85,  15),   # right cheek, upper
        (160, 105, 10),
        (128, 160, 6),    # chin area
        (115, 145, 9),
        (140, 140, 7),
        (90,  130, 5),
    ]
    for (cx, cy, r) in spots:
        cv2.circle(mask, (cx, cy), r, 255, -1)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, mask)
    print(f"[INFO] Synthetic test mask saved to: {output_path}")
    return mask


# ─── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate acne masks using HSV thresholding")
    parser.add_argument("--input",     type=str, default="dataset/acne/",  help="Input image directory")
    parser.add_argument("--output",    type=str, default="dataset/masks/", help="Output mask directory")
    parser.add_argument("--visualize", action="store_true",                 help="Save debug visualizations")
    parser.add_argument("--test",      action="store_true",                 help="Generate synthetic test mask only")
    args = parser.parse_args()

    if args.test:
        generate_synthetic_mask_for_testing("dataset/masks/test_mask.png")
    else:
        batch_generate_masks(args.input, args.output, visualize=args.visualize)