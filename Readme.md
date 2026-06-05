# AI-Based Acne Removal Pipeline
## U-Net + StarGAN + GFPGAN

---

## Project Structure
```
project/
├── dataset/
│   ├── acne/           ← Acne face images (.jpg/.png)
│   ├── celeba/         ← Clean CelebA-HQ images
│   └── masks/          ← Generated binary masks (auto-created)
├── models/
│   ├── unet.py                  ← U-Net segmentation model
│   ├── stargan.py               ← StarGAN Generator + Discriminator
│   ├── identity_preservation.py ← ArcFace/FaceNet embedding module
│   └── gfpgan_enhancer.py       ← GFPGAN enhancement wrapper
├── training/
│   ├── train_unet.py    ← U-Net training loop (LOCAL GPU)
│   └── train_stargan.py ← StarGAN training loop (COLAB)
├── utils/
│   ├── mask_generator.py ← OpenCV HSV acne mask generation
│   └── dataset.py        ← PyTorch Dataset & DataLoader classes
├── inference/
│   └── pipeline.py       ← Full end-to-end inference pipeline
├── checkpoints/          ← Saved model weights
├── outputs/              ← Visualisations and results
├── test_pipeline.py      ← Component tests (run this first!)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run tests (validates all code with synthetic data)
```bash
cd project/
python test_pipeline.py
```
Expected: All 8 tests PASS.

---

## Step-by-Step Training

### STEP 1 — Generate Acne Masks (Local)
```bash
python utils/mask_generator.py \
    --input  dataset/acne/ \
    --output dataset/masks/ \
    --visualize
```
Or test with a synthetic mask:
```bash
python utils/mask_generator.py --test
```

### STEP 2 — Train U-Net (Local, RTX 3050 4GB)
```bash
python training/train_unet.py \
    --acne_dir   dataset/acne/ \
    --mask_dir   dataset/masks/ \
    --epochs     50 \
    --batch_size 4 \
    --lr         1e-4
```
- Saves best model → `checkpoints/unet.pth`
- Visualisations  → `outputs/unet_vis/`
- Uses mixed precision (fp16) automatically on CUDA

If running out of VRAM, reduce to `--batch_size 2` or edit `unet.py` to use `base_features=32`.

### STEP 3 — Train StarGAN (Google Colab T4)
Upload the project to Colab, then:
```bash
!pip install -r requirements.txt
!python training/train_stargan.py \
    --acne_dir   dataset/acne/ \
    --celeba_dir dataset/celeba/ \
    --epochs     100 \
    --batch_size 2 \
    --image_size 256
```
- Saves Generator → `checkpoints/stargan_G.pth`
- Saves Discriminator → `checkpoints/stargan_D.pth`

**Download these files** and place in your local `checkpoints/` folder.

### STEP 4 — Install Optional Enhancements
```bash
# Identity preservation (FaceNet):
pip install facenet-pytorch

# Face enhancement (GFPGAN):
pip install gfpgan basicsr facexlib realesrgan
# Weights auto-download on first run to checkpoints/GFPGANv1.4.pth
```

### STEP 5 — Run Inference
```bash
# Single image:
python inference/pipeline.py \
    --input  path/to/face.jpg \
    --output outputs/result.jpg \
    --unet   checkpoints/unet.pth \
    --stargan checkpoints/stargan_G.pth

# Batch (folder):
python inference/pipeline.py \
    --input  dataset/test_images/ \
    --output outputs/batch_results/
```

---

## Pipeline Flow
```
Input Image
    ↓
Face Detection (RetinaFace / OpenCV fallback)
    ↓
U-Net Acne Segmentation → binary mask
    ↓
StarGAN Translation (acne → clear skin)
    ↓
Mask-Guided Blending (preserve non-acne regions)
    ↓
Identity Check (cosine similarity ≥ 0.75)
    ↓
GFPGAN Enhancement
    ↓
Paste back to original image
    ↓
Final Output
```

---

## Backend Integration
```python
from inference.pipeline import AcneRemovalPipeline

pipe = AcneRemovalPipeline(
    unet_checkpoint    = "checkpoints/unet.pth",
    stargan_checkpoint = "checkpoints/stargan_G.pth",
    device             = "cuda",
)

# From file path:
result = pipe.process("input.jpg", save_path="output.jpg")

# From numpy array (backend):
import cv2
img = cv2.imread("input.jpg")
result = pipe.process(img)

# result keys:
#   output               → (H,W,3) BGR uint8 — final image
#   mask                 → (H,W) uint8        — acne mask
#   identity_similarity  → float 0–1
#   acne_coverage        → float 0–1
#   timing               → dict of stage timings
```

---

## GPU Memory Usage (Estimated)
| Model       | fp32 B=4 | fp16 B=4 |
|-------------|----------|----------|
| U-Net (f64) | ~2.8 GB  | ~1.4 GB  |
| U-Net (f32) | ~1.8 GB  | ~0.9 GB  |
| StarGAN G   | ~1.2 GB  | ~0.6 GB  |
| StarGAN D   | ~0.6 GB  | ~0.3 GB  |

All training scripts default to fp16 mixed precision on CUDA.