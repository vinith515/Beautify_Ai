"""Test the new inpainting-based pipeline with a realistic acne face image."""
import sys, os, cv2, numpy as np
sys.path.insert(0, '.')

from inference.pipeline import AcneRemovalPipeline

print('=== Initializing Inpainting Pipeline ===')
pipe = AcneRemovalPipeline(
    unet_checkpoint='checkpoints/unet.pth',
    use_identity_check=True,
    smooth_strength=0.7,
    inpaint_radius=5,
)

# Create a realistic skin-tone face with acne spots
print('\n=== Creating test image with acne spots ===')
img = np.ones((256, 256, 3), dtype=np.uint8)

# Skin tone base with slight variation
for y in range(256):
    for x in range(256):
        img[y, x] = [
            int(140 + 10 * np.sin(x / 30) + 5 * np.cos(y / 25)),  # B
            int(165 + 8 * np.sin(x / 35) + 6 * np.cos(y / 20)),   # G
            int(195 + 12 * np.sin(x / 40) + 8 * np.cos(y / 30)),  # R
        ]

# Add acne spots (reddish, various sizes)
acne_spots = [
    (80, 80, 8, (90, 95, 210)),     # large red spot
    (120, 60, 5, (100, 100, 200)),   # medium
    (180, 100, 6, (85, 90, 215)),    # medium-large
    (160, 200, 4, (95, 105, 195)),   # small
    (70, 150, 7, (80, 85, 220)),     # large angry spot
    (200, 180, 3, (105, 110, 190)),  # small
    (100, 130, 5, (90, 95, 205)),    # medium
    (140, 90, 4, (100, 100, 200)),   # small
]

for cx, cy, r, color in acne_spots:
    cv2.circle(img, (cx, cy), r, color, -1)
    # Add slight inflammation ring around some spots
    if r > 5:
        cv2.circle(img, (cx, cy), r + 3, 
                   (color[0]+15, color[1]+15, min(255, color[2]+10)), 1)

os.makedirs('outputs', exist_ok=True)
cv2.imwrite('outputs/test_acne_input.jpg', img)

# Process
print('\n=== Processing ===')
result = pipe.process(img)
output = result['output']

# Analysis
diff = cv2.absdiff(img, output)
mean_diff = diff.mean()
max_diff = diff.max()

print(f'\nResults:')
print(f'  Spots detected:   {result["spots_detected"]}')
print(f'  Acne coverage:    {result["acne_coverage"]*100:.1f}%')
print(f'  Identity sim:     {result["identity_similarity"]:.4f}')
print(f'  Mean pixel diff:  {mean_diff:.2f}')
print(f'  Max pixel diff:   {max_diff}')
print(f'  Total time:       {result["timing"]["total"]:.3f}s')

for stage, t in result['timing'].items():
    if stage != 'total':
        print(f'    {stage}: {t*1000:.1f}ms')

if result['spots_detected'] > 0 and mean_diff > 0.5:
    print('\n[PASS] Acne spots detected and inpainted!')
else:
    print('\n[WARN] Check detection sensitivity.')

cv2.imwrite('outputs/test_acne_output.jpg', output)
cv2.imwrite('outputs/test_acne_mask.jpg', result['mask'])
cv2.imwrite('outputs/test_acne_diff.jpg', (diff * 5).clip(0, 255).astype(np.uint8))
print('\nSaved: outputs/test_acne_input.jpg, test_acne_output.jpg, test_acne_mask.jpg, test_acne_diff.jpg')
