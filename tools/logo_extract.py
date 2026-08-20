import numpy as np
import cv2
from PIL import Image

ref = np.asarray(Image.open("ref.png").convert("RGB")).astype(np.float32)

# logo sits in the macro panel (panel origin x=575)
crop = ref[168:280, 755:865]
lum = crop.mean(2)
gold = (crop[:, :, 0] - crop[:, :, 2])          # rose gold is strongly R>B, dial is neutral-dark
score = 0.55 * (lum / 255.0) + 0.45 * (gold / 90.0)
score = np.clip((score - score.min()) / (score.max() - score.min()), 0, 1)

K = 10
big = cv2.resize(score, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC)
big = cv2.GaussianBlur(big, (0, 0), K * 0.32)
thr = 0.42
mask = (big > thr).astype(np.uint8)

# clean: drop specks, close hairline gaps, then smooth the contour
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((K // 2, K // 2), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
keep = np.zeros_like(mask)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 40 * K:
        keep[lab == i] = 1
mask = keep
mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), K * 0.22)
mask = (mask > 0.5).astype(np.uint8)

ys, xs = np.nonzero(mask)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
mask = mask[y0:y1, x0:x1]
print("logo mask", mask.shape, "aspect w/h = %.3f" % (mask.shape[1] / mask.shape[0]))

# save a soft-edged alpha at a workable size
h = 900
w = int(round(h * mask.shape[1] / mask.shape[0]))
alpha = cv2.resize(mask.astype(np.float32) * 255, (w, h), interpolation=cv2.INTER_AREA)
Image.fromarray(alpha.astype(np.uint8)).save("logo_mask.png")

prev = np.zeros((h, w, 3), np.uint8)
prev[..., 0] = alpha.astype(np.uint8)
prev[..., 1] = (alpha * 0.72).astype(np.uint8)
prev[..., 2] = (alpha * 0.60).astype(np.uint8)
Image.fromarray(prev).save("logo_preview.png")

src = cv2.resize(np.clip(crop, 0, 255).astype(np.uint8), None, fx=8, fy=8, interpolation=cv2.INTER_LANCZOS4)
Image.fromarray(src).save("logo_src_big.png")
print("wrote logo_mask.png")
