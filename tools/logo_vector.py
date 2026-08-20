"""Snap the photo-extracted logo mask to straight-line polygons so it stays crisp at any size."""
import numpy as np
import cv2
from PIL import Image

m = np.asarray(Image.open("logo_mask.png")).astype(np.uint8)
m = (m > 127).astype(np.uint8)
H, W = m.shape

pad = 20
m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
print("contours", len(cnts))

SS = 4                       # supersample factor for the clean render
out = np.zeros((m.shape[0] * SS, m.shape[1] * SS), np.uint8)
eps = 0.0075 * max(W, H)     # straight-line snapping tolerance

order = sorted(range(len(cnts)), key=lambda i: -cv2.contourArea(cnts[i]))
for i in order:
    c = cnts[i]
    if cv2.contourArea(c) < 200:
        continue
    ap = cv2.approxPolyDP(c, eps, True)
    outer = hier[0][i][3] == -1
    cv2.fillPoly(out, [(ap * SS).astype(np.int32)], 255 if outer else 0)
    print("  poly verts=%d area=%.0f %s" % (len(ap), cv2.contourArea(c), "outer" if outer else "hole"))

out = cv2.GaussianBlur(out, (0, 0), SS * 0.45)
out = cv2.resize(out, (m.shape[1] * 2, m.shape[0] * 2), interpolation=cv2.INTER_AREA)

ys, xs = np.nonzero(out > 8)
out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
print("clean logo", out.shape, "aspect %.4f" % (out.shape[1] / out.shape[0]))
Image.fromarray(out).save("logo_clean.png")

prev = np.zeros(out.shape + (3,), np.uint8)
prev[..., 0] = out
prev[..., 1] = (out * 0.72).astype(np.uint8)
prev[..., 2] = (out * 0.60).astype(np.uint8)
Image.fromarray(prev).save("logo_clean_preview.png")
