"""Texture generation for the IARONE watch: dial, croc leather, caseback movement, date disc."""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

GOLD = (232, 176, 148)          # rose gold printed/applied colour on the dial (sRGB)
DIAL_BG = (10, 9, 9)


# ---------------------------------------------------------------- helpers
def _gold_layer(mask, base=GOLD):
    """mask float 0..1 -> RGB gold image with the mask as coverage."""
    out = np.zeros(mask.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = base[c]
    return out, mask


def _paste(img, layer, mask):
    m = mask[..., None]
    return img * (1 - m) + layer * m


# ---------------------------------------------------------------- dial
def dial_texture(S=2048, R_dial=17.0, cfg=None):
    """S x S albedo where the dial disc of radius R_dial maps to the inscribed circle."""
    c = dict(
        minute_r0=16.42, minute_r1=17.12, minute_w=0.115,
        hour_tick_r0=16.20, hour_tick_r1=17.12, hour_tick_w=0.0,
        logo_cy=7.05, logo_w=5.90,
        word_cy=-8.20, word_h=1.46, word_track=0.28,
    )
    if cfg:
        c.update(cfg)
    px_per_mm = (S / 2.0) / R_dial

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    X = (xx - S / 2 + 0.5) / px_per_mm
    Y = -(yy - S / 2 + 0.5) / px_per_mm
    Rr = np.sqrt(X * X + Y * Y)
    Th = np.arctan2(X, Y)                      # 0 at 12, growing clockwise

    # --- base: near-black with a faint vertical grain and a soft sunray
    img = np.zeros((S, S, 3), np.float32)
    for i in range(3):
        img[..., i] = DIAL_BG[i]
    rng = np.random.default_rng(7)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (1, 121), 0)
    grain /= max(grain.std(), 1e-6)
    sun = np.sin(Th * 220.0) * 0.5 + np.sin(Th * 97.0 + 1.3) * 0.5
    modul = 1.0 + 0.055 * grain + 0.030 * sun
    img *= modul[..., None]
    img += rng.normal(0, 0.9, (S, S, 1))

    # --- minute track (60 fine strokes) + slightly heavier strokes on the hours
    ang = (Th % (2 * np.pi))
    step = 2 * np.pi / 60
    d_min = np.abs(((ang + step / 2) % step) - step / 2) * Rr
    ring = (Rr > c["minute_r0"]) & (Rr < c["minute_r1"])
    m_min = np.clip((c["minute_w"] / 2 - d_min) * px_per_mm + 0.5, 0, 1) * ring

    step5 = 2 * np.pi / 12
    d_h = np.abs(((ang + step5 / 2) % step5) - step5 / 2) * Rr
    ringh = (Rr > c["hour_tick_r0"]) & (Rr < c["hour_tick_r1"])
    m_h = np.clip((c["hour_tick_w"] / 2 - d_h) * px_per_mm + 0.5, 0, 1) * ringh

    m_track = np.clip(m_min + m_h, 0, 1)
    layer, _ = _gold_layer(m_track)
    img = _paste(img, layer, m_track * 0.92)

    # --- logo (photo-derived, straight-line snapped)
    logo = np.asarray(Image.open("logo_clean.png").convert("L")).astype(np.float32) / 255.0
    lw = int(round(c["logo_w"] * px_per_mm))
    lh = int(round(lw * logo.shape[0] / logo.shape[1]))
    logo = cv2.resize(logo, (lw, lh), interpolation=cv2.INTER_AREA)
    lm = np.zeros((S, S), np.float32)
    cx = int(round(S / 2))
    cy = int(round(S / 2 - c["logo_cy"] * px_per_mm))
    x0, y0 = cx - lw // 2, cy - lh // 2
    lm[y0:y0 + lh, x0:x0 + lw] = logo
    layer, _ = _gold_layer(lm)
    img = _paste(img, layer, lm)

    # --- IARONE wordmark
    fs = int(round(c["word_h"] * px_per_mm * 1.36))
    font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", fs)
    tmp = Image.new("L", (S, fs * 2), 0)
    dr = ImageDraw.Draw(tmp)
    track = c["word_track"] * px_per_mm
    widths = [dr.textlength(ch, font=font) for ch in "IARONE"]
    total = sum(widths) + track * 5
    x = (S - total) / 2
    for ch, w in zip("IARONE", widths):
        dr.text((x, fs * 0.35), ch, fill=255, font=font)
        x += w + track
    wm = np.asarray(tmp).astype(np.float32) / 255.0
    ys, xs = np.nonzero(wm > 0.05)
    band = wm[ys.min():ys.max() + 1]
    tm = np.zeros((S, S), np.float32)
    ty = int(round(S / 2 - c["word_cy"] * px_per_mm - band.shape[0] / 2))
    tm[ty:ty + band.shape[0]] = band
    layer, _ = _gold_layer(tm, (226, 168, 141))
    img = _paste(img, layer, tm * 0.97)

    # outside the dial disc: black, so any UV slop stays invisible
    img[Rr > R_dial * 1.002] = 4.0

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def dial_orm(S=512):
    """occlusion / roughness / metallic for the dial: matte dielectric everywhere."""
    a = np.zeros((S, S, 3), np.uint8)
    a[..., 0] = 255          # AO
    a[..., 1] = 108          # roughness ~0.42
    a[..., 2] = 0            # metallic 0
    return Image.fromarray(a)


# ---------------------------------------------------------------- croc leather
def croc(S=1024, rows=(5, 4, 3, 4, 5), cols=7, seed=3):
    """Tileable alligator-embossed leather: albedo + normal + ORM.

    Scale rows run along the strap; the middle rows carry the big scales and the
    rows nearest the edges are progressively smaller, the way a real croc strap
    is cut from the belly.  v = across the strap, u = along it.
    """
    rng = np.random.default_rng(seed)
    nrow = len(rows)
    pts = []
    for j, per_row in enumerate(rows):
        n_here = int(round(cols * per_row / min(rows)))
        for i in range(n_here):
            u = (i + 0.5 + 0.5 * (j % 2)) / n_here + rng.normal(0, 0.16) / n_here
            v = (j + 0.5) / nrow + rng.normal(0, 0.10) / nrow
            pts.append((u % 1.0, v))
    pts = np.array(pts)
    P = np.vstack([pts + np.array([du, dv]) for du in (-1, 0, 1) for dv in (-1, 0, 1)])

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    U = xx / S
    V = yy / S
    d = np.full((S, S), 1e9, np.float32)
    d2 = np.full((S, S), 1e9, np.float32)
    for (pu, pv) in P:
        dd = np.sqrt((U - pu) ** 2 + (V - pv) ** 2)
        np.minimum(d2, np.maximum(d, dd), out=d2)
        np.minimum(d, dd, out=d)
    edge = d2 - d                                   # 0 exactly on a cell boundary

    groove = 0.55 / (nrow * min(rows) / min(rows)) / nrow   # groove half-width in uv
    t = np.clip(edge / max(groove, 1e-6), 0, 1)
    plateau = t * t * (3 - 2 * t)                   # smoothstep: flat crowns, rounded shoulders
    dome = cv2.GaussianBlur(plateau, (0, 0), S / (nrow * 6.0))
    height = np.clip(0.80 * plateau + 0.20 * dome, 0, 1)

    n = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 1.6)
    n /= max(n.std(), 1e-6)
    height = np.clip(height - 0.012 * np.abs(n), 0, 1)
    height = cv2.GaussianBlur(height, (0, 0), 2.6)

    base = np.array([25.0, 23.5, 23.0])
    alb = base[None, None, :] * (0.30 + 0.85 * height[..., None])
    alb += (1.8 * n)[..., None]
    albedo = Image.fromarray(np.clip(alb, 0, 255).astype(np.uint8))

    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=5) / 8.0
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=5) / 8.0
    st = 15.0
    nx, ny, nz = -gx * st, gy * st, np.ones_like(gx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = Image.fromarray((((np.stack([nx / ln, ny / ln, nz / ln], -1)) * 0.5 + 0.5) * 255).astype(np.uint8))

    orm = np.zeros((S, S, 3), np.uint8)
    orm[..., 0] = np.clip(160 + 95 * height, 0, 255).astype(np.uint8)
    orm[..., 1] = np.clip(205 - 70 * height, 0, 255).astype(np.uint8)
    orm[..., 2] = 0
    return albedo, normal, Image.fromarray(orm)


def croc_with_stitch(S=1024, v_rows=(0.135, 0.865), pitch=0.055, seed=3):
    """Croc albedo/normal/ORM plus a saddle stitch running along both edges.

    v maps once across the strap width, so the stitch rows sit at fixed v.
    """
    albedo, normal, orm = croc(S, seed=seed)
    A = np.asarray(albedo).astype(np.float32)
    N = np.asarray(normal).astype(np.float32)
    O = np.asarray(orm).astype(np.float32)
    xx = np.arange(S) / S
    for v in v_rows:
        y = int(v * S)
        half = max(2, int(S * 0.0055))
        ph = (xx % pitch) / pitch
        on = ((ph > 0.10) & (ph < 0.74)).astype(np.float32)
        on = cv2.GaussianBlur(on[None, :], (0, 0), S * 0.0016)[0]
        for dy in range(-half, half + 1):
            yy = np.clip(y + dy, 0, S - 1)
            fall = np.exp(-(dy / (half * 0.72)) ** 2)
            k = (on * fall)[:, None]
            A[yy] = A[yy] * (1 - k) + np.array([96.0, 88.0, 82.0]) * k
            O[yy, :, 1] = O[yy, :, 1] * (1 - k[:, 0]) + 200 * k[:, 0]
            # a shallow groove either side of the thread
            N[yy, :, 1] = np.clip(N[yy, :, 1] + 34 * np.sign(dy) * (on * fall), 0, 255)
    return (Image.fromarray(np.clip(A, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(N, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(O, 0, 255).astype(np.uint8)))


# ---------------------------------------------------------------- suede underside
def suede(S=512, seed=11):
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 0.9)
    n /= max(n.std(), 1e-6)
    fibre = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (7, 1), 0)
    fibre /= max(fibre.std(), 1e-6)
    base = np.array([132.0, 107.0, 90.0])
    a = base[None, None, :] * (1.0 + 0.055 * n + 0.035 * fibre)[..., None]
    # slightly darker along the two long edges, the way a cut edge burnishes
    v = np.linspace(0, 1, S)[:, None]
    edge = 1.0 - 0.20 * np.exp(-((v - 0.02) / 0.05) ** 2) - 0.20 * np.exp(-((v - 0.98) / 0.05) ** 2)
    a *= edge[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- crown cap
def crown_cap(S=256):
    """Rose-gold crown face with the monogram engraved into it."""
    logo = np.asarray(Image.open("logo_clean.png").convert("L")).astype(np.float32) / 255.0
    h = int(S * 0.52)
    w = int(round(h * logo.shape[1] / logo.shape[0]))
    lg = cv2.resize(logo, (w, h), interpolation=cv2.INTER_AREA)
    m = np.zeros((S, S), np.float32)
    y0, x0 = (S - h) // 2, (S - w) // 2
    m[y0:y0 + h, x0:x0 + w] = lg
    img = np.full((S, S, 3), 244.0, np.float32)
    img *= (1 - 0.42 * m)[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- caseback movement
def movement(S=1024):
    ref = np.asarray(Image.open("ref.png").convert("RGB"))
    # exhibition window inside the caseback panel
    cx, cy, r = 776, 858, 108
    crop = ref[cy - r:cy + r, cx - r:cx + r].astype(np.float32)
    crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(crop, (0, 0), 2.2)
    crop = np.clip(crop + 0.55 * (crop - blur), 0, 255)
    crop = np.clip((crop - 8) * 1.42, 0, 255)          # lift the window out of shadow
    crop = crop[:, ::-1]        # the disc is seen from behind, so pre-mirror it
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    rr = np.sqrt((xx - S / 2) ** 2 + (yy - S / 2) ** 2) / (S / 2)
    crop[rr > 0.995] = 14
    return Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- date disc
def date_disc(S=256, text="28"):
    im = Image.new("RGB", (S, S), (243, 241, 237))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("C:/Windows/Fonts/times.ttf", int(S * 0.62))
    w = d.textlength(text, font=f)
    bb = f.getbbox(text)
    d.text(((S - w) / 2, (S - (bb[3] - bb[1])) / 2 - bb[1]), text, fill=(24, 22, 20), font=f)
    return im


if __name__ == "__main__":
    dial_texture().save("t_dial.png")
    dial_orm().save("t_dial_orm.png")
    a, n, o = croc()
    a.save("t_croc.png"); n.save("t_croc_n.png"); o.save("t_croc_orm.png")
    movement().save("t_movement.png")
    date_disc().save("t_date.png")
    print("textures written")
