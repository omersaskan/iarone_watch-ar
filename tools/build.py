"""IARONE dress watch - parametric rebuild from the reference product shots.

Build frame (millimetres):  dial normal = +Z, 12 o-clock = +Y, crown = +X,
wrist axis = X, strap loop hangs to -Z.  Exported rotated to Y-up in metres.
"""
import numpy as np
from PIL import Image

from geom import (Mesh, merge, revolve, disc, prism, rect_poly, sweep, fan, frames,
                  spline_resample, rotX, rotY, rotZ, trans)
from glbout import GLB
import tex

# ------------------------------------------------------------------ params
P = dict(
    R_case=20.0,
    R_dial=17.55,
    R_crystal=18.10,
    z_dial=0.0,

    marker_out=16.35, marker_in=11.60, marker_w=1.06, marker_h=0.36,
    marker3_in=14.55, twelve_gap=1.45, twelve_w=0.92,

    date_r_in=9.95, date_r_out=14.10, date_half_ang=np.deg2rad(7.7),
    date_depth=0.55,

    hour_len=10.85, hour_w=1.12, hour_h=0.30, hour_tail=1.95,
    min_len=15.65, min_w=0.88, min_h=0.26, min_tail=2.05,
    sec_len=16.30, sec_w=0.21, sec_h=0.14, sec_tail=4.00,
    t_hour=10, t_min=9.5, t_sec=37,

    lug_w=3.20, strap_w=20.0, strap_th=2.60,
    wear_mode=False,
    wrist=dict(rx=28.5, rz=21.5, gap=1.35, skin_top=-5.30, drift=16.4, ease=17.0,
               lift=1.65, y_start=20.4, ends=0.75),
    loop_a=27.2, loop_bot=-44.0,
    NSEG=192,
)

GOLD = (0.920, 0.716, 0.605)          # rose gold reflectance, sRGB
GOLD_D = (0.900, 0.693, 0.585)        # slightly deeper for brushed surfaces


# ------------------------------------------------------------------ helpers
def lathe(profile, nseg=None, caps=None):
    return revolve(profile, nseg or P["NSEG"])


def withz(poly, z=0.0):
    poly = np.asarray(poly, float)
    return np.c_[poly, np.full(len(poly), z)]


def flipped(m):
    m.F = m.F[:, ::-1]
    return m


def baton(L, W, H, top=0.34, taper=1.0, tip=False, nseg=None):
    """A polished applied index / hand: flat base, faceted roof running lengthwise.
    Length along +Y from 0..L, width along X, height 0..H."""
    ys = np.linspace(0, L, 14 if tip else 3)
    secs = []
    for y in ys:
        t = y / max(L, 1e-9)
        w = W * (1 - (1 - taper) * t) / 2
        if tip:
            # narrow to a point over the last 18 %
            k = np.clip((t - 0.82) / 0.18, 0, 1)
            w *= (1 - k) ** 0.65
        w = max(w, 1e-4)
        h = H * (1 if not tip else (1 - 0.55 * np.clip((t - 0.82) / 0.18, 0, 1)))
        secs.append(np.array([[-w, 0], [w, 0], [top * w, h], [-top * w, h]]))
    path = np.stack([np.zeros_like(ys), ys, np.zeros_like(ys)], -1)
    return sweep(path, secs, cap_ends=True)


def dial_disc():
    """Dial plate with the date aperture cut out, plus the recess walls."""
    R = P["R_dial"]
    ri, ro = P["date_r_in"], P["date_r_out"]
    ha = P["date_half_ang"]
    rings = np.array([0, 2.5, 5.0, 7.4, 9.95, 10.98, 12.02, 13.06, 14.10, 15.5, 16.6, R])
    nse = 184
    a0, a1 = ha, 2 * np.pi - ha
    angs = np.linspace(a0, a1, nse + 1)

    def grid(rr, aa):
        nr, na = len(rr), len(aa)
        RR, AA = np.meshgrid(rr, aa, indexing="ij")
        X = RR * np.cos(AA)
        Y = RR * np.sin(AA)
        V = np.stack([X, Y, np.full_like(X, P["z_dial"])], -1).reshape(-1, 3)
        F = []
        for i in range(nr - 1):
            for j in range(na - 1):
                a = i * na + j
                b = a + 1
                c = (i + 1) * na + j + 1
                d = (i + 1) * na + j
                F.append([a, c, b])
                F.append([a, d, c])
        s = 0.5 / R
        UV = np.stack([0.5 + V[:, 0] * s, 0.5 - V[:, 1] * s], -1)
        return Mesh(V, np.array(F, np.int32), UV)

    main = grid(rings, angs)
    # the wedge that holds the date window: fill only inside ri and outside ro
    wa = np.linspace(-ha, ha, 9)
    inner = grid(np.array([0, 2.5, 5.0, 7.4, ri]), wa)
    outer = grid(np.array([ro, 15.5, 16.6, R]), wa)

    # recess walls around the aperture
    walls = []
    zt, zb = P["z_dial"], P["z_dial"] - P["date_depth"]
    for r in (ri, ro):
        aa = wa
        pts_t = np.stack([r * np.cos(aa), r * np.sin(aa), np.full_like(aa, zt)], -1)
        pts_b = pts_t.copy()
        pts_b[:, 2] = zb
        V = np.vstack([pts_t, pts_b])
        n = len(aa)
        F = []
        for j in range(n - 1):
            if r == ri:
                F.append([j, n + j, n + j + 1])
                F.append([j, n + j + 1, j + 1])
            else:
                F.append([j, j + 1, n + j + 1])
                F.append([j, n + j + 1, n + j])
        walls.append(Mesh(V, np.array(F, np.int32)))
    for sgn in (-1, 1):
        a = sgn * ha
        rr = np.array([ri, ro])
        pts_t = np.stack([rr * np.cos(a), rr * np.sin(a), np.full_like(rr, zt)], -1)
        pts_b = pts_t.copy()
        pts_b[:, 2] = zb
        V = np.vstack([pts_t, pts_b])
        if sgn > 0:
            F = [[0, 2, 3], [0, 3, 1]]
        else:
            F = [[0, 3, 2], [0, 1, 3]]
        walls.append(Mesh(V, np.array(F, np.int32)))

    return merge([main, inner, outer]), merge(walls)


def date_plate():
    ri, ro, ha = P["date_r_in"], P["date_r_out"], P["date_half_ang"]
    z = P["z_dial"] - P["date_depth"]
    aa = np.linspace(-ha, ha, 9)
    rr = np.array([ri, ro])
    RR, AA = np.meshgrid(rr, aa, indexing="ij")
    V = np.stack([RR * np.cos(AA), RR * np.sin(AA), np.full_like(RR, z)], -1).reshape(-1, 3)
    na = len(aa)
    F = []
    for j in range(na - 1):
        F += [[j, na + j + 1, j + 1], [j, na + j, na + j + 1]]
    u = (V[:, 0] - ri * np.cos(ha)) / (ro - ri * np.cos(ha))
    tang = np.arctan2(V[:, 1], V[:, 0])
    v = 0.5 - tang / (2 * ha)
    UV = np.stack([np.clip(u, 0, 1), np.clip(v, 0, 1)], -1)
    return Mesh(V, np.array(F, np.int32), UV)


def date_frame():
    """Polished rose-gold moulding around the date aperture."""
    ri, ro, ha = P["date_r_in"], P["date_r_out"], P["date_half_ang"]
    rc = (ri + ro) / 2
    hw = (ro - ri) / 2 + 0.09
    hh = rc * np.tan(ha) + 0.09
    path = withz(rect_poly(2 * hw, 2 * hh, cx=rc, r=0.22, nc=3), P["z_dial"] + 0.03)
    sec = np.array([[-0.19, -0.10], [0.19, -0.10], [0.19, 0.08], [0.06, 0.20], [-0.06, 0.20], [-0.19, 0.08]])
    return sweep(path, sec, closed=True)


def hands():
    out = []
    ang = {
        "h": np.deg2rad((P["t_hour"] % 12 + P["t_min"] / 60) / 12 * 360),
        "m": np.deg2rad(P["t_min"] / 60 * 360),
        "s": np.deg2rad(P["t_sec"] / 60 * 360),
    }
    # clock angle -> direction (sin, cos); build batons along +Y then rotate by -a
    def place(m, a, z):
        return m.transform(trans(0, 0, z) @ rotZ(-a))

    h = baton(P["hour_len"], P["hour_w"], P["hour_h"], top=0.30, taper=0.72, tip=True)
    ht = baton(P["hour_tail"], P["hour_w"] * 0.80, P["hour_h"], top=0.30, taper=0.9)
    ht.transform(rotZ(np.pi))
    out.append(place(merge([h, ht]), ang["h"], 0.96))

    m_ = baton(P["min_len"], P["min_w"], P["min_h"], top=0.30, taper=0.62, tip=True)
    mt = baton(P["min_tail"], P["min_w"] * 0.80, P["min_h"], top=0.30, taper=0.9)
    mt.transform(rotZ(np.pi))
    out.append(place(merge([m_, mt]), ang["m"], 1.34))

    s_ = baton(P["sec_len"], P["sec_w"], P["sec_h"], top=0.5, taper=0.75, tip=True)
    st = baton(P["sec_tail"], P["sec_w"], P["sec_h"], top=0.5, taper=1.0)
    st.transform(rotZ(np.pi))
    cw = revolve([(0.0, 0.0), (0.44, 0.0), (0.44, 0.20), (0.0, 0.20)], 24)
    cw.transform(trans(0, -3.35, -0.03))
    out.append(place(merge([s_, st, cw]), ang["s"], 1.70))

    hub = revolve([(0.00, 0.86), (0.98, 0.86), (1.02, 1.00), (0.92, 1.24),
                   (0.60, 1.34), (0.62, 1.72), (0.44, 1.96), (0.00, 2.02)], 48)
    out.append(hub)
    return merge(out)


def case_and_bezel():
    # outer shell, written bottom -> top so the normals face outwards
    prof = [
        (11.95, -4.94), (13.60, -5.10), (15.40, -5.32), (16.90, -5.50),
        (17.95, -5.58), (18.72, -5.54), (19.22, -5.30), (19.44, -4.86),
        (19.50, -4.10), (19.48, -3.00), (19.44, -2.30), (19.68, -2.00),
        (19.60, -1.10), (19.52, 0.20), (19.62, 1.20), (19.86, 1.98),
        (20.00, 2.34), (19.92, 2.58), (19.66, 2.84), (19.15, 3.02),
        (18.55, 2.90), (18.15, 2.52),
    ]
    case = lathe(prof)
    # rehaut: written top -> bottom so it faces inwards
    reh = [(18.15, 2.52), (18.00, 1.86), (17.70, 0.55), (17.59, 0.06)]
    rehaut = lathe(reh)
    return case, rehaut


def crystal():
    prof = [(0.0, 2.48), (17.78, 2.48), (18.10, 2.60), (18.10, 2.82),
            (17.66, 2.95), (0.0, 3.01)]
    body = revolve(prof, 128)
    bot = disc(17.78, 2.48, 128, up=False)
    return merge([body, bot])


def caseback_window():
    # slightly domed sapphire disc with a planar UV so the movement image maps flat
    d = disc(11.55, -4.98, 128, up=False)
    r = np.hypot(d.V[:, 0], d.V[:, 1])
    d.V[:, 2] += -0.12 * (1 - (r / 11.55) ** 2)
    rim = revolve([(11.55, -4.98), (12.02, -4.92)], 96)
    rim.F = rim.F[:, ::-1]
    return d, rim


def crown():
    """Fluted crown at 3 o-clock with the monogram on its face."""
    prof = [(0.00, 0.00), (1.75, 0.00), (2.28, 0.38), (2.52, 0.80),
            (2.54, 2.18), (2.30, 2.58), (2.06, 2.82), (1.68, 2.98), (0.00, 3.02)]
    m = revolve(prof, 96)
    ang = np.arctan2(m.V[:, 1], m.V[:, 0])
    r = np.hypot(m.V[:, 0], m.V[:, 1])
    band = np.clip((m.V[:, 2] - 0.72) / 0.12, 0, 1) * np.clip((2.24 - m.V[:, 2]) / 0.12, 0, 1)
    flute = 1.0 + 0.062 * band * np.cos(28 * ang)
    m.V[:, 0] = r * flute * np.cos(ang)
    m.V[:, 1] = r * flute * np.sin(ang)
    cap = disc(1.75, 3.03, 48)
    M = trans(18.85, 0, -0.20) @ rotY(np.pi / 2)
    # lay the crown on its side at 3 o-clock
    return m.transform(M), cap.transform(M)


def pusher():
    prof = [(0.00, 0.00), (1.18, 0.00), (1.18, 1.02), (0.94, 1.26), (0.00, 1.32)]
    m = merge([revolve(prof, 40), disc(0.94, 1.26, 40)])
    a = np.deg2rad(58.0)                      # just above the crown, near 2 o-clock
    d = np.array([np.sin(a), np.cos(a), 0.0])
    M = trans(*(d * 18.9 + np.array([0, 0, 0.30])))
    axis = np.arctan2(d[1], d[0])
    return m.transform(M @ rotZ(axis) @ rotY(np.pi / 2))


def lugs():
    out = []
    W = P["lug_w"]
    for sy in (1, -1):
        for sx in (1, -1):
            ctrl = np.array([
                [sx * 11.55, sy * 10.6, -1.40],
                [sx * 11.55, sy * 15.8, -1.62],
                [sx * 11.60, sy * 19.2, -2.30],
                [sx * 11.60, sy * 21.2, -3.35],
                [sx * 11.52, sy * 21.8, -4.25],
            ])
            path = spline_resample(ctrl, 30)
            secs = []
            n = len(path)
            for i in range(n):
                t = i / (n - 1)
                w = W * (1.00 - 0.34 * t ** 1.5)
                h = 5.0 * (1 - 0.50 * t ** 1.20)
                secs.append(rect_poly(w, h, r=min(w, h) * 0.36, nc=3))
            out.append(sweep(path, secs, fixed_right=np.array([1.0, 0, 0]), cap_ends=True))
    return merge(out)


def spring_bars():
    out = []
    for sy in (1, -1):
        m = revolve([(0.0, -10.2), (0.48, -10.2), (0.48, 10.2), (0.0, 10.2)], 20)
        out.append(m.transform(trans(0, sy * 20.6, -2.10) @ rotY(np.pi / 2)))
    return merge(out)


# ------------------------------------------------------------------ strap
def loop_path_wrist():
    """Saddle loop that really encircles a forearm.

    A planar loop cannot both attach at the 12 and 6 lugs (which are separated
    along the forearm) and encircle the arm - the two are geometrically
    incompatible.  A real strap resolves this by drifting along the forearm as it
    wraps: it leaves the 12 lug, spirals half a turn to the underside while
    drifting to the middle, then continues to the 6 lug.  That is what this path
    does, so the AR variant sits on a wrist instead of passing through it.
    """
    W = P["wrist"]
    Rx = W["rx"] + W["gap"]
    Rz = W["rz"] + W["gap"]
    zc = W["skin_top"] - W["rz"]
    A = W["drift"]
    th = np.linspace(0, 2 * np.pi, 300)
    x = Rx * np.sin(th)
    z = zc + Rz * np.cos(th)
    y = A * np.cos(th / 2)
    t0, t1 = th, 2 * np.pi - th
    # Anchor the two ends ON the spring bars.  The wrap-drift amplitude A shapes
    # how the strap spirals round the arm; it must not also decide where the strap
    # begins, or a gap opens between the lugs and the leather.
    sg = W["ends"]
    boost = (W["y_start"] - A) * (np.exp(-(t0 / sg) ** 2) - np.exp(-(t1 / sg) ** 2))
    y = y + boost
    # leave the lugs heading along the strap rather than straight across the wrist
    y = y - W["ease"] * t0 * np.exp(-t0 / 0.42) + W["ease"] * t1 * np.exp(-t1 / 0.42)
    # and lift the two ends up to the spring bars
    z = z + W["lift"] * (np.exp(-(t0 / 0.55) ** 2) + np.exp(-(t1 / 0.55) ** 2))
    return np.stack([x, y, z], -1)


def wrist_normal(pts):
    """Outward normal of the forearm ellipse at each path point (arm axis = Y)."""
    W = P["wrist"]
    zc = W["skin_top"] - W["rz"]
    n = np.zeros_like(pts)
    n[:, 0] = pts[:, 0] / (W["rx"] ** 2)
    n[:, 2] = (pts[:, 2] - zc) / (W["rz"] ** 2)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    bad = ln[:, 0] < 1e-9
    n[bad] = np.array([0.0, 0.0, 1.0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def loop_path():
    """Closed-ish wrist loop in the YZ plane, from the 12 lug round to the 6 lug."""
    a = P["loop_a"]
    zb = P["loop_bot"]
    ctrl = np.array([
        [0,  18.4,  -2.45],
        [0,  25.2,  -6.00],
        [0,  27.0, -12.50],
        [0,  27.2, -19.50],
        [0,  25.8, -27.00],
        [0,  22.0, -34.00],
        [0,  15.5, -40.00],
        [0,   6.5, -43.30],
        [0,  -3.0, -44.00],
        [0, -12.5, -41.80],
        [0, -20.0, -36.60],
        [0, -25.0, -29.50],
        [0, -27.2, -21.50],
        [0, -27.0, -13.00],
        [0, -25.2,  -6.00],
        [0, -18.4,  -2.45],
    ])
    return spline_resample(ctrl, 220)


def strap_section(w, th):
    """Cross-section points and their v coordinates.  Index 0..n_top-1 = top+sides."""
    hw = w / 2
    r = min(0.55, th * 0.42)
    top, vt = [], []
    # left wall (bottom-left -> top-left), top surface, right wall
    top += [(-hw + 0.05, -th * 0.42)]; vt += [0.0]
    top += [(-hw, -th * 0.12)]; vt += [0.030]
    top += [(-hw + 0.02, th * 0.30)]; vt += [0.062]
    top += [(-hw + 0.28, th * 0.50)]; vt += [0.085]
    n_mid = 9
    for i in range(n_mid):
        t = i / (n_mid - 1)
        x = (-hw + 0.28) + t * (2 * hw - 0.56)
        z = th * 0.50 + 0.10 * np.sin(np.pi * t)
        top.append((x, z)); vt.append(0.085 + t * (0.915 - 0.085))
    top += [(hw - 0.02, th * 0.30)]; vt += [0.938]
    top += [(hw, -th * 0.12)]; vt += [0.970]
    top += [(hw - 0.05, -th * 0.42)]; vt += [1.0]
    # underside, right -> left
    bot, vb = [], []
    n_b = 5
    for i in range(n_b):
        t = i / (n_b - 1)
        x = (hw - 0.05) - t * (2 * hw - 0.10)
        bot.append((x, -th * 0.50)); vb.append(1.0 - t)
    return np.array(top + bot), np.array(list(vt) + list(vb)), len(top)


def straps():
    path = loop_path_wrist() if P["wear_mode"] else loop_path()
    n = len(path)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    i_buck = int(n * (0.545 if P["wear_mode"] else 0.600))
    i_tip = int(n * (0.660 if P["wear_mode"] else 0.720))
    ctr = (np.array([0.0, 0.0, P["wrist"]["skin_top"] - P["wrist"]["rz"]])
           if P["wear_mode"] else np.array([0.0, 0.0, (P["loop_bot"] - 1.6) / 2]))

    def band(idx, w0, w1, th0, th1, taper_end=False, lift=None):
        pp = path[idx].copy()
        m = len(pp)
        if lift is not None:
            if P["wear_mode"]:
                out = wrist_normal(pp)
            else:
                out = pp - ctr
                out[:, 0] = 0
                out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
            t = np.clip((np.arange(m) / (m - 1) - lift[0]) / max(lift[1] - lift[0], 1e-6), 0, 1)
            pp += out * (t * t * (3 - 2 * t))[:, None] * lift[2]
        secs = []
        for i in range(m):
            t = i / max(m - 1, 1)
            w = w0 + (w1 - w0) * t
            th = th0 + (th1 - th0) * t
            if taper_end:
                k = np.clip((t - 0.88) / 0.12, 0, 1)
                w *= 1 - 0.26 * k
                th *= 1 - 0.30 * k
            s, v, nt = strap_section(w, th)
            secs.append(s)
        vv, ntop = v, nt
        uu = np.abs(arc[idx] - arc[idx][0]) / 34.0
        XR = np.array([-1.0, 0, 0])
        fu = wrist_normal(pp) if P["wear_mode"] else None
        fr = None if P["wear_mode"] else XR
        top = sweep(pp, [s[:ntop] for s in secs], fixed_right=fr, fixed_up=fu,
                    uv_u=uu, sec_v=vv[:ntop], section_closed=False)
        bot = sweep(pp, [np.vstack([s[ntop - 1:], s[:1]]) for s in secs],
                    fixed_right=fr, fixed_up=fu, uv_u=uu,
                    sec_v=np.r_[vv[ntop - 1:], vv[:1]], section_closed=False)
        # the strip winding runs the wrong way for an outward normal
        top.F = top.F[:, ::-1]
        bot.F = bot.F[:, ::-1]
        # close the two open ends of the tube
        _, rt, upv = frames(pp, False, fixed_right=fr, fixed_up=fu)
        caps = []
        for i, rev in ((0, False), (m - 1, True)):
            s = np.asarray(secs[i], float)
            ring = pp[i] + s[:, 0:1] * rt[i] + s[:, 1:2] * upv[i]
            caps.append(fan(ring, reverse=rev))
        return top, bot, merge(caps), pp

    t12, b12, c12, p12 = band(np.arange(0, i_tip), P["strap_w"], 17.4,
                              P["strap_th"], 2.10, taper_end=True, lift=(0.58, 0.74, 2.85))
    t6, b6, c6, p6 = band(np.arange(i_buck, n), 18.2, P["strap_w"],
                          2.40, P["strap_th"])
    return dict(top=merge([t12, t6]), bot=merge([b12, b6, c12, c6]),
                p12=p12, p6=p6, i_buck=i_buck, path=path, arc=arc)


def buckle(path, i_buck):
    """Rose-gold tang buckle at the end of the 6-side band."""
    i = max(i_buck - 7, 2)
    p = path[i]
    tg = path[i + 2] - path[i - 2]
    tg /= np.linalg.norm(tg)
    nrm = (wrist_normal(path[i:i + 1])[0] if P["wear_mode"]
           else np.cross(np.array([1.0, 0, 0]), tg))
    nrm = nrm - tg * np.dot(nrm, tg)
    nrm /= np.linalg.norm(nrm)
    ax = np.cross(tg, nrm)
    ax /= np.linalg.norm(ax)
    M = np.eye(4)
    M[:3, 0] = ax
    M[:3, 1] = tg
    M[:3, 2] = nrm
    M[:3, 3] = p + nrm * 0.25
    parts = []
    fr = withz(rect_poly(22.0, 15.0, r=3.6, nc=5))
    parts.append(sweep(fr, rect_poly(1.70, 1.95, r=0.55, nc=3), closed=True,
                       up_ref=np.array([0.0, 0, 1.0])))
    bar = revolve([(0, -10.3), (0.60, -10.3), (0.60, 10.3), (0, 10.3)], 20)
    parts.append(bar.transform(trans(0, 4.4, 0) @ rotY(np.pi / 2)))
    tongue = baton(11.6, 1.45, 0.66, top=0.5, taper=0.30, tip=True)
    parts.append(tongue.transform(trans(0, 4.4, -0.40) @ rotZ(np.pi)))
    return merge(parts).transform(M)


def keepers(path, i_buck):
    """Two leather loops holding the free end against the 6-side band."""
    out = []
    for off in (9, 20):
        i = i_buck + off
        if i >= len(path) - 3:
            continue
        p = path[i]
        tg = path[i + 1] - path[i - 1]
        tg /= np.linalg.norm(tg)
        nrm = (wrist_normal(path[i:i + 1])[0] if P["wear_mode"]
               else np.cross(np.array([1.0, 0, 0]), tg))
        nrm = nrm - tg * np.dot(nrm, tg)
        nrm /= np.linalg.norm(nrm)
        ax = np.cross(tg, nrm)
        ax /= np.linalg.norm(ax)
        M = np.eye(4)
        M[:3, 0] = ax
        M[:3, 1] = tg
        M[:3, 2] = nrm
        M[:3, 3] = p + nrm * 1.15
        ring = withz(rect_poly(23.4, 7.6, r=1.9, nc=4))
        m = sweep(ring, rect_poly(1.75, 1.35, r=0.42, nc=3), closed=True,
                  up_ref=np.array([0.0, 0, 1.0]))
        out.append(m.transform(M))
    return merge(out)


# ------------------------------------------------------------------ assemble
def build(out_path="iarone_watch.glb", nseg=None, wear=False, crystal_on=True):
    if nseg:
        P["NSEG"] = nseg
    P["wear_mode"] = wear
    g = GLB("IARONE_watch")

    t_dial = g.add_image(tex.dial_texture(2048, P["R_dial"]), "dial", jpeg_quality=94)
    a, nmap, orm = tex.croc_with_stitch(1024)
    t_croc = g.add_image(a, "croc", jpeg_quality=90)
    t_croc_n = g.add_image(nmap.resize((512, 512), Image.LANCZOS), "croc_n")
    t_croc_o = g.add_image(orm.resize((512, 512), Image.LANCZOS), "croc_orm", jpeg_quality=90)
    t_suede = g.add_image(tex.suede(512), "suede", jpeg_quality=90)
    t_crown = g.add_image(tex.crown_cap(256), "crown_face")
    t_mov = g.add_image(tex.movement(768), "movement", jpeg_quality=92)
    t_date = g.add_image(tex.date_disc(256), "date")

    m_gold = g.add_material("rose_gold_polished", GOLD, metallic=1.0, roughness=0.075)
    m_gold_s = g.add_material("rose_gold_satin", GOLD_D, metallic=1.0, roughness=0.155)
    m_dial = g.add_material("dial", (1, 1, 1), metallic=0.0, roughness=0.44,
                            base_tex=t_dial, specular=0.34)
    m_black = g.add_material("dial_recess", (0.035, 0.033, 0.032), metallic=0.0,
                             roughness=0.70, specular=0.25)
    m_date = g.add_material("date_disc", (0.92, 0.90, 0.88), metallic=0.0, roughness=0.46,
                            base_tex=t_date, specular=0.4)
    m_crystal = g.add_material("crystal", (1, 1, 1), alpha=0.055, metallic=0.0,
                               roughness=0.02, blend=True, ior=1.52, specular=1.0)
    m_leather = g.add_material("croc_leather", (1, 1, 1), metallic=0.0, roughness=1.0,
                               base_tex=t_croc, mr_tex=t_croc_o, occl_tex=t_croc_o,
                               normal_tex=t_croc_n, normal_scale=1.0, specular=0.42)
    m_under = g.add_material("strap_underside", (1, 1, 1), metallic=0.0,
                             roughness=0.97, base_tex=t_suede, specular=0.12)
    m_keeper = g.add_material("keeper_leather", (0.055, 0.052, 0.050), metallic=0.0,
                              roughness=0.62, specular=0.42)
    m_crown_cap = g.add_material("crown_face", (1, 1, 1), metallic=1.0, roughness=0.16,
                                 base_tex=t_crown)
    m_mov = g.add_material("movement", (1, 1, 1), metallic=0.55, roughness=0.30,
                           base_tex=t_mov)

    case, rehaut = case_and_bezel()
    dial, walls = dial_disc()

    # markers
    mk = []
    for i in range(12):
        a_ = np.deg2rad(i * 30.0)
        if i == 0:
            for s in (-1, 1):
                b = baton(P["marker_out"] - P["marker_in"], P["twelve_w"], P["marker_h"])
                b.transform(trans(s * P["twelve_gap"] / 2, P["marker_in"], P["z_dial"]))
                mk.append(b.transform(rotZ(0)) if i == 0 else b)
        else:
            rin = P["marker3_in"] if i == 3 else P["marker_in"]
            b = baton(P["marker_out"] - rin, P["marker_w"], P["marker_h"])
            b.transform(trans(0, rin, P["z_dial"]))
            mk.append(b.transform(rotZ(-a_)))
    markers = merge(mk)

    cbw, cbr = caseback_window()
    crn, crn_cap = crown()
    sp = straps()
    bk = buckle(sp["path"], sp["i_buck"])
    kp = keepers(sp["path"], sp["i_buck"])

    parts = [
        (merge([case, lugs(), spring_bars()]).crease_normals(19), m_gold_s),
        (merge([rehaut, crn, pusher(), markers, hands(), date_frame(), bk]).crease_normals(22), m_gold),
        (crn_cap.smooth_normals(), m_crown_cap),
        (dial.smooth_normals(), m_dial),
        (walls.flat_normals(), m_black),
        (date_plate().flat_normals(), m_date),
        *([(crystal().crease_normals(30), m_crystal)] if crystal_on else []),
        (cbw.smooth_normals(), m_mov),
        (cbr.crease_normals(22), m_gold),
        (sp["top"].smooth_normals(), m_leather),
        (sp["bot"].crease_normals(40), m_under),
        (kp.crease_normals(40), m_keeper),
    ]

    # mm -> m, and Z-up build frame -> Y-up (dial +Y, 12 at -Z, crown +X)
    M = rotX(-np.pi / 2)
    M[:3, :3] *= 0.001
    tri = 0
    for m, mat in parts:
        m.transform(M)
        g.add_prim(m, mat)
        tri += len(m.F)
    n = g.save(out_path)
    print("%s  %.2f MB  %d tris  %d prims" % (out_path, n / 1e6, tri, len(parts)))
    return out_path


if __name__ == "__main__":
    import sys
    if "--wear" in sys.argv:
        build("iarone_watch_ar.glb", wear=True)
    elif "--usdz" in sys.argv:
        # QuickLook renders thin transparent surfaces poorly, so the USDZ source
        # ships without the sapphire crystal
        build("iarone_watch_usdzsrc.glb", crystal_on=False)
    else:
        build()
