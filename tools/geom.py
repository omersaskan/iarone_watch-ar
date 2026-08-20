import numpy as np


class Mesh:
    def __init__(self, V, F, UV=None, N=None):
        self.V = np.asarray(V, np.float64)
        self.F = np.asarray(F, np.int32).reshape(-1, 3)
        self.UV = np.zeros((len(self.V), 2)) if UV is None else np.asarray(UV, np.float64)
        self.N = None if N is None else np.asarray(N, np.float64)

    def copy(self):
        return Mesh(self.V.copy(), self.F.copy(), self.UV.copy(),
                    None if self.N is None else self.N.copy())

    def transform(self, M):
        R = M[:3, :3]
        t = M[:3, 3]
        self.V = self.V @ R.T + t
        if self.N is not None:
            NI = np.linalg.inv(R).T
            n = self.N @ NI.T
            self.N = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return self

    def smooth_normals(self):
        n = np.zeros_like(self.V)
        p = self.V[self.F]
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        for k in range(3):
            np.add.at(n, self.F[:, k], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        self.N = n / np.maximum(ln, 1e-12)
        return self

    def flat_normals(self):
        # split every triangle so each face carries its own normal
        p = self.V[self.F]
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        V = p.reshape(-1, 3)
        UV = self.UV[self.F].reshape(-1, 2)
        N = np.repeat(fn, 3, axis=0)
        F = np.arange(len(V)).reshape(-1, 3)
        return Mesh(V, F, UV, N)

    def crease_normals(self, deg=42.0):
        # smooth within faces whose normals agree, hard elsewhere
        m = self.flat_normals()
        key = np.round(self.V[self.F].reshape(-1, 3) / 1e-7).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        acc = np.zeros((inv.max() + 1, 3))
        cnt = np.zeros(inv.max() + 1)
        np.add.at(acc, inv, m.N)
        np.add.at(cnt, inv, 1.0)
        avg = acc / np.maximum(cnt[:, None], 1)
        avg /= np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-12)
        cosd = np.cos(np.deg2rad(deg))
        use = (np.sum(avg[inv] * m.N, axis=1) > cosd)
        m.N = np.where(use[:, None], avg[inv], m.N)
        m.N /= np.maximum(np.linalg.norm(m.N, axis=1, keepdims=True), 1e-12)
        return m


def merge(meshes):
    Vs, Fs, UVs, Ns = [], [], [], []
    off = 0
    for m in meshes:
        if len(m.V) == 0:
            continue
        Vs.append(m.V)
        Fs.append(m.F + off)
        UVs.append(m.UV)
        Ns.append(m.N if m.N is not None else np.zeros_like(m.V))
        off += len(m.V)
    if not Vs:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    return Mesh(np.vstack(Vs), np.vstack(Fs), np.vstack(UVs), np.vstack(Ns))


def revolve(profile, nseg=192, v_span=(0.0, 1.0), a0=0.0, a1=2 * np.pi):
    # profile: list of (r, z) revolved about +Z
    prof = np.asarray(profile, float)
    ni = len(prof)
    full = abs((a1 - a0) - 2 * np.pi) < 1e-9
    nring = nseg if full else nseg + 1
    ang = a0 + (a1 - a0) * (np.arange(nring) / nseg)
    ca, sa = np.cos(ang), np.sin(ang)
    r = prof[:, 0][None, :]
    z = prof[:, 1][None, :]
    X = ca[:, None] * r
    Y = sa[:, None] * r
    Z = np.repeat(z, nring, axis=0)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
    d = d / max(d[-1], 1e-12)
    v = v_span[0] + (v_span[1] - v_span[0]) * d
    u = (ang - a0) / max(a1 - a0, 1e-12)
    UV = np.stack([np.repeat(u[:, None], ni, 1), np.repeat(v[None, :], nring, 0)], -1).reshape(-1, 2)
    F = []
    last = nring if full else nring - 1
    for j in range(last):
        j2 = (j + 1) % nring
        for i in range(ni - 1):
            a = j * ni + i
            b = j2 * ni + i
            F.append([a, b, b + 1])
            F.append([a, b + 1, a + 1])
    return Mesh(V, np.array(F, np.int32), UV)


def disc(r_outer, z, nseg=192, r_inner=0.0, up=True, uv_scale=None, uv_center=(0.5, 0.5)):
    ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    if r_inner <= 0:
        V = np.zeros((nseg + 1, 3))
        V[0] = [0, 0, z]
        V[1:, 0] = ca * r_outer
        V[1:, 1] = sa * r_outer
        V[1:, 2] = z
        F = [[0, i + 1, 1 + (i + 1) % nseg] for i in range(nseg)]
    else:
        V = np.zeros((nseg * 2, 3))
        V[0::2, 0] = ca * r_inner
        V[0::2, 1] = sa * r_inner
        V[1::2, 0] = ca * r_outer
        V[1::2, 1] = sa * r_outer
        V[:, 2] = z
        F = []
        for i in range(nseg):
            a = 2 * i
            b = 2 * i + 1
            c = 2 * ((i + 1) % nseg)
            d = c + 1
            F.append([a, b, d])
            F.append([a, d, c])
    F = np.array(F, np.int32)
    s = uv_scale if uv_scale else (0.5 / max(r_outer, 1e-9))
    UV = np.stack([uv_center[0] + V[:, 0] * s, uv_center[1] - V[:, 1] * s], -1)
    if not up:
        F = F[:, ::-1]
    return Mesh(V, F, UV)


def prism(poly, z0, z1, caps=True):
    poly = np.asarray(poly, float)
    n = len(poly)
    V = np.vstack([np.c_[poly, np.full(n, z0)], np.c_[poly, np.full(n, z1)]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, n + j])
        F.append([i, n + j, n + i])
    if caps:
        for i in range(1, n - 1):
            F.append([n, n + i, n + i + 1])
            F.append([0, i + 1, i])
    return Mesh(V, np.array(F, np.int32))


def rect_poly(w, h, cx=0.0, cy=0.0, r=0.0, nc=6):
    hw, hh = w / 2, h / 2
    r = min(r, hw, hh)
    if r <= 1e-9:
        return np.array([[cx - hw, cy - hh], [cx + hw, cy - hh],
                         [cx + hw, cy + hh], [cx - hw, cy + hh]])
    pts = []
    corners = [(cx + hw - r, cy - hh + r, -np.pi / 2), (cx + hw - r, cy + hh - r, 0.0),
               (cx - hw + r, cy + hh - r, np.pi / 2), (cx - hw + r, cy - hh + r, np.pi)]
    for (px, py, a0) in corners:
        for k in range(nc + 1):
            a = a0 + (np.pi / 2) * k / nc
            pts.append([px + r * np.cos(a), py + r * np.sin(a)])
    return np.array(pts)


def frames(path, closed=False, up_ref=np.array([0.0, 0.0, 1.0]), fixed_right=None, fixed_up=None):
    path = np.asarray(path, float)
    n = len(path)
    if closed:
        tang = np.roll(path, -1, 0) - np.roll(path, 1, 0)
    else:
        tang = np.zeros_like(path)
        tang[1:-1] = path[2:] - path[:-2]
        tang[0] = path[1] - path[0]
        tang[-1] = path[-1] - path[-2]
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    if fixed_up is not None:
        # the surface normal is given (e.g. the wrist); make the frame follow it
        ups = np.asarray(fixed_up, float).copy()
        ups /= np.maximum(np.linalg.norm(ups, axis=1, keepdims=True), 1e-12)
        rights = np.cross(ups, tang)
        rights /= np.maximum(np.linalg.norm(rights, axis=1, keepdims=True), 1e-12)
        ups = np.cross(tang, rights)
        ups /= np.maximum(np.linalg.norm(ups, axis=1, keepdims=True), 1e-12)
        return tang, rights, ups
    if fixed_right is not None:
        fr = np.asarray(fixed_right, float)
        fr = fr / np.linalg.norm(fr)
        rights = np.repeat(fr[None, :], n, axis=0)
        ups = np.cross(tang, rights)
        ups /= np.maximum(np.linalg.norm(ups, axis=1, keepdims=True), 1e-12)
        return tang, rights, ups
    rights = np.zeros_like(path)
    ups = np.zeros_like(path)
    r = np.cross(up_ref, tang[0])
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(np.array([1.0, 0, 0]), tang[0])
    r /= np.linalg.norm(r)
    for i in range(n):
        if i > 0:
            r = r - tang[i] * np.dot(r, tang[i])
            nr = np.linalg.norm(r)
            if nr > 1e-9:
                r = r / nr
            else:
                r = np.cross(up_ref, tang[i])
                r /= max(np.linalg.norm(r), 1e-9)
        rights[i] = r
        ups[i] = np.cross(tang[i], r)
    return tang, rights, ups


def sweep(path, secs, closed=False, up_ref=np.array([0.0, 0.0, 1.0]),
          uv_u=None, sec_v=None, cap_ends=False, section_closed=True, uv_u_scale=1.0,
          fixed_right=None, fixed_up=None):
    # secs: one (m,2) cross-section, or a list of (m,2) sections (one per path point)
    path = np.asarray(path, float)
    n = len(path)
    tang, rights, ups = frames(path, closed, up_ref, fixed_right, fixed_up)
    if np.ndim(secs) == 2:
        secs = [np.asarray(secs, float)] * n
    m = len(secs[0])
    V = np.zeros((n * m, 3))
    UV = np.zeros((n * m, 2))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    for i in range(n):
        s = np.asarray(secs[i], float)
        V[i * m:(i + 1) * m] = path[i] + s[:, 0:1] * rights[i] + s[:, 1:2] * ups[i]
        if sec_v is not None:
            UV[i * m:(i + 1) * m, 1] = sec_v
        else:
            per = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.vstack([s, s[:1]]), axis=0), axis=1))]
            UV[i * m:(i + 1) * m, 1] = per[:-1] / max(per[-1], 1e-12)
        UV[i * m:(i + 1) * m, 0] = (arc[i] if uv_u is None else uv_u[i]) * uv_u_scale
    F = []
    rng = range(n) if closed else range(n - 1)
    klast = m if section_closed else m - 1
    for i in rng:
        i2 = (i + 1) % n
        for k in range(klast):
            k2 = (k + 1) % m
            a = i * m + k
            b = i * m + k2
            c = i2 * m + k2
            d = i2 * m + k
            F.append([a, b, c])
            F.append([a, c, d])
    mesh = Mesh(V, np.array(F, np.int32), UV)
    if cap_ends and not closed and section_closed:
        caps = [fan(V[:m], reverse=True), fan(V[(n - 1) * m:n * m], reverse=False)]
        mesh = merge([mesh] + caps)
    return mesh


def spline_resample(pts, n, closed=False, k=3):
    """Periodic/open Catmull-Rom resampling of a control polyline (any dimension)."""
    P = np.asarray(pts, float)
    m = len(P)
    if closed:
        idx = lambda i: P[i % m]
    else:
        idx = lambda i: P[min(max(i, 0), m - 1)]
    segs = m if closed else m - 1
    out = []
    for s in range(segs):
        p0, p1, p2, p3 = idx(s - 1), idx(s), idx(s + 1), idx(s + 2)
        steps = max(2, int(np.ceil(n / segs)))
        for j in range(steps):
            t = j / steps
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-1])
    return np.array(out)


def fan(ring, reverse=False):
    P = np.asarray(ring, float)
    c = P.mean(0)
    V = np.vstack([c[None, :], P])
    n = len(P)
    F = np.array([[0, i + 1, 1 + (i + 1) % n] for i in range(n)], np.int32)
    if reverse:
        F = F[:, ::-1]
    return Mesh(V, F)


def rotZ(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4); M[0, 0] = c; M[0, 1] = -s; M[1, 0] = s; M[1, 1] = c
    return M


def rotX(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4); M[1, 1] = c; M[1, 2] = -s; M[2, 1] = s; M[2, 2] = c
    return M


def rotY(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4); M[0, 0] = c; M[0, 2] = s; M[2, 0] = -s; M[2, 2] = c
    return M


def trans(x, y, z):
    M = np.eye(4); M[:3, 3] = [x, y, z]
    return M
