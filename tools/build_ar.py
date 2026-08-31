"""Build the dedicated wrist-AR watch model.

The presentation watch uses a buckle/free-end construction.  That geometry is not
appropriate for live wrist AR: when pre-wrapped around a wrist it becomes a
helical loop, enters the lugs at the wrong angle and can show buckle/keeper parts
through the occlusion mask.

This builder keeps the watch head unchanged and replaces only the wear strap with
two clean arms.  Each arm starts square on its spring bar, runs along the forearm
and then bends below the wrist.  The hidden ends deliberately do not meet; the AR
wrist occluder covers that region.  This gives the renderer one unambiguous strap
surface per side and lets the runtime wrist-size deformer scale only leather.
"""

import numpy as np
import build as b


def _smooth_path(points, samples=96):
    """Centripetal-ish Catmull-Rom resampling without adding a new dependency."""
    p = np.asarray(points, dtype=float)
    if len(p) < 4:
        raise ValueError("need at least four path controls")

    # Duplicate end points so the curve begins and ends exactly on the spring bars.
    q = np.vstack([p[0], p, p[-1]])
    out = []
    spans = len(q) - 3
    per = max(8, int(np.ceil(samples / spans)))
    for i in range(spans):
        p0, p1, p2, p3 = q[i:i + 4]
        ts = np.linspace(0.0, 1.0, per, endpoint=(i == spans - 1))
        for t in ts:
            t2, t3 = t * t, t * t * t
            v = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(v)
    return np.asarray(out)


def _arm_path(sign):
    """One AR strap arm in the build frame (mm).

    +Y is 12 o'clock, -Y is 6 o'clock and +Z is the dial normal.  The first
    segment is intentionally almost parallel to Y, so the leather enters the lug
    square instead of immediately sweeping sideways around the wrist.
    """
    s = float(sign)
    controls = np.array([
        [0.0, s * 20.40, -2.30],  # spring-bar centre
        [0.0, s * 24.80, -2.65],  # straight lug exit
        [0.0, s * 31.50, -4.40],
        [0.0, s * 38.50, -8.50],
        [0.0, s * 44.00, -14.80],
        [0.0, s * 47.00, -22.50],
        [0.0, s * 47.50, -31.00], # safely below the wrist mask
    ])
    return _smooth_path(controls, 112)


def _band(path, w_lug=20.0, w_hidden=18.8, th_lug=2.60, th_hidden=2.35):
    """Generate leather top + underside for one strap arm."""
    path = np.asarray(path, float)
    n = len(path)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    secs = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # Keep the first 10 mm at full lug width, then taper very gently only in
        # the hidden region.  No pointed free end is present in the AR asset.
        k = t * t * (3.0 - 2.0 * t)
        w = w_lug + (w_hidden - w_lug) * k
        th = th_lug + (th_hidden - th_lug) * k
        sec, vv, ntop = b.strap_section(w, th)
        secs.append(sec)

    uv_u = arc / 34.0
    fixed_right = np.array([-1.0, 0.0, 0.0])

    top = b.sweep(
        path,
        [s[:ntop] for s in secs],
        fixed_right=fixed_right,
        uv_u=uv_u,
        sec_v=vv[:ntop],
        section_closed=False,
    )
    bot = b.sweep(
        path,
        [np.vstack([s[ntop - 1:], s[:1]]) for s in secs],
        fixed_right=fixed_right,
        uv_u=uv_u,
        sec_v=np.r_[vv[ntop - 1:], vv[:1]],
        section_closed=False,
    )
    top.F = top.F[:, ::-1]
    bot.F = bot.F[:, ::-1]

    # Close only the exposed lug end.  The remote end is buried in the wrist
    # occluder, so leaving it open avoids a visible cap flashing through the mask.
    _, rt, upv = b.frames(path, False, fixed_right=fixed_right)
    sec0 = np.asarray(secs[0], float)
    ring0 = path[0] + sec0[:, 0:1] * rt[0] + sec0[:, 1:2] * upv[0]
    cap0 = b.fan(ring0, reverse=False)
    return top, b.merge([bot, cap0])


def straps_ar():
    p12 = _arm_path(+1)
    p6 = _arm_path(-1)
    t12, b12 = _band(p12)
    t6, b6 = _band(p6)

    # Keep keys expected by build.py, but there is intentionally no buckle path in
    # the AR asset.  The concatenated path is used only by the compatibility stubs
    # below and never rendered.
    dummy = np.vstack([p12, p6[::-1]])
    return {
        "top": b.merge([t12, t6]),
        "bot": b.merge([b12, b6]),
        "p12": p12,
        "p6": p6,
        "i_buck": max(2, len(dummy) // 2),
        "path": dummy,
        "arc": np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dummy, axis=0), axis=1))],
    }


def _hidden_triangle(*_args, **_kwargs):
    """Compatibility primitive for buckle/keepers that build.py always requests.

    One sub-millimetre triangle is placed inside the opaque case so the legacy
    assembler can keep its primitive/material layout without rendering AR-only
    buckle or keeper geometry.
    """
    v = np.array([
        [0.000, 0.000, -3.80],
        [0.001, 0.000, -3.80],
        [0.000, 0.001, -3.80],
    ], dtype=float)
    f = np.array([[0, 1, 2]], dtype=np.int32)
    return b.Mesh(v, f)


# Patch only the wear-specific pieces; case, dial, hands, lugs, spring bars and
# all materials/textures remain the proven production geometry from build.py.
b.straps = straps_ar
b.buckle = _hidden_triangle
b.keepers = _hidden_triangle


if __name__ == "__main__":
    b.build("iarone_watch_ar.glb", wear=True)
