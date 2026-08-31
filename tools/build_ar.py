"""Build the dedicated wrist-AR watch model.

The presentation watch needs a complete buckle/free-end assembly; live wrist AR
does not.  The AR asset keeps the production watch head unchanged and uses two
independent leather wrap arms.  Each arm starts exactly on its spring bar, exits
the lug straight, follows the visible side of the wrist, then continues deep
behind the wrist where the depth occluder hides its end.

The arms intentionally do not meet and there is no rendered buckle, keeper or
free end.  Unlike the earlier short-stub version, however, the straps are long
enough that their cut ends cannot appear in the normal top/oblique try-on view.
"""

import numpy as np
import build as b


# Build-frame millimetres. +Y is 12 o'clock, -Y is 6 o'clock, +Z is dial normal.
# Production spring bars sit at about Y=+/-20.6, Z=-2.1.
LUG_Y = 20.40
LUG_Z = -2.30

# AR visual-wrap spec.  The remote end is deliberately below the authored wrist
# centre after export, so the depth occluder hides the physical termination.
END_Y = 41.00
END_Z = -31.50
LUG_EXIT_Y = 26.40       # 6.0 mm almost-straight spring-bar exit
BEND_Y = 39.00
BEND_Z = -11.00

W_LUG = 20.00
W_END = 19.20
TH_LUG = 2.60
TH_END = 2.15
TH_TIP = 1.90            # hidden tip only; avoids a thick square flash


def _bezier4(p0, p1, p2, p3, samples=112):
    """Sample one cubic Bezier curve including both exact endpoints."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    t = np.linspace(0.0, 1.0, samples)[:, None]
    u = 1.0 - t
    return (
        (u ** 3) * p0
        + 3.0 * (u ** 2) * t * p1
        + 3.0 * u * (t ** 2) * p2
        + (t ** 3) * p3
    )


def _arm_path(sign):
    """One monotonic AR strap arm that disappears behind the real wrist.

    P0->P1 keeps the spring-bar connection rigid and square. P1->P2 begins the
    natural side bend. P2->P3 turns steeply behind the wrist instead of ending in
    visible space. The 6-side is the exact Y mirror of the 12-side.
    """
    s = float(sign)
    p0 = np.array([0.0, s * LUG_Y, LUG_Z])
    p1 = np.array([0.0, s * LUG_EXIT_Y, -2.40])
    p2 = np.array([0.0, s * BEND_Y, BEND_Z])
    p3 = np.array([0.0, s * END_Y, END_Z])
    return _bezier4(p0, p1, p2, p3)


def _band(path):
    """Generate croc-leather top + suede underside for one AR wrap arm."""
    path = np.asarray(path, float)
    n = len(path)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]

    secs = []
    vv = None
    ntop = None
    for i in range(n):
        t = i / max(n - 1, 1)

        # Keep the visible lug/upper-wrist region essentially production width.
        # Taper is postponed until the arm is already turning out of view.
        k = np.clip((t - 0.38) / 0.62, 0.0, 1.0)
        k = k * k * (3.0 - 2.0 * k)
        w = W_LUG + (W_END - W_LUG) * k
        th = TH_LUG + (TH_END - TH_LUG) * k

        # Only the last hidden 12% softens slightly. This is not a visible pointed
        # free end; it merely makes any sub-pixel cap flash less block-like.
        tip = np.clip((t - 0.88) / 0.12, 0.0, 1.0)
        tip = tip * tip * (3.0 - 2.0 * tip)
        th += (TH_TIP - TH_END) * tip

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

    # Keep the mesh watertight for standalone inspection. The remote cap is deep
    # inside the occluded wrist volume during actual try-on.
    _, rt, upv = b.frames(path, False, fixed_right=fixed_right)
    caps = []
    for i, reverse in ((0, False), (n - 1, True)):
        sec = np.asarray(secs[i], float)
        ring = path[i] + sec[:, 0:1] * rt[i] + sec[:, 1:2] * upv[i]
        caps.append(b.fan(ring, reverse=reverse))

    return top, b.merge([bot, *caps])


def straps_ar():
    p12 = _arm_path(+1)
    p6 = _arm_path(-1)
    t12, b12 = _band(p12)
    t6, b6 = _band(p6)

    # Compatibility keys expected by build.py; no rendered buckle path exists.
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
    """Invisible compatibility placeholder for buckle/keepers requested by build.py."""
    v = np.array([
        [0.000, 0.000, -3.80],
        [0.001, 0.000, -3.80],
        [0.000, 0.001, -3.80],
    ], dtype=float)
    f = np.array([[0, 1, 2]], dtype=np.int32)
    return b.Mesh(v, f)


def _validate_paths():
    """Reject detached, short-stub or accidental full-loop regressions."""
    for sign in (+1, -1):
        p = _arm_path(sign)
        arc = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
        if not 34.0 <= arc <= 44.0:
            raise RuntimeError(f"AR strap arc length out of wrap range: {arc:.2f} mm")
        if abs(abs(p[0, 1]) - LUG_Y) > 0.05 or abs(p[0, 2] - LUG_Z) > 0.05:
            raise RuntimeError("AR strap no longer starts on the spring bar")
        if abs(abs(p[-1, 1]) - END_Y) > 0.05 or abs(p[-1, 2] - END_Z) > 0.05:
            raise RuntimeError("AR strap end no longer matches hidden wrap target")
        if p[-1, 2] > -28.0:
            raise RuntimeError("AR strap end is not deep enough to disappear behind wrist")


# Patch only wear-specific pieces. Case, dial, hands, lugs, spring bars,
# materials and textures remain the production geometry from build.py.
b.straps = straps_ar
b.buckle = _hidden_triangle
b.keepers = _hidden_triangle


if __name__ == "__main__":
    _validate_paths()
    b.build("iarone_watch_ar.glb", wear=True)
