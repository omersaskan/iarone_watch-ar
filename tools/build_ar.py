"""Build the dedicated wrist-AR watch model.

The presentation watch needs a complete buckle/free-end assembly; live wrist AR
does not.  The AR asset keeps the production watch head unchanged and uses only
two short leather stubs.  Each stub starts exactly on its spring bar, exits the
lug straight, bends naturally toward the wrist side and terminates inside the
depth-occluded region.  Buckle, keepers and free end are intentionally absent.
"""

import numpy as np
import build as b


# Build-frame millimetres. +Y is 12 o'clock, -Y is 6 o'clock, +Z is dial normal.
# Production spring bars sit at about Y=+/-20.6, Z=-2.1.
LUG_Y = 20.40
LUG_Z = -2.30

# Final AR strap spec v2.
END_Y = 38.00
END_Z = -17.50
LUG_EXIT_Y = 26.40       # 6.0 mm almost-straight spring-bar exit
BEND_Y = 35.00
BEND_Z = -7.00

W_LUG = 20.00
W_END = 19.20
TH_LUG = 2.60
TH_END = 2.15
TH_TIP = 1.85            # only the final hidden 10% softens to this value


def _bezier4(p0, p1, p2, p3, samples=88):
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
    """One monotonic AR strap stub.

    P0->P1 gives a 6 mm square lug exit.  P1->P2 introduces the bend gradually;
    P2->P3 turns down into the wrist occluder instead of continuing as a visible
    half-loop.  The 6-side is the exact Y mirror of the 12-side.
    """
    s = float(sign)
    p0 = np.array([0.0, s * LUG_Y, LUG_Z])
    p1 = np.array([0.0, s * LUG_EXIT_Y, -2.40])
    p2 = np.array([0.0, s * BEND_Y, BEND_Z])
    p3 = np.array([0.0, s * END_Y, END_Z])
    return _bezier4(p0, p1, p2, p3)


def _band(path):
    """Generate croc-leather top + suede underside for one AR strap stub."""
    path = np.asarray(path, float)
    n = len(path)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]

    secs = []
    vv = None
    ntop = None
    for i in range(n):
        t = i / max(n - 1, 1)

        # First 30% remains full lug width/thickness.  The visible strap then
        # tapers almost imperceptibly.  Only the final hidden 10% gets a small
        # thickness softening so a cap cannot read like a square cut block.
        k = np.clip((t - 0.30) / 0.70, 0.0, 1.0)
        k = k * k * (3.0 - 2.0 * k)
        w = W_LUG + (W_END - W_LUG) * k
        th = TH_LUG + (TH_END - TH_LUG) * k

        tip = np.clip((t - 0.90) / 0.10, 0.0, 1.0)
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

    # Keep the mesh watertight for standalone inspection.  The far cap uses the
    # underside material and is authored deep enough to be hidden by the AR wrist
    # occluder during try-on.
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
    """Reject regressions back to long/looping or detached AR straps."""
    for sign in (+1, -1):
        p = _arm_path(sign)
        arc = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
        if not 20.0 <= arc <= 28.0:
            raise RuntimeError(f"AR strap arc length out of final-spec range: {arc:.2f} mm")
        if abs(abs(p[0, 1]) - LUG_Y) > 0.05 or abs(p[0, 2] - LUG_Z) > 0.05:
            raise RuntimeError("AR strap no longer starts on the spring bar")
        if abs(abs(p[-1, 1]) - END_Y) > 0.05 or abs(p[-1, 2] - END_Z) > 0.05:
            raise RuntimeError("AR strap end no longer matches final occlusion target")
        if p[-1, 2] > -15.0:
            raise RuntimeError("AR strap end is not deep enough for wrist occlusion")


# Patch only wear-specific pieces. Case, dial, hands, lugs, spring bars,
# materials and textures remain the production geometry from build.py.
b.straps = straps_ar
b.buckle = _hidden_triangle
b.keepers = _hidden_triangle


if __name__ == "__main__":
    _validate_paths()
    b.build("iarone_watch_ar.glb", wear=True)
