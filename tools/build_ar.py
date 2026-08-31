"""Build the dedicated wrist-AR watch model.

The product/presentation watch needs a complete buckle and free-end assembly.
Live wrist AR does not.  Geometry below the wrist is hidden by the depth
occluder, and trying to model a complete pre-wrapped strap creates visible seams,
loops and buckle parts at oblique camera angles.

The AR asset therefore keeps the production watch head unchanged and uses two
short leather stubs only.  Each stub:
  * starts exactly at its spring bar,
  * leaves the lug almost straight for the first few millimetres,
  * bends smoothly down toward the side of the wrist,
  * terminates early inside the region covered by the wrist occluder.

There is no buckle, keeper or free end in this asset.  This is intentional: the
AR occluder supplies the hidden continuation around the user's real wrist.
"""

import numpy as np
import build as b


# Build-frame millimetres.  +Y is 12 o'clock, -Y is 6 o'clock, +Z is the
# dial normal.  The production spring bars sit at about Y=+/-20.6, Z=-2.1.
LUG_Y = 20.40
LUG_Z = -2.30

# The remote end is only ~20 mm farther along the forearm and ~16 mm below the
# lug.  That is enough to disappear behind the AR wrist occluder without making
# the standalone GLB look like it has two long hanging blades.
END_Y = 40.20
END_Z = -18.00


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
    """Short, monotonic AR strap stub for one side of the watch.

    P0->P1 is parallel to the forearm axis, so leather enters the spring bar
    square.  P2->P3 is mostly downward, so the far end turns behind the wrist
    instead of continuing as a long visible strip.
    """
    s = float(sign)
    p0 = np.array([0.0, s * LUG_Y, LUG_Z])
    p1 = np.array([0.0, s * 26.20, -2.45])   # ~5.8 mm straight lug exit
    p2 = np.array([0.0, s * 38.70, -9.20])   # begin the real wrist-side bend
    p3 = np.array([0.0, s * END_Y, END_Z])    # buried by the AR occluder
    return _bezier4(p0, p1, p2, p3)


def _band(path, w_lug=20.0, w_hidden=19.35, th_lug=2.60, th_hidden=2.25):
    """Generate leather top + suede underside for one short AR strap stub."""
    path = np.asarray(path, float)
    n = len(path)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]

    secs = []
    vv = None
    ntop = None
    for i in range(n):
        t = i / max(n - 1, 1)

        # Do not visibly taper at the lug.  Most taper happens after the first
        # third, where the strap is already turning behind the wrist.
        k = np.clip((t - 0.30) / 0.70, 0.0, 1.0)
        k = k * k * (3.0 - 2.0 * k)
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

    # Cap both ends so the GLB is also clean in a standalone viewer.  The far cap
    # uses the suede/underside material and sits in the region hidden by the AR
    # occluder during actual try-on.
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

    # build.py expects these compatibility keys even though the dedicated AR asset
    # intentionally has no real buckle path.
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
    """Compatibility placeholder for buckle/keepers requested by build.py."""
    v = np.array([
        [0.000, 0.000, -3.80],
        [0.001, 0.000, -3.80],
        [0.000, 0.001, -3.80],
    ], dtype=float)
    f = np.array([[0, 1, 2]], dtype=np.int32)
    return b.Mesh(v, f)


def _validate_paths():
    """Fail the build if a future edit makes the AR stubs long/looping again."""
    for sign in (+1, -1):
        p = _arm_path(sign)
        arc = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
        if not 22.0 <= arc <= 32.0:
            raise RuntimeError(f"AR strap arc length out of range: {arc:.2f} mm")
        if abs(abs(p[0, 1]) - LUG_Y) > 0.05 or abs(p[0, 2] - LUG_Z) > 0.05:
            raise RuntimeError("AR strap no longer starts on the spring bar")
        if p[-1, 2] > -15.0:
            raise RuntimeError("AR strap end is not deep enough for wrist occlusion")


# Patch only wear-specific pieces.  Case, dial, hands, lugs, spring bars,
# materials and textures remain the production geometry from build.py.
b.straps = straps_ar
b.buckle = _hidden_triangle
b.keepers = _hidden_triangle


if __name__ == "__main__":
    _validate_paths()
    b.build("iarone_watch_ar.glb", wear=True)
