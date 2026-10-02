// Runtime strap: swept around the wrist that was measured, not the one it
// was photographed on.
//
// The Lab exports the strap as a profile (lab_report.result.strap_profile):
// width and thickness by arc length, where it leaves the lugs at 12 and 6,
// and the clasp position.  This module ports tools/build.py's
// loop_path_wrist / wrist_normal / sweep to the browser and drives them from
// the live forearm ellipse (wristfit.js), so the strap encircles THIS arm:
// no gap on a thin wrist, no burying on a thick one.
//
// Frames.  The path is built in the build.py "wear" frame (mm): the forearm
// is the Y axis, Z is out of the back of the wrist, X across the wrist, and
// the watch sits on top at z = 0 (dial plane).  The AR rig frame (ar.html)
// is: +X forearm, +Y out of the wrist, -Z toward 12 o'clock.  The map is
//     rig = (y_build, z_build, -x_build) / 1000.
// Note: the real geometry of a strap on a wrist is NOT a planar loop: it
// drifts along the forearm as it wraps (see build.py's docstring), and that
// drift is what keeps it off the arm.

import * as THREE from '#three';

/** Geometry of the sample strap on the sample wrist, mm (build.py P["wrist"]).
 *  Everything a profile does not state falls back to these. */
export const STRAP_DEFAULTS = {
  gap: 1.35,        // strap stands this far off the skin
  drift: 16.4,      // how far along the forearm the loop spirals as it wraps
  ease: 17.0,       // how long the strap heads along the forearm before turning
  lift: 1.65,       // lift of the two ends up to the spring bars
  y_start: 20.4,    // where the strap leaves the lugs along the forearm (half lug-to-lug)
  ends: 0.75,
  width: 20.0,
  thickness: 2.6,
  n: 300,
};

/**
 * Loop centre line around a forearm ellipse (half-axes rx across, rz
 * back-to-front, mm) whose top skin is at skin_top (mm, negative: below the
 * dial plane).  Returns (n,3) float64 in the build frame.
 */
export function loopPathWrist(rx, rz, skinTop, opts = {}) {
  const o = Object.assign({}, STRAP_DEFAULTS, opts);
  const Rx = rx + o.gap, Rz = rz + o.gap;
  const zc = skinTop - rz;
  const A = o.drift, n = o.n;
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / (n - 1);
    const t0 = th, t1 = 2 * Math.PI - th;
    let x = Rx * Math.sin(th);
    let z = zc + Rz * Math.cos(th);
    let y = A * Math.cos(th / 2);
    // anchor the ends on the spring bars without letting the drift amplitude move them
    const sg = o.ends;
    y += (o.y_start - A) * (Math.exp(-((t0 / sg) ** 2)) - Math.exp(-((t1 / sg) ** 2)));
    // leave the lugs heading along the strap, not straight across the wrist
    y += -o.ease * t0 * Math.exp(-t0 / 0.42) + o.ease * t1 * Math.exp(-t1 / 0.42);
    // lift the two ends up to the spring bars
    z += o.lift * (Math.exp(-((t0 / 0.55) ** 2)) + Math.exp(-((t1 / 0.55) ** 2)));
    out[3 * i] = x; out[3 * i + 1] = y; out[3 * i + 2] = z;
  }
  return out;
}

/** Outward normal of the forearm ellipse at a path point (build frame). */
function ellipseNormal(x, z, rx, rz, zc, out) {
  let nx = x / (rx * rx), nz = (z - zc) / (rz * rz);
  const l = Math.hypot(nx, nz);
  if (l < 1e-9) { out[0] = 0; out[1] = 0; out[2] = 1; return out; }
  out[0] = nx / l; out[1] = 0; out[2] = nz / l;
  return out;
}

/**
 * Signed clearance of the strap's inner face from the skin, mm, at every
 * path point: positive = clear of the arm, negative = buried in it.  This is
 * the acceptance metric ("no gap, no burying": |clearance - gap| small).
 */
export function clearanceMm(path, rx, rz, skinTop, thickness) {
  const zc = skinTop - rz;
  const n = path.length / 3;
  const out = new Float64Array(n);
  const th = typeof thickness === 'number' ? () => thickness : thickness;
  for (let i = 0; i < n; i++) {
    const x = path[3 * i], z = path[3 * i + 2];
    // radial distance of the centre line from the ellipse surface along its normal
    // (an ellipse has no closed form for point distance; the normal-scaled
    // measure below is exact on the axes and within 2% between them for wrist
    // aspect ratios)
    const u = x / rx, v = (z - zc) / rz;
    const r = Math.hypot(u, v);               // 1 on the skin
    const nx = u / rx, nz = v / rz;
    const nl = Math.hypot(nx, nz) || 1e-9;
    const d = (r - 1) / nl;                   // mm from the skin along the normal
    out[i] = d - 0.5 * th(i / (n - 1));
  }
  return out;
}

/** Closed, rounded rectangle cross-section (w, th mm): 24 points, CCW in (right, up). */
export function strapSection(w, th, m = 24) {
  const hw = w / 2, ht = th / 2;
  const r = Math.min(0.55, th * 0.42);
  const pts = [];
  const corner = (cx, cz, a0, a1, k) => {
    for (let i = 0; i < k; i++) {
      const a = a0 + ((a1 - a0) * i) / (k - 1);
      pts.push([cx + r * Math.cos(a), cz + r * Math.sin(a)]);
    }
  };
  const k = Math.max(3, Math.round(m / 4));
  corner(hw - r, ht - r, 0, Math.PI / 2, k);               // top-right
  corner(-hw + r, ht - r, Math.PI / 2, Math.PI, k);        // top-left
  corner(-hw + r, -ht + r, Math.PI, 1.5 * Math.PI, k);     // bottom-left
  corner(hw - r, -ht + r, 1.5 * Math.PI, 2 * Math.PI, k);  // bottom-right
  return pts;
}

/** Linear interpolation of a profile array sampled at arc positions. */
function profileAt(arc, values, s) {
  if (!values || !values.length) return null;
  if (!arc || arc.length !== values.length) {
    const t = Math.min(1, Math.max(0, s));
    const i = Math.min(values.length - 2, Math.floor(t * (values.length - 1)));
    const f = t * (values.length - 1) - i;
    return values[i] * (1 - f) + values[i + 1] * f;
  }
  const total = arc[arc.length - 1];
  const ss = s * total;
  let i = 0;
  while (i < arc.length - 2 && arc[i + 1] < ss) i++;
  const f = (ss - arc[i]) / Math.max(arc[i + 1] - arc[i], 1e-9);
  return values[i] * (1 - f) + values[i + 1] * f;
}

/**
 * Build a strap BufferGeometry (rig frame, metres) around the ellipse.
 * profile: the Lab's strap_profile (arc_mm, width_mm, thickness_mm) or null
 * for the sample strap.  skinTopMm: dial plane -> skin, mm (negative).
 */
export function buildStrapGeometry(rx, rz, skinTopMm, profile, opts = {}) {
  const o = Object.assign({}, STRAP_DEFAULTS, opts);
  const path = loopPathWrist(rx, rz, skinTopMm, o);
  const n = path.length / 3;
  const zc = skinTopMm - rz;
  const W = (t) => (profile ? profileAt(profile.arc_mm, profile.width_mm, t) : null) ?? o.width;
  const T = (t) => (profile ? profileAt(profile.arc_mm, profile.thickness_mm, t) : null) ?? o.thickness;

  const m = 24;
  const V = new Float32Array(n * m * 3);
  const N = new Float32Array(n * m * 3);
  const UV = new Float32Array(n * m * 2);
  const tang = [0, 0, 0], up = [0, 0, 0], right = [0, 0, 0];
  let arcLen = 0;
  const arc = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    arcLen += Math.hypot(path[3 * i] - path[3 * i - 3], path[3 * i + 1] - path[3 * i - 2], path[3 * i + 2] - path[3 * i - 1]);
    arc[i] = arcLen;
  }
  for (let i = 0; i < n; i++) {
    const i0 = Math.max(0, i - 1), i1 = Math.min(n - 1, i + 1);
    tang[0] = path[3 * i1] - path[3 * i0]; tang[1] = path[3 * i1 + 1] - path[3 * i0 + 1]; tang[2] = path[3 * i1 + 2] - path[3 * i0 + 2];
    let tl = Math.hypot(tang[0], tang[1], tang[2]) || 1e-9;
    tang[0] /= tl; tang[1] /= tl; tang[2] /= tl;
    // the section's "up" is the wrist's outward normal, so the strap's face
    // follows the skin (build.py: fixed_up = wrist_normal)
    ellipseNormal(path[3 * i], path[3 * i + 2], rx, rz, zc, up);
    // right = up x tang ; up = tang x right (re-orthogonalised)
    right[0] = up[1] * tang[2] - up[2] * tang[1];
    right[1] = up[2] * tang[0] - up[0] * tang[2];
    right[2] = up[0] * tang[1] - up[1] * tang[0];
    let rl = Math.hypot(right[0], right[1], right[2]) || 1e-9;
    right[0] /= rl; right[1] /= rl; right[2] /= rl;
    up[0] = tang[1] * right[2] - tang[2] * right[1];
    up[1] = tang[2] * right[0] - tang[0] * right[2];
    up[2] = tang[0] * right[1] - tang[1] * right[0];

    const t = i / (n - 1);
    const sec = strapSection(W(t), T(t), m);
    for (let k = 0; k < m; k++) {
      const [sx, sz] = sec[k];
      const bx = path[3 * i] + sx * right[0] + sz * up[0];
      const by = path[3 * i + 1] + sx * right[1] + sz * up[1];
      const bz = path[3 * i + 2] + sx * right[2] + sz * up[2];
      const j = (i * m + k) * 3;
      // build frame -> rig frame, mm -> m
      V[j] = by / 1000; V[j + 1] = bz / 1000; V[j + 2] = -bx / 1000;
      // normal: the section's own outward direction (right*cos, up*sin of the local angle)
      const cx = sx, cz = sz;
      const cl = Math.hypot(cx, cz) || 1e-9;
      const nxb = (cx / cl) * right[0] + (cz / cl) * up[0];
      const nyb = (cx / cl) * right[1] + (cz / cl) * up[1];
      const nzb = (cx / cl) * right[2] + (cz / cl) * up[2];
      N[j] = nyb; N[j + 1] = nzb; N[j + 2] = -nxb;
      UV[(i * m + k) * 2] = arc[i] / 34.0;
      UV[(i * m + k) * 2 + 1] = k / m;
    }
  }
  const idx = new Uint32Array((n - 1) * m * 6);
  let q = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let k = 0; k < m; k++) {
      const k2 = (k + 1) % m;
      const a = i * m + k, b = i * m + k2, c = (i + 1) * m + k, d = (i + 1) * m + k2;
      idx[q++] = a; idx[q++] = c; idx[q++] = b;
      idx[q++] = b; idx[q++] = c; idx[q++] = d;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(V, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(N, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(UV, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.userData.path = path;
  g.userData.arcLengthMm = arcLen;
  return g;
}

/**
 * A strap mesh that re-sweeps itself when the wrist ellipse changes by more
 * than `tol` metres on either axis, and hides the GLB's own strap nodes.
 */
export class LiveStrap {
  constructor(material, profile = null, opts = {}) {
    this.material = material;
    this.profile = profile;
    this.opts = opts;
    this.tol = opts.tol ?? 0.0004;        // 0.4 mm: below the visible threshold
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    this.mesh.renderOrder = 1;
    this.mesh.frustumCulled = false;
    this.last = null;
    this.hidden = [];
  }
  /**
   * Hide the baked strap in a loaded GLB.  Lab GLBs name the strap node
   * (bracelet_strap / bracelet_*); the hand-authored sample is one node whose
   * strap is told apart by its materials (croc_leather, strap_underside,
   * keeper_leather).  Three.js splits a multi-material primitive into child
   * meshes that keep the material name, so both are matched here.
   */
  takeOverFrom(watchScene) {
    const byNode = /^(bracelet_|strap|band|keeper|buckle|clasp)/i;
    const byMat = /(strap|leather|keeper|bracelet|band)/i;
    watchScene.traverse(o => {
      if (!o.isMesh) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      const hit = byNode.test(o.name || '') || mats.some(m => m && byMat.test(m.name || ''));
      if (hit) { o.visible = false; this.hidden.push(o); }
    });
    return this.hidden.length;
  }
  release() { for (const o of this.hidden) o.visible = true; this.hidden = []; }
  /** Update for an ellipse {across, thick} in metres and skinTop (m, negative). */
  update(ellipse, skinTopM) {
    if (!ellipse) return false;
    const rx = ellipse.across * 1000, rz = ellipse.thick * 1000, st = skinTopM * 1000;
    if (this.last && Math.abs(this.last.rx - rx) < this.tol * 1000 && Math.abs(this.last.rz - rz) < this.tol * 1000
        && Math.abs(this.last.st - st) < this.tol * 1000) return false;
    const g = buildStrapGeometry(rx, rz, st, this.profile, this.opts);
    this.mesh.geometry.dispose();
    this.mesh.geometry = g;
    this.last = { rx, rz, st };
    return true;
  }
  /** Worst clearance error vs the design gap, mm (acceptance: ≤ 2 mm). */
  clearanceError() {
    if (!this.last) return null;
    const g = this.mesh.geometry;
    const o = Object.assign({}, STRAP_DEFAULTS, this.opts);
    const T = (t) => (this.profile ? profileAt(this.profile.arc_mm, this.profile.thickness_mm, t) : null) ?? o.thickness;
    const c = clearanceMm(g.userData.path, this.last.rx, this.last.rz, this.last.st, T);
    // ignore the first/last 12% of the loop: those run up to the lugs, off the skin by design
    const n = c.length, a = Math.round(0.12 * n), b = n - a;
    let worst = 0;
    for (let i = a; i < b; i++) worst = Math.max(worst, Math.abs(c[i] - o.gap));
    return worst;
  }
}
