// Forearm occluder shaped by the camera, not by a cylinder.
//
// The far side of the strap must hide behind the arm, and the case must slip
// behind the wrist's bone as the wrist turns.  A fixed ellipse cylinder gets
// that wrong at every frame in a different place: a real forearm is narrow at
// the wrist, widens toward the elbow, has the ulna's bump, and changes
// silhouette as it rotates.  Where the cylinder is too thin the strap floats;
// where it is too fat it swallows the strap.
//
// This module reads the forearm's WIDTH PROFILE off the body-skin mask the
// wrist measurement already computes (wristfit.js), at several stations along
// the forearm axis, converts it to metres with the hand's metric scale, and
// lofts a depth-only tube through those widths.  The mask is per frame, so
// the occluder follows the arm as it turns.
//
// Frame.  The tube is built in the rig's armPivot frame: +X along the forearm
// (hand at +X), +Y out of the back of the wrist, origin at the wrist centre.
// Thickness (Y) is not visible to the camera; it is the width times the
// anatomical ratio, like the wrist ellipse.

import * as THREE from '#three';
import { IDX } from './wrist.js';
import { WRIST_PRIOR } from './wristfit.js';

/** Where along the forearm the profile is read, in hand-widths from the wrist
 *  crease (negative = toward the hand).  Beyond ~2 hand-widths the mask often
 *  meets a sleeve, so the profile stops there and the tube continues straight. */
export const STATIONS = [-0.45, -0.2, 0.0, 0.25, 0.5, 0.8, 1.15, 1.5, 1.9];

/**
 * Width (px) of the mask across the forearm at one station.
 * Returns null when the station is off the mask (sleeve, frame edge, no arm).
 */
function widthAt(mask, W, H, cx, cy, nx, ny, maxR, isFloat) {
  const on = (x, y) => {
    const xi = Math.round(x), yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= W || yi >= H) return false;
    const v = mask[yi * W + xi];
    return isFloat ? v > 0.5 : v > 0;
  };
  if (!on(cx, cy)) return null;
  let r1 = 0, r2 = 0;
  while (r1 < maxR && on(cx + nx * (r1 + 1), cy + ny * (r1 + 1))) r1++;
  while (r2 < maxR && on(cx - nx * (r2 + 1), cy - ny * (r2 + 1))) r2++;
  const w = r1 + r2 + 1;
  if (w < 4 || w >= 2 * maxR - 2) return null;
  // also return the centre offset so a bent/offset arm keeps its axis
  return { w, offset: (r1 - r2) / 2 };
}

/**
 * Forearm width profile from a mask, metres, by station.
 * lm: hand landmarks (normalised) in the mask's pixel frame; mPerPx: from the
 * hand's metric width.  Returns { stations: [m along forearm], halfWidth: [m],
 * centreShift: [m across], n: stations found } or null when fewer than 3
 * stations are readable (then the caller keeps the ellipse cylinder).
 */
export function armProfile(mask, W, H, lm, mPerPx, isFloat = false) {
  const p0 = lm[IDX.WRIST], p9 = lm[IDX.MIDDLE_MCP];
  const ax = (p0.x - p9.x) * W, ay = (p0.y - p9.y) * H;      // toward the elbow, px
  const al = Math.hypot(ax, ay);
  if (al < 1e-6 || !(mPerPx > 0)) return null;
  const ux = ax / al, uy = ay / al;
  const nx = -uy, ny = ux;
  const span = Math.hypot((lm[IDX.INDEX_MCP].x - lm[IDX.PINKY_MCP].x) * W,
                          (lm[IDX.INDEX_MCP].y - lm[IDX.PINKY_MCP].y) * H);
  if (span < 8) return null;
  const maxR = 2.2 * span;
  const stations = [], halfWidth = [], centreShift = [];
  for (const s of STATIONS) {
    const d = s * span;
    const r = widthAt(mask, W, H, p0.x * W + ux * d, p0.y * H + uy * d, nx, ny, maxR, isFloat);
    if (!r) continue;
    // station distance: positive toward the elbow, which is -X in the rig (hand at +X)
    stations.push(-d * mPerPx);
    halfWidth.push(0.5 * r.w * mPerPx);
    centreShift.push(r.offset * mPerPx);
  }
  if (stations.length < 3) return null;
  return { stations, halfWidth, centreShift, n: stations.length };
}

/** Exponential smoothing of a profile, station by station, with outlier clamp. */
export class ArmProfileFilter {
  constructor(alpha = 0.2, maxJump = 0.004) { this.alpha = alpha; this.maxJump = maxJump; this.p = null; }
  reset() { this.p = null; }
  push(prof) {
    if (!prof) return this.p;
    if (!this.p || this.p.n !== prof.n) { this.p = { stations: prof.stations.slice(), halfWidth: prof.halfWidth.slice(),
                                                     centreShift: prof.centreShift.slice(), n: prof.n }; return this.p; }
    for (let i = 0; i < prof.n; i++) {
      const dw = Math.max(-this.maxJump, Math.min(this.maxJump, prof.halfWidth[i] - this.p.halfWidth[i]));
      this.p.halfWidth[i] += this.alpha * dw;
      const dc = Math.max(-this.maxJump, Math.min(this.maxJump, prof.centreShift[i] - this.p.centreShift[i]));
      this.p.centreShift[i] += this.alpha * dc;
      this.p.stations[i] += this.alpha * (prof.stations[i] - this.p.stations[i]);
    }
    return this.p;
  }
}

/**
 * Loft a closed tube along X through the profile.  Each ring is an ellipse:
 * Z half-axis = measured half width (across the wrist), Y half-axis = width x
 * the anatomical thickness ratio.  The tube is extended straight toward the
 * elbow and capped at the hand end with a hemisphere so the knuckles' side of
 * the strap is covered too.  Returns a BufferGeometry in the armPivot frame.
 */
export function buildArmGeometry(profile, opts = {}) {
  const ratio = opts.thickOverAcross ?? WRIST_PRIOR.thickOverAcross;
  const extendElbow = opts.extendElbow ?? 0.16;   // m past the last station
  const handCap = opts.handCap ?? 0.052;          // m toward the hand (ARM.hand)
  const segs = opts.segments ?? 32;
  // stations are negative toward the elbow; sort from hand (+X) to elbow (-X)
  const idx = profile.stations.map((_, i) => i).sort((a, b) => profile.stations[b] - profile.stations[a]);
  const xs = idx.map(i => profile.stations[i]);
  const ws = idx.map(i => profile.halfWidth[i]);
  const cs = idx.map(i => profile.centreShift[i]);
  // extend: hand end repeats the first width, elbow end continues the last slope gently
  // the hand end continues at the crease width (stations on the hand block are
  // wider than the wrist and would flare the tube); the elbow end eases out
  let wCrease = Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] <= 0.015 && xs[i] >= -0.02) wCrease = Math.min(wCrease, ws[i]);
  if (!isFinite(wCrease)) wCrease = ws[0];
  const wsHand = ws.map((w, i) => (xs[i] > 0.015 ? Math.min(w, wCrease * 1.05) : w));
  const X = [xs[0] + handCap, ...xs, xs[xs.length - 1] - extendElbow];
  const Wd = [wsHand[0], ...wsHand, wsHand[wsHand.length - 1] * 1.08];
  const C = [cs[0], ...cs, cs[cs.length - 1]];
  const n = X.length;
  const V = new Float32Array(n * segs * 3);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < segs; k++) {
      const a = (2 * Math.PI * k) / segs;
      const j = (i * segs + k) * 3;
      V[j] = X[i];
      V[j + 1] = Wd[i] * ratio * Math.cos(a);        // Y: thickness
      V[j + 2] = C[i] + Wd[i] * Math.sin(a);         // Z: width, shifted to the arm's own centre
    }
  }
  const idxArr = new Uint32Array((n - 1) * segs * 6);
  let q = 0;
  for (let i = 0; i < n - 1; i++) for (let k = 0; k < segs; k++) {
    const k2 = (k + 1) % segs;
    const a = i * segs + k, b = i * segs + k2, c = (i + 1) * segs + k, d = (i + 1) * segs + k2;
    idxArr[q++] = a; idxArr[q++] = b; idxArr[q++] = c;
    idxArr[q++] = b; idxArr[q++] = d; idxArr[q++] = c;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(V, 3));
  g.setIndex(new THREE.BufferAttribute(idxArr, 1));
  g.computeVertexNormals();
  g.userData.X = X; g.userData.W = Wd;
  return g;
}

/**
 * The mask-shaped occluder.  `update(profile)` rebuilds the tube when the
 * profile is fresh; `fallback()` hides it so the ellipse cylinder shows.
 */
export class ArmOccluder {
  constructor(material, opts = {}) {
    this.opts = opts;
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    this.mesh.renderOrder = -10;
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    this.cap = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), material);
    this.cap.renderOrder = -10; this.cap.visible = false;
    this.group = new THREE.Group();
    this.group.add(this.mesh, this.cap);
    this.active = false;
  }
  update(profile) {
    if (!profile) { this.fallback(); return false; }
    this.mesh.geometry.dispose();
    const g = buildArmGeometry(profile, this.opts);
    this.mesh.geometry = g;
    const ratio = this.opts.thickOverAcross ?? WRIST_PRIOR.thickOverAcross;
    // the cap closes the HAND end: size it to the narrowest station near the
    // crease (the hand block itself is wider than the wrist and must not read
    // as a bulb), and sit it at the hand end of the tube
    const X = g.userData.X, Wd = g.userData.W;
    let wCap = Infinity;
    for (let i = 0; i < X.length; i++) if (X[i] <= 0.015 && X[i] >= -0.02) wCap = Math.min(wCap, Wd[i]);
    if (!isFinite(wCap)) wCap = Wd[0];
    this.cap.scale.set(wCap * ratio, wCap * ratio, wCap);
    this.cap.position.set(X[0], 0, 0);
    this.mesh.visible = true; this.cap.visible = true; this.active = true;
    return true;
  }
  fallback() { this.mesh.visible = false; this.cap.visible = false; this.active = false; }
  /** Half width (m) of the occluder at x (m along the forearm), for tests/debug. */
  halfWidthAt(x) {
    const X = this.mesh.geometry.userData.X, W = this.mesh.geometry.userData.W;
    if (!X) return null;
    for (let i = 0; i < X.length - 1; i++) {
      if (x <= X[i] && x >= X[i + 1]) {
        const t = (X[i] - x) / Math.max(X[i] - X[i + 1], 1e-9);
        return W[i] * (1 - t) + W[i + 1] * t;
      }
    }
    return x > X[0] ? W[0] : W[W.length - 1];
  }
}
