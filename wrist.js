// Wrist pose estimation from MediaPipe hand landmarks.
//
// The exported watch sits in a Y-up frame: dial normal = +Y, 12 o-clock = -Z,
// crown and strap-loop axis = +X, loop hanging to -Y.
//
// HOW A WATCH ACTUALLY SITS (measured off a real wrist photo, not reasoned):
// the strap leaves the lugs along the 12-6 axis and wraps the arm, and the strap
// axis came out 77.5 degrees from the forearm axis - i.e. the strap encircles the
// arm perpendicular to it.  So the FOREARM RUNS THROUGH THE STRAP LOOP, along the
// watch's 3-9 (crown) axis = model +X.  The lugs are separated ACROSS the wrist,
// not along it.  Everything else follows from that.

import * as THREE from 'three';

export const IDX = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

// Anatomical fit, in metres.  outOfWrist MUST match the wear-mode strap geometry
// in build.py:  wrist centre z = P["wrist"].skin_top - P["wrist"].rz  (= -26.8 mm).
// The strap only clears the skin by ~2.6 mm, so a couple of millimetres of error
// here buries it inside the forearm.
export const FIT = {
  alongForearm: 0.030,     // wrist crease -> centre of the case, down the arm
  outOfWrist: 0.0268,      // wrist centre -> dial plane
};

/**
 * Stable 2-D wrist anchor.
 *
 * The raw WRIST landmark is the jitteriest point on the hand: it sits on the
 * silhouette boundary and slides with finger flexion.  The three metacarpal
 * knuckles are far steadier, and the wrist-to-knuckle distance is an anatomical
 * constant relative to knuckle span.  So we keep the *direction* from the knuckle
 * centroid to the wrist (low noise) and replace the *distance* with a temporally
 * smoothed ratio (kills the radial jitter that a raw blend cannot).
 */
export class WristAnchor {
  constructor(alpha = 0.06) {
    this.alpha = alpha;
    this.ratio = null;
  }
  reset() { this.ratio = null; }
  locate(lm) {
    const p0 = lm[IDX.WRIST], p5 = lm[IDX.INDEX_MCP];
    const p9 = lm[IDX.MIDDLE_MCP], p17 = lm[IDX.PINKY_MCP];
    const mx = (p5.x + p9.x + p17.x) / 3;
    const my = (p5.y + p9.y + p17.y) / 3;
    const span = Math.hypot(p5.x - p17.x, p5.y - p17.y);
    let dx = p0.x - mx, dy = p0.y - my;
    const dist = Math.hypot(dx, dy);
    if (span < 1e-6 || dist < 1e-6) return { x: p0.x, y: p0.y };
    const r = dist / span;
    this.ratio = this.ratio == null ? r : this.ratio + this.alpha * (r - this.ratio);
    const k = (this.ratio * span) / dist;      // rescale to the smoothed length
    return { x: mx + dx * k, y: my + dy * k };
  }
}

/**
 * Resolve the sign ambiguity in the metric world landmarks.
 * MediaPipe world landmarks are metric but their z axis direction is not worth
 * trusting blind, so we cross-check against a purely 2-D cue: the winding of
 * (wrist, index MCP, pinky MCP) in image space says unambiguously whether we are
 * looking at the back of the hand or the palm.
 *
 * Derivation (image coords: x right, y DOWN).  Anatomical position: the thumb is
 * the lateral digit, so a RIGHT hand held palm-to-camera shows its thumb on the
 * image LEFT.  Then index MCP is left of pinky MCP and both sit above the wrist:
 *   cz = (-)(-) - (-)(+) > 0.
 * So cz > 0 means the PALM faces the camera for a right hand.  Getting this
 * backwards puts the dial on the inside of the wrist.
 */
export function dorsalFacesCameraFromImage(lm, isRight) {
  const p0 = lm[IDX.WRIST], p5 = lm[IDX.INDEX_MCP], p17 = lm[IDX.PINKY_MCP];
  const cz = (p5.x - p0.x) * (p17.y - p0.y) - (p5.y - p0.y) * (p17.x - p0.x);
  return isRight ? cz < 0 : cz > 0;
}

/**
 * Build the wrist basis in three.js camera space (x right, y up, -z forward).
 * opts: { anchor: WristAnchor, depthFilter: DepthFilter }
 * Returns {quaternion, position, ...} or null when the pose is unusable.
 */
export function wristPose(lm, world, isRight, cam, viewW, viewH, opts = {}) {
  if (!lm || !world || lm.length < 21 || world.length < 21) return null;
  const dorsalToCam = dorsalFacesCameraFromImage(lm, isRight);

  // try both z conventions for the metric landmarks and keep the one that agrees
  let best = null;
  for (const zs of [-1, 1]) {
    const W = world.map(p => new THREE.Vector3(p.x, -p.y, zs * p.z));
    const across = W[IDX.INDEX_MCP].clone().sub(W[IDX.PINKY_MCP]);   // pinky -> index
    const along = W[IDX.MIDDLE_MCP].clone().sub(W[IDX.WRIST]);       // wrist -> fingers
    if (across.lengthSq() < 1e-8 || along.lengthSq() < 1e-8) continue;
    let n = new THREE.Vector3().crossVectors(across, along).normalize();
    if (!isRight) n.negate();
    const agrees = (n.z > 0) === dorsalToCam;
    if (agrees) { best = { W, across, along, n }; break; }
    if (!best) best = { W, across, along, n };
  }
  if (!best) return null;

  const { W, along, n } = best;

  // Frame: ex = forearm (model X, the axis the loop encircles), ey = dorsal.
  const u = along.clone().normalize();                      // toward the fingers
  let ey = n.clone();
  ey.sub(u.clone().multiplyScalar(ey.dot(u))).normalize();  // keep ey perpendicular to u
  // Manual override: negating the dorsal normal is a true 180 degree turn about
  // the forearm, which moves the watch to the other side of the wrist.
  if (opts.flipSide) ey.negate();
  const ex = (opts.crownToElbow ? u.clone().negate() : u.clone());
  const ez = new THREE.Vector3().crossVectors(ex, ey).normalize();
  ex.crossVectors(ey, ez).normalize();                      // re-orthogonalise

  const m = new THREE.Matrix4().makeBasis(ex, ey, ez);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(m);

  // ---- depth from apparent size -------------------------------------------
  const handW = W[IDX.INDEX_MCP].distanceTo(W[IDX.PINKY_MCP]);          // metres
  const p5 = lm[IDX.INDEX_MCP], p17 = lm[IDX.PINKY_MCP];
  const dx = (p5.x - p17.x) * viewW, dy = (p5.y - p17.y) * viewH;
  const handPx = Math.hypot(dx, dy);
  if (handPx < 8 || handW < 0.02) return null;

  const vfov = THREE.MathUtils.degToRad(cam.fov);
  const fpx = (viewH / 2) / Math.tan(vfov / 2);
  let depth = handW * fpx / handPx;
  if (opts.depthFilter) depth = opts.depthFilter.update(depth);

  // wrist placed on its own view ray at that depth, using the steadier anchor
  const a = opts.anchor ? opts.anchor.locate(lm) : lm[IDX.WRIST];
  const ndcX = 2 * a.x - 1, ndcY = 1 - 2 * a.y;
  const th = Math.tan(vfov / 2);
  const wrist = new THREE.Vector3(ndcX * th * cam.aspect * depth, ndcY * th * depth, -depth);

  const position = wrist.clone()
    .addScaledVector(u, -FIT.alongForearm)      // slide back down the forearm
    .addScaledVector(ey, FIT.outOfWrist);       // lift onto the skin

  return { quaternion, position, wrist, ex, ey, ez, u, depth, handW, dorsalToCam };
}

/** EMA depth filter with a spike clamp: depth is the noisiest axis by far. */
export class DepthFilter {
  constructor(alpha = 0.25, maxJump = 0.15) {
    this.alpha = alpha; this.maxJump = maxJump; this.value = null;
  }
  update(v) {
    if (v == null || !isFinite(v)) return this.value ?? v;
    if (this.value == null) { this.value = v; return v; }
    const d = v - this.value;
    const clamped = Math.abs(d) > this.maxJump ? this.value + Math.sign(d) * this.maxJump : v;
    this.value = this.alpha * clamped + (1 - this.alpha) * this.value;
    return this.value;
  }
  reset() { this.value = null; }
}

class LowPass {
  constructor() { this.a = 1; this.y = null; }
  setAlpha(a) { this.a = Math.min(Math.max(a, 0), 1); }
  filter(v) { this.y = this.y === null ? v : this.a * v + (1 - this.a) * this.y; return this.y; }
  reset() { this.y = null; }
}

/**
 * 1-Euro filter (Casiez et al., CHI 2012) over position and rotation.
 * A fixed-alpha EMA forces a choice between jitter when still and lag when moving;
 * this adapts the cutoff to the measured speed and avoids both.
 */
export class PoseFilter {
  constructor(opts = {}) {
    this.minCutoff = opts.minCutoff ?? 1.0;   // lower = steadier when still
    this.beta = opts.beta ?? 60.0;           // higher = less lag when moving
    // tuned by sweep: beta 40-80 beats a fixed EMA on BOTH jitter and lag because
    // the signal is in metres, so hand speeds are small numbers and beta must be large
    this.dCutoff = opts.dCutoff ?? 1.0;
    this.maxMissing = opts.maxMissing ?? 15;
    this.p = null; this.q = null;
    this.dx = new LowPass(); this.dy = new LowPass(); this.dz = new LowPass();
    this.dr = new LowPass();
    this.last = null; this.missing = 0;
  }
  _alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  push(pose, ts = performance.now()) {
    if (!pose) {
      this.missing++;
      return this.p ? { position: this.p, quaternion: this.q } : null;
    }
    if (!this.p || this.missing > this.maxMissing) {
      this.p = pose.position.clone(); this.q = pose.quaternion.clone();
      this.dx.reset(); this.dy.reset(); this.dz.reset(); this.dr.reset();
      this.last = ts; this.missing = 0;
      return { position: this.p, quaternion: this.q };
    }
    const dt = Math.max((ts - this.last) / 1000, 0.001);
    this.last = ts; this.missing = 0;

    const aD = this._alpha(this.dCutoff, dt);
    this.dx.setAlpha(aD); this.dy.setAlpha(aD); this.dz.setAlpha(aD);
    const vx = this.dx.filter((pose.position.x - this.p.x) / dt);
    const vy = this.dy.filter((pose.position.y - this.p.y) / dt);
    const vz = this.dz.filter((pose.position.z - this.p.z) / dt);
    this.p.lerp(pose.position, this._alpha(this.minCutoff + this.beta * Math.hypot(vx, vy, vz), dt));

    this.dr.setAlpha(aD);
    const w = this.dr.filter(this.q.angleTo(pose.quaternion) / dt);
    this.q.slerp(pose.quaternion, this._alpha(this.minCutoff + this.beta * Math.abs(w), dt));

    return { position: this.p, quaternion: this.q };
  }
  reset() {
    this.p = null; this.q = null; this.last = null; this.missing = 0;
    this.dx.reset(); this.dy.reset(); this.dz.reset(); this.dr.reset();
  }
}

/** Grace period after tracking loss, with a fade so the watch does not pop out. */
export class TrackingState {
  constructor(maxMissing = 12) { this.maxMissing = maxMissing; this.missing = 0; }
  update(hasPose) {
    if (hasPose) { this.missing = 0; return { lost: false, opacity: 1 }; }
    this.missing++;
    return { lost: this.missing >= this.maxMissing,
             opacity: Math.max(0, 1 - this.missing / this.maxMissing) };
  }
  reset() { this.missing = 0; }
}

/** Synthetic landmarks so the pose maths can be checked without a camera.
 *  The hand is built in metres, placed at `dist` in front of the camera and
 *  projected with the real camera parameters, so the resulting pose is realistic. */
export function demoHand(t = 0, isRight = true, palmToCam = false, cam = null,
                         dist = 0.32, poseType = 'default') {
  // +1 puts the thumb on the image RIGHT for a right hand, i.e. the back of the
  // hand toward the camera - which is what a wrist try-on actually shows.  The
  // old -1 generated a palm view while the detector was also inverted, so the two
  // errors cancelled and the synthetic test could never catch either.
  const spread = isRight ? 1 : -1;
  const base = [
    [0.000, 0.000, 0.000],
    [0.030 * spread, 0.020, 0.004], [0.045 * spread, 0.045, 0.006],
    [0.052 * spread, 0.068, 0.008], [0.058 * spread, 0.088, 0.010],
    [0.038 * spread, 0.078, 0.000],
    [0.040 * spread, 0.112, 0.000], [0.041 * spread, 0.134, 0.000], [0.042 * spread, 0.152, 0.000],
    [0.014 * spread, 0.083, 0.000],
    [0.015 * spread, 0.120, 0.000], [0.016 * spread, 0.144, 0.000], [0.017 * spread, 0.163, 0.000],
    [-0.008 * spread, 0.081, 0.000],
    [-0.009 * spread, 0.115, 0.000], [-0.010 * spread, 0.138, 0.000], [-0.011 * spread, 0.155, 0.000],
    [-0.030 * spread, 0.074, 0.000],
    [-0.034 * spread, 0.103, 0.000], [-0.036 * spread, 0.122, 0.000], [-0.038 * spread, 0.138, 0.000],
  ];
  const P = {
    default: { p: 0.35, y: 0.50, r: 0.25 },
    still:   { p: 0.02, y: 0.03, r: 0.02 },
    tilt:    { p: 0.85, y: 0.25, r: 0.15 },
    turn:    { p: 0.15, y: 1.15, r: 0.10 },
  }[poseType] || { p: 0.35, y: 0.50, r: 0.25 };

  const e = new THREE.Euler(Math.sin(t * 0.37) * P.p,
                            Math.sin(t * 0.6) * P.y + (palmToCam ? Math.PI : 0),
                            Math.sin(t * 0.23) * P.r, 'XYZ');
  const R = new THREE.Matrix4().makeRotationFromEuler(e);

  const pts = base.map(([x, y, z]) => new THREE.Vector3(x, y, z).applyMatrix4(R));
  const c = pts.reduce((a, p) => a.add(p), new THREE.Vector3()).multiplyScalar(1 / pts.length);
  const centred = pts.map(p => p.clone().sub(c));
  const camPts = centred.map(p => p.clone().add(new THREE.Vector3(0, 0.085, -dist)));

  const vfov = THREE.MathUtils.degToRad(cam ? cam.fov : 58);
  const aspect = cam ? cam.aspect : 0.8125;
  const th = Math.tan(vfov / 2);
  const lm = camPts.map(p => {
    const d = -p.z;
    return { x: (p.x / (th * aspect * d) + 1) / 2, y: (1 - p.y / (th * d)) / 2, z: p.z + dist };
  });
  // MediaPipe world landmarks: metric, origin at the hand centre, y down, z away
  const world = centred.map(p => ({ x: p.x, y: -p.y, z: -p.z }));
  return { lm, world };
}
