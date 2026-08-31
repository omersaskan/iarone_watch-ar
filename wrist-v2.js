// Wrist pose estimation for the wear-mode watch model.
//
// Wear-model frame after tools/build.py export:
//   +Y = dial normal / away from skin
//   -Z = 12 o'clock and the forearm direction toward the hand
//   +X = crown side / across the wrist
//
// The previous AR path treated +X as the forearm axis. That matched the display
// loop, but not the --wear geometry. It made the depth mask cut across the leather
// and is the main reason the case and strap could look like separate objects.

import * as THREE from 'three';

export const IDX = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

export const FIT = {
  alongForearm: 0.030,  // wrist crease -> case centre, toward the elbow
  outOfWrist: 0.0268,   // wrist centre -> dial plane; matches --wear geometry
};

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
    const k = (this.ratio * span) / dist;
    return { x: mx + dx * k, y: my + dy * k };
  }
}

export function dorsalFacesCameraFromImage(lm, isRight) {
  const p0 = lm[IDX.WRIST], p5 = lm[IDX.INDEX_MCP], p17 = lm[IDX.PINKY_MCP];
  const cz = (p5.x - p0.x) * (p17.y - p0.y) - (p5.y - p0.y) * (p17.x - p0.x);
  return isRight ? cz < 0 : cz > 0;
}

function mcpCentre(W) {
  return W[IDX.INDEX_MCP].clone()
    .add(W[IDX.MIDDLE_MCP])
    .add(W[IDX.RING_MCP])
    .add(W[IDX.PINKY_MCP])
    .multiplyScalar(0.25);
}

/** Build a model->camera-space wrist transform. */
export function wristPose(lm, world, isRight, cam, viewW, viewH, opts = {}) {
  if (!lm || !world || lm.length < 21 || world.length < 21) return null;
  const dorsalToCam = dorsalFacesCameraFromImage(lm, isRight);

  // MediaPipe world-z sign differs between implementations/devices. Try both and
  // retain the convention whose dorsal normal agrees with the 2-D hand winding.
  let best = null;
  for (const zs of [-1, 1]) {
    const W = world.map(p => new THREE.Vector3(p.x, -p.y, zs * p.z));
    const across = W[IDX.INDEX_MCP].clone().sub(W[IDX.PINKY_MCP]);
    // Use all four MCPs instead of only MIDDLE_MCP. This is noticeably steadier
    // when the user makes a fist and reduces the mask-angle jump seen in the video.
    const along = mcpCentre(W).sub(W[IDX.WRIST]);
    if (across.lengthSq() < 1e-8 || along.lengthSq() < 1e-8) continue;
    let n = new THREE.Vector3().crossVectors(across, along).normalize();
    if (!isRight) n.negate();
    const agrees = (n.z > 0) === dorsalToCam;
    const candidate = { W, across, along, n };
    if (agrees) { best = candidate; break; }
    if (!best) best = candidate;
  }
  if (!best) return null;

  const { W, along, n } = best;
  const u = along.clone().normalize(); // wrist -> fingers / hand direction

  let ey = n.clone();
  ey.sub(u.clone().multiplyScalar(ey.dot(u))).normalize();
  if (opts.flipSide) ey.negate();

  // Wear model: -Z points from wrist toward the hand; +X is across the wrist.
  let ez = u.clone().negate();
  let ex = new THREE.Vector3().crossVectors(ey, ez).normalize();

  // Rotate the whole watch 180 degrees on the dorsal plane to swap crown side.
  if (opts.rotate180) {
    ex.negate();
    ez.negate();
  }
  ey = new THREE.Vector3().crossVectors(ez, ex).normalize();

  const m = new THREE.Matrix4().makeBasis(ex, ey, ez);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(m);

  // Depth from metric hand width vs apparent pixel width.
  const handW = W[IDX.INDEX_MCP].distanceTo(W[IDX.PINKY_MCP]);
  const p5 = lm[IDX.INDEX_MCP], p17 = lm[IDX.PINKY_MCP];
  const dx = (p5.x - p17.x) * viewW, dy = (p5.y - p17.y) * viewH;
  const handPx = Math.hypot(dx, dy);
  if (handPx < 8 || handW < 0.02) return null;

  const vfov = THREE.MathUtils.degToRad(cam.fov);
  const fpx = (viewH / 2) / Math.tan(vfov / 2);
  let depth = handW * fpx / handPx;
  if (opts.depthFilter) depth = opts.depthFilter.update(depth);

  const a = opts.anchor ? opts.anchor.locate(lm) : lm[IDX.WRIST];
  const ndcX = 2 * a.x - 1, ndcY = 1 - 2 * a.y;
  const th = Math.tan(vfov / 2);
  const wrist = new THREE.Vector3(
    ndcX * th * cam.aspect * depth,
    ndcY * th * depth,
    -depth,
  );

  const position = wrist.clone()
    .addScaledVector(u, -FIT.alongForearm)
    .addScaledVector(ey, FIT.outOfWrist);

  return { quaternion, position, wrist, ex, ey, ez, u, depth, handW, dorsalToCam };
}

export class DepthFilter {
  constructor(alpha = 0.25, maxJump = 0.15) {
    this.alpha = alpha;
    this.maxJump = maxJump;
    this.value = null;
  }
  update(v) {
    if (v == null || !isFinite(v)) return this.value ?? v;
    if (this.value == null) { this.value = v; return v; }
    const d = v - this.value;
    const clamped = Math.abs(d) > this.maxJump
      ? this.value + Math.sign(d) * this.maxJump
      : v;
    this.value = this.alpha * clamped + (1 - this.alpha) * this.value;
    return this.value;
  }
  reset() { this.value = null; }
}

class LowPass {
  constructor() { this.a = 1; this.y = null; }
  setAlpha(a) { this.a = Math.min(Math.max(a, 0), 1); }
  filter(v) {
    this.y = this.y === null ? v : this.a * v + (1 - this.a) * this.y;
    return this.y;
  }
  reset() { this.y = null; }
}

export class PoseFilter {
  constructor(opts = {}) {
    this.minCutoff = opts.minCutoff ?? 1.0;
    this.beta = opts.beta ?? 60.0;
    this.dCutoff = opts.dCutoff ?? 1.0;
    this.maxMissing = opts.maxMissing ?? 15;
    this.p = null;
    this.q = null;
    this.dx = new LowPass(); this.dy = new LowPass(); this.dz = new LowPass();
    this.dr = new LowPass();
    this.last = null;
    this.missing = 0;
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
      this.p = pose.position.clone();
      this.q = pose.quaternion.clone();
      this.dx.reset(); this.dy.reset(); this.dz.reset(); this.dr.reset();
      this.last = ts;
      this.missing = 0;
      return { position: this.p, quaternion: this.q };
    }

    const dt = Math.max((ts - this.last) / 1000, 0.001);
    this.last = ts;
    this.missing = 0;

    const aD = this._alpha(this.dCutoff, dt);
    this.dx.setAlpha(aD); this.dy.setAlpha(aD); this.dz.setAlpha(aD);
    const vx = this.dx.filter((pose.position.x - this.p.x) / dt);
    const vy = this.dy.filter((pose.position.y - this.p.y) / dt);
    const vz = this.dz.filter((pose.position.z - this.p.z) / dt);
    this.p.lerp(
      pose.position,
      this._alpha(this.minCutoff + this.beta * Math.hypot(vx, vy, vz), dt),
    );

    this.dr.setAlpha(aD);
    const w = this.dr.filter(this.q.angleTo(pose.quaternion) / dt);
    this.q.slerp(
      pose.quaternion,
      this._alpha(this.minCutoff + this.beta * Math.abs(w), dt),
    );
    return { position: this.p, quaternion: this.q };
  }
  reset() {
    this.p = null; this.q = null; this.last = null; this.missing = 0;
    this.dx.reset(); this.dy.reset(); this.dz.reset(); this.dr.reset();
  }
}

export class TrackingState {
  constructor(maxMissing = 12) {
    this.maxMissing = maxMissing;
    this.missing = 0;
  }
  update(hasPose) {
    if (hasPose) {
      this.missing = 0;
      return { lost: false, opacity: 1 };
    }
    this.missing++;
    return {
      lost: this.missing >= this.maxMissing,
      opacity: Math.max(0, 1 - this.missing / this.maxMissing),
    };
  }
  reset() { this.missing = 0; }
}
