// Per-frame wrist cross-section from the camera, so the strap wraps THIS arm.
//
// The hand landmarker gives the hand's metric width (index MCP to pinky MCP,
// metres) and its pixel width, i.e. a mm-per-pixel scale at the hand's depth.
// A person/body-skin segmentation mask gives the forearm's silhouette.  The
// wrist's pixel width, read off the mask along the line through the wrist
// crease perpendicular to the forearm, times that scale, is the wrist's real
// width across (the X radius of the forearm ellipse, ARM.across).
//
// The wrist's thickness (back-to-front, ARM.thick) is not visible to a camera
// looking at the back of the hand.  It is taken from the measured width by the
// anatomical ratio the sample wrist had (21.5 / 28.5).  When the hand turns
// edge-on the mask width becomes the thickness instead; the dorsal normal's
// tilt says how much of each we see, and the two are mixed accordingly.
//
// Without a segmenter (model file missing, old device) nothing breaks: the
// estimate is null and the caller keeps the manual "Bilek çevresi" slider.

import * as THREE from '#three';
import { IDX } from './wrist.js';

/** Anatomy of the wrist the sample watch was fitted on (README "Kalibrasyon"). */
export const WRIST_PRIOR = {
  across: 0.0285,          // half-width of the forearm at the wrist, m
  thick: 0.0215,           // half-thickness, m
  thickOverAcross: 0.0215 / 0.0285,
};

/** Smoothed, outlier-clamped wrist ellipse (half-axes in metres). */
export class WristEllipse {
  constructor(alpha = 0.12, maxJump = 0.004) {
    this.alpha = alpha; this.maxJump = maxJump;
    this.across = null; this.thick = null; this.confidence = 0; this.samples = 0;
  }
  reset() { this.across = null; this.thick = null; this.confidence = 0; this.samples = 0; }
  push(m) {
    if (!m || !isFinite(m.across) || m.across <= 0) { this.confidence *= 0.9; return this.value(); }
    const a = Math.min(Math.max(m.across, 0.018), 0.045);   // 36..90 mm wide: a human wrist
    const t = Math.min(Math.max(m.thick, 0.013), 0.036);
    if (this.across == null) { this.across = a; this.thick = t; }
    else {
      const da = Math.max(-this.maxJump, Math.min(this.maxJump, a - this.across));
      const dt = Math.max(-this.maxJump, Math.min(this.maxJump, t - this.thick));
      this.across += this.alpha * da; this.thick += this.alpha * dt;
    }
    this.samples++;
    this.confidence = Math.min(1, 0.8 * this.confidence + 0.2 * (m.confidence ?? 1));
    return this.value();
  }
  value() {
    if (this.across == null) return null;
    return { across: this.across, thick: this.thick, confidence: this.confidence, samples: this.samples };
  }
}

/**
 * Read the forearm's width from a segmentation mask along the wrist line.
 *
 * mask: Uint8Array/Float32Array of W*H, non-zero (or > 0.5) = skin/body.
 * lm: hand landmarks in the SAME pixel frame as the mask (normalised 0..1).
 * Returns the wrist width in pixels, or null when the line finds no clean run.
 */
export function wristWidthPx(mask, W, H, lm, isFloat = false) {
  const p0 = lm[IDX.WRIST], p9 = lm[IDX.MIDDLE_MCP];
  // forearm direction in the image: from the knuckles toward the wrist and on
  const ax = p0.x - p9.x, ay = p0.y - p9.y;
  const al = Math.hypot(ax, ay);
  if (al < 1e-6) return null;
  const ux = ax / al, uy = ay / al;             // along the forearm (toward the elbow)
  const nx = -uy, ny = ux;                      // across the wrist
  // sample a little past the crease, where the forearm is at its narrowest
  const span = Math.hypot((lm[IDX.INDEX_MCP].x - lm[IDX.PINKY_MCP].x) * W,
                          (lm[IDX.INDEX_MCP].y - lm[IDX.PINKY_MCP].y) * H);
  const back = 0.35 * span;                     // ~1/3 of a hand width down the arm
  const cx = p0.x * W + ux * back, cy = p0.y * H + uy * back;
  const on = (x, y) => {
    const xi = Math.round(x), yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= W || yi >= H) return false;
    const v = mask[yi * W + xi];
    return isFloat ? v > 0.5 : v > 0;
  };
  if (!on(cx, cy)) return null;                 // the line must start inside the arm
  const maxR = 1.6 * span;
  let r1 = 0, r2 = 0;
  while (r1 < maxR && on(cx + nx * (r1 + 1), cy + ny * (r1 + 1))) r1++;
  while (r2 < maxR && on(cx - nx * (r2 + 1), cy - ny * (r2 + 1))) r2++;
  const w = r1 + r2 + 1;
  if (w < 6 || w >= 2 * maxR - 2) return null;  // nothing, or the mask ran off (not an arm)
  return w;
}

/**
 * One wrist measurement from a pose (wrist.js wristPose result) and a mask.
 * pose.handW is the hand's metric width; the pixel width is re-derived here
 * from the same landmarks so the two scales agree.
 */
export function measureWrist(pose, lm, mask, W, H, isFloat = false) {
  if (!pose || !lm || !mask) return null;
  const px = wristWidthPx(mask, W, H, lm, isFloat);
  if (px == null) return null;
  const handPx = Math.hypot((lm[IDX.INDEX_MCP].x - lm[IDX.PINKY_MCP].x) * W,
                            (lm[IDX.INDEX_MCP].y - lm[IDX.PINKY_MCP].y) * H);
  if (handPx < 8 || !(pose.handW > 0.02)) return null;
  const mPerPx = pose.handW / handPx;
  const seen = 0.5 * px * mPerPx;               // half of what the silhouette shows, m
  // how much of the width vs the thickness the camera sees: the dorsal normal's
  // z component is 1 looking straight at the back of the hand (we see the full
  // width) and 0 edge-on (we see the thickness)
  const k = Math.min(1, Math.max(0, Math.abs(pose.ey ? pose.ey.z : 1)));
  const r = WRIST_PRIOR.thickOverAcross;
  // seen = k*across + (1-k)*thick, thick = r*across  =>  across = seen / (k + (1-k) r)
  const across = seen / (k + (1 - k) * r);
  const thick = r * across;
  const confidence = 0.4 + 0.6 * k;             // the face-on view is the trustworthy one
  return { across, thick, confidence, px, mPerPx };
}

/**
 * Lazily create a MediaPipe ImageSegmenter for the body/skin mask.  Resolves
 * to null when the model file is not deployed, so the page works without it.
 */
export async function createWristSegmenter(visionBundle, wasmPath, modelPath) {
  try {
    const head = await fetch(modelPath, { method: 'HEAD' });
    if (!head.ok) return null;
    const fileset = await visionBundle.FilesetResolver.forVisionTasks(wasmPath);
    return await visionBundle.ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelPath, delegate: 'GPU' },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
  } catch (e) {
    console.warn('wrist segmenter unavailable:', e && e.message ? e.message : e);
    return null;
  }
}

/** selfie_multiclass categories: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 other. */
export const SKIN_CATEGORIES = new Set([2, 3]);

/** Turn a category mask into a 0/1 Uint8Array of skin pixels. */
export function skinMask(categoryMask) {
  const src = categoryMask.getAsUint8Array();
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = SKIN_CATEGORIES.has(src[i]) ? 1 : 0;
  return out;
}

/** The ellipse the strap and the occluder use: measured when confident, else the prior scaled by the slider. */
export function effectiveEllipse(est, manualScale = 1.0, minConfidence = 0.45, minSamples = 8) {
  if (est && est.confidence >= minConfidence && est.samples >= minSamples) {
    return { across: est.across, thick: est.thick, source: 'measured', confidence: est.confidence };
  }
  return { across: WRIST_PRIOR.across * manualScale, thick: WRIST_PRIOR.thick * manualScale,
           source: 'prior', confidence: 0 };
}

export const _three = THREE; // keep the import live for tree-shaking bundlers
