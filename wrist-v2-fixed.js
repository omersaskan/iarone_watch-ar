// Compatibility wrapper for ar-runtime-v3.
//
// Palm/dorsal handedness correction now has a single owner: ar-runtime-v3.
// The runtime converts MediaPipe's selfie-camera handedness once before calling
// wristPose(). This wrapper must therefore be a pure pass-through; applying a
// second mirror correction here cancels the runtime correction and makes the
// palm side look like the dorsal side (watch visible on the open palm).
//
// The query string intentionally bypasses ar.html's exact import-map entry for
// "./wrist-v2.js", avoiding self-import while also breaking stale module cache.

import * as base from './wrist-v2.js?core=2';

export const IDX = base.IDX;
export const FIT = base.FIT;
export const WristAnchor = base.WristAnchor;
export const DepthFilter = base.DepthFilter;
export const PoseFilter = base.PoseFilter;
export const TrackingState = base.TrackingState;
export const dorsalFacesCameraFromImage = base.dorsalFacesCameraFromImage;

export function wristPose(lm, world, isRight, cam, viewW, viewH, opts = {}) {
  return base.wristPose(lm, world, isRight, cam, viewW, viewH, opts);
}
