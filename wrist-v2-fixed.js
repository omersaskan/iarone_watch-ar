// Compatibility wrapper for ar-runtime-v3.
//
// ar-runtime-v3 already swaps MediaPipe handedness for the mirrored selfie UI.
// On the actual iPhone/Safari feed used in testing that produces the opposite
// palm/dorsal classification. Undo that extra swap only while the stage is
// mirrored, then delegate all pose math to the proven v2 implementation.
//
// IMPORTANT: ar.html maps the exact specifier "./wrist-v2.js" to this wrapper.
// Importing that exact specifier from here would therefore map back to this file
// and create a self-import / temporal-dead-zone error for IDX. The query string
// intentionally bypasses that exact import-map entry while loading the same base
// module from the server.

import * as base from './wrist-v2.js?core=1';

export const IDX = base.IDX;
export const FIT = base.FIT;
export const WristAnchor = base.WristAnchor;
export const DepthFilter = base.DepthFilter;
export const PoseFilter = base.PoseFilter;
export const TrackingState = base.TrackingState;
export const dorsalFacesCameraFromImage = base.dorsalFacesCameraFromImage;

export function wristPose(lm, world, isRight, cam, viewW, viewH, opts = {}) {
  const mirroredSelfie = document.getElementById('stage')?.classList.contains('mirror') === true;
  const correctedIsRight = mirroredSelfie ? !isRight : isRight;
  return base.wristPose(lm, world, correctedIsRight, cam, viewW, viewH, opts);
}
