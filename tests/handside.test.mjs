import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '#three';
import { demoHand, dorsalFacesCameraFromImage, isRightHandFromImage, wristPose } from '../wrist.js';

const cam = new THREE.PerspectiveCamera(58, 0.8125, 0.01, 6);

test('face and hand are read from the landmarks, for all four hand/face combinations', () => {
  for (const isRight of [true, false]) for (const palm of [false, true]) {
    for (const t of [0, 0.7, 1.9]) {
      const d = demoHand(t, isRight, palm, cam, 0.3, 'still');
      const dorsal = dorsalFacesCameraFromImage(d.lm, d.world);
      assert.equal(dorsal, !palm, `right=${isRight} palm=${palm} t=${t}: dorsal`);
      assert.equal(isRightHandFromImage(d.lm, dorsal), isRight, `right=${isRight} palm=${palm} t=${t}: hand`);
    }
  }
});

test('a WRONG handedness label no longer flips the dial off the wrist', () => {
  for (const isRight of [true, false]) {
    const d = demoHand(0.5, isRight, false, cam, 0.3, 'still');
    const good = wristPose(d.lm, d.world, isRight, cam, 540, 760, {});
    const lied = wristPose(d.lm, d.world, !isRight, cam, 540, 760, {});     // label mirrored (front camera)
    assert.ok(good && lied);
    assert.ok(good.quaternion.angleTo(lied.quaternion) < 1e-6, 'pose must not depend on the label');
    assert.equal(lied.isRight, isRight);
    // the dial normal (rig +Y) must point toward the camera (+z) on the back of the hand
    const ey = new THREE.Vector3(0, 1, 0).applyQuaternion(good.quaternion);
    assert.ok(ey.z > 0.5, `dial faces the camera: ey.z=${ey.z.toFixed(2)}`);
  }
});

test('on the palm the dial faces away from the camera (worn on the back of the wrist)', () => {
  const d = demoHand(0.5, true, true, cam, 0.3, 'still');
  const p = wristPose(d.lm, d.world, true, cam, 540, 760, {});
  const ey = new THREE.Vector3(0, 1, 0).applyQuaternion(p.quaternion);
  assert.ok(ey.z < -0.5, `ey.z=${ey.z.toFixed(2)}`);
});
