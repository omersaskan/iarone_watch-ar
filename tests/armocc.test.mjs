import { test } from 'node:test';
import assert from 'node:assert/strict';
import { armProfile, ArmProfileFilter, buildArmGeometry, ArmOccluder, STATIONS } from '../armocc.js';
import { WRIST_PRIOR } from '../wristfit.js';

// A vertical arm: hand block on top (width handPx), forearm below widening
// linearly from wristPx at the crease to wristPx*taper at 2 hand-widths,
// with the arm's centre drifting `drift` px to the right down the arm and a
// sleeve (mask off) below `sleeveY`.
function synth(W, H, { handPx = 90, wristPx = 60, taper = 1.35, drift = 0, sleeveY = Infinity } = {}) {
  const mask = new Uint8Array(W * H);
  const cx0 = W / 2, creaseY = Math.round(H * 0.42);
  for (let y = 0; y < H; y++) {
    if (y >= sleeveY) break;
    const below = Math.max(0, y - creaseY);
    const t = Math.min(1, below / (2 * handPx));
    const w = y < creaseY ? handPx : wristPx * (1 + (taper - 1) * t);
    const cx = cx0 + drift * t;
    for (let x = Math.round(cx - w / 2); x < Math.round(cx + w / 2); x++) if (x >= 0 && x < W) mask[y * W + x] = 1;
  }
  const lm = new Array(21).fill(null).map(() => ({ x: 0.5, y: 0.2, z: 0 }));
  lm[0] = { x: 0.5, y: creaseY / H, z: 0 };
  lm[9] = { x: 0.5, y: (creaseY - 0.6 * handPx) / H, z: 0 };
  lm[5] = { x: (cx0 - handPx / 2) / W, y: lm[9].y, z: 0 };
  lm[17] = { x: (cx0 + handPx / 2) / W, y: lm[9].y, z: 0 };
  return { mask, lm };
}

const W = 320, H = 640, mPerPx = 0.0009;

test('the profile reads the forearm widening toward the elbow, in metres along -X', () => {
  const { mask, lm } = synth(W, H);
  const p = armProfile(mask, W, H, lm, mPerPx);
  assert.ok(p && p.n >= 6, `stations ${p && p.n}`);
  // stations toward the elbow are negative X; widths grow with distance
  // on the forearm side of the crease (stations <= 0) the width grows toward the elbow
  const fore = p.stations.map((_, i) => i).filter(i => p.stations[i] <= 1e-9)
    .sort((a, b) => p.stations[b] - p.stations[a]);
  assert.ok(fore.length >= 4);
  for (let i = 1; i < fore.length; i++) {
    assert.ok(p.stations[fore[i]] < p.stations[fore[i - 1]]);
    assert.ok(p.halfWidth[fore[i]] >= p.halfWidth[fore[i - 1]] - 1e-6);
  }
  // at the crease the half width is 30 px = 27 mm
  const i0 = p.stations.findIndex(s => Math.abs(s) < 1e-9);
  assert.ok(i0 >= 0 && Math.abs(p.halfWidth[i0] - 0.027) < 0.0015, `crease ${p.halfWidth[i0]}`);
});

test('an arm whose centre drifts is followed (centreShift), not clipped', () => {
  const { mask, lm } = synth(W, H, { drift: 40 });
  const p = armProfile(mask, W, H, lm, mPerPx);
  assert.ok(p && p.n >= 5);
  const far = p.stations.indexOf(Math.min(...p.stations));
  assert.ok(Math.abs(p.centreShift[far]) > 0.01, `shift ${p.centreShift[far]}`);
  assert.ok(Math.abs(p.halfWidth[far] - 0.027 * 1.35 * 0.9) < 0.006);
});

test('a sleeve cuts the profile short but still yields a tube when >= 3 stations remain', () => {
  const { mask, lm } = synth(W, H, { sleeveY: Math.round(H * 0.42) + 70 });
  const p = armProfile(mask, W, H, lm, mPerPx);
  assert.ok(p && p.n >= 3 && p.n < STATIONS.length, `n=${p && p.n}`);
  // with the mask gone right below the crease only the hand-side stations
  // remain: fewer than 3 forearm readings is still "a tube" (hand side + crease),
  // but an empty mask is not
  const none = armProfile(new Uint8Array(W * H), W, H, lm, mPerPx);
  assert.equal(none, null);
});

test('the lofted tube matches the profile and extends past both ends', () => {
  const prof = { stations: [0.02, 0, -0.03, -0.06], halfWidth: [0.027, 0.027, 0.03, 0.033], centreShift: [0, 0, 0, 0], n: 4 };
  const occ = new ArmOccluder({});
  occ.update(prof);
  assert.ok(occ.active);
  assert.ok(Math.abs(occ.halfWidthAt(0) - 0.027) < 1e-6);
  assert.ok(Math.abs(occ.halfWidthAt(-0.045) - 0.0315) < 1e-4);   // interpolated
  assert.ok(occ.halfWidthAt(-0.3) > 0.033);                        // continues past the elbow end
  const g = buildArmGeometry(prof);
  const pos = g.getAttribute('position');
  let maxY = 0, maxZ = 0;
  for (let i = 0; i < pos.count; i++) { maxY = Math.max(maxY, Math.abs(pos.getY(i))); maxZ = Math.max(maxZ, Math.abs(pos.getZ(i))); }
  // Y is the thickness: width x anatomical ratio
  assert.ok(Math.abs(maxY / maxZ - WRIST_PRIOR.thickOverAcross) < 0.02);
});

test('the filter smooths and clamps a jumping profile', () => {
  const f = new ArmProfileFilter(0.5, 0.004);
  const a = { stations: [0, -0.03, -0.06], halfWidth: [0.027, 0.03, 0.033], centreShift: [0, 0, 0], n: 3 };
  f.push(a);
  const b = { ...a, halfWidth: [0.06, 0.06, 0.06] };
  const s = f.push(b);
  assert.ok(s.halfWidth[0] < 0.030, 'a 33 mm jump is clamped to 4 mm then halved');
  assert.equal(f.push(null), s);
});
