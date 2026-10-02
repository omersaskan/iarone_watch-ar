// node --test tests/  — strap wrap maths without a browser.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loopPathWrist, clearanceMm, buildStrapGeometry, strapSection, STRAP_DEFAULTS } from '../strap.js';
import { WristEllipse, effectiveEllipse, wristWidthPx, measureWrist, WRIST_PRIOR } from '../wristfit.js';

const SAMPLE = { rx: 28.5, rz: 21.5, skinTop: -5.3 };   // build.py P["wrist"]

function worstClearanceError(rx, rz, skinTop) {
  const path = loopPathWrist(rx, rz, skinTop);
  const c = clearanceMm(path, rx, rz, skinTop, STRAP_DEFAULTS.thickness);
  const n = c.length, a = Math.round(0.12 * n);
  let worst = 0;
  for (let i = a; i < n - a; i++) worst = Math.max(worst, Math.abs(c[i] - STRAP_DEFAULTS.gap));
  return worst;
}

test('the loop ends sit on the spring bars, drifting along the forearm as it wraps', () => {
  const p = loopPathWrist(SAMPLE.rx, SAMPLE.rz, SAMPLE.skinTop);
  const n = p.length / 3;
  assert.ok(Math.abs(p[1] - STRAP_DEFAULTS.y_start) < 0.3, `12 end y=${p[1]}`);
  assert.ok(Math.abs(p[3 * (n - 1) + 1] + STRAP_DEFAULTS.y_start) < 0.3, `6 end y=${p[3 * (n - 1) + 1]}`);
  // the two ends are above the skin (lifted to the lugs), the bottom is under the arm
  assert.ok(p[2] > SAMPLE.skinTop - SAMPLE.rz + SAMPLE.rz * 0.9);
  const mid = Math.floor(n / 2);
  assert.ok(p[3 * mid + 2] < SAMPLE.skinTop - 2 * SAMPLE.rz + 5, 'bottom of the loop passes under the arm');
});

test('on the sample wrist the strap clears the skin by the design gap (regression vs build.py)', () => {
  assert.ok(worstClearanceError(SAMPLE.rx, SAMPLE.rz, SAMPLE.skinTop) <= 2.0);
});

test('acceptance: across thin, medium and thick wrists the wrap error stays <= 2 mm', () => {
  const sizes = [[22, 16.5], [25, 19], [28.5, 21.5], [32, 24], [36, 27], [40, 30]];
  for (const [rx, rz] of sizes) {
    const skinTop = -(SAMPLE.skinTop < 0 ? 5.3 : 0);
    const e = worstClearanceError(rx, rz, skinTop);
    assert.ok(e <= 2.0, `rx=${rx} rz=${rz}: worst clearance error ${e.toFixed(2)} mm`);
  }
});

test('the swept geometry is a closed tube in metres with sane extents', () => {
  const g = buildStrapGeometry(SAMPLE.rx, SAMPLE.rz, SAMPLE.skinTop, null);
  const pos = g.getAttribute('position');
  assert.equal(g.getIndex().count, (STRAP_DEFAULTS.n - 1) * 24 * 6);
  let minY = Infinity, maxY = -Infinity, minX = Infinity, maxX = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    minY = Math.min(minY, pos.getY(i)); maxY = Math.max(maxY, pos.getY(i));
    minX = Math.min(minX, pos.getX(i)); maxX = Math.max(maxX, pos.getX(i));
  }
  // rig frame: Y out of the wrist. The loop spans from above the dial-plane lugs
  // down under the arm: ~2*rz + gap + thickness
  assert.ok(maxY - minY > 0.045 && maxY - minY < 0.065, `height ${(maxY - minY) * 1000} mm`);
  // X is the forearm: the loop drifts ~2*y_start along it
  assert.ok(maxX - minX > 0.035 && maxX - minX < 0.06, `length ${(maxX - minX) * 1000} mm`);
});

test('a Lab strap profile sets the section width and thickness along the arc', () => {
  const profile = { arc_mm: [0, 50, 100], width_mm: [20, 18, 16], thickness_mm: [2.6, 2.6, 2.6] };
  const g = buildStrapGeometry(SAMPLE.rx, SAMPLE.rz, SAMPLE.skinTop, profile);
  const pos = g.getAttribute('position');
  const m = 24, n = STRAP_DEFAULTS.n;
  // the section's width is the largest distance between any two of its 24 points
  const widthAt = (i) => {
    let w = 0;
    for (let a = 0; a < m; a++) for (let b = a + 1; b < m; b++) {
      const d = Math.hypot(pos.getX(i * m + a) - pos.getX(i * m + b), pos.getY(i * m + a) - pos.getY(i * m + b),
                           pos.getZ(i * m + a) - pos.getZ(i * m + b));
      w = Math.max(w, d);
    }
    return w * 1000;
  };
  const q = Math.round(0.25 * n), q3 = Math.round(0.75 * n);
  assert.ok(Math.abs(widthAt(q) - 19) < 0.6, `25%: ${widthAt(q)} mm`);
  assert.ok(Math.abs(widthAt(q3) - 17) < 0.6, `75%: ${widthAt(q3)} mm`);
});

test('strapSection is closed, rounded and the right size', () => {
  const s = strapSection(20, 2.6);
  assert.equal(s.length, 24);
  const xs = s.map(p => p[0]), zs = s.map(p => p[1]);
  assert.ok(Math.abs(Math.max(...xs) - 10) < 1e-9 && Math.abs(Math.min(...xs) + 10) < 1e-9);
  assert.ok(Math.abs(Math.max(...zs) - 1.3) < 1e-9 && Math.abs(Math.min(...zs) + 1.3) < 1e-9);
});

// ---- wrist measurement ------------------------------------------------------

test('WristEllipse smooths, clamps to human sizes and tracks confidence', () => {
  const e = new WristEllipse(0.5);
  assert.equal(e.push(null), null);
  e.push({ across: 0.030, thick: 0.022, confidence: 1 });
  for (let i = 0; i < 10; i++) e.push({ across: 0.030, thick: 0.022, confidence: 1 });
  const v = e.value();
  assert.ok(Math.abs(v.across - 0.030) < 1e-6 && v.samples === 11 && v.confidence > 0.9);
  e.push({ across: 0.300, thick: 0.2, confidence: 1 });          // a wild outlier is clamped
  assert.ok(e.value().across < 0.0345);
});

test('effectiveEllipse prefers a confident measurement and otherwise the scaled prior', () => {
  const meas = { across: 0.026, thick: 0.0196, confidence: 0.9, samples: 20 };
  assert.equal(effectiveEllipse(meas).source, 'measured');
  assert.equal(effectiveEllipse({ ...meas, samples: 2 }).source, 'prior');
  const pr = effectiveEllipse(null, 1.2);
  assert.ok(Math.abs(pr.across - WRIST_PRIOR.across * 1.2) < 1e-9 && pr.source === 'prior');
});

function synthMaskAndHand(W, H, wristPx, handPx) {
  // a vertical forearm of width wristPx below a hand of width handPx; hand at the top
  const mask = new Uint8Array(W * H);
  const cx = W / 2;
  for (let y = 0; y < H; y++) {
    const w = y < H * 0.4 ? handPx : wristPx;
    for (let x = Math.round(cx - w / 2); x < Math.round(cx + w / 2); x++) mask[y * W + x] = 1;
  }
  const lm = new Array(21).fill(null).map(() => ({ x: 0.5, y: 0.2, z: 0 }));
  lm[0]  = { x: 0.5, y: 0.42, z: 0 };                       // WRIST just below the hand block
  lm[9]  = { x: 0.5, y: 0.25, z: 0 };                       // MIDDLE_MCP above it
  lm[5]  = { x: (cx - handPx / 2) / W, y: 0.25, z: 0 };     // INDEX_MCP
  lm[17] = { x: (cx + handPx / 2) / W, y: 0.25, z: 0 };     // PINKY_MCP
  return { mask, lm };
}

test('wristWidthPx reads the forearm width off the mask along the wrist line', () => {
  const W = 320, H = 480;
  const { mask, lm } = synthMaskAndHand(W, H, 60, 90);
  const px = wristWidthPx(mask, W, H, lm);
  assert.ok(Math.abs(px - 60) <= 2, `got ${px}`);
});

test('measureWrist scales the pixel width by the hand metric width', () => {
  const W = 320, H = 480;
  const { mask, lm } = synthMaskAndHand(W, H, 60, 90);
  // hand 90 px wide = 0.081 m  =>  0.9 mm/px  =>  wrist 60 px = 54 mm across = 27 mm half
  const pose = { handW: 0.081, ey: { z: 1 } };
  const m = measureWrist(pose, lm, mask, W, H);
  assert.ok(Math.abs(m.across - 0.027) < 0.001, `across ${m.across}`);
  assert.ok(Math.abs(m.thick - 0.027 * WRIST_PRIOR.thickOverAcross) < 0.001);
  assert.ok(m.confidence > 0.9);
  // edge-on, the silhouette is the thickness: the same pixels mean a wider wrist
  const edge = measureWrist({ handW: 0.081, ey: { z: 0 } }, lm, mask, W, H);
  assert.ok(edge.across > m.across && edge.confidence < m.confidence);
});
