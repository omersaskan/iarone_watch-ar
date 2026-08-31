import * as THREE from 'three';

// Skin ellipse used when the AR model was authored.
export const WRIST_BASE = {
  rx: 0.0285,
  ry: 0.0215,
  centreY: -0.0268,
  strapGap: 0.00135,
};

// Exported wear frame: +Y is dial/dorsal normal and +/-Z is the forearm axis.
// The authored spring bars are at build Y=+/-20.4 mm -> exported Z=-/+20.4 mm.
const STRAP_LUG_Z = 0.0204;
const STRAP_PIN = 0.0080;
const STRAP_BLEND_END = 0.0140;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function smooth01(t) {
  t = clamp(t, 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * Estimate the wrist cross-section from MediaPipe metric hand landmarks.
 * HandLandmarker has only one wrist point, not the two silhouette edges, so the
 * wrist cannot be measured directly. Palm width + wrist-to-MCP length are used
 * as a stable anthropometric proxy and deliberately clamped to plausible wrists.
 */
export function estimateWristRadii(world, IDX) {
  if (!world || world.length < 21) {
    return { rx: WRIST_BASE.rx, ry: WRIST_BASE.ry, confidence: 0 };
  }

  const p = world.map(v => new THREE.Vector3(v.x, v.y, v.z));
  const mcp = p[IDX.INDEX_MCP].clone()
    .add(p[IDX.MIDDLE_MCP])
    .add(p[IDX.RING_MCP])
    .add(p[IDX.PINKY_MCP])
    .multiplyScalar(0.25);

  const palmWidth = p[IDX.INDEX_MCP].distanceTo(p[IDX.PINKY_MCP]);
  const palmLength = mcp.distanceTo(p[IDX.WRIST]);

  if (!isFinite(palmWidth) || !isFinite(palmLength) || palmWidth < 0.025 || palmLength < 0.025) {
    return { rx: WRIST_BASE.rx, ry: WRIST_BASE.ry, confidence: 0 };
  }

  const sizeProxy = 0.5 * (palmWidth + palmLength);
  const wristAcross = clamp(sizeProxy * 0.75, 0.046, 0.074);
  const rx = wristAcross * 0.5;
  const ry = clamp(rx * (WRIST_BASE.ry / WRIST_BASE.rx), 0.0170, 0.0280);

  const confidence = clamp(
    1 - Math.abs(palmWidth - palmLength) / Math.max(palmWidth + palmLength, 1e-6),
    0.35,
    1,
  );
  return { rx, ry, confidence, palmWidth, palmLength };
}

/** Slow EMA with per-frame jump clamp; wrist size should not breathe with tracking. */
export class WristSizeFilter {
  constructor(alpha = 0.10, maxJump = 0.0018) {
    this.alpha = alpha;
    this.maxJump = maxJump;
    this.rx = null;
    this.ry = null;
  }
  _step(prev, next) {
    if (prev == null) return next;
    const d = clamp(next - prev, -this.maxJump, this.maxJump);
    return prev + this.alpha * d;
  }
  update(raw) {
    if (!raw) return this.value();
    this.rx = this._step(this.rx, raw.rx);
    this.ry = this._step(this.ry, raw.ry);
    return this.value();
  }
  value() {
    return {
      rx: this.rx ?? WRIST_BASE.rx,
      ry: this.ry ?? WRIST_BASE.ry,
    };
  }
  reset() {
    this.rx = null;
    this.ry = null;
  }
}

function materialNames(material) {
  if (Array.isArray(material)) return material.map(m => m?.name || '');
  return [material?.name || ''];
}

function isDeformableStrapMesh(mesh) {
  const names = materialNames(mesh.material);
  return names.some(name =>
    name === 'croc_leather' ||
    name === 'strap_underside'
  );
}

/**
 * Bind a non-destructive wrist-fit deformer to the loaded AR GLB.
 *
 * Only leather primitives are deformable. Case, lugs, spring bars and all metal
 * remain rigid. The deformation weight is measured from the spring-bar along the
 * exported forearm axis rather than from dial depth:
 *   0..8 mm from spring bar   -> 0% deformation
 *   8..14 mm                  -> smooth 0..100% blend
 *   beyond 14 mm              -> full wrist fit
 *
 * This exactly preserves the authored lug connection while allowing only the
 * hidden/downward strap portion to adapt to thin or thick wrists.
 */
export function createWristLoopDeformer(root) {
  const targets = [];

  root.traverse(o => {
    const attr = o?.geometry?.attributes?.position;
    if (!o.isMesh || !attr || !attr.array || !isDeformableStrapMesh(o)) return;
    o.frustumCulled = false;
    targets.push({
      mesh: o,
      attr,
      original: new Float32Array(attr.array),
    });
  });

  if (!targets.length) {
    console.warn('[IARONE] no leather strap meshes found for wrist deformation');
  }

  let lastRx = null;
  let lastRy = null;

  function setRadii(rx, ry, force = false) {
    rx = clamp(rx, 0.022, 0.037);
    ry = clamp(ry, 0.0165, 0.0285);

    if (!force && lastRx != null &&
        Math.abs(rx - lastRx) < 0.00035 && Math.abs(ry - lastRy) < 0.00035) {
      return false;
    }
    lastRx = rx;
    lastRy = ry;

    const sx = (rx + WRIST_BASE.strapGap) / (WRIST_BASE.rx + WRIST_BASE.strapGap);
    const sy = (ry + WRIST_BASE.strapGap) / (WRIST_BASE.ry + WRIST_BASE.strapGap);

    for (const t of targets) {
      const a = t.attr.array;
      const src = t.original;
      for (let i = 0; i < src.length; i += 3) {
        const x0 = src[i];
        const y0 = src[i + 1];
        const z0 = src[i + 2];

        // Exported +/-Z runs along the two strap arms. Use absolute forearm
        // distance so the 12 and 6 sides get identical pinning.
        const alongFromLug = Math.max(0, Math.abs(z0) - STRAP_LUG_Z);
        const w = smooth01((alongFromLug - STRAP_PIN) / (STRAP_BLEND_END - STRAP_PIN));
        const fx = 1 + (sx - 1) * w;
        const fy = 1 + (sy - 1) * w;

        a[i] = x0 * fx;
        a[i + 1] = WRIST_BASE.centreY + (y0 - WRIST_BASE.centreY) * fy;
        a[i + 2] = z0;
      }
      t.attr.needsUpdate = true;
      t.mesh.geometry.computeVertexNormals();
      t.mesh.geometry.computeBoundingBox();
      t.mesh.geometry.computeBoundingSphere();
    }
    return true;
  }

  function reset() {
    lastRx = null;
    lastRy = null;
    for (const t of targets) {
      t.attr.array.set(t.original);
      t.attr.needsUpdate = true;
      t.mesh.geometry.computeVertexNormals();
      t.mesh.geometry.computeBoundingBox();
      t.mesh.geometry.computeBoundingSphere();
    }
  }

  return { setRadii, reset, targetCount: targets.length };
}
