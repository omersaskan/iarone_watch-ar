import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { HandLandmarker, FilesetResolver } from './vendor/tasks-vision/vision_bundle.mjs';
import { wristPose, PoseFilter, DepthFilter, TrackingState, WristAnchor, FIT, IDX } from './wrist-v2.js';
import { WRIST_BASE, WristSizeFilter, estimateWristRadii, createWristLoopDeformer } from './wrist-fit-v3.js';

const q = new URLSearchParams(location.search);
const DBG = q.get('dbg') === '1';
const SHOWARM = q.get('arm') === '1';
const MODEL_URL = q.get('m') || './iarone-watch-ar.glb';
const STORE_KEY = 'iarone-fit-v3';

const video = document.getElementById('cam');
const canvas = document.getElementById('gl');
const stage = document.getElementById('stage');
const hint = document.getElementById('hint');
const dbgEl = document.getElementById('arDbg');
if (DBG) dbgEl.style.display = 'block';

const renderer = new THREE.WebGLRenderer({ canvas, alpha:true, antialias:true, preserveDrawingBuffer:true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(58, 1, 0.01, 6);

function studioEnv() {
  const W=512,H=256,cv=document.createElement('canvas'); cv.width=W; cv.height=H;
  const c=cv.getContext('2d');
  const g=c.createLinearGradient(0,0,0,H);
  g.addColorStop(0,'#575450'); g.addColorStop(.46,'#8d8880'); g.addColorStop(.54,'#3a3836'); g.addColorStop(1,'#1a1918');
  c.fillStyle=g; c.fillRect(0,0,W,H);
  const key=c.createRadialGradient(150,46,4,150,46,110);
  key.addColorStop(0,'rgba(255,255,255,1)'); key.addColorStop(1,'rgba(255,255,255,0)'); c.fillStyle=key; c.fillRect(0,0,W,H);
  const fill=c.createRadialGradient(360,76,4,360,76,92);
  fill.addColorStop(0,'rgba(255,255,255,.62)'); fill.addColorStop(1,'rgba(255,255,255,0)'); c.fillStyle=fill; c.fillRect(0,0,W,H);
  const t=new THREE.CanvasTexture(cv); t.mapping=THREE.EquirectangularReflectionMapping; t.colorSpace=THREE.SRGBColorSpace; return t;
}
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromEquirectangular(studioEnv()).texture;
scene.environmentIntensity = 2.9;

const rig = new THREE.Group(); scene.add(rig);
const occMat = SHOWARM
  ? new THREE.MeshStandardMaterial({ color:0xc9a184, roughness:.85, metalness:0 })
  : new THREE.MeshBasicMaterial({ colorWrite:false, depthWrite:true, depthTest:true });
const occluder = new THREE.Mesh(new THREE.CylinderGeometry(1.04,.98,1,48,1,false), occMat);
occluder.renderOrder = -10;
const occGroup = new THREE.Group(); occGroup.add(occluder); rig.add(occGroup);

const armPivot = new THREE.Group();
const watchHolder = new THREE.Group();
armPivot.add(watchHolder); rig.add(armPivot);

let watch = null;
let deformer = null;
const mats = [];
new GLTFLoader().load(MODEL_URL, g => {
  watch = g.scene;
  watch.traverse(o => {
    if (!o.isMesh) return;
    o.renderOrder = 1;
    o.frustumCulled = false;
    o.material = o.material.clone();
    mats.push({ m:o.material, transparent:o.material.transparent, opacity:o.material.opacity });
  });
  watchHolder.add(watch);
  deformer = createWristLoopDeformer(watch);
  applyFit(true);
}, undefined, e => {
  hint.textContent = 'Bilek modeli yüklenemedi';
  hint.style.opacity = '1';
  console.error(e);
});

let curOpacity = 1;
function setWatchOpacity(v) {
  if (Math.abs(v-curOpacity) < .01) return;
  curOpacity = v;
  for (const e of mats) {
    if (v >= .999) { e.m.transparent=e.transparent; e.m.opacity=e.opacity; }
    else { e.m.transparent=true; e.m.opacity=e.opacity*v; }
    e.m.needsUpdate = true;
  }
}

const filter = new PoseFilter({ minCutoff:1.0, beta:60.0, dCutoff:1.0 });
const depthFilter = new DepthFilter(.25);
const sizeFilter = new WristSizeFilter(.10,.0018);
const anchor = new WristAnchor();
const tracking = new TrackingState(12);
const poseOpts = { anchor, depthFilter, flipSide:false, rotate180:false };
let wasLost = true;
let viewState = 'none';
let autoRadii = { rx:WRIST_BASE.rx, ry:WRIST_BASE.ry };

const ARM = { hand:.060, elbow:.090, inset:.0015 };
const DEF = { along:30, out:26.8, arm:1.00, roll:0, scale:1.00, flip:false, rotate180:false };
let CAL = { ...DEF };
try { Object.assign(CAL, JSON.parse(localStorage.getItem(STORE_KEY) || '{}')); } catch(e) {}
let liveFit = { rx:WRIST_BASE.rx, ry:WRIST_BASE.ry, out:FIT.outOfWrist };

function resetTracking(resetSize=false) {
  filter.reset(); depthFilter.reset(); anchor.reset(); tracking.reset(); wasLost=true;
  if (resetSize) {
    sizeFilter.reset();
    autoRadii={rx:WRIST_BASE.rx,ry:WRIST_BASE.ry};
    applyFit(true);
  }
}

function applyDynamicWrist(rawRx, rawRy, force=false) {
  const rx = THREE.MathUtils.clamp(rawRx * CAL.arm, .022, .037);
  const ry = THREE.MathUtils.clamp(rawRy * CAL.arm, .0165, .0285);
  const out = FIT.outOfWrist + (ry - WRIST_BASE.ry);
  liveFit = { rx, ry, out };

  armPivot.position.set(0, -out, -FIT.alongForearm);
  armPivot.rotation.set(0, 0, THREE.MathUtils.degToRad(CAL.roll));
  watchHolder.position.set(0, out, FIT.alongForearm);

  const len = ARM.hand + ARM.elbow;
  occGroup.position.set(0, -out, -FIT.alongForearm);
  occluder.rotation.set(Math.PI/2, 0, 0);
  occluder.scale.set(Math.max(.018,rx-ARM.inset), len, Math.max(.014,ry-ARM.inset));
  occluder.position.set(0, 0, (ARM.elbow-ARM.hand)/2);

  if (deformer) deformer.setRadii(rx, ry, force);
  return out;
}

function applyFit(force=false) {
  poseOpts.flipSide = !!CAL.flip;
  poseOpts.rotate180 = !!CAL.rotate180;
  FIT.alongForearm = CAL.along/1000;
  FIT.outOfWrist = CAL.out/1000;
  if (watch) watch.scale.setScalar(CAL.scale);
  applyDynamicWrist(autoRadii.rx, autoRadii.ry, force);
}
applyFit();

let stream = null;
let facing = 'user';
async function openCamera(mode) {
  if (stream) stream.getTracks().forEach(t=>t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio:false,
    video:{ facingMode:{ideal:mode}, width:{ideal:1280}, height:{ideal:720} },
  });
  video.srcObject = stream;
  await video.play();
  facing=mode;
  stage.classList.toggle('mirror', mode==='user');
}

let landmarker = null;
async function initTracker() {
  const fileset = await FilesetResolver.forVisionTasks('./vendor/tasks-vision/wasm');
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions:{ modelAssetPath:'./models/hand_landmarker.task', delegate:'GPU' },
    runningMode:'VIDEO', numHands:1,
    minHandDetectionConfidence:.55, minHandPresenceConfidence:.55, minTrackingConfidence:.55,
  });
}

function resize() {
  const w=stage.clientWidth,h=stage.clientHeight;
  renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();
}
addEventListener('resize',resize); resize();

function fitVideoCrop() {
  const vw=video.videoWidth,vh=video.videoHeight,cw=stage.clientWidth,ch=stage.clientHeight;
  if (!vw || !vh) return {sx:1,sy:1,ox:0,oy:0};
  const scale=Math.max(cw/vw,ch/vh),dw=vw*scale,dh=vh*scale;
  return { sx:dw/cw, sy:dh/ch, ox:(dw-cw)/2/dw, oy:(dh-ch)/2/dh };
}
function mapLandmarks(lm) {
  const c=fitVideoCrop();
  return lm.map(p=>({x:(p.x-c.ox)*c.sx,y:(p.y-c.oy)*c.sy,z:p.z}));
}

// MediaPipe handedness is defined for mirrored selfie imagery. We run inference on
// the raw (unmirrored) <video> and mirror only with CSS, so the selfie camera label
// must be swapped before using handedness to resolve palm-vs-dorsal winding.
function effectiveIsRight(categoryName) {
  const reportedRight = categoryName === 'Right';
  return facing === 'user' ? !reportedRight : reportedRight;
}

let running=false,lastVideoTime=-1,frames=0,t0=performance.now();
function loop() {
  if (!running) return;
  requestAnimationFrame(loop);
  let measured=null,fresh=false,info='';

  if (landmarker && video.readyState>=2 && video.currentTime!==lastVideoTime) {
    lastVideoTime=video.currentTime;
    fresh=true;
    const res=landmarker.detectForVideo(video,performance.now());
    const hands=res.handednesses || res.handedness || [];

    if (res.landmarks?.length && res.worldLandmarks?.length) {
      const cat=hands[0]?.[0]?.categoryName || 'Right';
      const isRight=effectiveIsRight(cat);
      const world=res.worldLandmarks[0];
      measured=wristPose(mapLandmarks(res.landmarks[0]),world,isRight,camera,stage.clientWidth,stage.clientHeight,poseOpts);

      if (measured && !measured.dorsalToCam) {
        // Never leave just the far-side strap visible on a palm view. That was the
        // black C-shape in the supplied video and looked like a second model.
        viewState='palm';
        measured=null;
        info=cat+'→'+(isRight?'R':'L')+' palm';
      } else if (measured) {
        viewState='dorsal';
        const rawSize=estimateWristRadii(world,IDX);
        autoRadii=sizeFilter.update(rawSize);
        const dynamicOut=applyDynamicWrist(autoRadii.rx,autoRadii.ry);
        measured.position.addScaledVector(measured.ey,dynamicOut-FIT.outOfWrist);
        measured.wristRx=liveFit.rx;
        measured.wristRy=liveFit.ry;
        info=cat+'→'+(isRight?'R':'L')+' d='+measured.depth.toFixed(2)+'m wrist='+
          (liveFit.rx*2000).toFixed(0)+'×'+(liveFit.ry*2000).toFixed(0)+'mm';
      } else {
        viewState='none';
        info=cat+' (poz yok)';
      }
    } else {
      viewState='none';
      info='el yok';
    }
  }

  let sm=filter.p ? {position:filter.p,quaternion:filter.q} : null;
  if (fresh) {
    if (viewState==='palm') {
      // Palm-side rejection is intentional, not tracking loss: hide immediately
      // instead of keeping the previous case/strap pose for the grace period.
      filter.reset(); tracking.reset(); wasLost=true; sm=null; setWatchOpacity(1);
    } else {
      sm=filter.push(measured);
      const st=tracking.update(!!measured);
      if (st.lost) {
        if (!wasLost) resetTracking(false);
        sm=null;
      } else {
        wasLost=false;
        setWatchOpacity(st.opacity);
      }
    }
  }

  if (viewState==='palm') {
    rig.visible=false;
    hint.textContent='Elinin sırtını kameraya çevir';
    hint.style.opacity='1';
  } else if (sm) {
    rig.position.copy(sm.position);
    rig.quaternion.copy(sm.quaternion);
    rig.visible=true;
    hint.style.opacity='0';
  } else {
    rig.visible=false;
    setWatchOpacity(1);
    hint.textContent='Bileğini kameraya göster';
    hint.style.opacity='1';
  }

  renderer.render(scene,camera);
  frames++;
  if (frames%20===0) {
    const el=(performance.now()-t0)/1000;
    document.getElementById('fps').textContent=(frames/el).toFixed(0)+' fps';
    if (DBG) dbgEl.textContent=info+'\n'+(sm?'p '+sm.position.toArray().map(v=>v.toFixed(3)).join(' '):viewState);
  }
}

document.getElementById('start').onclick=async()=>{
  const err=document.getElementById('gateErr'); err.textContent='';
  try {
    await openCamera(facing);
    await initTracker();
    document.getElementById('gate').remove();
    running=true; t0=performance.now(); frames=0; loop();
  } catch(e) {
    err.textContent='Kamera açılamadı: '+(e?.message||e)+' — HTTPS üzerinden açılması gerekir.';
  }
};

document.getElementById('flip').onclick=async()=>{
  resetTracking(true); viewState='none';
  try { await openCamera(facing==='user'?'environment':'user'); } catch(e) {}
};

document.getElementById('shot').onclick=()=>{
  const w=stage.clientWidth,h=stage.clientHeight,out=document.createElement('canvas');
  out.width=w*2; out.height=h*2;
  const c=out.getContext('2d');
  if (facing==='user') { c.translate(out.width,0); c.scale(-1,1); }
  const vw=video.videoWidth,vh=video.videoHeight;
  if (vw) {
    const s=Math.max(out.width/vw,out.height/vh);
    c.drawImage(video,(out.width-vw*s)/2,(out.height-vh*s)/2,vw*s,vh*s);
  }
  c.drawImage(renderer.domElement,0,0,out.width,out.height);
  out.toBlob(b=>{
    const a=document.createElement('a');
    a.href=URL.createObjectURL(b); a.download='iarone-bilek.png'; a.click();
    setTimeout(()=>URL.revokeObjectURL(a.href),4000);
  },'image/png');
};

const fitPanel=document.getElementById('fit');
const SL={along:'f_along',out:'f_out',arm:'f_arm',roll:'f_roll',scale:'f_scale'};
const UNIT={along:' mm',out:' mm',arm:'×',roll:'°',scale:'×'};
function fitString() {
  return `along=${CAL.along} out=${CAL.out} wristBias=${CAL.arm} roll=${CAL.roll} scale=${CAL.scale} auto=${(liveFit.rx*2000).toFixed(0)}x${(liveFit.ry*2000).toFixed(0)}mm flip=${CAL.flip?1:0} rotate180=${CAL.rotate180?1:0}`;
}
function syncPanel() {
  for(const k in SL) {
    document.getElementById(SL[k]).value=CAL[k];
    document.getElementById('v_'+k).textContent=CAL[k]+UNIT[k];
  }
  document.getElementById('f_str').textContent=fitString();
}
function saveFit() { try { localStorage.setItem(STORE_KEY,JSON.stringify(CAL)); } catch(e) {} }
for(const k in SL) document.getElementById(SL[k]).addEventListener('input',ev=>{
  CAL[k]=parseFloat(ev.target.value); applyFit(true); syncPanel(); saveFit();
});
document.getElementById('f_flip').onclick=()=>{ CAL.flip=!CAL.flip; applyFit(true); syncPanel(); resetTracking(false); saveFit(); };
document.getElementById('f_rotate').onclick=()=>{ CAL.rotate180=!CAL.rotate180; applyFit(true); syncPanel(); resetTracking(false); saveFit(); };
document.getElementById('f_reset').onclick=()=>{
  CAL={...DEF}; sizeFilter.reset(); autoRadii={rx:WRIST_BASE.rx,ry:WRIST_BASE.ry};
  applyFit(true); syncPanel(); resetTracking(false); saveFit();
};
document.getElementById('f_copy').onclick=async()=>{
  const t=fitString();
  try { await navigator.clipboard.writeText(t); hint.textContent='Kopyalandı'; }
  catch(e){ hint.textContent=t; }
  hint.style.opacity='1'; setTimeout(()=>hint.style.opacity='0',1800);
};
document.getElementById('f_close').onclick=()=>fitPanel.classList.remove('on');
document.getElementById('calib').onclick=()=>{ fitPanel.classList.toggle('on'); syncPanel(); };
syncPanel();
