// Ensure the rebuilt AR GLB is not shadowed by the one-year immutable model cache.
// Preserve an explicit ?m= override for debugging/custom models.
const u = new URL(location.href);
if (!u.searchParams.has('m')) {
  u.searchParams.set('m', './iarone-watch-ar.glb?v=wear5');
  history.replaceState(null, '', u);
}

// Runtime is versioned too so the final spring-bar pinning logic is reloaded.
await import('./ar-runtime-v3.js?v=7');
