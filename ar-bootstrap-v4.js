// Ensure the rebuilt wear GLB is not shadowed by the previous one-year immutable cache.
// Preserve an explicit ?m= override for debugging/custom models.
const u = new URL(location.href);
if (!u.searchParams.has('m')) {
  u.searchParams.set('m', './iarone-watch-ar.glb?v=wear2');
  history.replaceState(null, '', u);
}

await import('./ar-runtime-v3.js?v=5');
