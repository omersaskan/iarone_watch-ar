"""Patch the Blender USD stage to Y-up and package it as a Quick Look .usdz."""
import os
import shutil
import sys
import zipfile

from pxr import Usd, UsdGeom, UsdUtils, Sdf

src_dir = os.path.abspath(sys.argv[1])
out = os.path.abspath(sys.argv[2])
usdc = os.path.join(src_dir, "model.usdc")

stage = Usd.Stage.Open(usdc)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = stage.GetPrimAtPath("/root")
if root and root.IsValid():
    stage.SetDefaultPrim(root)
stage.GetRootLayer().Save()

if os.path.exists(out):
    os.remove(out)
ok = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(usdc), out)
print("packaged:", ok, out, os.path.getsize(out) if os.path.exists(out) else 0)

# Quick Look requires stored (uncompressed) entries aligned to 64 bytes;
# UsdUtils does this already, but verify rather than assume.
z = zipfile.ZipFile(out)
bad = [i.filename for i in z.infolist() if i.compress_type != zipfile.ZIP_STORED]
print("entries:", len(z.infolist()), "compressed:", bad or "none")
for i in z.infolist():
    print("  %-40s %9d  offset %d (mod64=%d)" %
          (i.filename, i.file_size, i.header_offset, i.header_offset % 64))

st = Usd.Stage.Open(out)
print("upAxis:", UsdGeom.GetStageUpAxis(st), " metersPerUnit:", UsdGeom.GetStageMetersPerUnit(st))
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
r = bb.ComputeWorldBound(st.GetPseudoRoot()).ComputeAlignedRange()
mn, mx = r.GetMin(), r.GetMax()
print("size mm:", [round((mx[i] - mn[i]) * 1000, 1) for i in range(3)],
      " min:", [round(v, 4) for v in mn])
