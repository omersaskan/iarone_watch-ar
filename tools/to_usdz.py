"""Blender 5.1 headless: GLB -> USD for iOS AR Quick Look.

    blender -b -P to_usdz.py -- <in.glb> <out_dir>

Blender always writes a Z-up stage; Quick Look wants Y-up, so the model is
pre-rotated here and the stage metadata is patched afterwards by pack_usdz.py.
"""
import sys
import os
import math

import bpy
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
src = os.path.abspath(argv[0])
out_dir = os.path.abspath(argv[1])
os.makedirs(out_dir, exist_ok=True)
dst = os.path.join(out_dir, "model.usdc")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)

objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
print("imported meshes:", [(o.name, len(o.data.polygons)) for o in objs])

for o in bpy.context.scene.objects:
    o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# glTF Y-up -> Blender Z-up on import; undo it so the geometry is Y-up again and
# the stage can simply declare upAxis = Y.
R = Matrix.Rotation(math.radians(-90.0), 4, "X")
for o in objs:
    o.matrix_world = R @ o.matrix_world
bpy.context.view_layer.update()
for o in bpy.context.scene.objects:
    o.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Quick Look culls back faces and renders thin transparency badly
for m in bpy.data.materials:
    m.use_backface_culling = False
    m.blend_method = "OPAQUE"
    if m.node_tree:
        for n in m.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED" and "Alpha" in n.inputs:
                n.inputs["Alpha"].default_value = 1.0

bpy.ops.wm.usd_export(filepath=dst, export_textures_mode="NEW",
                      root_prim_path="/root", convert_world_material=False)
print("wrote", dst, os.path.getsize(dst), "bytes")
