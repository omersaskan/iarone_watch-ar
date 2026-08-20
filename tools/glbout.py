import io
import json
import struct

import numpy as np
from PIL import Image


def srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class GLB:
    def __init__(self, name="scene"):
        self.name = name
        self.bin = bytearray()
        self.bufferViews = []
        self.accessors = []
        self.images = []
        self.samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        self.textures = []
        self.materials = []
        self.prims = []
        self.exts = set()

    # ---- buffers -------------------------------------------------------
    def _pad(self, align=4):
        while len(self.bin) % align:
            self.bin.append(0)

    def _view(self, data, target=None, stride=None):
        self._pad(4)
        off = len(self.bin)
        self.bin.extend(data)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            bv["target"] = target
        if stride:
            bv["byteStride"] = stride
        self.bufferViews.append(bv)
        return len(self.bufferViews) - 1

    def _acc(self, arr, typ, comp, target, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr)
        bv = self._view(arr.tobytes(), target)
        a = {"bufferView": bv, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            a["min"] = [float(x) for x in np.atleast_1d(arr.min(axis=0))]
            a["max"] = [float(x) for x in np.atleast_1d(arr.max(axis=0))]
        if normalized:
            a["normalized"] = True
        self.accessors.append(a)
        return len(self.accessors) - 1

    # ---- images / textures --------------------------------------------
    def add_image(self, pil_img, name="tex", jpeg_quality=None):
        buf = io.BytesIO()
        if jpeg_quality:
            pil_img.convert("RGB").save(buf, "JPEG", quality=jpeg_quality, subsampling=0)
            mime = "image/jpeg"
        else:
            pil_img.save(buf, "PNG", optimize=True)
            mime = "image/png"
        bv = self._view(buf.getvalue())
        self.images.append({"bufferView": bv, "mimeType": mime, "name": name})
        self.textures.append({"sampler": 0, "source": len(self.images) - 1})
        return len(self.textures) - 1

    # ---- materials -----------------------------------------------------
    def add_material(self, name, base_srgb=(1, 1, 1), alpha=1.0, metallic=1.0, roughness=0.3,
                     base_tex=None, mr_tex=None, normal_tex=None, normal_scale=1.0,
                     emissive_srgb=None, emissive_tex=None, blend=False, double=False,
                     specular=None, specular_color=None, ior=None, clearcoat=None,
                     clearcoat_rough=None, sheen=None, occl_tex=None, uv_scale=None):
        lin = srgb2lin(np.asarray(base_srgb, float))
        pbr = {"baseColorFactor": [float(lin[0]), float(lin[1]), float(lin[2]), float(alpha)],
               "metallicFactor": float(metallic), "roughnessFactor": float(roughness)}
        if base_tex is not None:
            pbr["baseColorTexture"] = {"index": base_tex}
        if mr_tex is not None:
            pbr["metallicRoughnessTexture"] = {"index": mr_tex}
        m = {"name": name, "pbrMetallicRoughness": pbr}
        if occl_tex is not None:
            m["occlusionTexture"] = {"index": occl_tex}
        if normal_tex is not None:
            m["normalTexture"] = {"index": normal_tex, "scale": float(normal_scale)}
        if emissive_srgb is not None:
            e = srgb2lin(np.asarray(emissive_srgb, float))
            m["emissiveFactor"] = [float(e[0]), float(e[1]), float(e[2])]
        if emissive_tex is not None:
            m["emissiveTexture"] = {"index": emissive_tex}
        if blend:
            m["alphaMode"] = "BLEND"
        if double:
            m["doubleSided"] = True
        ext = {}
        if specular is not None or specular_color is not None:
            s = {}
            if specular is not None:
                s["specularFactor"] = float(specular)
            if specular_color is not None:
                sc = srgb2lin(np.asarray(specular_color, float))
                s["specularColorFactor"] = [float(sc[0]), float(sc[1]), float(sc[2])]
            ext["KHR_materials_specular"] = s
            self.exts.add("KHR_materials_specular")
        if ior is not None:
            ext["KHR_materials_ior"] = {"ior": float(ior)}
            self.exts.add("KHR_materials_ior")
        if clearcoat is not None:
            cc = {"clearcoatFactor": float(clearcoat)}
            if clearcoat_rough is not None:
                cc["clearcoatRoughnessFactor"] = float(clearcoat_rough)
            ext["KHR_materials_clearcoat"] = cc
            self.exts.add("KHR_materials_clearcoat")
        if sheen is not None:
            sc = srgb2lin(np.asarray(sheen[0], float))
            ext["KHR_materials_sheen"] = {"sheenColorFactor": [float(sc[0]), float(sc[1]), float(sc[2])],
                                          "sheenRoughnessFactor": float(sheen[1])}
            self.exts.add("KHR_materials_sheen")
        if ext:
            m["extensions"] = ext
        self.materials.append(m)
        return len(self.materials) - 1

    # ---- geometry ------------------------------------------------------
    def add_prim(self, mesh, material, name=None):
        V = mesh.V.astype(np.float32)
        N = (mesh.N if mesh.N is not None else mesh.smooth_normals().N).astype(np.float32)
        UV = mesh.UV.astype(np.float32)
        F = mesh.F.astype(np.uint32).ravel()
        pos = self._acc(V, "VEC3", 5126, 34962, minmax=True)
        nrm = self._acc(N, "VEC3", 5126, 34962)
        uv = self._acc(UV, "VEC2", 5126, 34962)
        idx = self._acc(F.reshape(-1, 1), "SCALAR", 5125, 34963)
        self.prims.append({"attributes": {"POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": uv},
                           "indices": idx, "material": material, "_name": name})

    def save(self, path):
        prims = [{k: v for k, v in p.items() if not k.startswith("_")} for p in self.prims]
        gltf = {
            "asset": {"version": "2.0", "generator": "IARONE parametric watch builder"},
            "scene": 0,
            "scenes": [{"name": self.name, "nodes": [0]}],
            "nodes": [{"mesh": 0, "name": self.name}],
            "meshes": [{"name": self.name, "primitives": prims}],
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.bufferViews,
            "buffers": [{"byteLength": len(self.bin)}],
        }
        if self.images:
            gltf["images"] = self.images
            gltf["samplers"] = self.samplers
            gltf["textures"] = self.textures
        if self.exts:
            gltf["extensionsUsed"] = sorted(self.exts)
        js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        bl = bytes(self.bin)
        bl += b"\0" * ((4 - len(bl) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(bl)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack("<II", len(bl), 0x004E4942))
            f.write(bl)
        return total
