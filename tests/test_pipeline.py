"""Test end-to-end: obraz -> mapa głębi (model) -> mapa wysokości -> siatka -> STL."""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import depth as dm
from app import heightmap as hm
from app import mesh as me

MODEL = sys.argv[1] if len(sys.argv) > 1 else "dav2-small"
TILES = int(sys.argv[2]) if len(sys.argv) > 2 else 1

img = Image.open(ROOT / "tests" / "sample.jpg").convert("RGB")
if max(img.size) > 1400:
    s = 1400 / max(img.size)
    img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
print(f"obraz: {img.width}x{img.height}, urządzenie: {dm.device_info()}")

t0 = time.time()
res = dm.estimate(img, MODEL, dm.MODELS[MODEL]["default_size"], tiles=TILES,
                  progress=lambda m: print("  ", m))
t1 = time.time()
d = res.depth
print(f"głębia: {d.shape} zakres {d.min():.3f}..{d.max():.3f} w {t1 - t0:.1f} s")
assert np.isfinite(d).all() and d.shape == (img.height, img.width)
assert d.std() > 0.02, "mapa głębi jest płaska — model nie zadziałał"

gray = (np.asarray(img, np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114], np.float32)
p = {"detail": 0.5, "micro": 0.15, "contrast": 0.2, "relief_mm": 10.0,
     "base_mm": 3.0, "width_mm": 120.0, "resolution": 1000, "solid": True}
h = hm.build(d, gray, p)
print(f"wysokość: {h.shape} {h.min():.3f}..{h.max():.3f}")

t2 = time.time()
m = me.build(h, p, hm.shape_mask(*h.shape, {**hm.DEFAULTS, **p}))
print(f"siatka: {m.stats} w {time.time() - t2:.2f} s")

out = ROOT / "tests" / "out"
out.mkdir(exist_ok=True)
data = me.export(m, "stl")
(out / "sample_relief.stl").write_bytes(data)
Image.fromarray((d * 255).astype(np.uint8)).save(out / "depth.png")
Image.fromarray((h * 255).astype(np.uint8)).save(out / "height.png")
print(f"zapisano STL {len(data) / 1048576:.1f} MB -> {out}")

import trimesh
tm = trimesh.load(out / "sample_relief.stl")
print(f"weryfikacja STL: szczelny={tm.is_watertight}, objętość={tm.volume:.0f} mm3, "
      f"wymiary={np.round(tm.bounds[1] - tm.bounds[0], 1)}")
assert tm.is_watertight and tm.volume > 0
print("OK")
