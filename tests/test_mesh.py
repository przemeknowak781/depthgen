"""Testy poprawności siatki: szczelność, orientacja ścianek, wymiary."""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import heightmap as hm
from app import mesh as me


def make(hmap, params=None, mask=None):
    p = {"resolution": 120, "width_mm": 100.0, "relief_mm": 10.0, "base_mm": 3.0, "solid": True}
    p.update(params or {})
    m = me.build(hmap, p, mask)
    return trimesh.Trimesh(vertices=m.vertices.astype(np.float64), faces=m.faces, process=False), m


def gradient_map(h=180, w=240):
    y, x = np.mgrid[0:h, 0:w]
    return ((np.sin(x / 18.0) * np.cos(y / 14.0) * 0.5 + 0.5)).astype(np.float32)


def test_solid_is_watertight_and_outward():
    tm, m = make(gradient_map())
    assert tm.is_watertight, "bryła nie jest szczelna"
    assert tm.is_winding_consistent, "niespójna orientacja trójkątów"
    assert tm.volume > 0, f"objętość ujemna (ścianki do wewnątrz): {tm.volume}"
    assert not tm.is_empty
    print(f"  solid: {len(m.vertices)} v, {len(m.faces)} f, V={tm.volume:.1f} mm3")


def test_dimensions():
    tm, _ = make(gradient_map(h=100, w=200), {"width_mm": 150.0, "relief_mm": 6.0, "base_mm": 2.0})
    ext = tm.bounds[1] - tm.bounds[0]
    assert abs(ext[0] - 150.0) < 1e-3, ext
    assert abs(ext[1] - 75.0) < 1e-2, ext          # zachowany aspekt 200:100
    assert abs(ext[2] - 8.0) < 1e-2, ext           # baza 2 + relief 6
    assert abs(tm.bounds[0][2]) < 1e-6, "spód nie leży na Z=0"


def test_open_surface():
    tm, _ = make(gradient_map(), {"solid": False})
    assert not tm.is_watertight, "otwarta powierzchnia nie powinna być szczelna"
    assert tm.bounds[0][2] >= -1e-6, "powierzchnia schodzi poniżej Z=0"


def test_trim_to_ellipse():
    h = gradient_map()
    p = {**hm.DEFAULTS, "shape": "ellipse"}
    mask = hm.shape_mask(*h.shape, p)
    tm, m = make(h * mask, {"trim": True}, mask)
    assert tm.is_watertight, "przycięta bryła nie jest szczelna"
    assert tm.is_winding_consistent
    assert tm.volume > 0
    ext = tm.bounds[1] - tm.bounds[0]
    assert ext[0] < 101 and ext[1] < 76
    print(f"  ellipse trim: {len(m.faces)} f, V={tm.volume:.1f} mm3")


def test_heightmap_pipeline_bounds():
    d = gradient_map()
    g = np.random.default_rng(0).random(d.shape).astype(np.float32)
    for p in [
        {},
        {"invert": True, "gamma": 0.5, "contrast": 0.7},
        {"contrast": -0.8, "detail": 1.5, "micro": 0.6},
        {"bilateral": 5, "smooth": 2.0, "floor": 0.3, "edge_falloff": 0.25},
        {"shape": "rounded", "margin": 0.2, "corner": 0.4},
    ]:
        out = hm.build(d, g, p)
        assert out.shape == d.shape
        assert np.isfinite(out).all(), f"NaN/Inf dla {p}"
        assert out.min() >= 0.0 and out.max() <= 1.0, f"poza zakresem 0..1 dla {p}"


def test_export_formats():
    tm, m = make(gradient_map(), {"resolution": 60})
    for fmt in ("stl", "obj", "ply", "glb"):
        data = me.export(m, fmt)
        assert len(data) > 1000, fmt
        print(f"  {fmt}: {len(data) / 1024:.0f} KB")


def test_binary_packet():
    _, m = make(gradient_map(), {"resolution": 40})
    b = me.to_binary(m)
    nv, nf = np.frombuffer(b[:8], np.uint32)
    assert nv == len(m.vertices) and nf == len(m.faces)
    assert len(b) == 8 + nv * 12 * 2 + nf * 12
    n = me.vertex_normals(m)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-4)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK   {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"ERR  {name}: {type(e).__name__}: {e}")
    print("\nWynik:", "wszystko OK" if not fails else f"{fails} błędów")
    sys.exit(1 if fails else 0)
