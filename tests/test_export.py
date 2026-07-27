"""Eksport przez prawdziwe API: nazwy plików, formaty i przypadki brzegowe przycinania.

Mapę głębi wgrywamy gotową (endpoint dla własnych map), więc test nie potrzebuje
modelu ani GPU i wykonuje się w sekundy.
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BRELOK = dict(clip_low=4.3, clip_high=100, gamma=2.42, contrast=1.0, median=5,
              bilateral=5, detail=1.85, detail_radius=16.5, detail_guard=1.0,
              detail_clamp=0.135, micro=0.18, micro_radius=1.1, floor=0.14,
              floor_soft=0.175, edge_falloff=0.30, shape="rect", margin=0.06,
              trim=True, alpha_cut=False, cut_level=0.01, min_island=1.7,
              width_mm=100, relief_mm=14.3, base_mm=0.0, solid=True)


def _png(w=240, h=300) -> io.BytesIO:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    v = np.clip(255 - ((x - w / 2) ** 2 + (y - h / 2) ** 2) / 60, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.dstack([v] * 3)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def session(filename: str = "photo.png") -> str:
    r = client.post("/api/upload", files={"file": (filename, _png(), "image/png")})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    r = client.post("/api/depth-upload", data={"id": sid},
                    files={"file": ("d.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    return sid


def export(sid, params=None, fmt="stl", res=400):
    return client.post("/api/export", json={"id": sid, "params": {**BRELOK, **(params or {})},
                                            "resolution": res, "format": fmt})


def main() -> int:
    bad = 0

    print("--- nazwy plikow w naglowku HTTP (musi byc latin-1) ---")
    for name in ("photo.png", "zdjęcie mamy.png", "moj obraz - kopia.png",
                 "foto 😀.png", "ŻÓŁĆ.png"):
        try:
            r = export(session(name))
            cd = r.headers.get("content-disposition", "")
            ok = r.status_code == 200 and "filename*=UTF-8''" in cd
            cd.encode("latin-1")            # to właśnie wywalało serwer
        except Exception as e:              # noqa: BLE001
            print(f"  {name:24} WYJATEK {type(e).__name__}: {e}")
            bad += 1
            continue
        print(f"  {name:24} {'OK' if ok else 'BLAD <<<'}  {cd[:70]}")
        bad += not ok

    print("\n--- formaty ---")
    sid = session()
    for fmt in ("stl", "obj", "ply", "glb", "3mf"):
        r = export(sid, fmt=fmt)
        ok = r.status_code == 200 and len(r.content) > 500
        print(f"  {fmt:5} {'OK' if ok else 'BLAD <<<'}  {len(r.content) / 1024:8.1f} KB"
              f"{'' if ok else '  ' + r.text[:120]}")
        bad += not ok

    print("\n--- skrajne ustawienia przycinania: wolno zwrocic bryle albo czytelny 400,")
    print("    nigdy 500 ani wyjatek ---")
    sid = session()
    extremes = [
        ("prog wysokosci 0.99", {"cut_level": 0.99}),
        ("prog wysokosci 0.999", {"cut_level": 0.999}),
        ("maksymalny margines", {"margin": 0.4}),
        ("pelne zejscie krawedzi", {"edge_falloff": 0.5, "cut_level": 0.5}),
        ("wszystkie wyspy odsiane", {"min_island": 99}),
        ("prog tla przy suficie", {"floor": 0.99, "cut_level": 0.5}),
        ("wszystko naraz", {"margin": 0.4, "edge_falloff": 0.5, "floor": 0.9,
                            "cut_level": 0.95, "min_island": 50}),
    ]
    for label, prm in extremes:
        for endpoint, extra in (("/api/export", {"format": "stl", "resolution": 400}),
                                ("/api/mesh", {"resolution": 300})):
            try:
                r = client.post(endpoint, json={"id": sid, "params": {**BRELOK, **prm}, **extra})
            except Exception as e:                       # noqa: BLE001
                print(f"  {label:24} {endpoint:12} WYJATEK {type(e).__name__}: {e}")
                bad += 1
                continue
            if r.status_code == 200:
                stan = f"bryla {len(r.content) / 1024:7.0f} KB"
            elif r.status_code == 400 and r.json().get("detail"):
                stan = "400 " + r.json()["detail"][:44] + "..."
            else:
                stan = f"BLAD <<< {r.status_code}: {r.text[:90]}"
                bad += 1
            print(f"  {label:24} {endpoint:12} {stan}")

    print("\n--- pusta maska wprost (test jednostkowy) ---")
    from app import mesh as me

    h = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    for label, mask in (("maska pusta", np.zeros((64, 64), np.float32)),
                        ("same pojedyncze punkty", np.eye(64, dtype=np.float32) * 0)):
        try:
            me.build(h, {**BRELOK, "resolution": 64}, mask)
            print(f"  {label:24} BLAD <<< zbudowal bryle z niczego")
            bad += 1
        except me.EmptyMesh as e:
            print(f"  {label:24} OK EmptyMesh: {str(e)[:50]}...")
        except Exception as e:                           # noqa: BLE001
            print(f"  {label:24} BLAD <<< {type(e).__name__}: {e}")
            bad += 1

    print()
    print("OK" if not bad else f"{bad} problemow")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
