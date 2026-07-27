"""Wycinanie sylwetki: bryła musi zostać szczelna także dla kształtów wklęsłych,
z dziurami i rozpadających się na kilka wysp — a płaska płyta ma faktycznie zniknąć.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import heightmap as hm
from app import mesh as me

H, W = 260, 200


def scene(kind: str):
    """Zwraca (glebia, alfa) o zadanym ksztalcie sylwetki."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.zeros((H, W), np.float32)
    if kind == "wklesly":                       # litera C — mocno wklęsły obrys
        r = np.sqrt((y - H / 2) ** 2 + (x - W / 2) ** 2)
        a[(r < 80) & (r > 38)] = 1.0
        a[(x > W / 2) & (np.abs(y - H / 2) < 30)] = 0.0
    elif kind == "dziura":                      # pierścień — sylwetka z otworem
        r = np.sqrt((y - H / 2) ** 2 + (x - W / 2) ** 2)
        a[(r < 80) & (r > 34)] = 1.0
    elif kind == "wyspy":                       # trzy rozłączne obiekty
        for cx, cy, rr in ((55, 70, 32), (145, 80, 30), (100, 190, 36)):
            a[((y - cy) ** 2 + (x - cx) ** 2) < rr * rr] = 1.0
    elif kind == "okruchy":                     # duży obiekt + drobne śmieci
        a[((y - 130) ** 2 + (x - 100) ** 2) < 70 ** 2] = 1.0
        for cx, cy in ((15, 20), (185, 30), (20, 240), (180, 245)):
            a[cy - 2:cy + 3, cx - 2:cx + 3] = 1.0
    else:                                       # pełny prostokąt
        a[:] = 1.0
    d = 0.15 + 0.7 * cv2.GaussianBlur(a, (0, 0), 12)
    d += 0.05 * np.sin(x / 4.0) * a
    return np.clip(d, 0, 1).astype(np.float32), a


def build(kind, over=None):
    depth, alpha = scene(kind)
    p = {**hm.DEFAULTS, **me.MESH_DEFAULTS, "resolution": 200, "alpha_cut": True,
         "trim": True, "relief_mm": 8.0, "base_mm": 2.0, "width_mm": 100.0}
    p.update(over or {})
    h = hm.build(depth, None, p, alpha)
    mask = hm.cut_mask(p, h, alpha)
    m = me.build(h, p, mask)
    tm = trimesh.Trimesh(vertices=m.vertices.astype(np.float64), faces=m.faces, process=False)
    return tm, m, mask


def check(kind, expect_bodies=None):
    tm, m, mask = build(kind)
    parts = tm.split(only_watertight=False)
    ok = tm.is_watertight and tm.is_winding_consistent and tm.volume > 0
    print(f"{kind:10} {len(m.faces):>8} scian  bryl {len(parts):>2}  "
          f"V={tm.volume:9.1f}  szczelna={tm.is_watertight}  "
          f"orientacja={tm.is_winding_consistent}  {'OK' if ok else 'BLAD <<<'}")
    if expect_bodies is not None and len(parts) != expect_bodies:
        print(f"           BLAD: oczekiwano {expect_bodies} brył")
        return False
    return ok


def main() -> int:
    bad = 0
    print("--- ksztalty wyciecia (bryla musi byc szczelna) ---")
    for kind, bodies in (("pelny", 1), ("wklesly", 1), ("dziura", 1), ("wyspy", 3)):
        bad += not check(kind, bodies)

    print("\n--- plyta faktycznie znika ---")
    on, _, mask = build("wklesly")
    off, _, _ = build("wklesly", {"trim": False, "alpha_cut": False})
    shrink = 1 - on.volume / off.volume
    area = float((mask > 0.5).mean())
    print(f"objetosc: {off.volume:.0f} -> {on.volume:.0f} mm3  (mniej o {shrink * 100:.0f}%)"
          f"   sylwetka zajmuje {area * 100:.0f}% kadru")
    if shrink < 0.4:
        print("   BLAD: plyta nie zostala odcieta")
        bad += 1
    bb_on = on.bounds[1] - on.bounds[0]
    bb_off = off.bounds[1] - off.bounds[0]
    if not (bb_on[0] < bb_off[0] - 1 and bb_on[1] < bb_off[1] - 1):
        print(f"   BLAD: obrys sie nie zmniejszyl {bb_off} -> {bb_on}")
        bad += 1
    else:
        print(f"gabaryty: {np.round(bb_off, 1)} -> {np.round(bb_on, 1)} mm")

    print("\n--- usuwanie okruchow ---")
    _, _, m_keep = build("okruchy", {"min_island": 0.0})
    _, _, m_drop = build("okruchy", {"min_island": 0.5})
    n_keep = cv2.connectedComponents((m_keep > 0.5).astype(np.uint8), connectivity=8)[0] - 1
    n_drop = cv2.connectedComponents((m_drop > 0.5).astype(np.uint8), connectivity=8)[0] - 1
    print(f"wysp przed: {n_keep}, po odsianiu: {n_drop}")
    if not (n_keep >= 5 and n_drop == 1):
        print("   BLAD: filtr wysepek nie zadzialal")
        bad += 1

    print("\n--- ciecie progiem wysokosci (obrazy bez alfy) ---")
    tm_lvl, _, _ = build("pelny", {"alpha_cut": False, "cut_level": 0.45})
    tm_all, _, _ = build("pelny", {"alpha_cut": False, "cut_level": 0.0, "trim": False})
    print(f"objetosc {tm_all.volume:.0f} -> {tm_lvl.volume:.0f} mm3, "
          f"szczelna={tm_lvl.is_watertight}, orientacja={tm_lvl.is_winding_consistent}")
    if not (tm_lvl.is_watertight and tm_lvl.volume < tm_all.volume * 0.9):
        print("   BLAD: ciecie progiem wysokosci nie dziala")
        bad += 1

    print()
    print("OK" if not bad else f"{bad} problemow")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
