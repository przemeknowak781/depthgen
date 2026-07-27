"""Sprawdza, czy KAŻDY parametr z interfejsu realnie zmienia geometrię siatki.

Dla każdego parametru budujemy siatkę bazową i zmodyfikowaną, po czym porównujemy
wierzchołki. Parametr, który niczego nie zmienia, jest martwą kontrolką w UI.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import heightmap as hm
from app import mesh as me


def scene(h=300, w=240):
    """Syntetyczna scena: kopuła (obiekt) na płaskim tle + faktura + twarda sylwetka."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.45, w * 0.5
    r = np.sqrt(((y - cy) / (h * 0.30)) ** 2 + ((x - cx) / (w * 0.34)) ** 2)
    dome = np.clip(1.0 - r ** 2, 0.0, 1.0) ** 0.5
    depth = 0.12 + 0.8 * dome                      # tło 0.12, obiekt do 0.92
    rng = np.random.default_rng(7)
    depth += rng.normal(0, 0.006, depth.shape).astype(np.float32)   # szum
    depth += 0.02 * np.sin(x / 3.0) * (dome > 0)                    # drobna faktura
    gray = np.clip(0.5 + 0.4 * np.sin(x / 2.5) * np.cos(y / 3.5), 0, 1).astype(np.float32)
    # kanał alfa: nieregularna sylwetka + drobne okruchy (do testu filtra wysepek)
    alpha = np.clip((dome - 0.22) * 6.0, 0.0, 1.0).astype(np.float32)   # miękka krawędź
    alpha[10:14, 10:14] = 1.0
    alpha[h - 16:h - 12, w - 16:w - 12] = 1.0
    return np.clip(depth, 0, 1).astype(np.float32), gray, alpha


DEPTH, GRAY, ALPHA = scene()

BASE = {**hm.DEFAULTS, **me.MESH_DEFAULTS, "resolution": 160}


def build(over: dict):
    p = {**BASE, **over}
    h = hm.build(DEPTH, GRAY, p, ALPHA)
    return me.build(h, p, hm.cut_mask(p, h, ALPHA))


def diff(a, b) -> float:
    """Miara różnicy geometrii: rms po Z jeśli topologia ta sama, inaczej +inf."""
    if a.vertices.shape != b.vertices.shape or len(a.faces) != len(b.faces):
        return float("inf")
    return float(np.abs(a.vertices - b.vertices).max())


# parametr -> (kontekst wymagany, żeby parametr w ogóle działał, wartość testowa)
CASES = [
    ("invert",         {}, True),
    ("clip_low",       {}, 12.0),
    ("clip_high",      {}, 88.0),
    ("gamma",          {}, 0.45),
    ("contrast",       {}, 0.8),
    ("contrast(-)",    {}, None),          # obsłużone niżej
    ("median",         {}, 5),
    ("bilateral",      {}, 8),
    ("smooth",         {}, 3.0),
    ("detail",         {}, 1.2),
    ("detail_radius",  {"detail": 0.8}, 20.0),
    ("detail_guard",   {"detail": 1.0}, 0.05),
    ("detail_clamp",   {"detail": 1.0, "detail_guard": 0.0}, 0.005),
    ("micro",          {}, 0.6),
    ("micro_radius",   {"micro": 0.5}, 8.0),
    ("floor",          {}, 0.35),
    ("floor_soft",     {"floor": 0.35}, 0.3),
    ("edge_falloff",   {}, 0.3),
    ("shape",          {}, "ellipse"),
    ("corner",         {"shape": "rounded"}, 0.6),
    ("margin",         {}, 0.25),
    ("trim",           {"shape": "ellipse"}, True),
    ("resolution",     {}, 240),
    ("width_mm",       {}, 175.0),
    ("relief_mm",      {}, 22.0),
    ("base_mm",        {}, 9.0),
    ("solid",          {}, False),
    ("z_offset",       {}, 4.0),
    # wycinanie sylwetki
    ("alpha_cut",      {"trim": True}, True),
    ("alpha_threshold", {"trim": True, "alpha_cut": True}, 0.95),
    ("alpha_grow",     {"trim": True, "alpha_cut": True}, 6),
    ("cut_level",      {"trim": True}, 0.45),
    ("min_island",     {"trim": True, "alpha_cut": True, "min_island": 0.0}, 1.5),
]

TOL = 1e-4   # mm — poniżej tego uznajemy, że parametr nic nie zrobił


def main() -> int:
    bad = []
    print(f"{'parametr':18} {'zmiana [mm]':>12}   status")
    print("-" * 46)
    for name, ctx, val in CASES:
        if name == "contrast(-)":
            key, val = "contrast", -0.8
        else:
            key = name
        base = build(ctx)
        mod = build({**ctx, key: val})
        d = diff(base, mod)
        ok = d > TOL
        shown = "topologia" if d == float("inf") else f"{d:10.4f}"
        print(f"{name:18} {shown:>12}   {'OK' if ok else 'BRAK ZMIANY  <<<'}")
        if not ok:
            bad.append(name)

    # kontrola odwrotna: identyczne parametry muszą dać identyczną siatkę
    a, b = build({}), build({})
    if diff(a, b) != 0.0:
        print("\nBŁĄD: dwa przebiegi z tymi samymi parametrami dają różne siatki")
        bad.append("determinizm")

    print()
    if bad:
        print(f"MARTWE PARAMETRY ({len(bad)}): {', '.join(bad)}")
        return 1
    print(f"Wszystkie {len(CASES)} parametry realnie zmieniają geometrię.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
