"""Światła i cienie: kontrola nad tym, co ląduje na idealnej bieli i czerni.

Mocne wyostrzanie wypycha najwyższe punkty ponad zakres 0..1, a obcięcie do zakresu
robi z nich płaskie krążki — na twarzy widać to jako ścięty czubek nosa. Suwak „Światła”
na minusie ściąga te partie spod sufitu, zanim dojdzie do obcięcia. Test sprawdza, że
ta kontrola działa, oraz że z suwakami na zerze potok liczy dokładnie to, co wcześniej.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import heightmap as hm

H, W = 400, 320
CY, CX = 215, 160

BRELOK = dict(clip_low=4.3, clip_high=100, gamma=2.42, contrast=1.0, median=5,
              bilateral=5, detail=1.85, detail_radius=16.5, detail_guard=1.0,
              detail_clamp=0.135, micro=0.18, micro_radius=1.1, floor=0.14,
              floor_soft=0.175, edge_falloff=0.30, margin=0.06)


def scene(sharpness: float = 7.0):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    face = np.exp(-(((x - CX) / 130) ** 2 + ((y - 200) / 170) ** 2))
    nose = np.exp(-(((x - CX) / sharpness) ** 2 + ((y - CY) / (sharpness * 1.3)) ** 2))
    d = (0.08 + 0.50 * face + 0.22 * nose).astype(np.float32)
    assert d.max() < 0.95, "scena nie moze byc obcieta juz na wejsciu"
    gray = np.clip(0.5 + 0.2 * np.sin(x / 3), 0, 1).astype(np.float32)
    return d, gray


def build(over=None):
    d, g = scene()
    return hm.build(d, g, {**hm.DEFAULTS, **BRELOK, **(over or {})})


def at_ceiling(h, eps=1e-4):
    """Ile pikseli siedzi na idealnej bieli — tam relief jest ścięty na płasko."""
    return int((h >= 1.0 - eps).sum())


def at_floor(h, eps=1e-4):
    return int((h <= eps).sum())


def plateau(h, r=30, tol=1e-3):
    sub = h[CY - r:CY + r, CX - r:CX + r]
    return int((sub >= sub.max() - tol).sum())


def main() -> int:
    bad = 0
    base = build()
    print(f"punkt wyjscia (suwaki na zerze): na bieli {at_ceiling(base)} px, "
          f"placek na czubku {plateau(base)} px\n")

    print(f"{'swiatla':>9} {'na bieli':>10} {'placek':>8} {'sr. wys.':>10} {'szczyt':>8}")
    print("-" * 50)
    prev_white = None
    for hl in (0.0, -0.2, -0.4, -0.6, -0.8):
        h = build({"highlights": hl})
        obj = h > 0.05
        w, p = at_ceiling(h), plateau(h)
        print(f"{hl:9.2f} {w:10} {p:8} {h[obj].mean():10.4f} {h.max():8.4f}")
        if prev_white is not None and w > prev_white:
            print("   BLAD: ujemne swiatla nie zmniejszaja obszaru na bieli")
            bad += 1
        prev_white = w

    strong = build({"highlights": -0.6})
    if at_ceiling(strong) >= at_ceiling(base):
        print("   BLAD: suwak swiatel nie zdejmuje reliefu z sufitu")
        bad += 1
    if plateau(strong) >= plateau(base):
        print("   BLAD: suwak swiatel nie zmniejsza plaskiego placka na czubku")
        bad += 1

    # Cienie mierzymy bez progu tła i zejścia przy krawędziach — te i tak zerują tło
    # już po tym etapie, więc zasłoniłyby efekt suwaka.
    plain = {"floor": 0.0, "edge_falloff": 0.0, "margin": 0.0}
    ref_floor = at_floor(build(plain))
    print(f"\n{'cienie':>9} {'na czerni':>11} {'sr. wys.':>10}   (bez progu tla)")
    print("-" * 46)
    for sh in (-0.5, 0.0, 0.4, 0.8):
        h = build({**plain, "shadows": sh})
        obj = h > 0.05
        print(f"{sh:9.2f} {at_floor(h):11} {h[obj].mean():10.4f}")
    if at_floor(build({**plain, "shadows": 0.6})) >= ref_floor:
        print("   BLAD: dodatnie cienie nie odklejaja dolu od czerni")
        bad += 1
    if at_floor(build({**plain, "shadows": -0.5})) <= ref_floor:
        print("   BLAD: ujemne cienie nie dociskaja dolu ku czerni")
        bad += 1

    print("\nsrodek zakresu ma zostac nietkniety (gamma pracuje jak wczesniej):")
    d, g = scene()
    probe = np.linspace(0.0, 1.0, 11, dtype=np.float32).reshape(1, 11)
    for hl, sh in ((-0.6, 0.0), (0.0, 0.6)):
        out = hm._tone(probe.copy(), hl, sh)[0]
        mid = float(abs(out[5] - probe[0, 5]))
        print(f"  swiatla {hl:+.1f} cienie {sh:+.1f}: 0.5 -> {out[5]:.4f} "
              f"({'OK' if mid < 1e-6 else 'ZMIENIONE <<<'})")
        if mid >= 1e-6:
            bad += 1

    print("\nprzekroj przez czubek:")
    for hl in (0.0, -0.4, -0.8):
        h = build({"highlights": hl})
        print(f"  swiatla {hl:+.1f}: " +
              " ".join(f"{v:.4f}" for v in h[CY, CX - 12:CX + 13][::2]))

    print()
    print("OK" if not bad else f"{bad} problemow")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
