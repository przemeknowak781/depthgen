"""Najwyższy punkt reliefu nie może być ścinany w płaski krążek.

Wyostrzanie dokłada do wysokości amplitudę ograniczoną tylko limitem detalu razy siłę
— przy ustawieniach z presetu „Brelok" to nawet +0.5. Kiedy sufit 1.0 był egzekwowany
twardym obcięciem, czubek nosa zamieniał się w płaski placek. Test mierzy wielkość tego
placka i porównuje ją z tym samym szczytem policzonym bez wyostrzania.
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
    """Twarz z ostrym czubkiem — ostrym względem promienia wyostrzania (16.5 px)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    face = np.exp(-(((x - CX) / 130) ** 2 + ((y - 200) / 170) ** 2))
    nose = np.exp(-(((x - CX) / sharpness) ** 2 + ((y - CY) / (sharpness * 1.3)) ** 2))
    d = (0.08 + 0.50 * face + 0.22 * nose).astype(np.float32)
    assert d.max() < 0.95, "scena nie moze byc obcieta juz na wejsciu"
    gray = np.clip(0.5 + 0.2 * np.sin(x / 3), 0, 1).astype(np.float32)
    return d, gray


def plateau(h, r=30, tol=1e-3):
    """Ile pikseli wokół szczytu leży praktycznie na jednej wysokości."""
    sub = h[CY - r:CY + r, CX - r:CX + r]
    return int((sub >= sub.max() - tol).sum())


def monotonic_from_peak(h, r=22) -> bool:
    """Przekrój przez czubek musi opadać w obie strony bez płaskiego odcinka na górze."""
    line = h[CY, CX - r:CX + r + 1]
    top = int(np.argmax(line))
    left, right = line[:top + 1], line[top:]
    flat_top = np.sum(np.abs(np.diff(line[max(0, top - 3):top + 4])) < 1e-5)
    return bool(np.all(np.diff(left) >= -1e-6) and np.all(np.diff(right) <= 1e-6)
                and flat_top <= 2)


def main() -> int:
    depth, gray = scene()
    bad = 0

    ref = hm.build(depth, gray, {**hm.DEFAULTS, **BRELOK, "detail": 0.0, "micro": 0.0})
    ref_flat = plateau(ref)
    print(f"wzorzec bez wyostrzania:      placek {ref_flat:4} px, szczyt {ref.max():.4f}")

    print(f"\n{'ustawienia':34} {'placek':>7} {'szczyt':>8}  ocena")
    print("-" * 62)
    for label, over in (
        ("brelok (wyostrzenie 1.85)", {}),
        ("wyostrzenie 1.85 + limit 0.4", {"detail_clamp": 0.4}),
        ("wyostrzenie 2.0, promien 25", {"detail": 2.0, "detail_radius": 25}),
        ("mocny mikrodetal", {"micro": 0.8, "detail": 1.0}),
        ("bardzo ostry czubek", {"median": 3}),
    ):
        d, g = scene(4.0) if label == "bardzo ostry czubek" else (depth, gray)
        h = hm.build(d, g, {**hm.DEFAULTS, **BRELOK, **over})
        n = plateau(h)
        mono = monotonic_from_peak(h)
        # placek nie moze byc istotnie wiekszy niz naturalne zaokraglenie szczytu
        ok = n <= max(4 * ref_flat, 20) and mono
        print(f"{label:34} {n:7} {h.max():8.4f}  "
              f"{'OK' if ok else 'SCIETY CZUBEK <<<'}{'' if mono else ' (plaski odcinek)'}")
        bad += not ok

    # Osobna sprawa: filtr medianowy z zalozenia usuwa szczyty wezsze niz jego okno.
    # To nie jest usterka, ale warto miec to zmierzone i widoczne.
    print("\nwplyw odszumiania medianowego na waski czubek (4 px):")
    d4, g4 = scene(4.0)
    for med in (0, 3, 5):
        hm4 = hm.build(d4, g4, {**hm.DEFAULTS, **BRELOK, "median": med, "bilateral": 0})
        line = hm4[CY, CX - 8:CX + 9]
        top = int(np.argmax(line))
        flat = int(np.sum(np.abs(np.diff(line[max(0, top - 3):top + 4])) < 1e-5))
        print(f"  median {med} px: plaskich krokow na szczycie {flat}")

    print("\nprzekroj przez czubek (co 2 px):")
    h = hm.build(depth, gray, {**hm.DEFAULTS, **BRELOK})
    print("  " + " ".join(f"{v:.4f}" for v in h[CY, CX - 12:CX + 13][::2]))

    # zakres wyjsciowy musi zostac poprawny
    if not (0.0 <= h.min() and h.max() <= 1.0 + 1e-6):
        print(f"   BLAD: mapa poza zakresem 0..1 ({h.min()}..{h.max()})")
        bad += 1

    print()
    print("OK" if not bad else f"{bad} problemow")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
