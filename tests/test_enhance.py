"""Usuwanie artefaktów JPEG: czy naprawdę znikają i czy nie zabierają ze sobą faktury.

Bierzemy czysty obraz, zapisujemy go w bardzo niskiej jakości JPEG (to nasz „przypadek
z życia"), po czym mierzymy:
  * blokowość — energię gradientu na siatce 8x8 względem reszty obrazu,
  * błąd względem oryginału (PSNR) — czy zbliżamy się do prawdy, czy tylko rozmywamy,
  * ile artefaktów przenosi się na wysokość reliefu przez „mikrodetal z obrazu".
"""
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import enhance as en
from app import heightmap as hm


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), np.float32) / 255.0


def source() -> Image.Image:
    """Czysty obraz z fakturą w kilku skalach — żeby było co stracić."""
    h, w = 480, 640
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 128 + 60 * np.sin(x / 40) * np.cos(y / 55)
    base += 25 * np.sin((x + y) / 9)                     # średnia faktura
    base += 10 * np.sin(x / 2.5) * np.cos(y / 2.5)       # drobna faktura
    disc = ((x - 200) ** 2 + (y - 180) ** 2) < 90 ** 2   # twarda krawędź (dzwonienie)
    base[disc] += 60
    rgb = np.dstack([base, base * 0.95 + 8, base * 0.9 + 16])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def micro_energy(img: Image.Image) -> float:
    """Ile drobnego detalu z obrazu wchodzi do mapy wysokości (parametr `micro`)."""
    g = gray(img)
    depth = cv2.GaussianBlur(g, (0, 0), 20)              # gładka, sztuczna „głębia"
    p = {**hm.DEFAULTS, "detail": 0.0, "micro": 0.6, "micro_radius": 2.0,
         "detail_guard": 0.0, "detail_clamp": 1.0}
    h = hm.build(depth, g, p)
    return float(np.abs(h - cv2.GaussianBlur(h, (0, 0), 3)).mean())


def main() -> int:
    clean = source()
    ref = np.asarray(clean, np.uint8)
    bad = jpeg(clean, 18)                                # mocno zepsuty obraz
    ok_ref = en.blockiness(gray(clean))
    print(f"oryginal:            blokowosc {ok_ref:.3f}")
    print(f"po JPEG q18:         blokowosc {en.blockiness(gray(bad)):.3f}   "
          f"PSNR {psnr(ref, np.asarray(bad)):.1f} dB   "
          f"mikrodetal {micro_energy(bad) / micro_energy(clean):.2f}x oryginalu\n")

    b0 = en.blockiness(gray(bad))
    m_clean, m_bad = micro_energy(clean), micro_energy(bad)
    problems = 0

    print(f"{'wariant':28} {'blokowosc':>10} {'PSNR':>7} {'mikrodetal':>11}")
    print("-" * 60)
    results = {}
    for name, prm in (
        ("sam deblock 0.5", {"deblock": 0.5, "chroma": 0.5}),
        ("sam deblock 1.0", {"deblock": 1.0, "chroma": 1.0}),
        ("nadprobkowanie 2x", {"deblock": 0.0, "chroma": 0.0, "work_max": 320}),
        ("deblock + nadprobkowanie", {"deblock": 0.6, "chroma": 0.6, "work_max": 320}),
    ):
        out, meta = en.prepare(bad, prm)
        big = out.resize(clean.size, Image.LANCZOS)
        r = (en.blockiness(gray(out)), psnr(ref, np.asarray(big)),
             micro_energy(out) / m_clean)
        results[name] = r
        print(f"{name:28} {r[0]:10.3f} {r[1]:7.1f} {r[2]:10.2f}x")

    print()
    # 1. deblock musi realnie zbijać blokowość
    best = min(v[0] for v in results.values())
    print(f"blokowosc {b0:.3f} -> {best:.3f} "
          f"(usuniete {100 * (b0 - best) / max(b0 - 1, 1e-6):.0f}% nadmiaru)")
    if best > 1 + (b0 - 1) * 0.5:
        print("   BLAD: blokowosc spadla o mniej niz polowe")
        problems += 1

    # 2. nie wolno przy tym stracić wierności wobec oryginału
    p_bad = psnr(ref, np.asarray(bad))
    p_best = max(v[1] for v in results.values())
    print(f"PSNR {p_bad:.1f} -> {p_best:.1f} dB")
    if p_best < p_bad:
        print("   BLAD: czyszczenie oddala obraz od oryginalu (samo rozmycie)")
        problems += 1

    # 3. mikrodetal wracający do reliefu ma się zbliżyć do poziomu czystego obrazu
    m0 = m_bad / m_clean
    m_best = min(abs(v[2] - 1.0) for v in results.values()) + 1.0
    print(f"mikrodetal {m0:.2f}x -> {m_best:.2f}x poziomu czystego obrazu")
    if abs(m_best - 1) > abs(m0 - 1):
        print("   BLAD: do reliefu trafia gorszy detal niz przed czyszczeniem")
        problems += 1

    if "--sr" in sys.argv:
        problems += sr_check()

    print()
    print("OK" if not problems else f"{problems} problemow")
    return 1 if problems else 0


def sr_check() -> int:
    """Ścieżka z modelem upscalingu (pobiera wagi z HuggingFace)."""
    import time

    print("\n--- upscaling modelem ---")
    clean = source()
    small = jpeg(clean.resize((320, 240), Image.LANCZOS), 20)
    b_in = en.blockiness(gray(small))
    print(f"wejscie {small.size}, blokowosc {b_in:.3f}")

    bad = 0
    t0 = time.time()
    out, meta = en.prepare(small, {"deblock": 0.0, "chroma": 0.0,
                                   "sr_model": "compressed-x4", "work_max": 640})
    print(f"compressed-x4 -> {out.size} w {time.time() - t0:.1f}s, "
          f"blokowosc {meta['blockiness_after']:.3f}")
    if meta["blockiness_after"] > 1.15:
        print("   BLAD: model dla obrazow skompresowanych nie usunal blokowosci")
        bad += 1

    # szwy kafelkowania nie mogą zostawić linii w obrazie
    big, _ = en.prepare(small.resize((512, 384), Image.LANCZOS),
                        {"sr_model": "compressed-x4", "work_max": 2048})
    g = gray(big)
    prof = np.abs(np.diff(g, axis=1)).mean(axis=0)
    step = en.SR_TILE * 4
    xs = list(range(step, len(prof), step))
    if xs:
        seam = np.mean([prof[max(0, x - 2):x + 3].max() for x in xs]) / max(np.median(prof), 1e-9)
        print(f"szwy kafli SR: {seam:.3f}x mediany profilu")
        if seam > 1.30:
            print("   BLAD: widoczne szwy kafelkowania upscalingu")
            bad += 1
    return bad


if __name__ == "__main__":
    sys.exit(main())
