"""Kontrola artefaktów przebiegu kafelkowego.

Trzy mechanizmy psuły mapę z kafli:
1. dopasowanie kafla do przebiegu globalnego metodą najmniejszych kwadratów było
   źle uwarunkowane tam, gdzie głębia jest płaska (dzielenie przez wariancję ~0),
   więc kafel dostawał losowy poziom i wzmocniony szum — stąd prostokąty kafli;
2. kafle "dzwonią" na skoku głębi, co dawało poświatę wokół sylwetki;
3. uśrednianie kafli w szerokiej zakładce wygaszało drobny detal, czyli dokładnie
   to, po co w ogóle robi się przebieg kafelkowy.

Test porównuje obecny algorytm ze starym (wyłączone regularyzacja i maski pewności,
szerokie mieszanie) i jest samosprawdzalny: stary wariant musi przekroczyć progi.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import depth as dm

HALO_LIMIT = 0.005    # max. odchyłka od przebiegu globalnego w pierścieniu wokół sylwetki
FLAT_LIMIT = 1.30     # max. wzrost energii w płaskim tle (prostokąty kafli)
MIN_GAIN = 1.05       # kafle muszą realnie dołożyć mikrodetalu we wnętrzu obiektu

LEGACY = dict(TILE_ALIGN_REG=0.0, TILE_FLAT_GUARD=0.0, TILE_EDGE_GUARD=0.0,
              TILE_HI_BAND=400.0)


def _blur(d, s):
    return cv2.GaussianBlur(d, (0, 0), s, borderType=cv2.BORDER_REPLICATE)


def regions(ref):
    obj = (ref > 0.35).astype(np.uint8)
    grow = lambda r: cv2.dilate(obj, np.ones((r, r), np.uint8)) > 0
    shrink = lambda r: cv2.erode(obj, np.ones((r, r), np.uint8)) > 0
    return {
        "inside": shrink(41),            # wnętrze obiektu — tu chcemy zysk detalu
        "ring": grow(41) & ~grow(9),     # tuż za sylwetką — tu powstawała poświata
        "flat": ~grow(61),               # dalekie tło — tu mapa ma zostać płaska
    }


def measure(d, ref, R):
    mid = lambda x: np.abs(x - _blur(x, 40))
    micro = lambda x: np.abs(x - _blur(x, 3))
    return {
        "halo": float(np.abs(d - ref)[R["ring"]].mean()),
        "tlo": float(mid(d)[R["flat"]].mean() / max(mid(ref)[R["flat"]].mean(), 1e-9)),
        "detal": float(micro(d)[R["inside"]].mean() / max(micro(ref)[R["inside"]].mean(), 1e-9)),
    }


def run(img, n, legacy=False):
    keep = {k: getattr(dm, k) for k in LEGACY}
    if legacy:
        for k, v in LEGACY.items():
            setattr(dm, k, v)
    try:
        return dm.estimate(img, "dav2-small", 518, tiles=n).depth
    finally:
        for k, v in keep.items():
            setattr(dm, k, v)


def main() -> int:
    img = Image.open(ROOT / "tests" / "sample.jpg").convert("RGB")
    if max(img.size) > 1200:
        s = 1200 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)

    ref = dm.estimate(img, "dav2-small", 518, tiles=1).depth
    R = regions(ref)
    print(f"wnetrze {R['inside'].mean()*100:.0f}%  pierscien {R['ring'].mean()*100:.0f}%  "
          f"tlo {R['flat'].mean()*100:.0f}% kadru\n")
    print(f"{'wariant':18} {'halo':>8} {'tlo':>7} {'detal':>7}")
    print("-" * 44)

    bad = 0
    old = measure(run(img, 3, legacy=True), ref, R)
    print(f"{'stary 3x3':18} {old['halo']:8.4f} {old['tlo']:7.2f} {old['detal']:7.3f}"
          f"   <- tak bylo")
    if old["halo"] <= HALO_LIMIT:
        print("   BŁĄD: metryka nie wykrywa nawet starego artefaktu")
        bad += 1

    for n in (2, 3, 4):
        m = measure(run(img, n), ref, R)
        ok = [m["halo"] <= HALO_LIMIT, m["tlo"] <= FLAT_LIMIT, m["detal"] >= MIN_GAIN]
        flags = "".join(x for x, o in zip((" POSWIATA", " KRATKA", " BRAK-DETALU"), ok) if not o)
        print(f"{'nowy ' + str(n) + 'x' + str(n):18} {m['halo']:8.4f} {m['tlo']:7.2f} "
              f"{m['detal']:7.3f}   {flags or 'OK'}")
        bad += sum(1 for o in ok if not o)

    print()
    print("OK — kafle dodaja detal bez poswiaty i bez kratki" if not bad else f"{bad} problemow")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
