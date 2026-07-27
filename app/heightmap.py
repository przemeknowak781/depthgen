"""Obróbka mapy głębi -> mapa wysokości gotowa pod płaskorzeźbę."""
from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = {
    "invert": False,
    "clip_low": 0.5,        # percentyl odcięcia dołu
    "clip_high": 99.5,      # percentyl odcięcia góry
    "gamma": 1.0,           # <1 podbija wypukłości, >1 spłaszcza
    "contrast": 0.0,        # -1..1 krzywa S
    "highlights": 0.0,      # -1..1 — ujemne ściąga szczyty spod sufitu (koniec z idealną bielą)
    "shadows": 0.0,         # -1..1 — dodatnie odkleja dół od idealnej czerni
    "median": 0,            # 0/3/5 px — usuwa pojedyncze piksele (kolce na włosach)
    "smooth": 0.0,          # px, gaussian (redukcja szumu)
    "bilateral": 0.0,       # px, wygładza zachowując krawędzie
    "detail": 0.35,         # unsharp na głębi
    "detail_radius": 6.0,   # px
    "detail_guard": 0.6,    # 0..1 — chroni sylwetkę przed przestrzałem wyostrzania
    "detail_clamp": 0.08,   # maksymalna amplituda dokładanego detalu
    "micro": 0.0,           # detal z luminancji obrazu (faktura, włosy, tkanina)
    "micro_radius": 2.5,    # px
    "floor": 0.0,           # 0..1 — poniżej progu spłaszcz do tła
    "floor_soft": 0.05,
    "edge_falloff": 0.0,    # 0..0.5 — zejście do zera przy krawędziach
    "shape": "rect",        # rect | ellipse | rounded
    "corner": 0.15,         # promień narożnika dla 'rounded'
    "margin": 0.0,          # 0..0.4 — pusty margines wokół reliefu
    "trim": False,          # przycinaj siatkę do kształtu (zamiast płaskiego tła)
    # wycinanie sylwetki — pozwala odciąć płaską płytę i zostawić sam relief
    "alpha_cut": False,     # użyj kanału alfa obrazu jako kształtu
    "alpha_threshold": 0.5,
    "alpha_grow": 0,        # px: >0 poszerza sylwetkę, <0 zwęża
    "cut_level": 0.0,       # 0..0.9 — dodatkowo odetnij wszystko poniżej tej wysokości
    "min_island": 0.2,      # % powierzchni: mniejsze wysepki są usuwane
}


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


PIVOT = 0.5   # środek zakresu, ku któremu ściągają suwaki świateł i cieni


def _tone(h: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    """Światła i cienie — osobna kontrola nad górą i dołem zakresu.

    Ujemne „światła” ściągają najjaśniejsze partie spod sufitu, więc nic nie ląduje na
    idealnej bieli i nie zostaje ścięte przy obcinaniu do zakresu. Ujemne „cienie”
    dociskają dół ku zeru, dodatnie odklejają go od idealnej czerni. Środek zakresu
    zostaje nietknięty, więc gamma i kontrast działają dokładnie jak wcześniej.
    """
    if highlights:
        w = _smoothstep(np.clip((h - PIVOT) / (1.0 - PIVOT), 0.0, 1.0))
        h = h + highlights * w * ((1.0 - h) if highlights > 0 else (h - PIVOT))
    if shadows:
        w = _smoothstep(np.clip((PIVOT - h) / PIVOT, 0.0, 1.0))
        h = h + shadows * w * ((PIVOT - h) if shadows > 0 else h)
    return h


def _odd(v: float) -> int:
    k = int(max(1, round(v)))
    return k + 1 if k % 2 == 0 else k


def shape_mask(h: int, w: int, p: dict) -> np.ndarray:
    """Maska 0..1 kształtu reliefu (rozmiar h x w)."""
    m = float(np.clip(p.get("margin", 0.0), 0.0, 0.45))
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    if m > 0:
        s = 1.0 / max(1e-3, 1.0 - 2.0 * m)
        yy, xx = yy * s, xx * s
    shape = p.get("shape", "rect")
    if shape == "ellipse":
        d = np.sqrt(xx ** 2 + yy ** 2)
        mask = np.clip((1.0 - d) / 0.02, 0.0, 1.0)
    elif shape == "rounded":
        r = float(np.clip(p.get("corner", 0.15), 0.01, 1.0))
        qx = np.maximum(np.abs(xx) - (1.0 - r), 0.0)
        qy = np.maximum(np.abs(yy) - (1.0 - r), 0.0)
        d = np.sqrt(qx ** 2 + qy ** 2) - r
        inside = (np.abs(xx) <= 1.0) & (np.abs(yy) <= 1.0)
        mask = np.clip(-d / 0.02, 0.0, 1.0) * inside
    else:
        mask = ((np.abs(xx) <= 1.0) & (np.abs(yy) <= 1.0)).astype(np.float32)
    return mask.astype(np.float32)


def alpha_mask(shape: tuple[int, int], p: dict, alpha: np.ndarray | None) -> np.ndarray | None:
    """Sylwetka z kanału alfa: próg, poszerzenie/zwężenie."""
    if alpha is None or not p.get("alpha_cut"):
        return None
    a = alpha.astype(np.float32)
    if a.shape != shape:
        a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    m = (a >= float(np.clip(p.get("alpha_threshold", 0.5), 0.01, 0.99))).astype(np.uint8)
    g = int(p.get("alpha_grow", 0))
    if g:
        k = np.ones((abs(g) * 2 + 1,) * 2, np.uint8)
        m = cv2.dilate(m, k) if g > 0 else cv2.erode(m, k)
    return m.astype(np.float32)


def drop_islands(mask: np.ndarray, min_pct: float) -> np.ndarray:
    """Usuwa drobne wysepki — inaczej w druku zostają luźne okruchy."""
    if min_pct <= 0:
        return mask
    binary = (mask > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return mask
    limit = binary.size * min_pct / 100.0
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= limit
    if not keep.any():                      # nie zostawiaj pustej sceny
        keep[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = True
    return (keep[lab] & (mask > 0.5)).astype(np.float32)


def cut_mask(p: dict, height: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """Ostateczny kształt bryły: kształt płyty ∧ alfa ∧ próg wysokości, bez wysepek."""
    q = {**DEFAULTS, **(p or {})}
    m = shape_mask(*height.shape, q)
    am = alpha_mask(height.shape, q, alpha)
    if am is not None:
        m = m * am
    lvl = float(np.clip(q.get("cut_level", 0.0), 0.0, 0.95))
    if lvl > 0:
        m = m * (height >= lvl).astype(np.float32)
    return drop_islands(m, float(q.get("min_island", 0.0)))


def build(depth: np.ndarray, gray: np.ndarray | None, p: dict,
          alpha: np.ndarray | None = None) -> np.ndarray:
    """Zwraca mapę wysokości 0..1 (float32) w rozdzielczości `depth`."""
    q = {**DEFAULTS, **(p or {})}
    h = depth.astype(np.float32).copy()

    if q["invert"]:
        h = 1.0 - h

    lo = float(np.percentile(h, np.clip(q["clip_low"], 0, 49)))
    hi = float(np.percentile(h, np.clip(q["clip_high"], 51, 100)))
    h = np.clip((h - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    med = int(q.get("median", 0))
    if med >= 3:
        h = cv2.medianBlur(h, 5 if med >= 5 else 3)
    if q["bilateral"] > 0:
        d = int(np.clip(q["bilateral"], 1, 25))
        h = cv2.bilateralFilter(h, d * 2 + 1, 0.08, d)
    if q["smooth"] > 0:
        h = cv2.GaussianBlur(h, (0, 0), float(q["smooth"]))

    guard_w = None
    if q["detail"] != 0 or (q["micro"] != 0 and gray is not None):
        gd = float(np.clip(q.get("detail_guard", 0.6), 0.0, 1.0))
        if gd > 0:
            r = max(0.6, float(q["detail_radius"]))
            lo = cv2.GaussianBlur(h, (0, 0), r)
            gx = cv2.Sobel(lo, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(lo, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), r)
            t = (1.0 - gd) * 0.3 + 0.015
            guard_w = np.exp(-((grad / t) ** 2)).astype(np.float32)

    clamp = float(q.get("detail_clamp", 0.08))

    def _add(hp: np.ndarray, amount: float) -> None:
        nonlocal h
        if guard_w is not None:
            hp = hp * guard_w
        if clamp > 0:
            hp = np.clip(hp, -clamp, clamp)
        h = h + hp * amount

    if q["detail"] != 0:
        blur = cv2.GaussianBlur(h, (0, 0), max(0.3, float(q["detail_radius"])))
        _add(h - blur, float(q["detail"]) * 2.0)

    if q["micro"] != 0 and gray is not None:
        g = gray.astype(np.float32)
        if g.shape != h.shape:
            g = cv2.resize(g, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
        gb = cv2.GaussianBlur(g, (0, 0), max(0.3, float(q["micro_radius"])))
        _add(g - gb, float(q["micro"]) * 0.5)

    # Światła i cienie działają na surowej sumie, jeszcze przed obcięciem do zakresu —
    # tylko wtedy ujemne „światła” mogą uratować szczyty wypchnięte ponad sufit przez
    # wyostrzanie, zamiast ratować to, co już zostało ścięte.
    h = _tone(h, float(np.clip(q["highlights"], -1.0, 1.0)),
              float(np.clip(q["shadows"], -1.0, 1.0)))

    g_ = float(q["gamma"])
    if abs(g_ - 1.0) > 1e-3:
        h = np.clip(h, 0.0, 1.0) ** max(0.05, g_)

    c = float(np.clip(q["contrast"], -1.0, 1.0))
    if abs(c) > 1e-3:
        hc = np.clip(h, 0.0, 1.0)
        if c > 0:
            h = hc * (1 - c) + _smoothstep(hc) * c
        else:
            inv = 0.5 - np.sin(np.arcsin(np.clip(1 - 2 * hc, -1, 1)) / 3.0)
            h = hc * (1 + c) + inv * (-c)

    h = np.clip(h, 0.0, 1.0)

    f = float(np.clip(q["floor"], 0.0, 0.99))
    if f > 0:
        soft = max(1e-3, float(q["floor_soft"]))
        t = np.clip((h - f) / soft, 0.0, 1.0)
        h = h * _smoothstep(t)
        h = np.clip((h - 0.0) / max(1e-6, 1.0), 0.0, 1.0)

    ef = float(np.clip(q["edge_falloff"], 0.0, 0.5))
    if ef > 0:
        H, W = h.shape
        ry = np.minimum(np.arange(H), np.arange(H)[::-1]) / max(1.0, H * ef)
        rx = np.minimum(np.arange(W), np.arange(W)[::-1]) / max(1.0, W * ef)
        fall = _smoothstep(np.clip(ry, 0, 1))[:, None] * _smoothstep(np.clip(rx, 0, 1))[None, :]
        h = h * fall.astype(np.float32)

    mask = shape_mask(*h.shape, q)
    am = alpha_mask(h.shape, q, alpha)
    if am is not None:
        mask = mask * cv2.GaussianBlur(am, (0, 0), 0.8)   # miękka krawędź sylwetki
    if mask.min() < 1.0:
        h = h * mask

    return np.clip(h, 0.0, 1.0).astype(np.float32)


def preview_png(h: np.ndarray) -> bytes:
    img = (np.clip(h, 0, 1) * 65535.0).astype(np.uint16)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Nie udało się zakodować PNG")
    return buf.tobytes()
