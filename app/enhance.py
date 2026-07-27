"""Przygotowanie obrazu: usuwanie artefaktów JPEG i upscaling.

Artefakty kompresji wchodzą do reliefu dwiema drogami: bezpośrednio przez „mikrodetal
z obrazu" (wysokie częstotliwości luminancji to w dużej części kwadraty 8x8 i dzwonienie
przy krawędziach) oraz pośrednio przez sieć głębi, która na blokach widzi realną
strukturę. Dlatego czyścimy obraz zanim cokolwiek policzymy.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
import torch
from PIL import Image

SR_MODELS = {
    "none": {"repo": None, "scale": 1, "label": "bez upscalingu"},
    "compressed-x4": {
        "repo": "caidas/swin2SR-compressed-sr-x4-48",
        "scale": 4,
        "label": "Swin2SR 4x — dla obrazów z kompresją JPEG (zalecany)",
    },
    "classical-x2": {
        "repo": "caidas/swin2SR-classical-sr-x2-64",
        "scale": 2,
        "label": "Swin2SR 2x — dla obrazów czystych",
    },
    "realworld-x4": {
        "repo": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
        "scale": 4,
        "label": "Swin2SR 4x — zdjęcia z internetu, mocno zniszczone",
    },
}

DEFAULTS = {
    "deblock": 0.0,        # 0..1 — siła usuwania artefaktów JPEG
    "chroma": 0.5,         # 0..1 — dodatkowe czyszczenie kanałów koloru
    "sr_model": "none",
    "work_max": 2400,      # maksymalny bok obrazu roboczego
}

SR_TILE = 192              # px wejścia na kafel (pamięć GPU)
SR_OVERLAP = 24
SR_MAX_OUTPUT = 4200       # px — sufit rozmiaru po upscalingu (pamięć i czas)

_lock = threading.Lock()
_cache: dict[str, tuple] = {}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def blockiness(gray: np.ndarray, grid: int = 8) -> float:
    """Miara artefaktów blokowych: energia gradientu na siatce 8x8 vs poza nią.

    1.0 = brak struktury blokowej. Obrazy mocno skompresowane dają 1.3–3.
    """
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(g, axis=0)).mean(axis=1)
    out = []
    for prof in (dx, dy):
        idx = np.arange(len(prof))
        on = (idx + 1) % grid == 0
        if on.sum() < 2 or (~on).sum() < 2:
            continue
        out.append(prof[on].mean() / max(prof[~on].mean(), 1e-9))
    return float(np.mean(out)) if out else 1.0


def _deblock_grid(y: np.ndarray, strength: float, grid: int = 8) -> np.ndarray:
    """Wygładza wyłącznie granice bloków 8x8, i tylko tam, gdzie nie ma prawdziwej
    krawędzi — filtr w duchu deblockingu z H.264, ale liczony wektorowo.
    """
    out = y.copy()
    thr = 0.02 + 0.10 * strength          # skok uznawany jeszcze za artefakt
    amt = 0.35 + 0.55 * strength

    for axis in (1, 0):
        a = out if axis == 1 else out.T
        n = a.shape[1]
        cols = np.arange(grid - 1, n - 2, grid)
        if len(cols) == 0:
            continue
        p1, p0 = a[:, cols - 1], a[:, cols]
        q0, q1 = a[:, cols + 1], a[:, cols + 2]
        step = np.abs(q0 - p0)
        flat = (np.abs(p1 - p0) < thr) & (np.abs(q1 - q0) < thr) & (step < thr * 2)
        w = flat.astype(np.float32) * amt
        # przesuń krawędziowe próbki ku wspólnej średniej
        mid = (p0 + q0) * 0.5
        a[:, cols] = p0 + (mid - p0) * w
        a[:, cols + 1] = q0 + (mid - q0) * w
        a[:, cols - 1] = p1 + ((p1 + mid) * 0.5 - p1) * w * 0.5
        a[:, cols + 2] = q1 + ((q1 + mid) * 0.5 - q1) * w * 0.5
        out = a if axis == 1 else a.T
    return out


def deblock(img: Image.Image, strength: float, chroma: float = 0.5) -> Image.Image:
    """Usuwa bloki i dzwonienie JPEG, starając się nie zjeść faktury."""
    if strength <= 0 and chroma <= 0:
        return img
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]

    if strength > 0:
        y = _deblock_grid(y, float(np.clip(strength, 0, 1)))
        # dzwonienie przy krawędziach: filtr bilateralny o małym promieniu
        d = 5 if strength < 0.6 else 7
        y = cv2.bilateralFilter(y, d, 0.02 + 0.06 * strength, 3 + 3 * strength)

    if chroma > 0:
        # chroma jest w JPEG podpróbkowana i najbrudniejsza, można ją ciąć mocno
        for c in (cr, cb):
            c[:] = cv2.bilateralFilter(c, 7, 0.03 + 0.10 * chroma, 4 + 6 * chroma)

    ycc[..., 0], ycc[..., 1], ycc[..., 2] = y, cr, cb
    out = cv2.cvtColor((np.clip(ycc, 0, 1) * 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def load_sr(key: str, progress=None):
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

    with _lock:
        if key in _cache:
            return _cache[key]
        repo = SR_MODELS[key]["repo"]
        if progress:
            progress(f"Ładowanie modelu upscalingu {repo}...")
        proc = AutoImageProcessor.from_pretrained(repo)
        model = Swin2SRForImageSuperResolution.from_pretrained(repo)
        model.eval().to(_device())
        _cache[key] = (proc, model)
        return _cache[key]


def _sr_tile(model, tile: np.ndarray) -> np.ndarray:
    """Jeden kafel: HWC float 0..1 -> HWC float 0..1 w skali modelu."""
    t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(_device())
    ph = (8 - t.shape[2] % 8) % 8
    pw = (8 - t.shape[3] % 8) % 8
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    with torch.inference_mode():
        out = model(pixel_values=t).reconstruction
    out = out.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    s = out.shape[0] // t.shape[2]
    if ph or pw:
        out = out[: (t.shape[2] - ph) * s, : (t.shape[3] - pw) * s]
    return out


def _ramp(n: int, pad: int) -> np.ndarray:
    r = np.ones(n, np.float32)
    p = max(1, min(pad, n // 2))
    e = np.linspace(0.0, 1.0, p + 2, dtype=np.float32)[1:-1]
    r[:p], r[-p:] = e, e[::-1]
    return r


def upscale(img: Image.Image, key: str, progress=None) -> Image.Image:
    """Upscaling kafelkowy — cały obraz naraz nie mieści się w pamięci GPU."""
    if key == "none" or SR_MODELS[key]["repo"] is None:
        return img
    _, model = load_sr(key, progress)
    scale = SR_MODELS[key]["scale"]

    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    H, W = a.shape[:2]
    acc = np.zeros((H * scale, W * scale, 3), np.float32)
    wsum = np.zeros((H * scale, W * scale, 1), np.float32)

    step = SR_TILE - SR_OVERLAP
    ys = list(range(0, max(1, H - SR_OVERLAP), step))
    xs = list(range(0, max(1, W - SR_OVERLAP), step))
    total = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(H, y0 + SR_TILE), min(W, x0 + SR_TILE)
            y0c, x0c = max(0, y1 - SR_TILE), max(0, x1 - SR_TILE)
            out = _sr_tile(model, a[y0c:y1, x0c:x1])
            th, tw = out.shape[:2]
            w = (_ramp(th, SR_OVERLAP * scale)[:, None]
                 * _ramp(tw, SR_OVERLAP * scale)[None, :])[..., None]
            acc[y0c * scale:y0c * scale + th, x0c * scale:x0c * scale + tw] += out * w
            wsum[y0c * scale:y0c * scale + th, x0c * scale:x0c * scale + tw] += w
            done += 1
            if progress and done % 5 == 0:
                progress(f"Upscaling {done}/{total} kafli...")
    res = acc / np.maximum(wsum, 1e-6)
    return Image.fromarray((np.clip(res, 0, 1) * 255).round().astype(np.uint8))


def prepare(src: Image.Image, p: dict, progress=None) -> tuple[Image.Image, dict]:
    """Pełne przygotowanie: odblokowanie -> upscaling -> rozmiar roboczy."""
    q = {**DEFAULTS, **(p or {})}
    src = src.convert("RGBA") if src.mode in ("RGBA", "LA", "P") else src.convert("RGB")
    alpha = src.getchannel("A") if src.mode == "RGBA" else None
    rgb = src.convert("RGB")

    before = blockiness(np.asarray(rgb.convert("L"), np.float32) / 255.0)

    if float(q["deblock"]) > 0 or float(q["chroma"]) > 0:
        if progress:
            progress("Usuwanie artefaktów JPEG...")
        rgb = deblock(rgb, float(q["deblock"]), float(q["chroma"]))

    key = str(q["sr_model"])
    if key not in SR_MODELS:
        key = "none"
    work = int(np.clip(q["work_max"], 512, 6000))
    sr_input = None
    if key != "none":
        scale = SR_MODELS[key]["scale"]
        # Wejście do SR ograniczamy tak, żeby wynik dał się policzyć w rozsądnym czasie
        # i żeby po zejściu do rozmiaru roboczego zostało nadpróbkowanie (~1.5x).
        limit = max(256, min(SR_MAX_OUTPUT // scale, int(work * 1.5 / scale)))
        if max(rgb.size) > limit:
            s = limit / max(rgb.size)
            rgb = rgb.resize((max(1, int(rgb.width * s)), max(1, int(rgb.height * s))),
                             Image.LANCZOS)
        sr_input = list(rgb.size)
        rgb = upscale(rgb, key, progress)
        if alpha is not None:
            alpha = alpha.resize(rgb.size, Image.LANCZOS)

    if max(rgb.size) > work:
        s = work / max(rgb.size)
        size = (max(1, int(rgb.width * s)), max(1, int(rgb.height * s)))
        rgb = rgb.resize(size, Image.LANCZOS)   # nadpróbkowanie dodatkowo czyści artefakty
        if alpha is not None:
            alpha = alpha.resize(size, Image.LANCZOS)

    after = blockiness(np.asarray(rgb.convert("L"), np.float32) / 255.0)
    meta = {
        "width": rgb.width, "height": rgb.height,
        "blockiness_before": round(before, 3),
        "blockiness_after": round(after, 3),
        "scale": SR_MODELS[key]["scale"],
        "sr_input": sr_input,
    }
    if alpha is not None:
        rgb.putalpha(alpha)
    return rgb, meta
