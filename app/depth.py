"""Lokalne generowanie map głębi (Depth Anything V2 / DPT) — bez chmury."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODELS = {
    "dav2-small": {
        "repo": "depth-anything/Depth-Anything-V2-Small-hf",
        "label": "Depth Anything V2 — Small (szybki, ~50 MB)",
        "default_size": 518,
    },
    "dav2-base": {
        "repo": "depth-anything/Depth-Anything-V2-Base-hf",
        "label": "Depth Anything V2 — Base (~390 MB)",
        "default_size": 518,
    },
    "dav2-large": {
        "repo": "depth-anything/Depth-Anything-V2-Large-hf",
        "label": "Depth Anything V2 — Large (najlepszy detal, ~1.3 GB)",
        "default_size": 700,
    },
    "dpt-large": {
        "repo": "Intel/dpt-large",
        "label": "MiDaS DPT-Large (Intel, alternatywny charakter)",
        "default_size": 384,
    },
    "dpt-beit-large": {
        "repo": "Intel/dpt-beit-large-512",
        "label": "MiDaS 3.1 BEiT-L 512 (bardzo ostry, ~1.4 GB)",
        "default_size": 512,
    },
}

# Strojenie łączenia kafli.
# Filtr dryfu w px: powyżej tej skali struktura pochodzi z przebiegu globalnego,
# poniżej — z kafli. Musi być mały; szeroki filtr rozmazuje skok głębi na sylwetce
# i tworzy wokół niej halo.
TILE_DRIFT_PX = 8.0
TILE_BAND_SPLIT = 6.0     # px — granica pasma niskiego/wysokiego
TILE_HI_BAND = 6.0        # px — szerokość przejścia dla pasma wysokiego
TILE_ALIGN_REG = 0.02     # regularyzacja dopasowania kafla (ułamek wariancji mapy)
TILE_FLAT_GUARD = 0.03    # próg "płaskości", poniżej którego detal z kafli jest tłumiony
TILE_EDGE_GUARD = 0.60    # tłumienie detalu na skokach głębi (poświata wokół sylwetki)

_lock = threading.Lock()
_cache: dict[str, tuple] = {}


def device_info() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda else "cpu",
        "name": torch.cuda.get_device_name(0) if cuda else "CPU",
        "torch": torch.__version__,
    }


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(model_key: str, progress=None):
    """Ładuje (i cache'uje) model. Pierwsze użycie pobiera wagi z HuggingFace."""
    if model_key not in MODELS:
        raise ValueError(f"Nieznany model: {model_key}")
    with _lock:
        if model_key in _cache:
            return _cache[model_key]
        repo = MODELS[model_key]["repo"]
        if progress:
            progress(f"Ładowanie modelu {repo} (pierwszy raz = pobieranie wag)...")
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForDepthEstimation.from_pretrained(repo)
        model.eval().to(_device())
        _cache[model_key] = (processor, model)
        return _cache[model_key]


@dataclass
class DepthResult:
    depth: np.ndarray  # float32, HxW, znormalizowana 0..1 (1 = najbliżej kamery)
    raw_min: float
    raw_max: float


def _infer(processor, model, img: Image.Image, size: int) -> np.ndarray:
    """Jedno przejście sieci; zwraca surową mapę w rozdzielczości obrazu wejściowego."""
    inputs = processor(images=img, return_tensors="pt", size={"height": size, "width": size})
    inputs = {k: v.to(_device()) for k, v in inputs.items()}
    with torch.inference_mode():
        if _device().type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**inputs).predicted_depth
        else:
            out = model(**inputs).predicted_depth
    out = out.float()
    if out.ndim == 3:
        out = out.unsqueeze(1)
    out = torch.nn.functional.interpolate(
        out, size=(img.height, img.width), mode="bicubic", align_corners=False
    )
    return out[0, 0].cpu().numpy().astype(np.float32)


def _align(src: np.ndarray, ref: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """Dopasowuje skalę i przesunięcie `src` do `ref` metodą najmniejszych kwadratów.

    `reg` to regularyzacja Tichonowa. Bez niej kafel złożony niemal wyłącznie z płaskiego
    tła ma wariancję bliską zeru, dopasowanie skali robi się źle uwarunkowane i wzmacnia
    szum — w efekcie na tle pojawiają się prostokąty kafli.
    """
    a = src.ravel().astype(np.float64)
    b = ref.ravel().astype(np.float64)
    ma, mb = a.mean(), b.mean()
    va = ((a - ma) ** 2).mean()
    cov = ((a - ma) * (b - mb)).mean()
    scale = cov / (va + reg) if (va + reg) > 1e-12 else 1.0
    scale = float(np.clip(scale, 0.25, 4.0))
    return (src - ma) * scale + mb


def _axis_weight(length: int, n: int, idx: int, band: float) -> np.ndarray:
    """Waga kafla `idx` wzdłuż osi, w układzie całego obrazu.

    Przejście jest wyśrodkowane na granicy terytoriów kafli i ma szerokość `band`.
    Dzięki smoothstepowi wagi sąsiadów sumują się dokładnie do 1 — nie trzeba
    niczego normalizować, a łączenie nie wprowadza schodka.
    """
    x = np.arange(length, dtype=np.float32)
    step = length / n
    t0, t1 = idx * step, (idx + 1) * step
    band = max(1e-3, band)
    ones = np.ones(length, dtype=np.float32)
    left = _smooth(np.clip((x - (t0 - band / 2)) / band, 0, 1)) if idx > 0 else ones
    right = _smooth(np.clip(((t1 + band / 2) - x) / band, 0, 1)) if idx < n - 1 else ones
    return (left * right).astype(np.float32)


def _smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def estimate(
    img: Image.Image,
    model_key: str = "dav2-large",
    input_size: int = 700,
    tiles: int = 1,
    tile_overlap: float = 0.25,
    tile_blend: float = 0.7,
    progress=None,
) -> DepthResult:
    """Mapa głębi. `tiles>1` uruchamia dodatkowy przebieg kafelkowy dla detalu."""
    processor, model = load(model_key, progress)
    img = img.convert("RGB")

    if progress:
        progress("Przebieg globalny...")
    base = _infer(processor, model, img, input_size)

    if tiles > 1:
        import cv2

        W, H = img.width, img.height
        n = int(tiles)
        step_x, step_y = W / n, H / n
        ov_x, ov_y = step_x * tile_overlap, step_y * tile_overlap

        # Łączenie dwupasmowe. Uśrednianie kafli w szerokiej zakładce wygasza drobny
        # detal (dwa nieco różne wzory znoszą się nawzajem) — to właśnie daje widoczną
        # kratkę gładszych pasów. Dlatego pasmo niskie mieszamy szeroko i miękko,
        # a wysokie przełączamy w wąskim pasie, gdzie nie ma czego kasować.
        lo_acc = np.zeros((H, W), np.float32)
        hi_acc = np.zeros((H, W), np.float32)
        lo_w = np.zeros((H, W), np.float32)
        hi_w = np.zeros((H, W), np.float32)
        reg = TILE_ALIGN_REG * float(base.var())
        total = n * n
        for i in range(n):
            for j in range(n):
                x0 = int(max(0, j * step_x - ov_x))
                x1 = int(min(W, (j + 1) * step_x + ov_x))
                y0 = int(max(0, i * step_y - ov_y))
                y1 = int(min(H, (i + 1) * step_y + ov_y))
                if progress:
                    progress(f"Kafel detalu {i * n + j + 1}/{total}...")
                crop = img.crop((x0, y0, x1, y1))
                d = _infer(processor, model, crop, input_size)
                d = _align(d, base[y0:y1, x0:x1], reg=reg)

                lo = cv2.GaussianBlur(d, (0, 0), TILE_BAND_SPLIT,
                                      borderType=cv2.BORDER_REPLICATE)
                hi = d - lo
                wl = (_axis_weight(H, n, i, 2 * ov_y)[y0:y1, None]
                      * _axis_weight(W, n, j, 2 * ov_x)[None, x0:x1])
                wh = (_axis_weight(H, n, i, TILE_HI_BAND)[y0:y1, None]
                      * _axis_weight(W, n, j, TILE_HI_BAND)[None, x0:x1])
                lo_acc[y0:y1, x0:x1] += lo * wl
                lo_w[y0:y1, x0:x1] += wl
                hi_acc[y0:y1, x0:x1] += hi * wh
                hi_w[y0:y1, x0:x1] += wh
        detail = lo_acc / np.maximum(lo_w, 1e-6) + hi_acc / np.maximum(hi_w, 1e-6)
        # Kluczowe dla braku "kratki": kafle różnią się od przebiegu globalnego przede
        # wszystkim niskimi częstotliwościami (każdy jest normalizowany na swoim wycinku
        # sceny). Te różnice zmieniają się na skali całego kafla, więc odejmujemy je
        # filtrem o sigmie rzędu ćwierci kafla — zostają same wysokie częstotliwości.
        import cv2

        drift = cv2.GaussianBlur(detail - base, (0, 0), TILE_DRIFT_PX,
                                 borderType=cv2.BORDER_REPLICATE)
        merged = detail - drift

        # W obszarach bez struktury (płaskie tło) kafle nie mają czego wnieść — zostaje
        # tam sam szum, który układa się w prostokąty kafli. Wpuszczamy detal tylko tam,
        # gdzie przebieg globalny widzi realną strukturę.
        s = 12.0
        conf = np.float32(1.0)
        mean = cv2.GaussianBlur(base, (0, 0), s, borderType=cv2.BORDER_REPLICATE)
        sq = cv2.GaussianBlur(base * base, (0, 0), s, borderType=cv2.BORDER_REPLICATE)
        act = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        if TILE_FLAT_GUARD > 0:
            thr = float(np.percentile(act, 90)) * TILE_FLAT_GUARD
            if thr > 0:
                conf = _smooth(np.clip(act / thr, 0.0, 1.0)).astype(np.float32)

        # Na skoku głębi (obrys postaci) kafle "dzwonią" — przestrzelony detal tworzy
        # poświatę wokół sylwetki. Tam też odcinamy wstrzykiwanie detalu.
        if TILE_EDGE_GUARD > 0:
            gx = cv2.Sobel(mean, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(mean, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), s,
                                    borderType=cv2.BORDER_REPLICATE)
            gt = float(np.percentile(grad, 97)) * TILE_EDGE_GUARD
            if gt > 0:
                conf = conf * np.exp(-((grad / gt) ** 2)).astype(np.float32)

        base = base + (merged - base) * (tile_blend * conf)

    lo, hi = float(base.min()), float(base.max())
    norm = (base - lo) / max(hi - lo, 1e-6)
    return DepthResult(depth=norm.astype(np.float32), raw_min=lo, raw_max=hi)
