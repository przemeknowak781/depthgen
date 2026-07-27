"""DepthGen — lokalny generator płaskorzeźb 3D z obrazu."""
from __future__ import annotations

import io
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import depth as depth_mod
from . import heightmap as hm_mod
from . import mesh as mesh_mod

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MAX_SIDE = 2400  # górny limit rozdzielczości roboczej obrazu

app = FastAPI(title="DepthGen")

SESSIONS: dict[str, dict] = {}
PROGRESS: dict[str, str] = {}


def _sess(sid: str) -> dict:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Nie znaleziono obrazu — wgraj go ponownie.")
    return s


def _trim_sessions(keep: int = 6) -> None:
    if len(SESSIONS) <= keep:
        return
    for k in sorted(SESSIONS, key=lambda k: SESSIONS[k]["t"])[:-keep]:
        SESSIONS.pop(k, None)
        PROGRESS.pop(k, None)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/info")
def info() -> dict:
    return {
        "device": depth_mod.device_info(),
        "models": [
            {"key": k, "label": v["label"], "default_size": v["default_size"]}
            for k, v in depth_mod.MODELS.items()
        ],
        "defaults": {**hm_mod.DEFAULTS, **mesh_mod.MESH_DEFAULTS},
    }


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    data = file.file.read()
    try:
        src = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Nie udało się odczytać obrazu.")
    if max(src.size) > MAX_SIDE:
        s = MAX_SIDE / max(src.size)
        src = src.resize((int(src.width * s), int(src.height * s)), Image.LANCZOS)

    # Przezroczystość zachowujemy osobno — posłuży do wycięcia sylwetki. Sieć głębi
    # dostaje obraz podłożony na neutralnej szarości, żeby nie brała pustego tła
    # za czarną ścianę tuż przed obiektem.
    alpha = None
    if src.mode in ("RGBA", "LA") or (src.mode == "P" and "transparency" in src.info):
        rgba = src.convert("RGBA")
        a = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
        if a.min() < 0.99:
            alpha = a
        bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        img = src.convert("RGB")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = (arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)

    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"img": img, "gray": gray, "alpha": alpha, "depth": None,
                     "t": time.time(), "name": file.filename or "obraz"}
    _trim_sessions()
    return {"id": sid, "width": img.width, "height": img.height,
            "name": file.filename, "has_alpha": alpha is not None}


@app.get("/api/image/{sid}")
def image(sid: str) -> Response:
    buf = io.BytesIO()
    _sess(sid)["img"].save(buf, format="JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/api/progress/{sid}")
def progress(sid: str) -> dict:
    return {"msg": PROGRESS.get(sid, "")}


@app.post("/api/depth")
def make_depth(payload: dict = Body(...)) -> dict:
    sid = payload.get("id", "")
    s = _sess(sid)
    t0 = time.time()
    PROGRESS[sid] = "Start..."
    try:
        res = depth_mod.estimate(
            s["img"],
            model_key=payload.get("model", "dav2-large"),
            input_size=int(payload.get("input_size", 700)),
            tiles=int(payload.get("tiles", 1)),
            tile_overlap=float(payload.get("tile_overlap", 0.25)),
            tile_blend=float(payload.get("tile_blend", 0.7)),
            progress=lambda m: PROGRESS.__setitem__(sid, m),
        )
    except Exception as e:  # noqa: BLE001
        PROGRESS[sid] = ""
        raise HTTPException(500, f"Błąd generowania głębi: {e}")
    s["depth"] = res.depth
    s["t"] = time.time()
    PROGRESS[sid] = ""
    return {"ok": True, "ms": int((time.time() - t0) * 1000), "shape": list(res.depth.shape)}


@app.post("/api/depth-upload")
def depth_upload(id: str = Form(...), file: UploadFile = File(...)) -> dict:
    """Wgranie własnej mapy głębi (np. z innego narzędzia)."""
    s = _sess(id)
    data = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "Nie udało się odczytać mapy głębi.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = img.astype(np.float32)
    d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
    d = cv2.resize(d, (s["img"].width, s["img"].height), interpolation=cv2.INTER_CUBIC)
    s["depth"] = d.astype(np.float32)
    return {"ok": True, "shape": list(d.shape)}


def _height(sid: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    s = _sess(sid)
    if s["depth"] is None:
        raise HTTPException(400, "Najpierw wygeneruj mapę głębi.")
    alpha = s.get("alpha")
    h = hm_mod.build(s["depth"], s["gray"], params, alpha)
    return h, hm_mod.cut_mask(params, h, alpha)


@app.post("/api/heightmap")
def heightmap_png(payload: dict = Body(...)) -> Response:
    h, _ = _height(payload.get("id", ""), payload.get("params", {}))
    return Response(hm_mod.preview_png(h), media_type="image/png")


@app.post("/api/mesh")
def mesh_preview(payload: dict = Body(...)) -> Response:
    sid = payload.get("id", "")
    params = payload.get("params", {})
    h, mask = _height(sid, params)
    m = mesh_mod.build(h, {**params, "resolution": int(payload.get("resolution", 400))}, mask)
    body = mesh_mod.to_binary(m)
    st = m.stats
    return Response(
        body,
        media_type="application/octet-stream",
        headers={
            "X-Mesh-Vertices": str(st["vertices"]),
            "X-Mesh-Faces": str(st["faces"]),
            "X-Mesh-Size": ",".join(str(v) for v in st["size_mm"]),
        },
    )


@app.post("/api/export")
def export(payload: dict = Body(...)) -> Response:
    sid = payload.get("id", "")
    params = payload.get("params", {})
    fmt = str(payload.get("format", "stl")).lower()
    h, mask = _height(sid, params)
    m = mesh_mod.build(h, {**params, "resolution": int(payload.get("resolution", 1200))}, mask)
    try:
        data = mesh_mod.export(m, fmt)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Błąd eksportu: {e}")
    name = Path(_sess(sid)["name"]).stem or "relief"
    return Response(
        data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{name}_relief.{fmt}"',
            "X-Mesh-Faces": str(m.stats["faces"]),
        },
    )


@app.get("/favicon.ico")
def favicon() -> Response:
    p = STATIC / "favicon.ico"
    return FileResponse(p) if p.exists() else Response(status_code=204)


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
