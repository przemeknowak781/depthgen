"""Mapa wysokości -> siatka 3D (płaskorzeźba)."""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np

MESH_DEFAULTS = {
    "resolution": 512,      # próbki wzdłuż dłuższej krawędzi siatki
    "width_mm": 100.0,      # szerokość fizyczna
    "relief_mm": 8.0,       # maksymalna wysokość reliefu
    "base_mm": 2.0,         # grubość płyty bazowej
    "solid": True,          # zamknięta bryła (do druku) vs sama powierzchnia
    "trim": False,          # przycięcie siatki do kształtu maski
    "z_offset": 0.0,
}


class EmptyMesh(ValueError):
    """Po przycięciu nie zostało nic, z czego dałoby się zbudować bryłę."""


@dataclass
class Mesh:
    vertices: np.ndarray  # (N,3) float32
    faces: np.ndarray     # (M,3) int32
    normals: np.ndarray | None = None

    @property
    def stats(self) -> dict:
        bb = self.vertices.max(0) - self.vertices.min(0)
        return {
            "vertices": int(len(self.vertices)),
            "faces": int(len(self.faces)),
            "size_mm": [round(float(v), 2) for v in bb],
        }


def _grid_shape(h: int, w: int, res: int) -> tuple[int, int]:
    res = int(max(16, min(6000, res)))
    if w >= h:
        gw = res
        gh = max(16, int(round(res * h / w)))
    else:
        gh = res
        gw = max(16, int(round(res * w / h)))
    return gh, gw


def build(height: np.ndarray, p: dict, mask: np.ndarray | None = None) -> Mesh:
    q = {**MESH_DEFAULTS, **(p or {})}
    H, W = height.shape
    gh, gw = _grid_shape(H, W, q["resolution"])

    interp = cv2.INTER_AREA if gw < W else cv2.INTER_CUBIC
    hm = cv2.resize(height.astype(np.float32), (gw, gh), interpolation=interp)
    hm = np.clip(hm, 0.0, 1.0)

    solid = bool(q["solid"])
    base = float(q["base_mm"]) if solid else 0.0
    relief = float(q["relief_mm"])
    width_mm = float(q["width_mm"])
    height_mm = width_mm * (H / W)

    xs = (np.arange(gw, dtype=np.float32) / (gw - 1) - 0.5) * width_mm
    ys = (0.5 - np.arange(gh, dtype=np.float32) / (gh - 1)) * height_mm
    X = np.broadcast_to(xs[None, :], (gh, gw))
    Y = np.broadcast_to(ys[:, None], (gh, gw))
    Z = base + hm * relief + float(q["z_offset"])

    if q["trim"] and mask is not None:
        mk = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR) > 0.5
    else:
        mk = np.ones((gh, gw), dtype=bool)

    if not mk.any():
        raise EmptyMesh(
            "Przycinanie usunęło całą bryłę. Zmniejsz „Odetnij poniżej wysokości”, "
            "„Margines” lub „Próg tła”."
        )

    idx = np.full((gh, gw), -1, dtype=np.int64)
    n_top = int(mk.sum())
    idx[mk] = np.arange(n_top, dtype=np.int64)

    verts = [np.stack([X[mk], Y[mk], Z[mk]], axis=1).astype(np.float32)]

    v00 = idx[:-1, :-1]
    v10 = idx[:-1, 1:]
    v01 = idx[1:, :-1]
    v11 = idx[1:, 1:]
    cell = (v00 >= 0) & (v10 >= 0) & (v01 >= 0) & (v11 >= 0)

    if not cell.any():
        raise EmptyMesh(
            "Po przycięciu zostały tylko pojedyncze punkty — nie da się z nich zbudować "
            "powierzchni. Poluzuj progi przycinania albo podnieś rozdzielczość."
        )

    a, b, c, d = v01[cell], v11[cell], v10[cell], v00[cell]
    faces = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]

    if solid:
        # Dno: wachlarz od środka jest tani, ale poprawny tylko dla kształtów wypukłych.
        # Przy wycinaniu do sylwetki (kanał alfa, próg wysokości) obrys bywa wklęsły
        # i może się rozpaść na kilka wysp — wtedy dno musi być lustrem siatki wierzchu.
        bottom = str(q.get("bottom", "auto"))
        if bottom == "auto":
            complex_cut = bool(q.get("alpha_cut")) or float(q.get("cut_level", 0.0)) > 0
            bottom = "grid" if (q["trim"] and complex_cut) else "fan"

        cu = np.zeros_like(cell)   # sąsiad wyżej (wiersz-1)
        cd = np.zeros_like(cell)   # niżej
        cl = np.zeros_like(cell)
        cr = np.zeros_like(cell)
        cu[1:, :] = cell[:-1, :]
        cd[:-1, :] = cell[1:, :]
        cl[:, 1:] = cell[:, :-1]
        cr[:, :-1] = cell[:, 1:]

        edges = []
        # pętla CCW komórki: v01 -> v11 -> v10 -> v00 -> v01
        e = cell & ~cd                       # dolna krawędź: v01 -> v11
        edges.append(np.stack([v01[e], v11[e]], 1))
        e = cell & ~cr                       # prawa: v11 -> v10
        edges.append(np.stack([v11[e], v10[e]], 1))
        e = cell & ~cu                       # górna: v10 -> v00
        edges.append(np.stack([v10[e], v00[e]], 1))
        e = cell & ~cl                       # lewa: v00 -> v01
        edges.append(np.stack([v00[e], v01[e]], 1))
        edge = np.concatenate(edges, 0)

        z_bot = float(q["z_offset"])
        if bottom == "grid":
            lut = np.arange(n_top, dtype=np.int64) + n_top
            bot = verts[0].copy()
            bot[:, 2] = z_bot
            verts.append(bot)
            # dno = siatka wierzchu z odwróconą kolejnością wierzchołków
            faces.append(np.stack([a + n_top, c + n_top, b + n_top], 1))
            faces.append(np.stack([a + n_top, d + n_top, c + n_top], 1))
        else:
            ring = np.unique(edge.ravel())
            lut = np.full(n_top, -1, dtype=np.int64)
            lut[ring] = np.arange(len(ring), dtype=np.int64) + n_top
            bot = verts[0][ring].copy()
            bot[:, 2] = z_bot
            verts.append(bot)
            centroid = np.array([[bot[:, 0].mean(), bot[:, 1].mean(), z_bot]], dtype=np.float32)
            verts.append(centroid)

        at, bt = edge[:, 0], edge[:, 1]
        ab, bb = lut[at], lut[bt]
        faces.append(np.stack([at, ab, bb], 1))     # ściana boczna (na zewnątrz)
        faces.append(np.stack([at, bb, bt], 1))
        if bottom != "grid":
            cc = np.full(len(edge), n_top + len(ring), dtype=np.int64)
            faces.append(np.stack([cc, bb, ab], 1))  # spód (wachlarz od środka)

    V = np.concatenate(verts, 0).astype(np.float32)
    F = np.concatenate(faces, 0).astype(np.int32)
    return Mesh(vertices=V, faces=F)


def vertex_normals(m: Mesh) -> np.ndarray:
    v, f = m.vertices, m.faces.astype(np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for i in range(3):
        np.add.at(n, f[:, i], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(ln, 1e-12)).astype(np.float32)


def to_binary(m: Mesh) -> bytes:
    """Pakiet dla podglądu WebGL: pozycje f32, normalne f32, indeksy u32."""
    n = vertex_normals(m)
    header = np.array([len(m.vertices), len(m.faces)], dtype=np.uint32)
    return (
        header.tobytes()
        + m.vertices.astype(np.float32).tobytes()
        + n.tobytes()
        + m.faces.astype(np.uint32).tobytes()
    )


def export(m: Mesh, fmt: str) -> bytes:
    import trimesh

    tm = trimesh.Trimesh(vertices=m.vertices.astype(np.float64), faces=m.faces, process=False)
    fmt = fmt.lower()
    if fmt == "stl":
        return trimesh.exchange.stl.export_stl(tm)
    if fmt == "obj":
        return trimesh.exchange.obj.export_obj(tm).encode("utf-8")
    if fmt == "ply":
        return trimesh.exchange.ply.export_ply(tm, encoding="binary")
    if fmt == "glb":
        return trimesh.exchange.gltf.export_glb(trimesh.Scene(tm))
    if fmt == "3mf":
        buf = io.BytesIO()
        tm.export(buf, file_type="3mf")
        return buf.getvalue()
    raise ValueError(f"Nieobsługiwany format: {fmt}")
