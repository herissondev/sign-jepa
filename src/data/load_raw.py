"""Lecture des .npz raw produits par extract_raw.py.

    from src.data.load_raw import load_stream, to_full_frame, iter_streams

`load_stream` renvoie un dict (np.load, allow_pickle=False) + "path".
Les coordonnées x, y sont normalisées 0..1 dans le quadrant (`crop_xyxy`) ; `to_full_frame`
les ramène en coordonnées normalisées pleine frame (z, visibility, presence inchangés).
"""
from pathlib import Path

import numpy as np


def load_stream(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    out["path"] = str(path)
    return out


def to_full_frame(points, crop_xyxy, frame_size_wh):
    """points [T, N, 5] (x, y normalisés dans le crop) -> x, y normalisés pleine frame.

    crop_xyxy = (x0, y0, x1, y1) en pixels dans la frame ; frame_size_wh = (W, H).
    z est une profondeur relative (MediaPipe), elle ne dépend pas du crop : copiée telle quelle.
    """
    x0, y0, x1, y1 = [float(v) for v in crop_xyxy]
    w, h = [float(v) for v in frame_size_wh]
    out = np.array(points, dtype=np.float32, copy=True)
    out[..., 0] = (out[..., 0] * (x1 - x0) + x0) / w
    out[..., 1] = (out[..., 1] * (y1 - y0) + y0) / h
    return out


def iter_streams(out_root):
    """Itère sur tous les .npz raw sous out_root (récursif, fichiers temporaires exclus)."""
    return sorted(p for p in Path(out_root).rglob("*.npz") if not p.name.startswith("tmp"))


def part(stream, name):
    """Raccourci : stream[name] -> array [T, N, 5], ou None si la région n'a jamais été détectée."""
    arr = stream.get(name)
    if arr is None or (np.isnan(arr).all() if arr.size else True):
        return None
    return arr
