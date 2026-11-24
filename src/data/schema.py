"""Schéma des fichiers raw produits par extract_raw.py + découverte des vidéos.

Conventions
-----------
- Un run = un dataset : `--root` pointe sur la racine du corpus et `--dataset` (défaut: nom du
  dossier racine) estampille la sortie -> OUT/<dataset>/<arborescence des vidéos>/...
  Plusieurs corpus peuvent donc partager le même --out sans collision.
- Géométrie (corpus JSL_DATASET) : chaque vidéo est une grille 2x2 (participant gauche/droite
  x caméra front/side, séparateurs auto-détectés). Chaque quadrant est traité par son propre
  HolisticLandmarker (mode VIDEO) et produit `<stem>__<side>_<view>.npz`.
- Coordonnées normalisées 0..1 **dans le quadrant**, avec `crop_xyxy` et `frame_size_wh` stockés :
  tout est ramenable aux coordonnées pleine frame (voir load_raw.to_full_frame).
- 5 colonnes par point : x, y, z, visibility, presence. NaN = région ou champ absent sur la frame.
  Aucun filtrage de points, aucune interpolation, aucune annotation ELAN lue.
"""
import json
import re
from pathlib import Path

# name -> nombre de points, dans l'ordre des attributs de HolisticLandmarkerResult
REGIONS = {"pose": 33, "face": 478, "left_hand": 21, "right_hand": 21,
           "pose_world": 33, "left_hand_world": 21, "right_hand_world": 21}
COLUMNS = ["x", "y", "z", "visibility", "presence"]
SIDES = ("left", "right")
VIEWS = ("front", "side")
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
MEDIA_DIR_NAMES = {"media_data", "videos", "video", "mp4", "clips"}   # dossier "conteneur" ignoré pour region/topic
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "holistic_landmarker.task"

_PREFIX = r"[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*"
_NUM = r"\d+[A-Za-z]?"
# préfixe répété : TY_11_TY_30F_... ; préfixe partagé : NR_13_14_... / GM_11-12_...
_PAIR_RES = [
    re.compile(rf"^({_PREFIX})[_\-]({_NUM})[_\-]({_PREFIX})[_\-]({_NUM})(?:[_\-]|$)"),
    re.compile(rf"^({_PREFIX})[_\-]({_NUM})[_\-]({_NUM})(?:[_\-]|$)"),
]


def parse_pair_ids(stem):
    """'NR_13_14_AniN' -> ['NR_13','NR_14'] ; 'TY_11_TY_30F_Moto' -> ['TY_11','TY_30F'] ; sinon []."""
    m = _PAIR_RES[0].match(stem)          # préfixe répété : TY_11_TY_30F
    if m:
        p1, n1, p2, n2 = m.groups()
        return [f"{p1}_{n1}", f"{p2}_{n2}"]
    m = _PAIR_RES[1].match(stem)          # préfixe partagé : NR_13_14 / GM_11-12
    if m:
        p1, n1, n2 = m.groups()
        return [f"{p1}_{n1}", f"{p1}_{n2}"]
    return []


def pid_number(pid):
    m = re.search(r"\d+", pid)
    return int(m.group(0)) if m else None


def side_for_pid(pid, odd_side):
    """Convention du corpus JSL : le participant au numéro impair est d'un côté donné (défaut: droite)."""
    n = pid_number(pid)
    if n is None:
        return None
    return odd_side if n % 2 else ("left" if odd_side == "right" else "right")


def find_videos(root, exclude=("zz-Backup",)):
    """Toutes les vidéos sous root (récursif), triées, en ignorant les chemins contenant un motif exclu."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        if any(x and x in p.relative_to(root).as_posix() for x in exclude):
            continue
        out.append(p)
    return out


def _norm_stem(s):
    return re.sub(r"[-\s]+", "_", s)     # 'NR_13-14_AniN' et 'NR_13_14_AniN' sont le même fichier


def video_meta(root, video_path, dataset):
    """Métadonnées de provenance : dataset, region/topic (best effort depuis les dossiers),
    ids de paire parsés du nom de fichier, eaf voisin s'il existe (chemin seul, jamais lu)."""
    root, video_path = Path(root), Path(video_path)
    rel = video_path.relative_to(root)
    dirs = list(rel.parts[:-1])
    while dirs and dirs[-1].lower() in MEDIA_DIR_NAMES:
        dirs.pop()
    region = dirs[0] if len(dirs) >= 1 else ""
    topic = dirs[1] if len(dirs) >= 2 else ""
    stem = video_path.stem
    eaf = ""
    for d in (video_path.parent, video_path.parent.parent):
        if d.is_dir():
            for cand in d.glob("*.eaf"):
                if _norm_stem(cand.stem) == _norm_stem(stem):
                    eaf = cand.relative_to(root).as_posix()
                    break
        if eaf:
            break
    return dict(dataset=dataset, region=region, topic=topic, pair_ids=parse_pair_ids(stem),
                source_video=rel.as_posix(), eaf=eaf)


def stream_path(out_root, dataset, rel_video, side, view):
    """Chemin de sortie d'un flux : OUT/<dataset>/<arborescence vidéo>/<stem>__<side>_<view>.npz"""
    rel = Path(rel_video)
    return Path(out_root) / dataset / rel.parent / f"{rel.stem}__{side}_{view}.npz"


def streams_for_video(out_root, dataset, rel_video, views):
    return {(side, view): stream_path(out_root, dataset, rel_video, side, view)
            for side in SIDES for view in views}


def options_json(**kwargs):
    return json.dumps(kwargs, sort_keys=True)
