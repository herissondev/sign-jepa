"""Extraction raw des landmarks Holistic de toutes les vidéos du corpus JSL_DATASET.

Usage (depuis la racine de sign-jepa) :

    python -m src.data.extract_raw \
        --root /chemin/JSL_DATASET --out /chemin/sortie_raw [--dataset JSL_DATASET] \
        [--front-only] [--views both] [--workers N] [--limit N] [--start-frame N] \
        [--max-frames N] [--overwrite]

Un run = un corpus : les .npz vont dans OUT/<dataset>/<arborescence des vidéos>/ (--dataset
défaut = nom du dossier --root), donc plusieurs corpus peuvent partager le même --out.
Chaque vidéo (grille 2x2 : participant gauche/droite x caméra front/side, séparateurs
détectés automatiquement) produit un .npz par quadrant : `<stem>__<side>_<view>.npz`.
Contenu : pose/face/left_hand/right_hand (+ _world) en [T, N, 5] = x, y, z, visibility,
presence, normalisés 0..1 dans le quadrant ; NaN si absent. S'y ajoutent timestamps_ms,
decoder_timestamps_ms, frame_index, fps, frame_size_wh, crop_xyxy, sep_row, sep_col, side,
view, participant (heuristique, voir --odd-side), pair_ids, region, topic, source_video, eaf,
mediapipe_version et les options du modèle en JSON. Aucune annotation ELAN n'est lue.
Les coordonnées pleine frame se retrouvent avec load_raw.to_full_frame.
"""
import argparse
import math
import multiprocessing
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # best effort : silence les logs TF/mediapipe
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # permet `python src/data/extract_raw.py` aussi

from src.data.schema import (COLUMNS, DEFAULT_MODEL, REGIONS, SIDES, VIEWS, find_videos,
                             options_json, side_for_pid, stream_path, streams_for_video,
                             video_meta)

# ---------------------------------------------------------------- détection de la grille
def detect_separators(frame):
    """Ligne et colonne les plus sombres dans la bande médiane = séparateurs de la grille 2x2."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    lo, hi = int(h * .3), int(h * .6)
    row = lo + int(np.argmin(g[lo:hi].mean(1)))
    loc, hic = int(w * .4), int(w * .6)
    col = loc + int(np.argmin(g[:, loc:hic].mean(0)))
    return row, col


def quadrants(shape, sep_row, sep_col, views):
    """-> {(side, view): (x0, y0, x1, y1)} dans la frame complète."""
    h, w = shape[:2]
    out = {}
    for side, x0, x1 in (("left", 0, sep_col), ("right", sep_col, w)):
        if "front" in views:
            out[(side, "front")] = (x0, 0, x1, sep_row)
        if "side" in views:
            out[(side, "side")] = (x0, sep_row, x1, h)
    return out


# ---------------------------------------------------------------- landmarks
def read_landmarks(result, name):
    points = getattr(result, name + "_landmarks")
    if not points:
        return None
    return np.asarray([(p.x, p.y, p.z, getattr(p, "visibility", None),
                        getattr(p, "presence", None)) for p in points], dtype=np.float32)


def stack_frames(frames, default_count):
    shape = next((f.shape for f in frames if f is not None), (default_count, len(COLUMNS)))
    array = np.full((len(frames), *shape), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if f is not None:
            array[i] = f
    return array


def make_options(model_path, running_mode, conf, delegate):
    return mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.GPU if delegate == "gpu"
            else mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=running_mode,
        min_face_detection_confidence=conf["face_detection"],
        min_face_landmarks_confidence=conf["face_landmarks"],
        min_pose_detection_confidence=conf["pose_detection"],
        min_pose_landmarks_confidence=conf["pose_landmarks"],
        min_hand_landmarks_confidence=conf["hand_landmarks"],
    )


def gpu_works(model_path):
    """Sonde le delegate GPU sur une petite image ; False -> on retombe sur CPU."""
    try:
        import numpy as _np
        options = make_options(model_path, mp.tasks.vision.RunningMode.IMAGE, DEFAULT_CONF, "gpu")
        det = mp.tasks.vision.HolisticLandmarker.create_from_options(options)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=_np.zeros((64, 64, 3), _np.uint8))
        det.detect(img)
        det.close()
        return True
    except Exception:
        return False


DEFAULT_CONF = dict(face_detection=0.5, face_landmarks=0.5, pose_detection=0.5,
                    pose_landmarks=0.5, hand_landmarks=0.5)


# ---------------------------------------------------------------- une vidéo
def process_video(job):
    """Extrait tous les quadrants d'une vidéo -> un .npz par flux. Retourne (n_frames, n_streams)."""
    (video_path, out_root, rel_video, meta, model_path, conf, views, max_frames,
     front_fraction, odd_side, start_frame, delegate) = job
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    with ExitStack() as stack:
        stack.callback(cap.release)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        expected = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # séparateurs : première frame non noire, sauf si --front-fraction force la coupure
        probe = None
        for _ in range(300):
            ok, frame = cap.read()
            if not ok:
                break
            if frame.mean() > 10:
                probe = frame
                break
        if probe is None:
            raise ValueError("Aucune frame non noire décodée")
        h, w = probe.shape[:2]
        if front_fraction is not None:
            sep_row, sep_col = int(round(h * front_fraction)), w // 2
        else:
            sep_row, sep_col = detect_separators(probe)
        quads = quadrants(probe.shape, sep_row, sep_col, views)
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)   # on repart du début (ou de --start-frame)

        options = make_options(model_path, mp.tasks.vision.RunningMode.VIDEO, conf, delegate)
        tracks = {key: stack.enter_context(mp.tasks.vision.HolisticLandmarker.create_from_options(options))
                  for key in quads}
        records = {key: {} for key in quads}
        times, decoder_times, indices = [], [], []
        n = start_frame
        while max_frames is None or n - start_frame < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (h, w):
                raise ValueError("Dimensions variables dans la vidéo")
            pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            prev = times[-1] if times else -1
            if math.isfinite(pos) and round(pos) > prev:
                ts = round(pos)
            elif math.isfinite(fps) and fps > 0:
                ts = max(prev + 1, round(n * 1000 / fps))
            else:
                raise ValueError("Timestamp et cadence indisponibles")
            times.append(ts)
            decoder_times.append(float(pos))
            indices.append(n)
            for key, (x0, y0, x1, y1) in quads.items():
                rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                result = tracks[key].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                for name in REGIONS:
                    if hasattr(result, name + "_landmarks"):
                        records[key].setdefault(name, []).append(read_landmarks(result, name))
            n += 1

        if n == start_frame:
            raise ValueError("Aucune frame décodée")
        if max_frames is None and start_frame == 0 and math.isfinite(expected) and expected > 0 \
                and n < round(expected):
            raise ValueError(f"Lecture interrompue : {n}/{round(expected)} frames")

        pair_ids = meta["pair_ids"]
        side_rule = f"odd->{odd_side}"
        staged = []
        for (side, view), (x0, y0, x1, y1) in sorted(quads.items()):
            out_path = stream_path(out_root, meta["dataset"], rel_video, side, view)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            participant = next((p for p in pair_ids if side_for_pid(p, odd_side) == side), "")
            arrays = {name: stack_frames(frames, REGIONS[name]) for name, frames in records[(side, view)].items()}
            payload = dict(
                **arrays,
                columns=np.array(COLUMNS),
                timestamps_ms=np.asarray(times, dtype=np.int64),
                decoder_timestamps_ms=np.asarray(decoder_times, dtype=np.float64),
                frame_index=np.asarray(indices, dtype=np.int64),
                fps=np.float64(fps),
                frame_size_wh=np.array([w, h], dtype=np.int64),
                sep_row=np.int64(sep_row), sep_col=np.int64(sep_col),
                crop_xyxy=np.array([x0, y0, x1, y1], dtype=np.int64),
                side=side, view=view, participant=participant, pair_ids=np.array(pair_ids),
                side_rule=side_rule, dataset=meta["dataset"], region=meta["region"], topic=meta["topic"],
                source_video=meta["source_video"], eaf=meta["eaf"],
                mediapipe_version=mp.__version__,
                model=Path(model_path).name,
                options=options_json(**conf, running_mode="VIDEO", views=",".join(views),
                                     delegate=delegate),
            )
            with NamedTemporaryFile(dir=out_path.parent, suffix=".npz", delete=False) as tmp:
                temp_path = Path(tmp.name)
                stack.callback(temp_path.unlink, missing_ok=True)
                np.savez_compressed(tmp, **payload)
            staged.append((temp_path, out_path))
        for temp_path, out_path in staged:
            os.replace(temp_path, out_path)
        return n - start_frame, len(staged), sep_row, sep_col


def run_job(job):
    import cv2 as _cv2
    _cv2.setNumThreads(1)
    try:
        n, n_streams, sep_row, sep_col = process_video(job)
        return n, n_streams, "", sep_row, sep_col
    except (OSError, RuntimeError, ValueError, _cv2.error) as exc:
        return 0, 0, str(exc), -1, -1


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="racine du dataset (ex. .../JSL_DATASET)")
    ap.add_argument("--out", required=True, type=Path, help="racine de sortie des .npz raw")
    ap.add_argument("--dataset", default=None,
                    help="nom du dossier de sortie pour ce corpus (défaut: nom du dossier --root) ; "
                         "les .npz iront dans OUT/<dataset>/...")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="chemin du holistic_landmarker.task")
    ap.add_argument("--views", choices=("front", "side", "both"), default="both",
                    help="quels quadrants extraire (défaut: les 4)")
    ap.add_argument("--front-only", action="store_true",
                    help="raccourci pour --views front (2 flux par vidéo au lieu de 4)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processus parallèles (défaut: tous les cores ; chaque worker charge "
                         "son propre HolisticLandmarker, 1 thread OpenCV par worker)")
    ap.add_argument("--limit", type=int, help="traiter au plus N vidéos")
    ap.add_argument("--start-frame", type=int, default=0, help="frame de départ par vidéo")
    ap.add_argument("--max-frames", type=int, help="cap de frames par vidéo (test)")
    ap.add_argument("--heartbeat", type=int, default=60,
                    help="secondes entre deux lignes de statut pendant l'attente (multi-worker)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exclude", action="append", default=["zz-Backup"],
                    help="motif de chemin à ignorer (répétable, défaut: zz-Backup)")
    ap.add_argument("--odd-side", choices=SIDES, default="right",
                    help="côté du participant au numéro impair (heuristique sans ELAN)")
    ap.add_argument("--front-fraction", type=float, default=None,
                    help="force la coupure front/side à cette fraction de hauteur (sinon auto-détectée)")
    ap.add_argument("--min-face-detection-confidence", type=float, default=0.5)
    ap.add_argument("--min-face-landmarks-confidence", type=float, default=0.5)
    ap.add_argument("--min-pose-detection-confidence", type=float, default=0.5)
    ap.add_argument("--min-pose-landmarks-confidence", type=float, default=0.5)
    ap.add_argument("--min-hand-landmarks-confidence", type=float, default=0.5)
    ap.add_argument("--delegate", choices=("cpu", "gpu"), default="cpu",
                    help="delegate MediaPipe (gpu = CUDA/Blackwell sur la Spark ; repli auto sur cpu si indisponible)")
    args = ap.parse_args()

    if not args.model.is_file():
        ap.error(f"Modèle .task introuvable : {args.model}")
    if args.workers < 1:
        ap.error("--workers doit être >= 1")
    views = ("front",) if args.front_only else (VIEWS if args.views == "both" else (args.views,))
    conf = dict(face_detection=args.min_face_detection_confidence,
                face_landmarks=args.min_face_landmarks_confidence,
                pose_detection=args.min_pose_detection_confidence,
                pose_landmarks=args.min_pose_landmarks_confidence,
                hand_landmarks=args.min_hand_landmarks_confidence)
    delegate = args.delegate
    if delegate == "gpu" and not gpu_works(args.model):
        print("[WARN] delegate GPU indisponible ici -> repli sur CPU", flush=True)
        delegate = "cpu"

    dataset = args.dataset or args.root.resolve().name
    videos = find_videos(args.root, exclude=args.exclude)
    if not videos:
        raise SystemExit(f"Aucune vidéo trouvée sous {args.root}")
    jobs = []
    skipped = 0
    for v in videos:
        rel = v.relative_to(args.root).as_posix()
        outs = streams_for_video(args.out, dataset, rel, views)
        if not args.overwrite and all(p.is_file() for p in outs.values()):
            skipped += 1
            continue
        jobs.append((str(v), str(args.out), rel, video_meta(args.root, v, dataset), str(args.model),
                     conf, views, args.max_frames, args.front_fraction, args.odd_side,
                     args.start_frame, delegate))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"dataset={dataset} | {len(videos)} vidéos trouvées | {skipped} déjà traitées | "
          f"{len(jobs)} à traiter | vues={views} | workers={args.workers} | delegate={delegate}",
          flush=True)
    if not jobs:
        return

    failures = frames = streams = 0
    start = monotonic()
    if args.workers == 1:
        for i, job in enumerate(jobs, 1):
            n, ns, err, sep_row, sep_col = run_job(job)
            frames += n
            streams += ns
            failures += bool(err)
            el = monotonic() - start
            print(f"[{i}/{len(jobs)}] {job[2]} | "
                  + (f"ÉCHEC {err}" if err else f"{n} frames x {ns} flux (sep {sep_row}/{sep_col}) | {n/el:.0f} f/s")
                  + (f" | reste ~{el/i*(len(jobs)-i)/60:.1f} min" if i < len(jobs) else ""), flush=True)
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_job, job): job[2] for job in jobs}
            pending = set(futures)
            done = 0
            next_beat = start + args.heartbeat
            while pending:
                finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in finished:
                    done += 1
                    n, ns, err, sep_row, sep_col = future.result()
                    frames += n
                    streams += ns
                    failures += bool(err)
                    el = monotonic() - start
                    print(f"[{done}/{len(jobs)}] {futures[future]} | "
                          + (f"ÉCHEC {err}" if err else f"{n} frames x {ns} flux (sep {sep_row}/{sep_col}) | {frames/el:.0f} f/s")
                          + (f" | reste ~{el/done*(len(jobs)-done)/60:.1f} min" if done < len(jobs) else ""),
                          flush=True)
                now = monotonic()
                if now >= next_beat:
                    el = now - start
                    rate = frames / el if el else 0.0
                    left = el / done * (len(jobs) - done) / 60 if done else float("nan")
                    print(f"[heartbeat] {done}/{len(jobs)} vidéos terminées | {rate:.1f} f/s | "
                          f"écoulé {el/60:.0f} min | reste ~{left:.0f} min", flush=True)
                    next_beat = now + args.heartbeat
    print(f"Terminé : {len(jobs) - failures}/{len(jobs)} vidéos, {streams} flux, {failures} échec(s).", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
