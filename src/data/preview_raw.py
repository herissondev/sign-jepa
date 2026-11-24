"""Rendu visuel d'un .npz raw sur la vidéo source (vérification à l'oeil).

    # une frame -> PNG (index positionnel dans le .npz, défaut: milieu)
    python -m src.data.preview_raw --npz OUT/JSL/.../X__left_front.npz \
        --video-root /chemin/JSL_DATASET --frame 1500 --out prev.png

    # un extrait -> MP4
    python -m src.data.preview_raw --npz ... --video-root /chemin/JSL_DATASET \
        --frames 3000-3120 --stride 4 --out prev.mp4

La vidéo est retrouvée automatiquement via `source_video` dans le .npz + `--video-root`.
Les landmarks sont reprojetés en coordonnées pleine frame (to_full_frame) et dessinés
par région ; le rectangle blanc = le quadrant analysé (crop_xyxy).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.load_raw import load_stream, to_full_frame
from src.data.schema import REGIONS

COLORS = dict(face=(0, 255, 255), pose=(255, 0, 0), left_hand=(0, 0, 255),
              right_hand=(0, 255, 0), pose_world=(160, 160, 160),
              left_hand_world=(120, 120, 255), right_hand_world=(120, 255, 120))


def draw(frame, stream, i, radius=2):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in stream["crop_xyxy"])
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)
    for name in REGIONS:
        arr = stream.get(name)
        if arr is None or i >= len(arr):
            continue
        pts = to_full_frame(arr[i], stream["crop_xyxy"], stream["frame_size_wh"])
        for x, y, *_ in pts:
            if np.isfinite(x):
                cv2.circle(frame, (int(x * w), int(y * h)), radius, COLORS[name], -1)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--video-root", required=True, type=Path, help="racine du dataset (pour retrouver source_video)")
    ap.add_argument("--frame", type=int, default=None, help="index positionnel dans le .npz (défaut: milieu)")
    ap.add_argument("--frames", default=None, help="'a:b' ou 'a-b' -> extrait MP4 (index positionnels a..b)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path, help=".png (1 frame) ou .mp4 (extrait)")
    ap.add_argument("--radius", type=int, default=2)
    args = ap.parse_args()

    stream = load_stream(args.npz)
    video = args.video_root / str(stream["source_video"])
    if not video.is_file():
        raise SystemExit(f"Vidéo source introuvable : {video}")
    T = len(stream["frame_index"])
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    if args.frames:
        sep = ":" if ":" in args.frames else "-"
        a, b = (int(x) for x in args.frames.split(sep))
        a, b = max(0, a), min(T, b)
        writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride,
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        for i in range(a, b, args.stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(stream["frame_index"][i]))
            ok, frame = cap.read()
            if not ok:
                continue
            writer.write(draw(frame, stream, i, args.radius))
        writer.release()
        print(f"{args.out} : {len(range(a, b, args.stride))} frames (flux {stream['side']}_{stream['view']}, "
              f"{stream['dataset']}/{stream['source_video']})")
    else:
        i = args.frame if args.frame is not None else T // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(stream["frame_index"][i]))
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Frame illisible")
        cv2.imwrite(str(args.out), draw(frame, stream, i, args.radius))
        det = {name: bool(np.isfinite(stream[name][i, :, 0]).any()) for name in REGIONS if name in stream}
        print(f"{args.out} : frame video {int(stream['frame_index'][i])} (t={stream['timestamps_ms'][i]} ms), "
              f"détecté: {det}")
    cap.release()


if __name__ == "__main__":
    main()
