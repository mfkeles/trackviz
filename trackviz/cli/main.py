from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2

from trackviz.gui.viewer import ViewerConfig, run_viewer
from trackviz.io.predictions import Predictions


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trackviz", description="Scrubbable viewer for tracking predictions.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gui", help="Open the GUI viewer (drag-and-drop a video to load predictions).")
    g.add_argument("--autoplay", action="store_true", help="Start playing immediately after loading a video.")

    v = sub.add_parser("view", help="Open a GUI viewer.")
    v.add_argument("video", type=str, help="Path to the video file.")
    v.add_argument(
        "--preds",
        type=str,
        required=True,
        help="Predictions file (.npz) OR a directory containing custom exports (.npy/.csv).",
    )
    v.add_argument("--bboxes", type=str, default=None, help="Optional explicit bboxes file (.npy or .csv).")
    v.add_argument("--conf", type=str, default=None, help="Optional explicit confidences file (.npy or .csv).")
    v.add_argument("--track-ids", type=str, default=None, help="Optional explicit track-ids file (.npy or .csv).")
    v.add_argument("--meta", type=str, default=None, help="Optional metadata (.npz).")
    v.add_argument(
        "--yolo",
        action="store_true",
        help="Interpret CSV bbox columns (x,y,w,h) as YOLO format (x/y are center).",
    )
    v.add_argument("--autoplay", action="store_true", help="Start playing immediately.")
    return p


def _video_frame_count(video_path: Path) -> Optional[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n if n > 0 else None


def _load_preds(args: argparse.Namespace) -> Predictions:
    preds_path = Path(args.preds)

    # If user passes an NPZ directly, that's the simplest case.
    if preds_path.is_file() and preds_path.suffix.lower() == ".npz":
        return Predictions.from_npz(preds_path)

    # Otherwise, interpret --preds as a directory (unless bboxes/conf paths are explicitly provided).
    root = preds_path if preds_path.is_dir() else preds_path.parent

    # Explicit overrides
    bboxes = Path(args.bboxes) if args.bboxes else None
    conf = Path(args.conf) if args.conf else None
    tids = Path(args.track_ids) if args.track_ids else None
    meta = Path(args.meta) if args.meta else None

    # If not provided explicitly, try common filenames in the directory.
    if bboxes is None:
        for cand in ("ds_bboxes.npy", "ds_bboxes.csv", "bboxes.npy", "bboxes.csv"):
            p = root / cand
            if p.exists():
                bboxes = p
                break

    if conf is None:
        for cand in ("ds_confidences.npy", "ds_confidences.csv", "confidences.npy", "confidences.csv", "conf.npy", "conf.csv"):
            p = root / cand
            if p.exists():
                conf = p
                break

    if tids is None:
        for cand in ("ds_track_ids.npy", "ds_track_ids.csv", "track_ids.npy", "track_ids.csv", "track_ids.csv", "ids.csv", "ids.npy"):
            p = root / cand
            if p.exists():
                tids = p
                break

    if meta is None:
        for cand in ("ds_metadata.npz", "metadata.npz"):
            p = root / cand
            if p.exists():
                meta = p
                break

    if bboxes is None or not bboxes.exists():
        raise SystemExit("Could not find bboxes. Provide --bboxes explicitly or place ds_bboxes.(npy|csv) in --preds directory.")

    # Determine expected frame count from video (helps when CSV doesn't include the last frames)
    expected = _video_frame_count(Path(args.video))

    suf = bboxes.suffix.lower()
    if suf == ".npy":
        return Predictions.from_custom_npy_triplet(
            bboxes_npy=bboxes,
            confidences_npy=conf if conf and conf.suffix.lower() == ".npy" else None,
            track_ids_npy=tids if tids and tids.suffix.lower() == ".npy" else None,
            metadata_npz=meta if meta and meta.exists() else None,
        )
    if suf == ".csv":
        return Predictions.from_custom_csv_triplet(
            bboxes_csv=bboxes,
            confidences_csv=conf if conf and conf.suffix.lower() == ".csv" else None,
            track_ids_csv=tids if tids and tids.suffix.lower() == ".csv" else None,
            metadata_npz=meta if meta and meta.exists() else None,
            expected_total_frames=expected,
            xywh_is_center=bool(getattr(args, "yolo", False)),
        )

    raise SystemExit("Unsupported predictions format. Use .npz, or custom .npy/.csv exports.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "gui":
        cfg = ViewerConfig(start_paused=not args.autoplay)
        run_viewer(None, None, cfg)
        return

    if args.cmd == "view":
        preds = _load_preds(args)
        cfg = ViewerConfig(start_paused=not args.autoplay)
        run_viewer(args.video, preds, cfg)
