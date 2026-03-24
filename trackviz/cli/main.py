from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2

from trackviz.gui.viewer import ViewerConfig, run_viewer
from trackviz.io.auto import autoload_predictions
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

    e = sub.add_parser("export", help="Render video with prediction overlays and save to a file.")
    e.add_argument("video", type=str, help="Path to the source video file.")
    e.add_argument("--preds", type=str, default=None,
                   help="Predictions file (.pkl/.npz/.npy) or directory. "
                        "Auto-detected from the video directory if omitted.")
    e.add_argument("--output", "-o", type=str, default=None,
                   help="Output file path (default: <video_stem>_overlay.mp4 next to source).")
    e.add_argument("--start", type=float, default=None,
                   help="Start time in seconds (default: beginning of video).")
    e.add_argument("--end", type=float, default=None,
                   help="End time in seconds (default: end of video).")
    e.add_argument("--quality", type=int, default=8, metavar="1-10",
                   help="Encoder quality 1 (smallest) – 10 (best). Default: 8.")
    e.add_argument("--no-annotations", action="store_true",
                   help="Skip loading <video_stem>_annotations.json (corrected boxes).")
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

    if args.cmd == "export":
        from trackviz.render.export import export_video, load_annotations

        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"Video not found: {video_path}")

        # Resolve output path
        output_path = Path(args.output) if args.output else \
            video_path.parent / (video_path.stem + "_overlay.mp4")

        # Determine frame range from times
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        start_frame = int(args.start * fps) if args.start is not None else 0
        end_frame = int(args.end * fps) if args.end is not None else total_frames - 1
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total_frames - 1)
        n_frames = end_frame - start_frame + 1
        duration_sec = n_frames / fps

        # Warn if export is longer than 10 minutes
        if duration_sec > 600:
            mins = duration_sec / 60
            print(f"\nWarning: you are about to export {mins:.1f} minutes of video "
                  f"({n_frames:,} frames).")
            answer = input("Continue? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Export cancelled.")
                sys.exit(0)

        preds_hint = Path(args.preds) if args.preds else None
        preds = autoload_predictions(video_path, total_frames, preds_hint)
        annotations = {} if args.no_annotations else load_annotations(video_path)

        quality = max(1, min(10, args.quality))

        def _progress(done: int, total: int) -> None:
            pct = done / total * 100
            bar_len = 40
            filled = int(bar_len * done / total)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total} frames", end="", flush=True)

        print(f"\nExporting {video_path.name} → {output_path}")
        print(f"  frames {start_frame}–{end_frame}  ({duration_sec:.1f}s)  quality={quality}\n")

        export_video(
            video_path=video_path,
            preds=preds,
            output_path=output_path,
            start_frame=start_frame,
            end_frame=end_frame,
            annotations=annotations,
            quality=quality,
            on_progress=_progress,
        )
        print(f"\nDone → {output_path}")
