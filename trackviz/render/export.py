from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from trackviz.io.predictions import Detection, Predictions
from trackviz.render.overlay import OverlayStyle, draw_overlays

# BGR colours matching the viewer
_COLOR_PREDICTION = (0, 255, 0)
_COLOR_CORRECTION = (0, 200, 255)

# Codec preference order per container
_CODECS_MP4 = ["avc1", "H264", "mp4v"]
_CODECS_AVI = ["XVID", "MJPG"]

# Quality thresholds
_QUALITY_MAP: Dict[int, float] = {
    # quality level (1-10) → VIDEOWRITER_PROP_QUALITY (0-100)
    1: 20,
    2: 30,
    3: 40,
    4: 50,
    5: 60,
    6: 70,
    7: 80,
    8: 88,
    9: 94,
    10: 100,
}


def _pick_codec(output_path: Path, frame: np.ndarray) -> Tuple[int, str]:
    """Try codecs in preference order; return the first one cv2 accepts."""
    ext = output_path.suffix.lower()
    candidates = _CODECS_AVI if ext == ".avi" else _CODECS_MP4

    h, w = frame.shape[:2]
    for name in candidates:
        fourcc = cv2.VideoWriter_fourcc(*name)
        tmp = str(output_path.with_suffix(".tmp_probe" + output_path.suffix))
        writer = cv2.VideoWriter(tmp, fourcc, 24.0, (w, h))
        if writer.isOpened():
            writer.release()
            Path(tmp).unlink(missing_ok=True)
            return fourcc, name
        writer.release()
        Path(tmp).unlink(missing_ok=True)

    # Last-resort fallback
    return cv2.VideoWriter_fourcc(*candidates[-1]), candidates[-1]


def export_video(
    video_path: Path,
    preds: Predictions,
    output_path: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    style: Optional[OverlayStyle] = None,
    annotations: Optional[dict] = None,
    quality: int = 8,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Render video frames with overlays and write to *output_path*.

    Args:
        video_path:   Source video file.
        preds:        Predictions to overlay (green boxes).
        output_path:  Destination file (.mp4 recommended, .avi also supported).
        start_frame:  First frame index to export (inclusive, default 0).
        end_frame:    Last frame index to export (inclusive). None = last frame.
        style:        Overlay style; defaults to OverlayStyle().
        annotations:  Dict of ``{str(frame): {"cls", "bbox", "corrected"}}``
                      loaded from ``_annotations.json``. Corrected boxes are
                      drawn in cyan on top of the green prediction box.
        quality:      Integer 1–10 controlling encoder quality (default 8).
                      Higher → better image quality, larger file.
        on_progress:  Optional callback ``(frames_done, total_frames)`` called
                      after each frame — useful for GUI progress bars.
    """
    if style is None:
        style = OverlayStyle()
    if annotations is None:
        annotations = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, start_frame)
    if end_frame is None or end_frame < 0:
        end_frame = max(0, total_frames - 1)
    end_frame = min(end_frame, total_frames - 1)
    n_frames = end_frame - start_frame + 1

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read one frame to probe codec support
    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        raise RuntimeError("Failed to read first frame — check video file.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc, codec_name = _pick_codec(output_path, probe)
    print(f"[trackviz export] codec={codec_name}  size={width}x{height}  fps={fps:.2f}  "
          f"frames={n_frames}  quality={quality}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Apply quality hint (most effective for MJPEG; mp4v/avc1 may ignore it)
    q_value = _QUALITY_MAP.get(max(1, min(10, quality)), 88)
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, q_value)

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        idx = start_frame + i
        dets: List[Detection] = preds.for_frame(idx)

        # Green: model predictions
        if dets:
            frame = draw_overlays(frame, dets, style, box_color=_COLOR_PREDICTION)

        # Cyan: corrected annotation bbox (if any)
        anno = annotations.get(str(idx))
        if anno and anno.get("corrected") and "bbox" in anno:
            corr_det = Detection(
                frame=idx,
                bbox_xyxy=tuple(anno["bbox"]),
                cls=anno.get("cls"),
                confidence=None,
                track_id=None,
            )
            frame = draw_overlays(frame, [corr_det], style, box_color=_COLOR_CORRECTION)

        writer.write(frame)

        if on_progress is not None:
            on_progress(i + 1, n_frames)

    writer.release()
    cap.release()


def load_annotations(video_path: Path) -> dict:
    """Load ``_annotations.json`` that lives next to *video_path*, if present."""
    ann_path = video_path.parent / (video_path.stem + "_annotations.json")
    if ann_path.exists():
        try:
            with open(ann_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}
