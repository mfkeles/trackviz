from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from trackviz.io.class_names import resolve_class_names
from trackviz.io.predictions import Detection, Predictions
from trackviz.render.overlay import OverlayStyle

# BGR colours matching the viewer
_COLOR_PREDICTION = (0, 255, 0)
_COLOR_CORRECTION = (0, 200, 255)

# CRF values for libx264: lower = better quality / larger file.
# quality 1 → CRF 32 (small file), quality 10 → CRF 15 (high quality)
_QUALITY_TO_CRF = {
    1: 32, 2: 30, 3: 28, 4: 26, 5: 24,
    6: 22, 7: 20, 8: 18, 9: 16, 10: 15,
}


# ---------------------------------------------------------------------------
# Enhanced overlay drawing (export-only)
# ---------------------------------------------------------------------------

def _draw_overlays_export(
    frame: np.ndarray,
    detections: List[Detection],
    style: OverlayStyle,
    box_color: Tuple[int, int, int],
) -> np.ndarray:
    """Draw overlays with polished visuals suited for exported video.

    Differences from the live-viewer renderer:
    - Anti-aliased box edges with a dark drop-shadow for contrast on any background.
    - Semi-transparent filled label badge (75 % opaque, darkened box colour).
    - White label text with padding — always readable regardless of background.
    """
    h, w = frame.shape[:2]
    class_names = style.class_names

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        # ── Bounding box ────────────────────────────────────────────────
        # Dark shadow offset by +2 px for contrast on light backgrounds
        sx1 = min(x1 + 2, w - 1)
        sy1 = min(y1 + 2, h - 1)
        sx2 = min(x2 + 2, w - 1)
        sy2 = min(y2 + 2, h - 1)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 0),
                      style.box_thickness, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color,
                      style.box_thickness, cv2.LINE_AA)

        # ── Label ────────────────────────────────────────────────────────
        parts: List[str] = []
        if style.show_class and det.cls is not None:
            if class_names and 0 <= det.cls < len(class_names):
                parts.append(class_names[det.cls])
            else:
                parts.append(f"cls:{det.cls}")
        if style.show_track_id and det.track_id is not None:
            parts.append(f"id:{det.track_id}")
        if style.show_confidence and det.confidence is not None:
            parts.append(f"{det.confidence:.2f}")
        if not parts:
            continue

        label = " ".join(parts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = style.font_scale
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

        pad_x, pad_y = 6, 3
        badge_h = th + baseline + pad_y * 2
        badge_w = tw + pad_x * 2

        # Prefer badge above the box; shift below if too close to top edge
        by2 = y1
        by1 = y1 - badge_h
        if by1 < 0:
            by1 = y2
            by2 = y2 + badge_h
        by1 = max(0, min(h - 1, by1))
        by2 = max(0, min(h, by2))
        bx1 = max(0, x1)
        bx2 = min(w, x1 + badge_w)

        if by2 <= by1 or bx2 <= bx1:
            continue

        # Semi-transparent badge: blend darkened box colour over the frame ROI
        badge_bgr = np.array(
            [max(0, int(c * 0.55)) for c in box_color], dtype=np.uint8
        )
        roi = frame[by1:by2, bx1:bx2]
        filled = np.empty_like(roi)
        filled[:] = badge_bgr
        cv2.addWeighted(filled, 0.80, roi, 0.20, 0, roi)
        frame[by1:by2, bx1:bx2] = roi

        # White text
        text_x = bx1 + pad_x
        text_y = by2 - pad_y - baseline
        cv2.putText(frame, label, (text_x, text_y), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _crf_from_quality(quality: int) -> int:
    return _QUALITY_TO_CRF.get(max(1, min(10, quality)), 18)


def _export_via_ffmpeg(
    cap: cv2.VideoCapture,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    start_frame: int,
    n_frames: int,
    preds: Predictions,
    style: OverlayStyle,
    annotations: dict,
    quality: int,
    on_progress: Optional[Callable[[int, int], None]],
) -> None:
    crf = _crf_from_quality(quality)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    print(f"[trackviz export] FFmpeg  crf={crf}  cmd: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = _render_frame(frame, start_frame + i, preds, style, annotations)
            proc.stdin.write(frame.tobytes())
            if on_progress is not None:
                on_progress(i + 1, n_frames)
    finally:
        proc.stdin.close()

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg exited with code {ret}.")


def _export_via_opencv(
    cap: cv2.VideoCapture,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    start_frame: int,
    n_frames: int,
    preds: Predictions,
    style: OverlayStyle,
    annotations: dict,
    quality: int,
    on_progress: Optional[Callable[[int, int], None]],
) -> None:
    ext = output_path.suffix.lower()
    candidates = ["XVID", "MJPG"] if ext == ".avi" else ["avc1", "H264", "mp4v"]
    fourcc, codec_name = _pick_opencv_codec(output_path, width, height, candidates)

    # Map quality 1–10 to VIDEOWRITER_PROP_QUALITY 20–95
    q_prop = 20 + int((quality - 1) / 9 * 75)

    print(f"[trackviz export] OpenCV  codec={codec_name}  quality_prop={q_prop}")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter could not open: {output_path}")
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, q_prop)

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = _render_frame(frame, start_frame + i, preds, style, annotations)
        writer.write(frame)
        if on_progress is not None:
            on_progress(i + 1, n_frames)

    writer.release()


def _pick_opencv_codec(
    output_path: Path,
    width: int,
    height: int,
    candidates: List[str],
) -> Tuple[int, str]:
    for name in candidates:
        fourcc = cv2.VideoWriter_fourcc(*name)
        tmp = str(output_path.with_suffix(".tmp_probe" + output_path.suffix))
        writer = cv2.VideoWriter(tmp, fourcc, 24.0, (width, height))
        opened = writer.isOpened()
        writer.release()
        Path(tmp).unlink(missing_ok=True)
        if opened:
            return fourcc, name
    return cv2.VideoWriter_fourcc(*candidates[-1]), candidates[-1]


# ---------------------------------------------------------------------------
# Per-frame rendering (shared between both backends)
# ---------------------------------------------------------------------------

def _render_frame(
    frame: np.ndarray,
    idx: int,
    preds: Predictions,
    style: OverlayStyle,
    annotations: dict,
) -> np.ndarray:
    dets: List[Detection] = preds.for_frame(idx)
    if dets:
        frame = _draw_overlays_export(frame, dets, style, _COLOR_PREDICTION)

    anno = annotations.get(str(idx))
    if anno and anno.get("corrected") and "bbox" in anno:
        corr_det = Detection(
            frame=idx,
            bbox_xyxy=tuple(anno["bbox"]),
            cls=anno.get("cls"),
            confidence=None,
            track_id=None,
        )
        frame = _draw_overlays_export(frame, [corr_det], style, _COLOR_CORRECTION)

    return frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
        output_path:  Destination file (.mp4 strongly recommended).
        start_frame:  First frame index to export (inclusive, default 0).
        end_frame:    Last frame index to export (inclusive). None = last frame.
        style:        Overlay style. Defaults to :class:`OverlayStyle`.
        annotations:  ``{str(frame): {"cls", "bbox", "corrected"}}`` from
                      ``_annotations.json``. Corrected boxes render in cyan.
        quality:      Integer 1–10. With FFmpeg this maps to CRF 32–15
                      (lower CRF = higher quality, larger file).
                      With OpenCV fallback it maps to VIDEOWRITER_PROP_QUALITY.
        on_progress:  Optional ``(frames_done, total_frames)`` callback.
    """
    if style is None:
        style = OverlayStyle()
    if annotations is None:
        annotations = {}

    # Always resolve class names: replaces generic "cls0/cls1" with real names
    style.class_names = resolve_class_names(preds.meta.get("class_names"))

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

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(
        cap=cap,
        output_path=output_path,
        fps=fps,
        width=width,
        height=height,
        start_frame=start_frame,
        n_frames=n_frames,
        preds=preds,
        style=style,
        annotations=annotations,
        quality=quality,
        on_progress=on_progress,
    )

    if _ffmpeg_available():
        try:
            _export_via_ffmpeg(**common)
        except Exception as exc:
            print(f"\n[trackviz export] FFmpeg failed ({exc}); retrying with OpenCV …")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            _export_via_opencv(**common)
    else:
        print("[trackviz export] FFmpeg not found on PATH — using OpenCV encoder.")
        _export_via_opencv(**common)

    cap.release()


def load_annotations(video_path: Path) -> dict:
    """Load ``<stem>_annotations.json`` next to *video_path*, if present."""
    ann_path = video_path.parent / (video_path.stem + "_annotations.json")
    if ann_path.exists():
        try:
            with open(ann_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}
