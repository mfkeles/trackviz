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

# Stop the export loop after this many consecutive unreadable frames.
# CAP_PROP_FRAME_COUNT is often a few frames over the real count, so a small
# tolerance prevents stopping prematurely on a single decode hiccup while still
# terminating cleanly at genuine end-of-stream.
_MAX_CONSECUTIVE_FAILURES = 5


# ---------------------------------------------------------------------------
# Heatmap generator (rolling motion accumulator)
# ---------------------------------------------------------------------------

class _HeatmapGenerator:
    """Produces a motion-heatmap overlay per frame.

    Mirrors ``trackviz.gui.viewer._generate_heatmap`` so exported heatmap
    videos match what the viewer shows on-screen.
    """

    def __init__(self, window_size: int = 6, threshold: int = 10) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self._buffer: List[np.ndarray] = []  # grayscale past frames

    def prewarm_from_capture(
        self, cap: "cv2.VideoCapture", start_frame: int
    ) -> None:
        """Fill the rolling buffer with the frames immediately preceding
        *start_frame* so the first exported frame matches what the viewer
        shows when you jump to that frame (the viewer reads its own window
        on demand).

        Leaves the capture positioned at *start_frame* on return.
        """
        n = self.window_size - 1
        warm_start = max(0, start_frame - n)
        if warm_start >= start_frame:
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm_start)
        for _ in range(start_frame - warm_start):
            ok, f = cap.read()
            if not ok or f is None:
                break
            self.process(f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if frame_bgr.ndim == 3
            else frame_bgr
        )
        self._buffer.append(gray)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        ref = self._buffer[-1]
        accumulator = np.zeros(ref.shape, dtype=np.float32)
        for past in self._buffer[:-1]:
            diff = cv2.absdiff(ref, past)
            _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            accumulator += mask
        mx = accumulator.max()
        if mx > 0:
            acc_u8 = (accumulator * (255.0 / mx)).astype(np.uint8)
        else:
            acc_u8 = np.zeros_like(ref, dtype=np.uint8)
        heatmap = cv2.applyColorMap(acc_u8, cv2.COLORMAP_HOT)
        ref_bgr = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(ref_bgr, 0.6, heatmap, 0.4, 0)


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


def _read_frame(
    cap: cv2.VideoCapture,
    absolute_idx: int,
    last_good: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], bool]:
    """Read the next frame from *cap*, recovering from transient decode errors.

    Tries a sequential read first.  On failure, seeks to *absolute_idx* and
    retries once — this recovers frames that are present in the file but whose
    decoder state was corrupted by an earlier bad frame.  If both attempts fail
    (true end-of-stream or genuinely missing frame), returns
    (*last_good*, False) so the caller can decide to duplicate or stop.

    Returns (frame, is_real) where *is_real* is False when a duplicate was
    substituted.
    """
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame, True

    # Retry after an explicit seek — recovers from mid-stream decode glitches
    cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_idx)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame, True

    return last_good, False


def _export_via_ffmpeg(
    cap: cv2.VideoCapture,
    output_path: Path,
    fps: float,
    out_width: int,
    out_height: int,
    start_frame: int,
    n_frames: int,
    preds: Optional[Predictions],
    style: OverlayStyle,
    annotations: dict,
    quality: int,
    scale: float,
    include_predictions: bool,
    include_annotations: bool,
    include_heatmap: bool,
    on_progress: Optional[Callable[[int, int], None]],
) -> None:
    crf = _crf_from_quality(quality)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_width}x{out_height}",
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

    heatmap_gen = _HeatmapGenerator() if include_heatmap else None
    if heatmap_gen is not None:
        heatmap_gen.prewarm_from_capture(cap, start_frame)
    try:
        last_good: Optional[np.ndarray] = None
        consecutive_failures = 0
        for i in range(n_frames):
            raw, is_real = _read_frame(cap, start_frame + i, last_good)
            if raw is None:
                break  # no frame at all (very start of video failed)
            if is_real:
                last_good = raw
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > _MAX_CONSECUTIVE_FAILURES:
                    break
            if heatmap_gen is not None:
                raw = heatmap_gen.process(raw)
            frame = _render_frame(raw, start_frame + i, preds, style, annotations,
                                   out_width, out_height, scale,
                                   include_predictions, include_annotations)
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
    out_width: int,
    out_height: int,
    start_frame: int,
    n_frames: int,
    preds: Optional[Predictions],
    style: OverlayStyle,
    annotations: dict,
    quality: int,
    scale: float,
    include_predictions: bool,
    include_annotations: bool,
    include_heatmap: bool,
    on_progress: Optional[Callable[[int, int], None]],
) -> None:
    ext = output_path.suffix.lower()
    candidates = ["XVID", "MJPG"] if ext == ".avi" else ["avc1", "H264", "mp4v"]
    fourcc, codec_name = _pick_opencv_codec(output_path, out_width, out_height, candidates)

    # Map quality 1–10 to VIDEOWRITER_PROP_QUALITY 20–95
    q_prop = 20 + int((quality - 1) / 9 * 75)

    print(f"[trackviz export] OpenCV  codec={codec_name}  quality_prop={q_prop}")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter could not open: {output_path}")
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, q_prop)

    heatmap_gen = _HeatmapGenerator() if include_heatmap else None
    if heatmap_gen is not None:
        heatmap_gen.prewarm_from_capture(cap, start_frame)
    last_good: Optional[np.ndarray] = None
    consecutive_failures = 0
    for i in range(n_frames):
        raw, is_real = _read_frame(cap, start_frame + i, last_good)
        if raw is None:
            break
        if is_real:
            last_good = raw
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures > _MAX_CONSECUTIVE_FAILURES:
                break
        if heatmap_gen is not None:
            raw = heatmap_gen.process(raw)
        frame = _render_frame(raw, start_frame + i, preds, style, annotations,
                               out_width, out_height, scale,
                               include_predictions, include_annotations)
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

def _scale_det(det: Detection, scale: float) -> Detection:
    x1, y1, x2, y2 = det.bbox_xyxy
    return Detection(
        frame=det.frame,
        bbox_xyxy=(x1 * scale, y1 * scale, x2 * scale, y2 * scale),
        cls=det.cls,
        confidence=det.confidence,
        track_id=det.track_id,
    )


def _render_frame(
    frame: np.ndarray,
    idx: int,
    preds: Optional[Predictions],
    style: OverlayStyle,
    annotations: dict,
    out_width: int,
    out_height: int,
    scale: float,
    include_predictions: bool = True,
    include_annotations: bool = True,
) -> np.ndarray:
    # Downscale first so overlays are drawn at output resolution
    if scale != 1.0:
        frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

    if include_predictions and preds is not None:
        dets: List[Detection] = preds.for_frame(idx)
        if scale != 1.0:
            dets = [_scale_det(d, scale) for d in dets]
        if dets:
            frame = _draw_overlays_export(frame, dets, style, _COLOR_PREDICTION)

    if include_annotations:
        anno = annotations.get(str(idx))
        if anno and anno.get("corrected") and "bbox" in anno:
            bbox = anno["bbox"]
            if scale != 1.0:
                bbox = [c * scale for c in bbox]
            corr_det = Detection(
                frame=idx,
                bbox_xyxy=tuple(bbox),
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
    preds: Optional[Predictions],
    output_path: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    style: Optional[OverlayStyle] = None,
    annotations: Optional[dict] = None,
    quality: int = 8,
    scale: float = 1.0,
    include_predictions: bool = True,
    include_annotations: bool = True,
    include_heatmap: bool = False,
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
        scale:        Output resolution multiplier (default 1.0 = original size).
                      0.5 halves width and height (quarter the pixel count).
                      H.264 requires even dimensions — values are rounded up to
                      the nearest even pixel automatically.
        include_predictions:  Draw model prediction boxes (green).  Set False
                      to omit them — e.g. for a clean export.
        include_annotations:  Draw corrected annotation boxes (cyan).  Set
                      False to omit them.
        include_heatmap:  Replace each frame with a motion-heatmap overlay
                      (same formula the viewer uses for its heatmap mode).
                      Overlays still draw on top of the heatmap.
        on_progress:  Optional ``(frames_done, total_frames)`` callback.
    """
    if style is None:
        style = OverlayStyle()
    if annotations is None:
        annotations = {}

    if preds is None:
        # No tracking available — drawing predictions is impossible.
        include_predictions = False
    else:
        # Always resolve class names: replaces generic "cls0/cls1" with real names
        style.class_names = resolve_class_names(preds.meta.get("class_names"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute output dimensions — H.264 requires even width and height
    scale = max(0.01, min(1.0, scale))
    out_width = max(2, int(width * scale))
    out_height = max(2, int(height * scale))
    out_width += out_width % 2    # round up to even
    out_height += out_height % 2

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
        out_width=out_width,
        out_height=out_height,
        start_frame=start_frame,
        n_frames=n_frames,
        preds=preds,
        style=style,
        annotations=annotations,
        quality=quality,
        scale=scale,
        include_predictions=include_predictions,
        include_annotations=include_annotations,
        include_heatmap=include_heatmap,
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
