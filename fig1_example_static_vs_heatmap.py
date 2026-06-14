"""
Figure 1 — plain frame vs heatmap frame for selected indices.

For each frame index two SVGs are produced:
  <stem>_frame<idx>_plain.svg
  <stem>_frame<idx>_heatmap.svg

Both use the same full-brightness base image so the only visual
difference is the heatmap overlay on motion pixels.

Output folder: <DATA_DIR>/fig1_frames/

─────────────────────────────────────────
Source files
─────────────────────────────────────────
Video : 20260220_200000_raw.mp4
PKL   : 20260220_182213_yolo_fast.pkl
─────────────────────────────────────────
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ── Source files ──────────────────────────────────────────────────────────────
DATA_DIR = Path("/Volumes/Lab_Files3/mfk/projects/2025_flyvista2/PE_Origins_Project/ClosedLoopArousal/Data/20260220")
VIDEO    = DATA_DIR / "20260220_200000_raw.mp4"
PKL      = DATA_DIR / "20260220_182213_yolo_fast.pkl"

# ── Frame indices ─────────────────────────────────────────────────────────────
FRAME_INDICES = [26127, 60784, 43635, 273807, 159465, 199325]

# ── Heatmap settings ──────────────────────────────────────────────────────────
WINDOW_SIZE   = 6    # number of frames in sliding window (inclusive of target)
THRESHOLD     = 10   # min pixel-intensity change to count as motion (0–255)
MOTION_ALPHA  = 0.5  # heatmap weight on motion pixels (plain frame = 1 − this)

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = DATA_DIR / "fig1_frames"
OUT_DIR.mkdir(exist_ok=True)


# ── Heatmap generator (motion-mask blend) ────────────────────────────────────

def generate_heatmap(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Return a BGR heatmap composite for *idx* using motion-mask blending.

    Static pixels keep full original brightness.
    Motion pixels are blended (MOTION_ALPHA) with the HOT colormap.
    """
    gray_frames: list[np.ndarray] = []
    for i in range(max(0, idx - WINDOW_SIZE + 1), idx + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok and frame is not None:
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if not gray_frames:
        return None

    # Use the current frame (newest) as reference so past positions accumulate
    # motion color — history trail, not current-position highlight.
    ref_gray = gray_frames[-1]
    h, w = ref_gray.shape
    accumulator = np.zeros((h, w), dtype=np.float32)

    for g in gray_frames[:-1]:
        diff = cv2.absdiff(ref_gray, g)
        _, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        accumulator += mask

    mx     = accumulator.max()
    acc_u8 = (
        (accumulator * (255.0 / mx)).astype(np.uint8)
        if mx > 0
        else np.zeros((h, w), dtype=np.uint8)
    )

    heatmap = cv2.applyColorMap(acc_u8, cv2.COLORMAP_HOT)
    ref_bgr = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)

    # Blend heatmap only where motion was detected; static pixels are unchanged.
    motion_mask = (acc_u8 > 0).astype(np.float32)[:, :, np.newaxis]
    blended     = cv2.addWeighted(ref_bgr, 1 - MOTION_ALPHA, heatmap, MOTION_ALPHA, 0)
    return (blended * motion_mask + ref_bgr * (1 - motion_mask)).astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from trackviz.io.predictions import Predictions
    from trackviz.io.class_names import resolve_class_names
    from trackviz.render.overlay import OverlayStyle
    from trackviz.render.snapshot import save_snapshot

    print(f"Loading predictions from {PKL.name} …")
    preds = Predictions.from_yolo_pickle(PKL, expected_total_frames=None)

    style = OverlayStyle(
        show_confidence=True,
        show_track_id=False,
        show_class=True,
    )
    style.class_names = resolve_class_names(preds.meta.get("class_names"))

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    stem = VIDEO.stem  # "20260220_200000_raw"

    for idx in FRAME_INDICES:
        print(f"\nFrame {idx}")
        dets = preds.for_frame(idx)
        anno = None  # no annotation corrections used for figures

        # ── Plain frame ───────────────────────────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, plain_frame = cap.read()
        if not ok or plain_frame is None:
            print(f"  [skip] could not read plain frame {idx}")
            continue

        plain_out = OUT_DIR / f"{stem}_frame{idx:06d}_plain.svg"
        save_snapshot(plain_frame, dets, anno, idx, style, plain_out)
        print(f"  saved plain  → {plain_out.name}")

        # ── Heatmap frame ─────────────────────────────────────────────────────
        heatmap_frame = generate_heatmap(cap, idx)
        if heatmap_frame is None:
            print(f"  [skip] could not generate heatmap for frame {idx}")
            continue

        heatmap_out = OUT_DIR / f"{stem}_frame{idx:06d}_heatmap.svg"
        save_snapshot(heatmap_frame, dets, anno, idx, style, heatmap_out)
        print(f"  saved heatmap → {heatmap_out.name}")

    cap.release()
    print(f"\nDone. SVGs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
