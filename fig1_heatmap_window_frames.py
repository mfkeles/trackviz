"""
Extract the 6 constituent frames of the heatmap sliding window for two
target indices (26127 and 273810), saved as plain SVGs (no heatmap overlay).

These are the frames that feed into the heatmap visualisation, useful for
illustrating what the heatmap represents in a figure.

─────────────────────────────────────────
Source files
─────────────────────────────────────────
Video : 20260220_200000_raw.mp4
PKL   : 20260220_182213_yolo_fast.pkl
─────────────────────────────────────────

Output: <DATA_DIR>/fig1_window_frames/
Naming: <stem>_target<target_idx>_window<window_idx>.svg
"""
from __future__ import annotations

from pathlib import Path

import cv2

# ── Source files ──────────────────────────────────────────────────────────────
DATA_DIR = Path("/Volumes/Lab_Files3/mfk/projects/2025_flyvista2/PE_Origins_Project/ClosedLoopArousal/Data/20260220")
VIDEO    = DATA_DIR / "20260220_200000_raw.mp4"
PKL      = DATA_DIR / "20260220_182213_yolo_fast.pkl"

# ── Target heatmap frames ─────────────────────────────────────────────────────
TARGET_INDICES = [26127, 273807]
WINDOW_SIZE    = 6   # must match viewer setting

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = DATA_DIR / "fig1_window_frames"
OUT_DIR.mkdir(exist_ok=True)


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

    stem = VIDEO.stem

    for target in TARGET_INDICES:
        window = list(range(max(0, target - WINDOW_SIZE + 1), target + 1))
        print(f"\nTarget {target} — window frames: {window}")

        for frame_idx in window:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"  [skip] could not read frame {frame_idx}")
                continue

            dets = preds.for_frame(frame_idx)
            out_path = OUT_DIR / f"{stem}_target{target:06d}_window{frame_idx:06d}.svg"
            save_snapshot(frame, dets, None, frame_idx, style, out_path)
            print(f"  saved → {out_path.name}")

    cap.release()
    print(f"\nDone. SVGs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
