from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from trackviz.io.predictions import Detection


@dataclass
class OverlayStyle:
    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    show_confidence: bool = True
    show_track_id: bool = True


def draw_overlays(
    frame_bgr: np.ndarray,
    detections: List[Detection],
    style: Optional[OverlayStyle] = None,
) -> np.ndarray:
    """Return a copy of frame with overlays."""
    if style is None:
        style = OverlayStyle()

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        # clip
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), style.box_thickness)

        parts = []
        if style.show_track_id and det.track_id is not None:
            parts.append(f"id:{det.track_id}")
        if style.show_confidence and det.confidence is not None:
            parts.append(f"{det.confidence:.2f}")
        if parts:
            label = " ".join(parts)
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale,
                style.font_thickness,
            )

            # cv2.putText uses the y coordinate as the text baseline.
            # If the bbox is near the top, "y1 - 5" can clip the text.
            y_text = y1 - 5
            min_baseline_y = text_h + baseline
            if y_text < min_baseline_y:
                # Prefer below the box; otherwise clamp inside the image.
                y_text = min(h - 1, y2 + text_h + baseline + 5)
                if y_text < min_baseline_y:
                    y_text = min_baseline_y
            cv2.putText(
                out,
                label,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale,
                (0, 255, 0),
                style.font_thickness,
                cv2.LINE_AA,
            )

    return out
