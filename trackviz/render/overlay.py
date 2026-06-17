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
    show_class: bool = True
    class_names: Optional[List[str]] = None


# Type alias for a placed-label rectangle in image space (x1, y1, x2, y2),
# with y increasing downward.
LabelRect = Tuple[float, float, float, float]


def resolve_label_top(
    top: float,
    height: float,
    x1: float,
    x2: float,
    placed: List[LabelRect],
    *,
    push_down: bool,
    gap: float = 2.0,
) -> float:
    """Return a non-overlapping top-y for a label rectangle.

    When two (or more) detections share almost the same bounding box — which
    happens when the model is unsure and emits two low-confidence predictions
    for one animal — their labels would otherwise be drawn on top of each
    other and become unreadable.  This shifts the candidate rectangle away
    from its box (downward if *push_down*, else upward) until it no longer
    overlaps any rectangle already in *placed*.

    The caller is responsible for appending the final (post-clamp) rectangle
    to *placed* so subsequent labels stack against it.
    """
    bottom = top + height
    moved = True
    # Bounded iteration guards against pathological cascades.
    for _ in range(len(placed) + 1):
        if not moved:
            break
        moved = False
        for px1, py1, px2, py2 in placed:
            horizontal_overlap = x1 < px2 and x2 > px1
            vertical_overlap = top < py2 and bottom > py1
            if horizontal_overlap and vertical_overlap:
                if push_down:
                    shift = py2 - top + gap
                else:
                    shift = -(bottom - py1 + gap)
                top += shift
                bottom += shift
                moved = True
                break
    return top


def draw_overlays(
    frame_bgr: np.ndarray,
    detections: List[Detection],
    style: Optional[OverlayStyle] = None,
    box_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Return a copy of frame with overlays.

    Args:
        box_color: BGR color for boxes and labels (default green). Pass
            ``(0, 200, 255)`` for cyan to distinguish corrected boxes.
    """
    if style is None:
        style = OverlayStyle()

    class_names = style.class_names

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Rectangles of labels already drawn on this frame, so co-located
    # detections stack instead of overprinting one another.
    placed_labels: List[LabelRect] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        # clip
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, style.box_thickness)

        parts = []
        if style.show_class and det.cls is not None:
            if class_names and 0 <= det.cls < len(class_names):
                parts.append(class_names[det.cls])
            else:
                parts.append(f"cls:{det.cls}")
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
            label_h = text_h + baseline

            # Prefer placing the label above the box; fall back below when the
            # box hugs the top edge (where "above" would clip the text).
            below = (y1 - 5 - text_h) < 0
            rect_top = (y2 + 5) if below else (y1 - 5 - text_h)

            # Stack against any label already drawn for this frame, then clamp
            # back inside the image.
            rect_top = resolve_label_top(
                rect_top, label_h, x1, x1 + text_w, placed_labels, push_down=below
            )
            rect_top = max(0.0, min(float(h - label_h), rect_top))
            placed_labels.append((float(x1), rect_top, float(x1 + text_w), rect_top + label_h))

            # cv2.putText uses the y coordinate as the text baseline.
            y_text = int(round(rect_top + text_h))
            cv2.putText(
                out,
                label,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale,
                box_color,
                style.font_thickness,
                cv2.LINE_AA,
            )

    return out
