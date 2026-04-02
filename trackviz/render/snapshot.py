"""Frame snapshot export — PNG, SVG, and PDF output.

PNG  : OpenCV rendering, identical look to the video exporter.
SVG  : matplotlib vector output; text stays editable in Illustrator
       (svg.fonttype = 'none').  Boxes and labels are separate objects.
PDF  : matplotlib vector output; TrueType fonts embedded (pdf.fonttype = 42).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from trackviz.io.predictions import Detection
from trackviz.render.export import _COLOR_CORRECTION, _COLOR_PREDICTION, _draw_overlays_export
from trackviz.render.overlay import OverlayStyle

# Hex equivalents of the BGR export colours (for matplotlib / SVG)
_PRED_HEX = "#00ff00"
_CORR_HEX = "#00c8ff"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_snapshot(
    raw_frame_bgr: np.ndarray,
    detections: List[Detection],
    annotation: Optional[dict],
    frame_idx: int,
    style: OverlayStyle,
    out_path: Path,
) -> None:
    """Render *raw_frame_bgr* with polished overlays and save to *out_path*.

    Format is determined by the file extension:

    ``.png``
        High-quality raster via OpenCV — identical look to the video exporter
        (anti-aliased boxes, drop-shadow, semi-transparent label badges).

    ``.svg``
        Vector output via matplotlib.  Text labels are kept as real ``<text>``
        elements (``svg.fonttype = "none"``) so they can be edited in
        Illustrator.  Boxes are separate path objects; colours are easily
        changed in the Illustrator Appearance panel.

    ``.pdf``
        Vector output via matplotlib.  TrueType fonts are embedded
        (``pdf.fonttype = 42``) for maximum Illustrator compatibility.

    Parameters
    ----------
    raw_frame_bgr:
        The video frame (or heatmap composite) *before* any viewer overlays
        have been applied, in BGR channel order.
    detections:
        Model predictions for the frame.
    annotation:
        The annotation dict for the frame, or ``None``.  If the dict contains
        ``corrected=True`` and a ``"bbox"`` key the corrected box is drawn in
        cyan on top of the prediction boxes.
    frame_idx:
        Frame index (used to construct the correction Detection object).
    style:
        Overlay display settings (show_class, show_confidence, …).
        ``style.class_names`` must be populated before calling.
    out_path:
        Destination path.  The suffix must be ``.png``, ``.svg``, or ``.pdf``.
    """
    suffix = out_path.suffix.lower()
    if suffix not in (".png", ".svg", ".pdf"):
        raise ValueError(
            f"Unsupported format {suffix!r}. Choose .png, .svg, or .pdf."
        )

    corr_det = _corr_det_from_annotation(annotation, frame_idx)

    if suffix == ".png":
        _save_png(raw_frame_bgr, detections, corr_det, style, out_path)
    else:
        _save_vector(raw_frame_bgr, detections, corr_det, style, out_path)


# ---------------------------------------------------------------------------
# PNG path (OpenCV — matches video export look)
# ---------------------------------------------------------------------------

def _save_png(
    raw_frame_bgr: np.ndarray,
    detections: List[Detection],
    corr_det: Optional[Detection],
    style: OverlayStyle,
    out_path: Path,
) -> None:
    frame = raw_frame_bgr.copy()
    _draw_overlays_export(frame, detections, style, _COLOR_PREDICTION)
    if corr_det is not None:
        _draw_overlays_export(frame, [corr_det], style, _COLOR_CORRECTION)
    cv2.imwrite(str(out_path), frame)


# ---------------------------------------------------------------------------
# Vector path (SVG / PDF via matplotlib)
# ---------------------------------------------------------------------------

def _save_vector(
    raw_frame_bgr: np.ndarray,
    detections: List[Detection],
    corr_det: Optional[Detection],
    style: OverlayStyle,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        import matplotlib.patches as mpatches  # noqa: F401 — imported for type checking
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for SVG/PDF export.\n"
            "Install it with:  pip install matplotlib"
        ) from exc

    # Keep text as real <text> / TrueType — essential for Illustrator editing
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42

    frame_rgb = cv2.cvtColor(raw_frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    dpi = 150
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    # add_axes([left, bottom, width, height]) in figure-fraction units
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(frame_rgb, interpolation="lanczos", aspect="equal")
    # Match matplotlib's default imshow extent so patch coords = pixel coords
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)   # y increases downward (image convention)
    ax.axis("off")

    _draw_mpl_boxes(ax, detections, _PRED_HEX, style, h)
    if corr_det is not None:
        _draw_mpl_boxes(ax, [corr_det], _CORR_HEX, style, h)

    fmt = out_path.suffix[1:].lower()
    fig.savefig(str(out_path), format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _draw_mpl_boxes(
    ax,
    detections: List[Detection],
    color_hex: str,
    style: OverlayStyle,
    img_h: int,
) -> None:
    import matplotlib.patches as mpatches

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        # Drop-shadow (dark, offset +1.5 px, semi-transparent)
        ax.add_patch(mpatches.Rectangle(
            (x1 + 1.5, y1 + 1.5), bw, bh,
            linewidth=1.5,
            edgecolor="#000000",
            facecolor="none",
            alpha=0.45,
            zorder=3,
        ))

        # Main bounding box — editable as a path in Illustrator
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), bw, bh,
            linewidth=1.5,
            edgecolor=color_hex,
            facecolor="none",
            zorder=4,
        ))

        label = _make_label(det, style)
        if not label:
            continue

        # Prefer label above the box; fall back below when near the top edge
        if y1 > 20:
            ty, va = y1 - 2, "bottom"
        else:
            ty, va = y2 + 2, "top"

        # bbox= adds a coloured rectangle behind the text — both the rect and
        # the <text> element are separate objects in the SVG/PDF, so colours
        # and content can be changed independently in Illustrator.
        ax.text(
            x1 + 2, ty, label,
            fontsize=7,
            color="white",
            va=va,
            ha="left",
            fontfamily="Arial",
            zorder=5,
            bbox=dict(
                boxstyle="square,pad=0.25",
                facecolor=color_hex,
                alpha=0.85,
                edgecolor="none",
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corr_det_from_annotation(
    annotation: Optional[dict],
    frame_idx: int,
) -> Optional[Detection]:
    """Build a Detection from a corrected annotation dict, or return None."""
    if annotation and annotation.get("corrected") and "bbox" in annotation:
        return Detection(
            frame=frame_idx,
            bbox_xyxy=tuple(annotation["bbox"]),
            cls=annotation.get("cls"),
        )
    return None


def _make_label(det: Detection, style: OverlayStyle) -> str:
    """Compose the label string for a detection, respecting OverlayStyle flags."""
    parts: List[str] = []
    class_names = style.class_names or []
    if style.show_class and det.cls is not None:
        if 0 <= det.cls < len(class_names):
            parts.append(class_names[det.cls])
        else:
            parts.append(f"cls:{det.cls}")
    if style.show_track_id and det.track_id is not None:
        parts.append(f"id:{det.track_id}")
    if style.show_confidence and det.confidence is not None:
        parts.append(f"{det.confidence:.2f}")
    return " ".join(parts)
