from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from trackviz.io.predictions import Detection, Predictions
from trackviz.render.overlay import OverlayStyle, draw_overlays

# BGR colors
_COLOR_ORIGINAL = (0, 255, 0)    # green — model predictions
_COLOR_CORRECTION = (0, 200, 255)  # cyan  — user corrections
_HANDLE_RADIUS = 7  # px in label space

# ---------------------------------------------------------------------------
# Dark stylesheet
# ---------------------------------------------------------------------------
_STYLESHEET = """
/* ── Base ─────────────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1a1b2e;
    color: #dde1f0;
    font-family: "Inter", "Segoe UI", "SF Pro Text", sans-serif;
    font-size: 13px;
}

/* ── Video panel ──────────────────────────────────────────────────── */
QWidget#videoPanel {
    background-color: #1a1b2e;
}

/* ── Right sidebar ────────────────────────────────────────────────── */
QWidget#sidePanel {
    background-color: #21223a;
    border-left: 1px solid #383958;
}

/* ── Buttons ──────────────────────────────────────────────────────── */
QPushButton {
    background-color: #2e3052;
    color: #dde1f0;
    border: 1px solid #454772;
    border-radius: 6px;
    padding: 5px 14px;
    min-height: 28px;
}
QPushButton:hover  { background-color: #383b68; border-color: #6870b0; }
QPushButton:pressed { background-color: #252747; }
QPushButton:disabled { color: #555878; border-color: #2e3052; }

/* Primary action */
QPushButton#btnLabel {
    background-color: #3d5fcc;
    border-color: #5578e0;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#btnLabel:hover  { background-color: #4a6fe0; }
QPushButton#btnLabel:pressed { background-color: #2e4fb8; }

/* Play */
QPushButton#btnPlay {
    background-color: #1f5c3a;
    border-color: #2e8c58;
    color: #a6e3c1;
    font-weight: 600;
    min-width: 72px;
}
QPushButton#btnPlay:hover  { background-color: #266b45; }
QPushButton#btnPlay:pressed { background-color: #184d30; }

/* Pause */
QPushButton#btnPause {
    background-color: #5c3a10;
    border-color: #8c6020;
    color: #f5c07a;
    font-weight: 600;
    min-width: 72px;
}
QPushButton#btnPause:hover  { background-color: #6e4614; }
QPushButton#btnPause:pressed { background-color: #4a2e0c; }

/* Step buttons */
QPushButton#btnStep {
    background-color: #2a2b48;
    border-color: #454772;
    min-width: 32px;
    padding: 5px 8px;
    font-size: 14px;
}
QPushButton#btnStep:hover { background-color: #383b68; }

/* Delete — danger tint */
QPushButton#btnDelete {
    background-color: #4a1e2a;
    border-color: #7a3040;
    color: #f0a0b0;
}
QPushButton#btnDelete:hover  { background-color: #5c2535; }
QPushButton#btnDelete:pressed { background-color: #3d1820; }

/* Jump */
QPushButton#btnJump {
    min-width: 52px;
    padding: 5px 10px;
}

/* ── Slider ───────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #383958;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #4a6fe0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #7a9ef5;
    border: 2px solid #4a6fe0;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #a0bcff; }

/* ── CheckBox ─────────────────────────────────────────────────────── */
QCheckBox {
    spacing: 8px;
    color: #c0c4e0;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #545878;
    background: #2a2b48;
}
QCheckBox::indicator:checked {
    background-color: #4a6fe0;
    border-color: #4a6fe0;
    image: url(none);
}
QCheckBox::indicator:hover { border-color: #7a9ef5; }

/* ── ComboBox ─────────────────────────────────────────────────────── */
QComboBox {
    background-color: #2a2b48;
    border: 1px solid #454772;
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 26px;
    color: #dde1f0;
}
QComboBox:hover { border-color: #6870b0; }
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
}
QComboBox QAbstractItemView {
    background-color: #2a2b48;
    border: 1px solid #454772;
    selection-background-color: #3d5fcc;
    color: #dde1f0;
    padding: 2px;
}

/* ── SpinBox ──────────────────────────────────────────────────────── */
QSpinBox {
    background-color: #2a2b48;
    border: 1px solid #454772;
    border-radius: 6px;
    padding: 4px 8px;
    color: #dde1f0;
    min-height: 26px;
}
QSpinBox:hover { border-color: #6870b0; }

/* ── List widget ──────────────────────────────────────────────────── */
QListWidget {
    background-color: #1e1f36;
    border: 1px solid #383958;
    border-radius: 6px;
    outline: none;
    padding: 2px;
}
QListWidget::item {
    padding: 5px 8px;
    border-radius: 4px;
    color: #c8cce8;
}
QListWidget::item:hover     { background-color: #2a2b48; }
QListWidget::item:selected  {
    background-color: #3d5fcc;
    color: #ffffff;
}

/* ── Labels ───────────────────────────────────────────────────────── */
QLabel#sectionHeader {
    color: #8890cc;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
QLabel#frameCounter {
    color: #8890cc;
    font-size: 12px;
    font-family: monospace;
    min-width: 48px;
}
QLabel#savedInfo {
    font-size: 12px;
    color: #8890cc;
}

/* ── Separator ────────────────────────────────────────────────────── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    color: #383958;
    max-height: 1px;
    border: none;
    background: #383958;
}

/* ── Status bar ───────────────────────────────────────────────────── */
QStatusBar {
    background-color: #14152a;
    color: #6870b0;
    font-size: 12px;
    border-top: 1px solid #2e2f50;
}
"""


def _bgr_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    bytes_per_line = 3 * w
    return QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# Interactive image widget
# ---------------------------------------------------------------------------

class InteractiveImageWidget(QtWidgets.QLabel):
    """QLabel subclass that handles mouse interactions for bbox editing.

    The widget stores the *base* pixmap (frame + original overlays already
    baked in by cv2) and draws selection handles / rubber-band on top via
    paintEvent without re-encoding the frame on every mouse move.
    """

    # Emitted when the user finishes drawing a new box (image-space coords).
    box_drawn = QtCore.Signal(list)  # [x1, y1, x2, y2]

    # Emitted when the user selects an existing box.
    # kind = 'det' | 'corr', idx = index into the respective list
    box_selected = QtCore.Signal(str, int)

    # Emitted when edit mode changes
    edit_mode_changed = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtGui.QColor("#000000"))
        self.setPalette(pal)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._edit_mode = False

        # Current-frame data (image-space coordinates)
        self._img_w: int = 1
        self._img_h: int = 1
        self._detections: List[Detection] = []
        self._corrections: List[dict] = []  # [{bbox:[x1,y1,x2,y2], cls:int}]

        # Selection / drag state
        self._selected: Optional[Tuple[str, int]] = None  # ('det'|'corr', idx)
        self._drag_op: Optional[str] = None  # 'move' | 'resize_TL' | ... | None
        self._drag_anchor_img: Optional[QtCore.QPointF] = None  # img-space mouse pos at drag start
        self._drag_bbox_start: Optional[List[float]] = None  # [x1,y1,x2,y2] at drag start

        # New-box drawing state
        self._drawing = False
        self._draw_start_img: Optional[QtCore.QPointF] = None
        self._draw_current_img: Optional[QtCore.QPointF] = None

        # The "pending" bbox that was drawn/selected but not yet saved
        self._pending_bbox: Optional[List[float]] = None  # image coords
        self._pending_kind: Optional[str] = None  # 'new' | 'det' | 'corr'
        self._pending_idx: Optional[int] = None   # only for 'corr'

    # ------------------------------------------------------------------
    # Public API called by TrackVizWindow
    # ------------------------------------------------------------------

    def set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = enabled
        if not enabled:
            self._reset_interaction()
        self.update()

    def set_frame_data(
        self,
        qimg: QtGui.QImage,
        img_w: int,
        img_h: int,
        detections: List[Detection],
        corrections: List[dict],
    ) -> None:
        """Called each time the frame changes."""
        self._img_w = img_w
        self._img_h = img_h
        self._orig_qimage = qimg  # keep original for re-scaling on resize
        self._detections = detections
        self._corrections = corrections
        self._reset_interaction()
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        """Scale the stored image to fit the label (aspect-ratio preserved) and set it."""
        if not hasattr(self, "_orig_qimage") or self._orig_qimage is None:
            return
        lw, lh = self.width(), self.height()
        if lw <= 0 or lh <= 0:
            return
        pix = QtGui.QPixmap.fromImage(self._orig_qimage).scaled(
            lw, lh, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.setPixmap(pix)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def get_pending(self) -> Tuple[Optional[List[float]], Optional[str], Optional[int]]:
        """Return (bbox_xyxy, kind, idx) of the pending (not-yet-saved) selection."""
        return self._pending_bbox, self._pending_kind, self._pending_idx

    def clear_pending(self) -> None:
        self._pending_bbox = None
        self._pending_kind = None
        self._pending_idx = None
        self._selected = None
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.clear_pending()
            event.accept()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _get_pixmap_rect(self) -> QtCore.QRect:
        """Return the rect occupied by the (already-scaled) pixmap inside the label."""
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return QtCore.QRect(0, 0, self.width(), self.height())
        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        return QtCore.QRect(ox, oy, pw, ph)

    def _to_img(self, pt: QtCore.QPointF) -> QtCore.QPointF:
        """Convert label-space point to image-space point."""
        r = self._get_pixmap_rect()
        if r.width() == 0 or r.height() == 0:
            return pt
        sx = self._img_w / r.width()
        sy = self._img_h / r.height()
        return QtCore.QPointF((pt.x() - r.x()) * sx, (pt.y() - r.y()) * sy)

    def _to_label(self, pt: QtCore.QPointF) -> QtCore.QPointF:
        """Convert image-space point to label-space point."""
        r = self._get_pixmap_rect()
        if self._img_w == 0 or self._img_h == 0:
            return pt
        sx = r.width() / self._img_w
        sy = r.height() / self._img_h
        return QtCore.QPointF(r.x() + pt.x() * sx, r.y() + pt.y() * sy)

    def _bbox_to_label(self, bbox: List[float]) -> QtCore.QRectF:
        tl = self._to_label(QtCore.QPointF(bbox[0], bbox[1]))
        br = self._to_label(QtCore.QPointF(bbox[2], bbox[3]))
        return QtCore.QRectF(tl, br)

    # ------------------------------------------------------------------
    # Handle positions (8 handles: 4 corners + 4 edge midpoints)
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_positions(rect: QtCore.QRectF) -> Dict[str, QtCore.QPointF]:
        cx = (rect.left() + rect.right()) / 2
        cy = (rect.top() + rect.bottom()) / 2
        return {
            "TL": QtCore.QPointF(rect.left(), rect.top()),
            "TC": QtCore.QPointF(cx, rect.top()),
            "TR": QtCore.QPointF(rect.right(), rect.top()),
            "ML": QtCore.QPointF(rect.left(), cy),
            "MR": QtCore.QPointF(rect.right(), cy),
            "BL": QtCore.QPointF(rect.left(), rect.bottom()),
            "BC": QtCore.QPointF(cx, rect.bottom()),
            "BR": QtCore.QPointF(rect.right(), rect.bottom()),
        }

    def _hit_handle(self, pos: QtCore.QPointF, rect: QtCore.QRectF) -> Optional[str]:
        for name, hpt in self._handle_positions(rect).items():
            dx = pos.x() - hpt.x()
            dy = pos.y() - hpt.y()
            if dx * dx + dy * dy <= _HANDLE_RADIUS ** 2:
                return name
        return None

    # ------------------------------------------------------------------
    # Hit testing on detections
    # ------------------------------------------------------------------

    def _hit_test(self, pos_label: QtCore.QPointF):
        """Return (kind, idx, handle_name|None) of the topmost hit, or None.

        The pending box is checked first because it may have been dragged away
        from its original stored position — we must use its current visual location.
        """
        # Pending box takes priority (it reflects the current visual position after any drag)
        if self._pending_bbox and len(self._pending_bbox) == 4:
            b = self._pending_bbox
            rx1, ry1 = min(b[0], b[2]), min(b[1], b[3])
            rx2, ry2 = max(b[0], b[2]), max(b[1], b[3])
            r = self._bbox_to_label([rx1, ry1, rx2, ry2])
            h = self._hit_handle(pos_label, r)
            if h:
                return self._pending_kind, self._pending_idx, h
            if r.contains(pos_label):
                return self._pending_kind, self._pending_idx, None

        # Check corrections (they visually sit on top of original detections)
        for i, corr in reversed(list(enumerate(self._corrections))):
            r = self._bbox_to_label(corr["bbox"])
            h = self._hit_handle(pos_label, r)
            if h:
                return "corr", i, h
            if r.contains(pos_label):
                return "corr", i, None
        # Then original detections
        for i, det in reversed(list(enumerate(self._detections))):
            r = self._bbox_to_label(
                [det.bbox_xyxy[0], det.bbox_xyxy[1], det.bbox_xyxy[2], det.bbox_xyxy[3]]
            )
            h = self._hit_handle(pos_label, r)
            if h:
                return "det", i, h
            if r.contains(pos_label):
                return "det", i, None
        return None

    def _get_bbox(self, kind: str, idx: int) -> List[float]:
        if kind == "corr":
            return list(self._corrections[idx]["bbox"])
        det = self._detections[idx]
        return [det.bbox_xyxy[0], det.bbox_xyxy[1], det.bbox_xyxy[2], det.bbox_xyxy[3]]

    def _get_cls(self, kind: str, idx: int) -> Optional[int]:
        if kind == "corr":
            return self._corrections[idx].get("cls")
        return self._detections[idx].cls

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._edit_mode:
            super().mousePressEvent(event)
            return
        pos = event.position() if hasattr(event, "position") else QtCore.QPointF(event.pos())
        hit = self._hit_test(pos)
        if hit:
            kind, idx, handle = hit
            self._selected = (kind, idx)
            self._drag_anchor_img = self._to_img(pos)
            self._drag_op = f"resize_{handle}" if handle else "move"
            self._drawing = False

            # If the hit landed on the already-pending item, use its current
            # (possibly dragged) position as the drag start, not the stale
            # stored coords — this is what allows multiple adjustments in a row.
            if (
                self._pending_bbox is not None
                and kind == self._pending_kind
                and idx == self._pending_idx
            ):
                self._drag_bbox_start = list(self._pending_bbox)
            else:
                self._drag_bbox_start = self._get_bbox(kind, idx)
                self._pending_bbox = list(self._drag_bbox_start)
                self._pending_kind = kind
                self._pending_idx = idx if kind == "corr" else None
                # Only notify the parent for a genuinely new selection
                if kind in ("det", "corr"):
                    self.box_selected.emit(kind, idx)
        else:
            # Start drawing a new box
            self._selected = None
            self._pending_bbox = None
            self._pending_kind = None
            self._pending_idx = None
            self._drag_op = None
            self._drawing = True
            self._draw_start_img = self._to_img(pos)
            self._draw_current_img = self._draw_start_img
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._edit_mode:
            super().mouseMoveEvent(event)
            return
        pos = event.position() if hasattr(event, "position") else QtCore.QPointF(event.pos())
        img_pos = self._to_img(pos)

        if self._drawing and self._draw_start_img is not None:
            self._draw_current_img = img_pos
            self.update()
            return

        if self._drag_op and self._drag_anchor_img and self._drag_bbox_start:
            dx = img_pos.x() - self._drag_anchor_img.x()
            dy = img_pos.y() - self._drag_anchor_img.y()
            b = list(self._drag_bbox_start)
            op = self._drag_op

            if op == "move":
                w = b[2] - b[0]
                h = b[3] - b[1]
                nx1 = b[0] + dx
                ny1 = b[1] + dy
                b = [nx1, ny1, nx1 + w, ny1 + h]
            else:
                handle = op.replace("resize_", "")
                if "L" in handle:
                    b[0] = b[0] + dx
                if "R" in handle:
                    b[2] = b[2] + dx
                if "T" in handle:
                    b[1] = b[1] + dy
                if "B" in handle:
                    b[3] = b[3] + dy

            # Clamp to image bounds
            b[0] = max(0.0, min(float(self._img_w - 1), b[0]))
            b[1] = max(0.0, min(float(self._img_h - 1), b[1]))
            b[2] = max(0.0, min(float(self._img_w - 1), b[2]))
            b[3] = max(0.0, min(float(self._img_h - 1), b[3]))
            self._pending_bbox = b
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._edit_mode:
            super().mouseReleaseEvent(event)
            return
        pos = event.position() if hasattr(event, "position") else QtCore.QPointF(event.pos())
        img_pos = self._to_img(pos)

        if self._drawing:
            self._drawing = False
            if self._draw_start_img:
                x1 = min(self._draw_start_img.x(), img_pos.x())
                y1 = min(self._draw_start_img.y(), img_pos.y())
                x2 = max(self._draw_start_img.x(), img_pos.x())
                y2 = max(self._draw_start_img.y(), img_pos.y())
                # Clamp
                x1 = max(0.0, min(float(self._img_w - 1), x1))
                y1 = max(0.0, min(float(self._img_h - 1), y1))
                x2 = max(0.0, min(float(self._img_w - 1), x2))
                y2 = max(0.0, min(float(self._img_h - 1), y2))
                if x2 - x1 > 4 and y2 - y1 > 4:
                    self._pending_bbox = [x1, y1, x2, y2]
                    self._pending_kind = "new"
                    self._pending_idx = None
                    self.box_drawn.emit(self._pending_bbox)
            self._draw_start_img = None
            self._draw_current_img = None
            self.update()
            return

        self._drag_op = None
        self._drag_anchor_img = None
        self._drag_bbox_start = None
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)  # draws the pixmap
        if not self._edit_mode:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Draw the pending/selected bbox with handles
        if self._pending_bbox and len(self._pending_bbox) == 4:
            b = self._pending_bbox
            # Ensure valid order for display
            rx1, ry1 = min(b[0], b[2]), min(b[1], b[3])
            rx2, ry2 = max(b[0], b[2]), max(b[1], b[3])
            rect = self._bbox_to_label([rx1, ry1, rx2, ry2])
            color = QtGui.QColor(0, 200, 255)
            pen = QtGui.QPen(color, 2, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(rect)
            # Draw handles
            handle_brush = QtGui.QBrush(color)
            painter.setBrush(handle_brush)
            painter.setPen(QtCore.Qt.NoPen)
            for hpt in self._handle_positions(rect).values():
                painter.drawEllipse(hpt, _HANDLE_RADIUS, _HANDLE_RADIUS)

        # Draw rubber-band for new box being drawn
        if self._drawing and self._draw_start_img and self._draw_current_img:
            tl = self._to_label(self._draw_start_img)
            br = self._to_label(self._draw_current_img)
            rect = QtCore.QRectF(tl, br).normalized()
            pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 1, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(rect)

        painter.end()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_interaction(self) -> None:
        self._selected = None
        self._drag_op = None
        self._drag_anchor_img = None
        self._drag_bbox_start = None
        self._drawing = False
        self._draw_start_img = None
        self._draw_current_img = None
        self._pending_bbox = None
        self._pending_kind = None
        self._pending_idx = None


# ---------------------------------------------------------------------------
# Video reader
# ---------------------------------------------------------------------------

class VideoReader:
    """Thin wrapper around cv2.VideoCapture with frame-count inference and caching."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._last_idx = -1
        self._cache = {}

    def read_frame(self, idx: int) -> Optional[np.ndarray]:
        idx = int(idx)
        if idx < 0:
            return None
        if idx != self._last_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._last_idx = idx
        return frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Viewer config
# ---------------------------------------------------------------------------

@dataclass
class ViewerConfig:
    start_paused: bool = True
    playback_fps: Optional[float] = None
    overlay_style: OverlayStyle = OverlayStyle()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class TrackVizWindow(QtWidgets.QMainWindow):
    """Main application window.

    Annotation model (single source of truth)
    ------------------------------------------
    _annotations: Dict[str, dict]
        key   = str(frame_index)
        value = {"cls": int, "bbox": [x1,y1,x2,y2], "corrected": bool}

    * corrected=False  → bbox is the model's prediction (auto-filled)
    * corrected=True   → bbox was manually drawn/moved/resized by the user
    Saved to  <video_stem>_annotations.json
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        preds: Optional[Predictions] = None,
        config: Optional[ViewerConfig] = None,
    ):
        super().__init__()
        self.setWindowTitle("trackviz")
        self.setAcceptDrops(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setStyleSheet(_STYLESHEET)

        self.video: Optional[VideoReader] = None
        self.preds: Optional[Predictions] = None
        self.cfg = config or ViewerConfig()
        self.playback_fps = 30.0
        self.max_frames = 0

        # ── Root layout ───────────────────────────────────────────────
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_h_layout = QtWidgets.QHBoxLayout(central)
        main_h_layout.setContentsMargins(0, 0, 0, 0)
        main_h_layout.setSpacing(0)

        # ── Left panel ────────────────────────────────────────────────
        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("videoPanel")
        v = QtWidgets.QVBoxLayout(left_panel)
        v.setContentsMargins(12, 12, 12, 8)
        v.setSpacing(8)
        main_h_layout.addWidget(left_panel, stretch=3)

        # Image area — pure black background for letterboxing
        self.image_label = InteractiveImageWidget()
        self.image_label.setStyleSheet("background-color: #000000; border-radius: 6px;")
        v.addWidget(self.image_label, stretch=1)

        # Transport controls row
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)

        self.btn_prev = QtWidgets.QPushButton("◀")
        self.btn_prev.setObjectName("btnStep")
        self.btn_prev.setFixedWidth(36)
        self.btn_toggle_play = QtWidgets.QPushButton("▶  Play")
        self.btn_toggle_play.setObjectName("btnPlay")
        self.btn_next = QtWidgets.QPushButton("▶")
        self.btn_next.setObjectName("btnStep")
        self.btn_next.setFixedWidth(36)
        controls.addWidget(self.btn_prev)
        controls.addWidget(self.btn_toggle_play)
        controls.addWidget(self.btn_next)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        controls.addWidget(self.slider, stretch=1)

        self.lbl_frame = QtWidgets.QLabel("0 / 0")
        self.lbl_frame.setObjectName("frameCounter")
        self.lbl_frame.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_frame.setMinimumWidth(70)
        controls.addWidget(self.lbl_frame)

        self.jump_box = QtWidgets.QSpinBox()
        self.jump_box.setRange(0, 0)
        self.jump_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.jump_box.setFixedWidth(72)
        self.jump_btn = QtWidgets.QPushButton("Go")
        self.jump_btn.setObjectName("btnJump")
        self.jump_btn.setFixedWidth(44)
        controls.addWidget(self.jump_box)
        controls.addWidget(self.jump_btn)
        v.addLayout(controls)

        # Options row
        opts = QtWidgets.QHBoxLayout()
        opts.setSpacing(16)
        self.chk_overlay = QtWidgets.QCheckBox("Overlay")
        self.chk_overlay.setChecked(True)
        self.chk_heatmap = QtWidgets.QCheckBox("Heatmap")
        self.chk_heatmap.setChecked(False)
        self.chk_edit = QtWidgets.QCheckBox("Edit Mode")
        self.chk_edit.setChecked(False)
        opts.addWidget(self.chk_overlay)
        opts.addWidget(self.chk_heatmap)
        opts.addWidget(self.chk_edit)
        opts.addStretch()
        v.addLayout(opts)

        # Behavior / save row
        save_row = QtWidgets.QHBoxLayout()
        save_row.setSpacing(8)
        lbl_behavior = QtWidgets.QLabel("Behavior")
        lbl_behavior.setObjectName("sectionHeader")
        self.combo_classes = QtWidgets.QComboBox()
        self.behavior_names = [
            "0: ProbPumping", "1: Moving", "2: Grooming",
            "3: Feeding", "4: Quiescent", "5: HaltereSwitch", "6: Twitching",
            "7: Defecation",
        ]
        self.combo_classes.addItems(self.behavior_names)
        self.combo_classes.setMinimumWidth(160)
        self.btn_label = QtWidgets.QPushButton("Save  [S]")
        self.btn_label.setObjectName("btnLabel")
        self.lbl_saved_info = QtWidgets.QLabel("")
        self.lbl_saved_info.setObjectName("savedInfo")
        save_row.addWidget(lbl_behavior)
        save_row.addWidget(self.combo_classes, stretch=1)
        save_row.addWidget(self.btn_label)
        save_row.addWidget(self.lbl_saved_info)
        v.addLayout(save_row)

        # ── Right sidebar ─────────────────────────────────────────────
        right_panel = QtWidgets.QWidget()
        right_panel.setObjectName("sidePanel")
        right_panel.setFixedWidth(280)
        v_right = QtWidgets.QVBoxLayout(right_panel)
        v_right.setContentsMargins(12, 14, 12, 12)
        v_right.setSpacing(8)
        main_h_layout.addWidget(right_panel)

        lbl_anno_header = QtWidgets.QLabel("ANNOTATIONS")
        lbl_anno_header.setObjectName("sectionHeader")
        v_right.addWidget(lbl_anno_header)

        self.list_annotations = QtWidgets.QListWidget()
        self.list_annotations.setAlternatingRowColors(False)
        v_right.addWidget(self.list_annotations, stretch=1)

        self.btn_delete_anno = QtWidgets.QPushButton("Delete Selected")
        self.btn_delete_anno.setObjectName("btnDelete")
        v_right.addWidget(self.btn_delete_anno)

        v_right.addWidget(_hsep())

        self.lbl_edit_hint = QtWidgets.QLabel(
            "0–7  select class  ·  S  save  ·  Del  delete annotation\n"
            "←/→  step  ·  Space  play/pause  ·  Esc  clear selection\n"
            "Edit Mode: click/drag a box, then press S."
        )
        self.lbl_edit_hint.setWordWrap(True)
        self.lbl_edit_hint.setStyleSheet("color: #555878; font-size: 11px;")
        v_right.addWidget(self.lbl_edit_hint)

        # ── State ─────────────────────────────────────────────────────
        self._playing = False
        self._current_frame = 0
        self._annotations: Dict[str, dict] = {}
        self._annotation_path: Optional[Path] = None
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        # ── Signals ───────────────────────────────────────────────────
        self.btn_toggle_play.clicked.connect(self.toggle_play)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(+1))
        self.slider.valueChanged.connect(self.set_frame)
        self.chk_overlay.stateChanged.connect(lambda: self.set_frame(self._current_frame, force=True))
        self.chk_heatmap.stateChanged.connect(lambda: self.set_frame(self._current_frame, force=True))
        self.jump_btn.clicked.connect(lambda: self.set_frame(self.jump_box.value()))
        self.btn_label.clicked.connect(self._save_annotation)
        self.btn_delete_anno.clicked.connect(self._delete_annotation)
        self.list_annotations.itemDoubleClicked.connect(self._on_anno_clicked)
        self.chk_edit.stateChanged.connect(self._on_edit_mode_changed)
        self.image_label.box_drawn.connect(self._on_box_drawn)
        self.image_label.box_selected.connect(self._on_box_selected)

        # ── Status bar ────────────────────────────────────────────────
        self._status_bar = self.statusBar()
        self._status_bar.showMessage("Ready — drag a video file to load.")

        self._show_blank()
        if video_path is not None and preds is not None:
            self.load_video_and_predictions(video_path, preds)
            if not self.cfg.start_paused:
                self.play()

    # ------------------------------------------------------------------
    # Key / close events
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self.toggle_play()
            event.accept()
        elif key == QtCore.Qt.Key_Left:
            self.step(-1)
            event.accept()
        elif key == QtCore.Qt.Key_Right:
            self.step(1)
            event.accept()
        elif key == QtCore.Qt.Key_Escape:
            self.image_label.clear_pending()
            self._status_bar.showMessage("Selection cleared")
            event.accept()
        elif key == QtCore.Qt.Key_Delete:
            self._delete_current_annotation()
            event.accept()
        elif key == QtCore.Qt.Key_S:
            self._save_annotation()
            event.accept()
        elif QtCore.Qt.Key_0 <= key <= QtCore.Qt.Key_9:
            digit = key - QtCore.Qt.Key_0
            if digit < self.combo_classes.count():
                self.combo_classes.setCurrentIndex(digit)
                self._status_bar.showMessage(f"Class → {self.behavior_names[digit]}")
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.video is not None:
            self.video.release()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        for url in urls:
            p = Path(url.toLocalFile())
            if not p.exists() or p.is_dir():
                continue
            if p.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
                continue
            try:
                self.load_from_video_path(p)
            except SystemExit as e:
                self._status_bar.showMessage(f"Error: {e}")
                QtWidgets.QMessageBox.critical(self, "trackviz", str(e))
                event.ignore()
                return
            except Exception as e:
                self._status_bar.showMessage(f"Error: {e}")
                QtWidgets.QMessageBox.critical(self, "trackviz", f"Failed to load video/predictions:\n{e}")
                event.ignore()
                return
            event.acceptProposedAction()
            return
        event.ignore()

    # ------------------------------------------------------------------
    # Blank / load
    # ------------------------------------------------------------------

    def _show_blank(self) -> None:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        qimg = _bgr_to_qimage(blank)
        self.image_label.set_frame_data(qimg, 640, 360, [], [])

    def load_video_and_predictions(self, video_path: str, preds: Predictions) -> None:
        self.pause()
        self._status_bar.showMessage(f"Loading {Path(video_path).name}…")
        QtWidgets.QApplication.processEvents()
        if self.video is not None:
            self.video.release()
        self.video = VideoReader(video_path)
        self.preds = preds
        self.playback_fps = self.cfg.playback_fps or self.video.fps
        v_path = Path(video_path)
        self._annotation_path = v_path.parent / f"{v_path.stem}_annotations.json"
        self._load_annotations()
        self.max_frames = min(self.video.frame_count or self.preds.total_frames, self.preds.total_frames)
        if self.max_frames <= 0:
            self.max_frames = self.preds.total_frames
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self.max_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.jump_box.setMaximum(max(0, self.max_frames - 1))
        self.set_frame(0)
        self._status_bar.showMessage(
            f"Loaded {Path(video_path).name} — {self.max_frames} frames @ {self.video.fps:.1f} fps"
        )

    def load_from_video_path(self, video_path: Path) -> None:
        self.pause()
        self._status_bar.showMessage(f"Loading {video_path.name}…")
        QtWidgets.QApplication.processEvents()
        if self.video is not None:
            self.video.release()
        self.video = VideoReader(str(video_path))
        self.playback_fps = self.cfg.playback_fps or self.video.fps
        self._status_bar.showMessage(f"Loading predictions for {video_path.name}…")
        QtWidgets.QApplication.processEvents()
        preds = self._autoload_predictions_for_video(video_path)
        self._annotation_path = video_path.parent / f"{video_path.stem}_annotations.json"
        self._load_annotations()
        self.preds = preds
        v_count = self.video.frame_count
        p_count = self.preds.total_frames
        if v_count > 0 and p_count > 0 and abs(v_count - p_count) > max(10, int(0.01 * v_count)):
            print(
                f"[trackviz] Warning: video has {v_count} frames but predictions cover "
                f"{p_count} frames. Overlays may be misaligned."
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Frame count mismatch",
                f"Video: {v_count} frames\nPredictions: {p_count} frames\n\n"
                "Overlays may be misaligned.",
            )
        self.max_frames = v_count if v_count > 0 else p_count
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self.max_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.jump_box.setMaximum(max(0, self.max_frames - 1))
        self.set_frame(0)
        self._status_bar.showMessage(
            f"Loaded {video_path.name} — {self.max_frames} frames @ {self.video.fps:.1f} fps"
        )

    # ------------------------------------------------------------------
    # Annotations  (unified: class + bbox + corrected flag)
    # ------------------------------------------------------------------

    def _save_annotation(self) -> None:
        if self._annotation_path is None:
            return

        cls_idx = self.combo_classes.currentIndex()

        # bbox source priority: pending edit → first model prediction → None
        pending_bbox, _, _ = self.image_label.get_pending()
        corrected = pending_bbox is not None

        bbox = pending_bbox
        if bbox is None and self.preds is not None:
            dets = self.preds.for_frame(self._current_frame)
            if dets:
                b = dets[0].bbox_xyxy
                bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

        entry: dict = {"cls": cls_idx, "corrected": corrected}
        if bbox is not None:
            entry["bbox"] = [float(x) for x in bbox]

        self._annotations[str(self._current_frame)] = entry
        if corrected:
            self.image_label.clear_pending()

        with open(self._annotation_path, "w") as f:
            json.dump(self._annotations, f, indent=2)

        self._update_annotation_list()
        self.set_frame(self._current_frame, force=True)
        self._status_bar.showMessage(
            f"Saved frame {self._current_frame}: {self.behavior_names[cls_idx]}"
            + ("  [corrected box]" if corrected else "  [predicted box]")
        )

    def _load_annotations(self) -> None:
        """Load annotations from JSON, migrating old formats and box-corrections files."""
        self._annotations = {}
        if self._annotation_path and self._annotation_path.exists():
            try:
                with open(self._annotation_path, "r") as f:
                    raw = json.load(f)
                for k, v in raw.items():
                    if isinstance(v, int):
                        # Oldest format: {frame: class_int}
                        self._annotations[k] = {"cls": v, "corrected": False}
                    else:
                        # Ensure corrected key exists
                        if "corrected" not in v:
                            v["corrected"] = "bbox" in v
                        self._annotations[k] = v
            except Exception as e:
                print(f"[trackviz] Failed to load annotations: {e}")

        # Migrate legacy _box_corrections.json if present
        if self._annotation_path:
            corr_path = self._annotation_path.parent / (
                self._annotation_path.stem.replace("_annotations", "") + "_box_corrections.json"
            )
            if corr_path.exists():
                try:
                    with open(corr_path, "r") as f:
                        old_corr = json.load(f)
                    migrated = False
                    for frame_str, corr_list in old_corr.items():
                        if corr_list and frame_str not in self._annotations:
                            c = corr_list[0]
                            self._annotations[frame_str] = {
                                "cls": c.get("cls", 0),
                                "bbox": c["bbox"],
                                "corrected": True,
                            }
                            migrated = True
                    if migrated:
                        with open(self._annotation_path, "w") as f:
                            json.dump(self._annotations, f, indent=2)
                        print(f"[trackviz] Migrated box corrections from {corr_path.name}")
                except Exception as e:
                    print(f"[trackviz] Failed to migrate box corrections: {e}")

        self._update_annotation_list()

    @staticmethod
    def _anno_cls(entry) -> int:
        return entry["cls"] if isinstance(entry, dict) else int(entry)

    def _update_annotation_list(self) -> None:
        """Rebuild the sidebar annotation list."""
        count = len(self._annotations)
        self.list_annotations.blockSignals(True)
        self.list_annotations.clear()

        _cyan = QtGui.QColor(0, 200, 220)

        for k in sorted(self._annotations.keys(), key=lambda x: int(x)):
            entry = self._annotations[k]
            cls = self._anno_cls(entry)
            name = self.behavior_names[cls] if 0 <= cls < len(self.behavior_names) else f"cls:{cls}"
            corrected = isinstance(entry, dict) and entry.get("corrected", False)
            has_bbox = isinstance(entry, dict) and "bbox" in entry
            if corrected:
                suffix = "  ✓"
            elif has_bbox:
                suffix = "  □"
            else:
                suffix = ""
            item = QtWidgets.QListWidgetItem(f"Frame {k}:  {name}{suffix}")
            if corrected:
                item.setForeground(_cyan)
            self.list_annotations.addItem(item)

        self.list_annotations.blockSignals(False)

        # Status bar label
        curr = self._annotations.get(str(self._current_frame))
        if curr is not None:
            cls = self._anno_cls(curr)
            name = self.behavior_names[cls] if 0 <= cls < len(self.behavior_names) else f"cls:{cls}"
            corrected = isinstance(curr, dict) and curr.get("corrected", False)
            tag = "  ✓ corrected" if corrected else ""
            self.lbl_saved_info.setText(f"Total: {count} | Current: {name}{tag}")
            self.lbl_saved_info.setStyleSheet("color: #00AA00; font-weight: bold;")
        else:
            self.lbl_saved_info.setText(f"Total: {count} | Current: none")
            self.lbl_saved_info.setStyleSheet("")

    def _on_anno_clicked(self, item) -> None:
        try:
            f_str = item.text().split(":")[0].split(" ")[1]
            self.set_frame(int(f_str))
        except Exception as e:
            print(f"[trackviz] Failed to navigate to annotation: {e}")

    def _delete_annotation(self) -> None:
        item = self.list_annotations.currentItem()
        if not item:
            return
        try:
            f_str = item.text().split(":")[0].split(" ")[1]
            if f_str in self._annotations:
                del self._annotations[f_str]
                if self._annotation_path:
                    with open(self._annotation_path, "w") as f:
                        json.dump(self._annotations, f, indent=2)
                self._update_annotation_list()
                self.set_frame(self._current_frame, force=True)
        except Exception as e:
            print(f"[trackviz] Failed to delete annotation: {e}")

    def _delete_current_annotation(self) -> None:
        f_str = str(self._current_frame)
        if f_str not in self._annotations:
            self._status_bar.showMessage("No annotation on this frame")
            return
        del self._annotations[f_str]
        if self._annotation_path:
            with open(self._annotation_path, "w") as f:
                json.dump(self._annotations, f, indent=2)
        self._update_annotation_list()
        self.set_frame(self._current_frame, force=True)
        self._status_bar.showMessage(f"Annotation deleted — frame {self._current_frame}")

    # ------------------------------------------------------------------
    # Edit mode callbacks
    # ------------------------------------------------------------------

    def _on_edit_mode_changed(self, state: int) -> None:
        enabled = bool(state)
        if enabled:
            self.pause()
        self.image_label.set_edit_mode(enabled)
        if enabled:
            self._presselect_class_for_frame()
            self._status_bar.showMessage("Edit Mode on — click/drag a box, then press S to save.")
        else:
            self._status_bar.showMessage("Edit Mode off.")

    def _presselect_class_for_frame(self) -> None:
        """Set the class dropdown to match the current frame's annotation or prediction."""
        # Existing annotation takes priority
        anno = self._annotations.get(str(self._current_frame))
        if anno is not None:
            cls = anno.get("cls")
            if cls is not None and 0 <= cls < self.combo_classes.count():
                self.combo_classes.setCurrentIndex(cls)
                return
        # Fall back to first predicted detection
        if self.preds is not None:
            dets = self.preds.for_frame(self._current_frame)
            if dets and dets[0].cls is not None:
                cls = int(dets[0].cls)
                if 0 <= cls < self.combo_classes.count():
                    self.combo_classes.setCurrentIndex(cls)

    def _on_box_drawn(self, bbox: list) -> None:
        self._status_bar.showMessage("New box drawn — pick a class and press S to save.")

    def _on_box_selected(self, kind: str, idx: int) -> None:
        source = "corrected" if kind == "corr" else "predicted"
        self._status_bar.showMessage(
            f"Selected {source} box — drag to adjust, then press S to save."
        )
        # Pre-select the class from the existing annotation for this frame
        anno = self._annotations.get(str(self._current_frame))
        if anno:
            cls = self._anno_cls(anno)
            if 0 <= cls < self.combo_classes.count():
                self.combo_classes.setCurrentIndex(cls)

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self._playing:
            return
        self._playing = True
        self.btn_toggle_play.setText("⏸  Pause")
        self.btn_toggle_play.setObjectName("btnPause")
        self.btn_toggle_play.style().unpolish(self.btn_toggle_play)
        self.btn_toggle_play.style().polish(self.btn_toggle_play)
        interval_ms = int(round(1000.0 / max(1e-3, self.playback_fps)))
        self._timer.start(interval_ms)

    def pause(self) -> None:
        self._playing = False
        self.btn_toggle_play.setText("▶  Play")
        self.btn_toggle_play.setObjectName("btnPlay")
        self.btn_toggle_play.style().unpolish(self.btn_toggle_play)
        self.btn_toggle_play.style().polish(self.btn_toggle_play)
        self._timer.stop()

    def step(self, delta: int) -> None:
        self.pause()
        self.set_frame(self._current_frame + int(delta))

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def _generate_heatmap(self, idx: int) -> Optional[np.ndarray]:
        if self.video is None:
            return None
        window_size = 6
        frames = []
        for i in range(max(0, idx - window_size + 1), idx + 1):
            if i in self.video._cache:
                frames.append(self.video._cache[i])
            else:
                f = self.video.read_frame(i)
                if f is not None:
                    if f.ndim == 3:
                        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    self.video._cache[i] = f
                    frames.append(f)
        if len(self.video._cache) > 20:
            oldest = min(self.video._cache.keys())
            if oldest < idx - window_size - 10:
                del self.video._cache[oldest]
        if not frames:
            return None
        ref_gray = frames[0]
        h, w = ref_gray.shape
        accumulator = np.zeros((h, w), dtype=np.float32)
        threshold = 10
        for i in range(1, len(frames)):
            diff = cv2.absdiff(ref_gray, frames[i])
            _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            accumulator += mask
        mx = accumulator.max()
        acc_u8 = (accumulator * (255.0 / mx)).astype(np.uint8) if mx > 0 else np.zeros_like(ref_gray, dtype=np.uint8)
        heatmap = cv2.applyColorMap(acc_u8, cv2.COLORMAP_HOT)
        ref_bgr = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(ref_bgr, 0.6, heatmap, 0.4, 0)

    # ------------------------------------------------------------------
    # set_frame
    # ------------------------------------------------------------------

    def set_frame(self, idx: int, force: bool = False) -> None:
        idx = int(idx)
        if self.video is None or self.preds is None or self.max_frames <= 0:
            self._current_frame = 0
            self.lbl_frame.setText("— / —")
            return

        idx = max(0, min(self.max_frames - 1, idx))
        if not force and idx == self._current_frame and self.image_label.pixmap() is not None:
            self._update_annotation_list()
            return

        self._current_frame = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.jump_box.blockSignals(True)
        self.jump_box.setValue(idx)
        self.jump_box.blockSignals(False)
        self.lbl_frame.setText(f"{idx} / {self.max_frames - 1}")

        if self.chk_edit.isChecked():
            self._presselect_class_for_frame()

        if self.chk_heatmap.isChecked():
            frame = self._generate_heatmap(idx)
        else:
            frame = self.video.read_frame(idx)

        if frame is None:
            self._show_blank()
            return

        # Always fetch dets (needed for widget hit-testing even when overlay is off)
        dets = self.preds.for_frame(idx)
        style = self.cfg.overlay_style
        if self.preds.meta.get("class_names"):
            style.class_names = self.preds.meta["class_names"]

        if self.chk_overlay.isChecked():
            frame = draw_overlays(frame, dets, style, box_color=_COLOR_ORIGINAL)

        # Render corrected annotation bbox in cyan
        anno = self._annotations.get(str(idx))
        widget_corrections: List[dict] = []
        if anno and isinstance(anno, dict) and anno.get("corrected") and "bbox" in anno:
            corr_det = Detection(
                frame=idx,
                bbox_xyxy=tuple(anno["bbox"]),
                cls=anno.get("cls"),
            )
            frame = draw_overlays(frame, [corr_det], style, box_color=_COLOR_CORRECTION)
            widget_corrections = [{"bbox": anno["bbox"], "cls": anno.get("cls", 0)}]

        img_h, img_w = frame.shape[:2]
        qimg = _bgr_to_qimage(frame)
        self.image_label.set_frame_data(qimg, img_w, img_h, dets, widget_corrections)
        self._update_annotation_list()

    def _on_tick(self) -> None:
        if self.video is None or self.preds is None:
            self.pause()
            return
        nxt = self._current_frame + 1
        if nxt >= self.max_frames:
            self.pause()
            return
        self.set_frame(nxt)

    # ------------------------------------------------------------------
    # Auto-load predictions
    # ------------------------------------------------------------------

    def _autoload_predictions_for_video(self, video_path: Path) -> Predictions:
        root = video_path.parent
        stem = video_path.stem

        def _pick(cands: list) -> Optional[Path]:
            for c in cands:
                if c.exists():
                    return c
            return None

        run_id = stem
        if stem.endswith("_raw"):
            run_id = stem[:-4]

        yolo_pkl = _pick(
            [
                root / f"{run_id}_yolo_fast.pkl",
                root / f"{stem}_yolo_fast.pkl",
                root / "yolo_fast.pkl",
                root / "results" / f"{run_id}_yolo_fast.pkl",
                root / "results" / f"{stem}_yolo_fast.pkl",
                root / "results" / "yolo_fast.pkl",
                root / ".." / "results" / f"{run_id}_yolo_fast.pkl",
                root / f"{run_id}_tracking.pkl",
                root / f"{stem}_tracking.pkl",
                root / "tracking.pkl",
            ]
        )

        if not yolo_pkl:
            possible_pkls = list(root.glob("*_yolo_fast.pkl")) + list(root.glob("*_tracking.pkl"))
            if not possible_pkls:
                possible_pkls = list(root.glob("results/*_yolo_fast.pkl")) + list(root.glob("results/*_tracking.pkl"))
            if len(possible_pkls) == 1:
                yolo_pkl = possible_pkls[0]
            elif len(possible_pkls) > 1:
                date_part = run_id.split("_")[0]
                matches = [p for p in possible_pkls if p.stem.startswith(date_part)]
                if len(matches) == 1:
                    yolo_pkl = matches[0]

        if yolo_pkl:
            return Predictions.from_yolo_pickle(
                yolo_pkl,
                expected_total_frames=self.video.frame_count if self.video else None,
            )

        results_npy = _pick([root / f"{stem}_results.npy", root / "results.npy"])
        if results_npy:
            return Predictions.from_results_npy(
                results_npy,
                expected_total_frames=self.video.frame_count if self.video else None,
            )

        npz_file = _pick(
            [root / f"{stem}.npz", root / f"{stem}_preds.npz",
             root / "predictions.npz", root / "preds.npz"]
        )
        if npz_file:
            return Predictions.from_npz(npz_file)

        bbox = _pick(
            [root / f"{stem}_bboxes.npy", root / f"{stem}_bboxes.csv",
             root / "bboxes.npy", root / "bboxes.csv"]
        )
        if bbox is None:
            raise SystemExit(
                f"Could not find any supported prediction files next to video: {video_path}"
            )

        conf = _pick(
            [root / f"{stem}_confidences.npy", root / f"{stem}_confidences.csv",
             root / "confidences.npy", root / "confidences.csv",
             root / "conf.npy", root / "conf.csv"]
        )
        tids = _pick(
            [root / f"{stem}_track_ids.npy", root / f"{stem}_track_ids.csv",
             root / "track_ids.npy", root / "track_ids.csv",
             root / "ids.npy", root / "ids.csv"]
        )
        meta = _pick([root / f"{stem}_metadata.npz", root / "metadata.npz"])

        if bbox.suffix.lower() == ".npy":
            return Predictions.from_custom_npy_triplet(
                bboxes_npy=bbox,
                confidences_npy=conf if conf and conf.suffix.lower() == ".npy" else None,
                track_ids_npy=tids if tids and tids.suffix.lower() == ".npy" else None,
                metadata_npz=meta if meta else None,
                bbox_format="auto",
            )

        if bbox.suffix.lower() == ".csv":
            return Predictions.from_custom_csv_triplet(
                bboxes_csv=bbox,
                confidences_csv=conf if conf and conf.suffix.lower() == ".csv" else None,
                track_ids_csv=tids if tids and tids.suffix.lower() == ".csv" else None,
                metadata_npz=meta if meta else None,
                expected_total_frames=self.video.frame_count if self.video is not None else None,
                xywh_is_center=True,
            )

        raise SystemExit(f"Unsupported bbox file type: {bbox}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hsep() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


def run_viewer(
    video_path: Optional[str] = None,
    preds: Optional[Predictions] = None,
    config: Optional[ViewerConfig] = None,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = TrackVizWindow(video_path=video_path, preds=preds, config=config)
    win.resize(1200, 800)
    win.show()
    app.exec()
