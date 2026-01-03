from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from trackviz.io.predictions import Predictions
from trackviz.render.overlay import OverlayStyle, draw_overlays


def _bgr_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    bytes_per_line = 3 * w
    return QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


class VideoReader:
    """Thin wrapper around cv2.VideoCapture with frame-count inference."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read_frame(self, idx: int) -> Optional[np.ndarray]:
        idx = int(idx)
        if idx < 0:
            return None
        # random access; can be slow for some codecs
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


@dataclass
class ViewerConfig:
    start_paused: bool = True
    playback_fps: Optional[float] = None  # if None, use video fps
    overlay_style: OverlayStyle = OverlayStyle()


class TrackVizWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        video_path: Optional[str] = None,
        preds: Optional[Predictions] = None,
        config: Optional[ViewerConfig] = None,
    ):
        super().__init__()
        self.setWindowTitle("trackviz")

        self.setAcceptDrops(True)

        self.video: Optional[VideoReader] = None
        self.preds: Optional[Predictions] = None
        self.cfg = config or ViewerConfig()
        self.playback_fps = 30.0
        self.max_frames = 0

        # --- UI ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        v.addWidget(self.image_label, stretch=1)

        controls = QtWidgets.QHBoxLayout()

        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_prev = QtWidgets.QPushButton("◀")
        self.btn_next = QtWidgets.QPushButton("▶")
        controls.addWidget(self.btn_prev)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_next)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        controls.addWidget(self.slider, stretch=1)

        self.lbl_frame = QtWidgets.QLabel("0")
        controls.addWidget(self.lbl_frame)
        v.addLayout(controls)

        self.chk_overlay = QtWidgets.QCheckBox("Overlay")
        self.chk_overlay.setChecked(True)
        v.addWidget(self.chk_overlay)

        # --- state ---
        self._playing = False
        self._current_frame = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        # --- signals ---
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(+1))
        self.slider.valueChanged.connect(self.set_frame)

        # init: show blank until a video is loaded
        self._show_blank()
        if video_path is not None and preds is not None:
            self.load_video_and_predictions(video_path, preds)
            if not self.cfg.start_paused:
                self.play()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.video is not None:
            self.video.release()
        super().closeEvent(event)

    # --- drag & drop ---
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
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
                QtWidgets.QMessageBox.critical(self, "trackviz", str(e))
                event.ignore()
                return
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "trackviz", f"Failed to load video/predictions:\n{e}")
                event.ignore()
                return

            event.acceptProposedAction()
            return
        event.ignore()

    def _show_blank(self) -> None:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        qimg = _bgr_to_qimage(blank)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def load_video_and_predictions(self, video_path: str, preds: Predictions) -> None:
        self.pause()

        if self.video is not None:
            self.video.release()
        self.video = VideoReader(video_path)
        self.preds = preds
        self.playback_fps = self.cfg.playback_fps or self.video.fps

        self.max_frames = min(self.video.frame_count or self.preds.total_frames, self.preds.total_frames)
        if self.max_frames <= 0:
            self.max_frames = self.preds.total_frames

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self.max_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.set_frame(0)

    def load_from_video_path(self, video_path: Path) -> None:
        # Open the video first (needed for fps/frame_count and for CSV expected_total_frames)
        self.pause()
        if self.video is not None:
            self.video.release()
        self.video = VideoReader(str(video_path))
        self.playback_fps = self.cfg.playback_fps or self.video.fps

        preds = self._autoload_predictions_for_video(video_path)

        # Finish wiring up state/UI
        self.preds = preds
        self.max_frames = min(self.video.frame_count or self.preds.total_frames, self.preds.total_frames)
        if self.max_frames <= 0:
            self.max_frames = self.preds.total_frames

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self.max_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.set_frame(0)

    def _autoload_predictions_for_video(self, video_path: Path) -> Predictions:
        """Auto-load prediction files next to a video.

        Expected naming (preferred):
          <video_stem>_bboxes.(npy|csv)
          <video_stem>_confidences.(npy|csv)   (optional)
          <video_stem>_track_ids.(npy|csv)     (optional)
          <video_stem>_metadata.npz            (optional)

        Also accepts generic fallback names in the same directory:
          bboxes.(npy|csv), confidences.(npy|csv), track_ids.(npy|csv), metadata.npz
        """
        root = video_path.parent
        stem = video_path.stem

        def _pick(cands: list[Path]) -> Optional[Path]:
            for c in cands:
                if c.exists():
                    return c
            return None

        bbox = _pick(
            [
                root / f"{stem}_bboxes.npy",
                root / f"{stem}_bboxes.csv",
                root / "bboxes.npy",
                root / "bboxes.csv",
            ]
        )
        if bbox is None:
            raise SystemExit(f"Could not find bboxes next to video: {video_path}")

        conf = _pick(
            [
                root / f"{stem}_confidences.npy",
                root / f"{stem}_confidences.csv",
                root / "confidences.npy",
                root / "confidences.csv",
                root / "conf.npy",
                root / "conf.csv",
            ]
        )
        tids = _pick(
            [
                root / f"{stem}_track_ids.npy",
                root / f"{stem}_track_ids.csv",
                root / "track_ids.npy",
                root / "track_ids.csv",
                root / "ids.npy",
                root / "ids.csv",
            ]
        )
        meta = _pick(
            [
                root / f"{stem}_metadata.npz",
                root / "metadata.npz",
            ]
        )

        if bbox.suffix.lower() == ".npy":
            return Predictions.from_custom_npy_triplet(
                bboxes_npy=bbox,
                confidences_npy=conf if conf and conf.suffix.lower() == ".npy" else None,
                track_ids_npy=tids if tids and tids.suffix.lower() == ".npy" else None,
                metadata_npz=meta if meta else None,
                bbox_format="auto",
            )

        if bbox.suffix.lower() == ".csv":
            # For YOLO-style exports, x/y are typically center coordinates.
            return Predictions.from_custom_csv_triplet(
                bboxes_csv=bbox,
                confidences_csv=conf if conf and conf.suffix.lower() == ".csv" else None,
                track_ids_csv=tids if tids and tids.suffix.lower() == ".csv" else None,
                metadata_npz=meta if meta else None,
                expected_total_frames=self.video.frame_count if self.video is not None else None,
                xywh_is_center=True,
            )

        raise SystemExit(f"Unsupported bbox file type: {bbox}")

    # --- controls ---
    def play(self) -> None:
        if self._playing:
            return
        self._playing = True
        interval_ms = int(round(1000.0 / max(1e-3, self.playback_fps)))
        self._timer.start(interval_ms)

    def pause(self) -> None:
        self._playing = False
        self._timer.stop()

    def step(self, delta: int) -> None:
        self.pause()
        self.set_frame(self._current_frame + int(delta))

    def set_frame(self, idx: int) -> None:
        idx = int(idx)
        if self.video is None or self.preds is None or self.max_frames <= 0:
            self._current_frame = 0
            self.lbl_frame.setText("-")
            return

        idx = max(0, min(self.max_frames - 1, idx))
        if idx == self._current_frame and self.image_label.pixmap() is not None:
            return
        self._current_frame = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(str(idx))

        frame = self.video.read_frame(idx)
        if frame is None:
            # graceful fallback: show blank
            self._show_blank()
            return

        if self.chk_overlay.isChecked():
            dets = self.preds.for_frame(idx)
            frame = draw_overlays(frame, dets, self.cfg.overlay_style)

        qimg = _bgr_to_qimage(frame)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)

    def _on_tick(self) -> None:
        if self.video is None or self.preds is None:
            self.pause()
            return
        nxt = self._current_frame + 1
        if nxt >= self.max_frames:
            self.pause()
            return
        self.set_frame(nxt)


def run_viewer(video_path: Optional[str] = None, preds: Optional[Predictions] = None, config: Optional[ViewerConfig] = None) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = TrackVizWindow(video_path=video_path, preds=preds, config=config)
    win.resize(1100, 800)
    win.show()
    app.exec()
