from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

from trackviz.io.predictions import Predictions
from trackviz.render.export import export_video
from trackviz.render.overlay import OverlayStyle


class _ExportWorker(QtCore.QObject):
    """Runs ``export_video`` on a worker thread, bridging progress back via signals."""

    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, kwargs: dict) -> None:
        super().__init__()
        self._kwargs = kwargs

    @QtCore.Slot()
    def run(self) -> None:
        try:
            export_video(
                on_progress=lambda d, t: self.progress.emit(int(d), int(t)),
                **self._kwargs,
            )
            self.finished.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


_DIALOG_STYLESHEET = """
QLineEdit {
    background-color: #2a2b48;
    border: 1px solid #454772;
    border-radius: 6px;
    padding: 4px 10px;
    color: #dde1f0;
    selection-background-color: #3d5fcc;
    min-height: 26px;
}
QLineEdit:focus    { border-color: #6870b0; }
QLineEdit:disabled { color: #555878; border-color: #2e3052; }

QSpinBox, QDoubleSpinBox {
    background-color: #2a2b48;
    border: 1px solid #454772;
    border-radius: 6px;
    padding: 4px 8px;
    color: #dde1f0;
    min-height: 26px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #6870b0; }

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    background-color: #383b68;
    border: none;
    width: 18px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button       { subcontrol-position: top right;    border-top-right-radius: 5px; }
QSpinBox::down-button, QDoubleSpinBox::down-button   { subcontrol-position: bottom right; border-bottom-right-radius: 5px; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background-color: #4a4e85; }

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #dde1f0;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #dde1f0;
}
"""


class ExportDialog(QtWidgets.QDialog):
    """Modal dialog wrapping :func:`trackviz.render.export.export_video`."""

    def __init__(
        self,
        video_path: Path,
        preds: Predictions,
        annotations: dict,
        max_frames: int,
        style: OverlayStyle,
        fps: Optional[float] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Video")
        self.setModal(True)
        self.setMinimumWidth(540)
        self.setStyleSheet(_DIALOG_STYLESHEET)

        self._video_path = video_path
        self._preds = preds
        self._annotations = annotations
        self._style = style
        self._fps = float(fps) if fps else None
        self._max_frames = int(max_frames)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_ExportWorker] = None
        # Flipped to True once the user edits the path manually or picks one
        # via Browse — after which we stop auto-regenerating the default name.
        self._user_edited_output = False

        last = max(0, int(max_frames) - 1)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        # ── Output file ─────────────────────────────────────────────
        self.edit_output = QtWidgets.QLineEdit()
        # textEdited fires only when the *user* types; programmatic setText
        # does not — that's what lets us auto-update the default name
        # until the user takes control.
        self.edit_output.textEdited.connect(self._on_output_edited)
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self._choose_output)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.edit_output, stretch=1)
        out_row.addWidget(btn_browse)
        form.addRow("Output file", out_row)

        # ── Quality ─────────────────────────────────────────────────
        self.slider_quality = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_quality.setRange(1, 10)
        self.slider_quality.setValue(8)
        self.slider_quality.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_quality.setTickInterval(1)
        self.lbl_quality = QtWidgets.QLabel("8")
        self.lbl_quality.setMinimumWidth(24)
        self.lbl_quality.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.slider_quality.valueChanged.connect(
            lambda v: self.lbl_quality.setText(str(v))
        )
        q_row = QtWidgets.QHBoxLayout()
        q_row.addWidget(self.slider_quality, stretch=1)
        q_row.addWidget(self.lbl_quality)
        form.addRow("Quality (1 low – 10 high)", q_row)

        # ── Scale ───────────────────────────────────────────────────
        self.spin_scale = QtWidgets.QDoubleSpinBox()
        self.spin_scale.setRange(0.1, 1.0)
        self.spin_scale.setSingleStep(0.1)
        self.spin_scale.setDecimals(2)
        self.spin_scale.setValue(1.0)
        form.addRow("Output scale", self.spin_scale)

        # ── Frame range ─────────────────────────────────────────────
        self.spin_start = QtWidgets.QSpinBox()
        self.spin_start.setRange(0, last)
        self.spin_start.setValue(0)
        self.spin_end = QtWidgets.QSpinBox()
        self.spin_end.setRange(0, last)
        self.spin_end.setValue(last)
        # Use editingFinished (not valueChanged) so intermediate keystrokes
        # while typing a multi-digit number don't snap the "to" box back up
        # to the current "from" value on every digit entered.
        self.spin_start.editingFinished.connect(self._sync_range)
        self.spin_end.editingFinished.connect(self._sync_range)
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(self.spin_start)
        range_row.addWidget(QtWidgets.QLabel("to"))
        range_row.addWidget(self.spin_end)
        range_row.addStretch()
        form.addRow("Frame range", range_row)

        # ── Overlays ────────────────────────────────────────────────
        self.chk_predictions = QtWidgets.QCheckBox("Include predictions (green)")
        self.chk_predictions.setChecked(True)
        self.chk_annotations = QtWidgets.QCheckBox("Include annotations (cyan)")
        self.chk_annotations.setChecked(True)
        self.chk_heatmap = QtWidgets.QCheckBox("Render as motion heatmap")
        self.chk_heatmap.setChecked(False)
        form.addRow("Overlays", self.chk_predictions)
        form.addRow("", self.chk_annotations)
        form.addRow("", self.chk_heatmap)

        # ── Progress ────────────────────────────────────────────────
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.lbl_progress = QtWidgets.QLabel("")
        self.lbl_progress.setVisible(False)
        self.lbl_progress.setStyleSheet("color: #8890cc; font-size: 12px;")

        # ── Buttons ─────────────────────────────────────────────────
        self.btn_start = QtWidgets.QPushButton("Start Export")
        self.btn_start.setDefault(True)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_start.clicked.connect(self._start_export)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_start)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)
        layout.addLayout(form)
        layout.addSpacing(4)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_progress)
        layout.addLayout(btn_row)

        # Regenerate the default filename whenever a setting that's encoded
        # in it changes — unless the user has taken control of the path.
        self.slider_quality.valueChanged.connect(self._maybe_refresh_filename)
        self.spin_scale.valueChanged.connect(self._maybe_refresh_filename)
        self.spin_start.editingFinished.connect(self._maybe_refresh_filename)
        self.spin_end.editingFinished.connect(self._maybe_refresh_filename)
        self.chk_predictions.toggled.connect(self._maybe_refresh_filename)
        self.chk_annotations.toggled.connect(self._maybe_refresh_filename)
        self.chk_heatmap.toggled.connect(self._maybe_refresh_filename)

        # Prime the line edit with the initial default.
        self.edit_output.setText(self._build_default_filename())

    # ------------------------------------------------------------------

    def _sync_range(self) -> None:
        if self.spin_end.value() < self.spin_start.value():
            self.spin_end.setValue(self.spin_start.value())

    def _on_output_edited(self, _text: str) -> None:
        self._user_edited_output = True

    def _maybe_refresh_filename(self) -> None:
        if self._user_edited_output:
            return
        self.edit_output.blockSignals(True)
        self.edit_output.setText(self._build_default_filename())
        self.edit_output.blockSignals(False)

    def _build_default_filename(self) -> str:
        """Encode the current settings into the default output path.

        Format: ``{stem}_f{start}-{end}_q{quality}_s{scale_pct}[_flags].mp4``
        where *flags* is the concatenation of ``h`` (heatmap), ``p``
        (predictions), ``a`` (annotations) — omitted entirely if all are off.
        """
        stem = self._video_path.stem
        start = int(self.spin_start.value())
        end = int(self.spin_end.value())
        quality = int(self.slider_quality.value())
        scale_pct = int(round(self.spin_scale.value() * 100))
        flags = ""
        if self.chk_heatmap.isChecked():
            flags += "h"
        if self.chk_predictions.isChecked():
            flags += "p"
        if self.chk_annotations.isChecked():
            flags += "a"
        parts = [stem, f"f{start}-{end}", f"q{quality}", f"s{scale_pct}"]
        if flags:
            parts.append(flags)
        name = "_".join(parts) + ".mp4"
        return str(self._video_path.with_name(name))

    def _choose_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Video",
            self.edit_output.text(),
            "MP4 (*.mp4);;AVI (*.avi);;MOV (*.mov)",
        )
        if path:
            self.edit_output.setText(path)
            self._user_edited_output = True
            # Re-activate the dialog and move focus back to the line edit so
            # the field visibly reflects the new path and is ready to edit.
            self.activateWindow()
            self.raise_()
            self.edit_output.setFocus()
            self.edit_output.setCursorPosition(len(path))

    # ------------------------------------------------------------------

    _LONG_EXPORT_MINUTES = 20

    def _confirm_long_export(self) -> bool:
        """If the user selected the full range of a video >20 min, confirm.

        Returns True if the export should proceed, False if the user backed
        out. Without an fps we can't know duration, so we proceed silently.
        """
        if not self._fps or self._max_frames <= 0:
            return True
        last = self._max_frames - 1
        full_range = (
            int(self.spin_start.value()) == 0 and int(self.spin_end.value()) == last
        )
        if not full_range:
            return True
        duration_min = self._max_frames / self._fps / 60.0
        if duration_min <= self._LONG_EXPORT_MINUTES:
            return True
        ret = QtWidgets.QMessageBox.question(
            self,
            "Long Export",
            (
                f"You're about to export the full video — roughly "
                f"{duration_min:.1f} minutes ({self._max_frames} frames).\n\n"
                "This may take a while. Continue?"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return ret == QtWidgets.QMessageBox.Yes

    def _start_export(self) -> None:
        out_text = self.edit_output.text().strip()
        if not out_text:
            QtWidgets.QMessageBox.warning(
                self, "Export Video", "Please choose an output file."
            )
            return
        out_path = Path(out_text).expanduser()
        if out_path.exists():
            ret = QtWidgets.QMessageBox.question(
                self,
                "Overwrite?",
                f"{out_path.name} already exists. Overwrite?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                return

        if not self._confirm_long_export():
            return

        kwargs = dict(
            video_path=self._video_path,
            preds=self._preds,
            output_path=out_path,
            start_frame=int(self.spin_start.value()),
            end_frame=int(self.spin_end.value()),
            style=self._style,
            annotations=self._annotations,
            quality=int(self.slider_quality.value()),
            scale=float(self.spin_scale.value()),
            include_predictions=self.chk_predictions.isChecked(),
            include_annotations=self.chk_annotations.isChecked(),
            include_heatmap=self.chk_heatmap.isChecked(),
        )

        # Lock inputs during export
        for w in (
            self.edit_output,
            self.slider_quality,
            self.spin_scale,
            self.spin_start,
            self.spin_end,
            self.chk_predictions,
            self.chk_annotations,
            self.chk_heatmap,
            self.btn_start,
            self.btn_cancel,
        ):
            w.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_progress.setVisible(True)
        self.lbl_progress.setText("Starting…")

        self._thread = QtCore.QThread(self)
        self._worker = _ExportWorker(kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    @QtCore.Slot(int, int)
    def _on_progress(self, done: int, total: int) -> None:
        pct = int(100 * done / total) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.lbl_progress.setText(f"Encoding frame {done} / {total}  ({pct}%)")

    @QtCore.Slot()
    def _on_finished(self) -> None:
        self.progress_bar.setValue(100)
        self.lbl_progress.setText("Done.")
        QtWidgets.QMessageBox.information(
            self,
            "Export Video",
            f"Exported to:\n{self.edit_output.text()}",
        )
        self.accept()

    @QtCore.Slot(str)
    def _on_failed(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Export Failed", msg)
        self.reject()

    # ------------------------------------------------------------------

    def _export_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def reject(self) -> None:
        if self._export_running():
            QtWidgets.QMessageBox.information(
                self,
                "Export Video",
                "Export is in progress — please wait for it to finish.",
            )
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._export_running():
            QtWidgets.QMessageBox.information(
                self,
                "Export Video",
                "Export is in progress — please wait for it to finish.",
            )
            event.ignore()
            return
        super().closeEvent(event)
