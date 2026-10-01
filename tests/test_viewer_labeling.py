"""Offscreen GUI tests for labeling mode (no display needed)."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6 import QtWidgets

from trackviz.gui.viewer import TrackVizWindow
from trackviz.io.class_names import BEHAVIOR_NAMES
from trackviz.io.predictions import Predictions
from trackviz.io.project import project_from_dict


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """Message boxes would block an offscreen test; record them instead."""
    shown = []
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            staticmethod(lambda *a, _n=name, **k: shown.append((_n, a[1:]))))

    def question(*a, **k):
        shown.append(("question", a[1:]))
        return QtWidgets.QMessageBox.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(question))
    return shown


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "mouse01.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


def _project(classes, name="mice"):
    return project_from_dict({"name": name, "classes": classes})


def _window(app, project, video):
    win = TrackVizWindow(project=project)
    win.load_from_video_path(video)
    return win


def _label(win, frame, cls_idx, bbox):
    win.set_frame(frame)
    win.combo_classes.setCurrentIndex(cls_idx)
    win.image_label._pending_bbox = list(bbox)
    win.image_label._pending_kind = "new"
    win._save_annotation()


def test_default_mode_keeps_builtin_classes(app):
    win = TrackVizWindow()
    assert win.project is None
    assert win.combo_classes.count() == len(BEHAVIOR_NAMES)
    assert win.combo_classes.itemText(0) == f"0: {BEHAVIOR_NAMES[0]}"


def test_labeling_mode_saves_by_key_and_requires_box(app, video):
    project = _project(["Rearing", "Grooming", {"name": "Freezing", "hotkey": 9}])
    win = _window(app, project, video)
    assert [win.combo_classes.itemText(i) for i in range(3)] == [
        "0: Rearing", "1: Grooming", "9: Freezing"]
    assert win._class_index_for_digit(9) == 2
    assert win._class_index_for_digit(5) is None

    # No predictions and no drawn box → refuse to save.
    win.set_frame(3)
    win._save_annotation()
    labels_path = video.parent / "mouse01_mice_labels.json"
    assert not labels_path.exists()
    assert win.chk_edit.isChecked()

    _label(win, 3, 2, [5, 5, 20, 30])
    _label(win, 7, 0, [1, 1, 10, 10])
    saved = json.loads(labels_path.read_text())
    assert saved["labels"] == {
        "3": {"class": "freezing", "bbox": [5.0, 5.0, 20.0, 30.0]},
        "7": {"class": "rearing", "bbox": [1.0, 1.0, 10.0, 10.0]},
    }
    # Default-mode file is untouched.
    assert not (video.parent / "mouse01_annotations.json").exists()

    # Re-classing a labeled frame keeps its box without redrawing.
    win.set_frame(3)
    win.combo_classes.setCurrentIndex(1)
    win._save_annotation()
    assert json.loads(labels_path.read_text())["labels"]["3"]["class"] == "grooming"

    # Rendering helpers use the project name and class color.
    anno = win._annotations_for_render()["3"]
    assert anno["name"] == "Grooming" and anno["color"] == project.classes[1].bgr


def test_reordered_project_keeps_meaning(app, video):
    win = _window(app, _project(["Rearing", "Grooming"]), video)
    _label(win, 4, 1, [0, 0, 9, 9])
    win.close()

    win2 = _window(app, _project(["Grooming", "Rearing", "Sniffing"]), video)
    assert win2._annotations["4"]["cls"] == 0   # Grooming, now first


def test_deleted_class_forces_reassignment(app, video, monkeypatch):
    win = _window(app, _project(["Rearing", "Grooming"]), video)
    _label(win, 2, 1, [0, 0, 9, 9])
    win.close()
    labels_path = video.parent / "mouse01_mice_labels.json"

    # Cancelling the reassignment refuses to open the video in this project.
    monkeypatch.setattr(TrackVizWindow, "_ask_class_reassignment", lambda self, u, n: None)
    win2 = _window(app, _project(["Rearing", "Sniffing"]), video)
    assert win2.video is None
    assert json.loads(labels_path.read_text())["labels"]["2"]["class"] == "grooming"

    seen = {}

    def choose_sniffing(self, unknown, old_names):
        seen.update(unknown=unknown, old_names=old_names)
        return {"grooming": "sniffing"}

    monkeypatch.setattr(TrackVizWindow, "_ask_class_reassignment", choose_sniffing)
    win3 = _window(app, _project(["Rearing", "Sniffing"]), video)
    assert seen["unknown"] == {"grooming": 1}
    assert seen["old_names"]["grooming"] == "Grooming"
    assert win3._annotations["2"]["cls"] == 1
    assert json.loads(labels_path.read_text())["labels"]["2"]["class"] == "sniffing"
    assert labels_path.with_name(labels_path.name + ".bak").exists()


def test_model_class_preselects_linked_class(app, video):
    project = _project([{"name": "Regurgitation"}, {"name": "Grooming", "model_class": 2}], name="flies")
    preds = Predictions(
        total_frames=20,
        dense_bbox_xyxy=np.tile([2.0, 2.0, 20.0, 20.0], (20, 1)),
        dense_cls=np.full(20, 2.0),
    )
    win = TrackVizWindow(project=project)
    win.load_video_and_predictions(str(video), preds)
    win.combo_classes.setCurrentIndex(0)
    win.set_frame(5)                     # edit mode is on in labeling mode → preselect runs
    assert win.combo_classes.currentIndex() == 1

    # A predicted box counts as the label's box.
    win._save_annotation()
    labels = json.loads((video.parent / "mouse01_flies_labels.json").read_text())["labels"]
    assert labels["5"] == {"class": "grooming", "bbox": [2.0, 2.0, 20.0, 20.0]}


def test_switching_modes_swaps_class_set_and_files(app, video):
    win = _window(app, None, video)
    assert win._annotation_path.name == "mouse01_annotations.json"
    win.set_project(_project(["Rearing"]))
    assert win.combo_classes.count() == 1
    assert win._annotation_path.name == "mouse01_mice_labels.json"
    win.set_project(None)
    assert win.combo_classes.count() == len(BEHAVIOR_NAMES)
    assert win.windowTitle() == "trackviz"


def test_default_mode_file_format_unchanged(app, video):
    win = _window(app, None, video)
    _label(win, 6, 3, [1, 2, 3, 4])
    saved = json.loads((video.parent / "mouse01_annotations.json").read_text())
    assert saved == {"6": {"cls": 3, "corrected": True, "bbox": [1.0, 2.0, 3.0, 4.0]}}


def _write_default_annotations(video):
    path = video.parent / "mouse01_annotations.json"
    path.write_text(json.dumps({
        "3": {"cls": 2, "bbox": [1, 2, 3, 4], "corrected": False},   # Grooming, predicted box
        "5": {"cls": 6, "bbox": [5, 6, 7, 8], "corrected": True},    # Twitching
        "9": 4,                                                     # oldest format, no box
        "11": {"cls": 0, "corrected": False},                       # no box
    }))
    return path


def test_default_annotations_import_on_first_open(app, video, no_modal_dialogs):
    default_path = _write_default_annotations(video)
    original = default_path.read_text()
    project = project_from_dict({"name": "flies", "classes": list(BEHAVIOR_NAMES) + ["Regurgitation"]})
    win = _window(app, project, video)

    assert [d[0] for d in no_modal_dialogs] == ["question", "information"]   # offer, then skipped report
    assert "9, 11" in no_modal_dialogs[1][1][1]
    saved = json.loads((video.parent / "mouse01_flies_labels.json").read_text())["labels"]
    assert saved == {
        "3": {"class": "grooming", "bbox": [1.0, 2.0, 3.0, 4.0]},
        "5": {"class": "twitching", "bbox": [5.0, 6.0, 7.0, 8.0]},
    }
    assert win._annotations["3"]["cls"] == 2
    assert default_path.read_text() == original

    # Once the project has its own label file, there's no second offer.
    no_modal_dialogs.clear()
    _window(app, project, video)
    assert no_modal_dialogs == []


def test_declined_import_starts_empty(app, video, monkeypatch):
    _write_default_annotations(video)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
    win = _window(app, _project(["Grooming"]), video)
    assert win._annotations == {}
    assert not (video.parent / "mouse01_mice_labels.json").exists()


def test_import_unmatched_class_goes_through_reassignment(app, video, monkeypatch):
    _write_default_annotations(video)
    seen = {}

    def choose(self, unknown, old_names):
        seen.update(unknown=unknown, old_names=old_names)
        return {"twitching": "grooming"}

    monkeypatch.setattr(TrackVizWindow, "_ask_class_reassignment", choose)
    win = _window(app, _project(["Grooming"]), video)
    assert seen == {"unknown": {"twitching": 1}, "old_names": {"grooming": "Grooming", "twitching": "Twitching"}}
    assert {k: e["cls"] for k, e in win._annotations.items()} == {"3": 0, "5": 0}
