import csv

import cv2
import numpy as np
import pytest

from trackviz.io.labels import Label, LabelFile, write_labels
from trackviz.io.project import project_from_dict
from trackviz.render.dataset import (
    ExistingDataset,
    export_dataset,
    find_labeled_videos,
    motion_heatmap,
    yolo_line,
)

W, H = 64, 48


def _project(extra=()):
    return project_from_dict({"name": "flies", "classes": ["Grooming", "Feeding", *extra]})


def _video(path, n=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
    for i in range(n):
        frame = np.full((H, W, 3), 40, np.uint8)
        cv2.rectangle(frame, (2 + 2 * i, 10), (12 + 2 * i, 20), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _labeled(tmp_path, project, labels, name="fly01"):
    video = _video(tmp_path / f"{name}.avi")
    write_labels(project.labels_path(video), LabelFile(labels=labels), project)
    return video


def _dataset(root, entries, names=("Grooming", "Feeding")):
    """entries: (split, image stem) pairs."""
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text(
        "nc: %d\nnames:\n" % len(names) + "".join(f"  {i}: '{n}'\n" for i, n in enumerate(names))
    )
    for split, stem in entries:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "images" / f"{stem}.jpg").write_bytes(b"")
    return ExistingDataset.load(root)


def test_yolo_line_matches_flyolo_format():
    assert yolo_line([10, 20, 30, 60], 3, 100, 100) == "3 0.200000 0.400000 0.200000 0.400000\n"
    assert yolo_line([10, 20, 10, 60], 0, 100, 100) is None      # zero width
    assert yolo_line([150, 20, 170, 60], 0, 100, 100) is None    # center outside


def test_heatmap_marks_past_positions_on_current_frame():
    frames = []
    for x in range(6):
        f = np.zeros((10, 20, 3), np.uint8)
        f[4:6, 2 * x:2 * x + 2] = 200
        frames.append(f)
    heat = motion_heatmap(frames)
    assert heat.shape == frames[-1].shape
    assert heat[5, 0].sum() > 0          # oldest position lights up
    assert heat[5, 19].sum() == 0        # untouched background stays black


def test_flat_export_writes_images_labels_yaml_manifest(tmp_path):
    project = _project()
    video = _labeled(tmp_path, project, {
        0: Label("grooming", [1, 1, 5, 5]),      # no earlier frames → skipped for heatmaps
        8: Label("feeding", [16, 10, 28, 20]),
        12: Label("grooming", [20, 10, 32, 20]),
    })
    videos, problems = find_labeled_videos(project, [tmp_path])
    assert problems == [] and [lv.video for lv in videos] == [video]

    out = tmp_path / "out"
    res = export_dataset(project, videos, out)
    assert res.total_exported == 2
    assert res.skipped == {"skipped: no earlier frames for heatmap": 1}
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "fly01_Feeding_8_heatmap.jpg", "fly01_Grooming_12_heatmap.jpg"]
    assert (out / "labels" / "fly01_Feeding_8_heatmap.txt").read_text().startswith("1 ")
    assert cv2.imread(str(out / "images" / "fly01_Feeding_8_heatmap.jpg")).shape == (H, W, 3)
    assert "  1: 'Feeding'" in (out / "data.yaml").read_text()
    rows = list(csv.DictReader(open(out / "manifest.csv")))
    assert [r["status"] for r in rows] == [
        "skipped: no earlier frames for heatmap", "exported", "exported"]

    raw = export_dataset(project, videos, tmp_path / "raw", heatmap=False)
    assert raw.total_exported == 3
    assert (tmp_path / "raw" / "images" / "fly01_Grooming_0.jpg").exists()


def test_class_filter_and_unknown_class(tmp_path):
    project = _project(["Regurgitation"])
    _labeled(tmp_path, project, {5: Label("grooming", [1, 1, 5, 5]),
                                 6: Label("regurgitation", [1, 1, 5, 5])})
    videos, _ = find_labeled_videos(project, [tmp_path])
    res = export_dataset(project, videos, tmp_path / "out", classes=["Regurgitation"])
    assert res.exported == {("", "Regurgitation"): 1}
    assert res.skipped == {"skipped: class not selected": 1}
    with pytest.raises(ValueError, match="Unknown class 'Nope'"):
        export_dataset(project, videos, tmp_path / "out2", classes=["Nope"])


def test_match_dataset_inherits_splits_and_skips_existing(tmp_path):
    project = _project(["Regurgitation"])
    labels = {
        5: Label("grooming", [1, 1, 5, 5]),        # already in dataset, same class
        6: Label("regurgitation", [1, 1, 5, 5]),   # in dataset as Feeding → conflict
        7: Label("regurgitation", [1, 1, 5, 5]),   # new
    }
    _labeled(tmp_path, project, labels, name="fly01")
    _labeled(tmp_path, project, {9: Label("regurgitation", [1, 1, 5, 5])}, name="fly02")
    ds = _dataset(tmp_path / "ds", [
        ("val", "fly01_Grooming_5_heatmap"),
        ("val", "fly01_Feeding_6_heatmap"),
        ("train", "other_Grooming_3_heatmap"),
    ])
    assert ds.video_splits == {"fly01": "val", "other": "train"}

    videos, _ = find_labeled_videos(project, [tmp_path])
    out = tmp_path / "out"
    res = export_dataset(project, videos, out, match_dataset=ds)
    assert res.exported == {("val", "Regurgitation"): 1, ("unassigned", "Regurgitation"): 1}
    assert res.skipped == {"skipped: already in dataset": 1, "skipped: in dataset as Feeding": 1}
    assert (out / "val" / "labels" / "fly01_Regurgitation_7_heatmap.txt").read_text().startswith("2 ")
    assert (out / "unassigned" / "images" / "fly02_Regurgitation_9_heatmap.jpg").exists()
    assert "train: train/images" in (out / "data.yaml").read_text()
    # The existing dataset is untouched.
    assert sorted(p.name for p in (tmp_path / "ds" / "val" / "images").iterdir()) == [
        "fly01_Feeding_6_heatmap.jpg", "fly01_Grooming_5_heatmap.jpg"]


def test_match_dataset_safety_checks(tmp_path):
    project = _project()
    _labeled(tmp_path, project, {5: Label("grooming", [1, 1, 5, 5])})
    videos, _ = find_labeled_videos(project, [tmp_path])

    swapped = _dataset(tmp_path / "swapped", [("train", "x_Feeding_1_heatmap")],
                       names=("Feeding", "Grooming"))
    with pytest.raises(ValueError, match="don't line up"):
        export_dataset(project, videos, tmp_path / "o1", match_dataset=swapped)

    ds = _dataset(tmp_path / "ds", [("train", "x_Grooming_1_heatmap")])
    with pytest.raises(ValueError, match="heatmap dataset"):
        export_dataset(project, videos, tmp_path / "o2", match_dataset=ds, heatmap=False)
    with pytest.raises(ValueError, match="never modified"):
        export_dataset(project, videos, tmp_path / "ds", match_dataset=ds)

    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("")
    with pytest.raises(ValueError, match="not empty"):
        export_dataset(project, videos, tmp_path / "full")


def test_video_in_two_splits_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="more than one split"):
        _dataset(tmp_path / "ds", [("train", "v_Grooming_1_heatmap"), ("val", "v_Grooming_2_heatmap")])
