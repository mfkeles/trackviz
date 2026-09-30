import json

import pytest

from trackviz.io.labels import Label, LabelFile, LabelFileError, read_labels, write_labels
from trackviz.io.project import project_from_dict


def _project(*names):
    return project_from_dict({"name": "flies", "classes": list(names)})


def test_missing_file_reads_empty(tmp_path):
    assert read_labels(tmp_path / "nope.json").labels == {}


def test_round_trip_stores_keys_not_positions(tmp_path):
    path = tmp_path / "v_flies_labels.json"
    project = _project("Grooming", "Regurgitation")
    lf = LabelFile(labels={12: Label("regurgitation", [1, 2, 30.456, 40])})
    write_labels(path, lf, project, video_name="v.mp4")

    raw = json.loads(path.read_text())
    assert raw["labels"] == {"12": {"class": "regurgitation", "bbox": [1.0, 2.0, 30.46, 40.0]}}
    assert raw["classes"] == {"regurgitation": "Regurgitation"}
    assert raw["video"] == "v.mp4"

    # Reordering the project doesn't change what the label means.
    reordered = _project("Regurgitation", "Grooming")
    back = read_labels(path)
    assert back.labels[12].cls == "regurgitation"
    assert back.unknown_classes(reordered) == {}


def test_removed_class_is_reported_and_reassigned(tmp_path):
    path = tmp_path / "labels.json"
    write_labels(path, LabelFile(labels={
        1: Label("grooming", [0, 0, 5, 5]),
        2: Label("feeding", [0, 0, 5, 5]),
        3: Label("feeding", [0, 0, 5, 5]),
    }), _project("Grooming", "Feeding"))

    smaller = _project("Grooming", "Eating")
    lf = read_labels(path)
    assert lf.unknown_classes(smaller) == {"feeding": 2}
    assert lf.class_names["feeding"] == "Feeding"   # old name survives for the prompt

    lf.reassign({"feeding": "eating"})
    assert lf.unknown_classes(smaller) == {}
    assert [lab.cls for _, lab in sorted(lf.labels.items())] == ["grooming", "eating", "eating"]


@pytest.mark.parametrize("content, msg", [
    ("not json", "Could not read"),
    ('{"labels": {}}', "not a trackviz label file"),
    ('{"format": "trackviz-labels", "version": 99}', "newer trackviz"),
    ('{"format": "trackviz-labels", "version": 1, "labels": {"3": {"class": "a", "bbox": [1, 2]}}}',
     "bad label"),
])
def test_bad_files_raise(tmp_path, content, msg):
    path = tmp_path / "labels.json"
    path.write_text(content)
    with pytest.raises(LabelFileError, match=msg):
        read_labels(path)
