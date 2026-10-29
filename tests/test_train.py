import pytest

from trackviz.train import PRESET, TrainError, check_dataset


def _ds(root, train="train/images", val="val/images", images=True):
    root.mkdir()
    (root / "data.yaml").write_text(f"train: {train}\nval: {val}\nnc: 1\nnames:\n  0: 'Grooming'\n")
    for split in {train, val}:
        (root / split).mkdir(parents=True)
        if images:
            (root / split / "a.jpg").write_bytes(b"")
    return root


def test_check_dataset_accepts_split_export(tmp_path):
    data = check_dataset(_ds(tmp_path / "ds"))
    assert data["names"] == {0: "Grooming"}
    assert data["_yaml"] == tmp_path / "ds" / "data.yaml"


def test_check_dataset_rejects_unsplit_and_empty(tmp_path):
    with pytest.raises(TrainError, match="same folder"):
        check_dataset(_ds(tmp_path / "flat", train="images", val="images"))
    with pytest.raises(TrainError, match="No images"):
        check_dataset(_ds(tmp_path / "empty", images=False))
    with pytest.raises(TrainError, match="No data.yaml"):
        check_dataset(tmp_path / "missing")


def test_preset_preserves_heatmap_colors():
    assert PRESET["hsv_h"] == 0.0 and PRESET["hsv_s"] == 0.0
    assert PRESET["imgsz"] == 800 and PRESET["rect"] is True
