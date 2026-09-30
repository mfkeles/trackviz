from pathlib import Path

import pytest

from trackviz.io.project import (
    DEFAULT_COLORS,
    ProjectError,
    load_project,
    project_from_dict,
    resolve_project_path,
    write_project_template,
)


def test_minimal_project_fills_defaults():
    p = project_from_dict({"name": "Mouse Study", "classes": ["Rearing", "Grooming", "Freezing"]})
    assert p.name == "Mouse Study"
    assert p.keys == ["rearing", "grooming", "freezing"]
    assert [c.hotkey for c in p.classes] == [0, 1, 2]
    assert [c.color for c in p.classes] == DEFAULT_COLORS[:3]
    assert all(c.model_class is None for c in p.classes)
    assert p.export_heatmap is True
    assert p.labels_path(Path("/data/m1.mp4")) == Path("/data/m1_mouse_study_labels.json")


def test_explicit_values_are_respected_and_defaults_avoid_them():
    p = project_from_dict({
        "name": "flies",
        "classes": [
            {"name": "Regurgitation"},
            {"name": "Grooming", "key": "groom", "color": "#e69f00", "hotkey": 0, "model_class": 2},
            {"name": "Feeding", "hotkey": None},
        ],
        "export": {"heatmap": False},
    })
    regurg, groom, feed = p.classes
    assert groom.key == "groom" and groom.color == "#E69F00" and groom.hotkey == 0
    assert regurg.hotkey == 1            # 0 is taken explicitly
    assert regurg.color != "#E69F00"     # palette skips the explicit color
    assert feed.hotkey is None           # explicit null → no hotkey
    assert p.index_for_hotkey(0) == 1
    assert p.index_for_model_class(2) == 1
    assert p.index_for_model_class(5) is None
    assert p.export_heatmap is False
    assert groom.bgr == (0x00, 0x9F, 0xE6)


@pytest.mark.parametrize("data, msg", [
    ({"classes": ["A"]}, "'name'"),
    ({"name": "x", "classes": []}, "'classes'"),
    ({"name": "x", "classes": ["A", "a"]}, "same key"),
    ({"name": "x", "classes": [{"name": "A", "hotkey": 1}, {"name": "B", "hotkey": 1}]}, "same hotkey"),
    ({"name": "x", "classes": [{"name": "A", "hotkey": 12}]}, "0-9"),
    ({"name": "x", "classes": [{"name": "A", "color": "red"}]}, "#RRGGBB"),
    ({"name": "x", "classes": [{"name": "A", "model_class": -1}]}, "model_class"),
    ({"name": "x", "classes": [{"name": "A", "model_class": 1}, {"name": "B", "model_class": 1}]},
     "same model_class"),
    ({"name": "x", "classes": [{"name": "A", "colour": "#000000"}]}, "unknown field"),
    ({"name": "x", "classes": [{"name": "A", "key": "Has Space"}]}, "'key'"),
])
def test_invalid_projects_raise_clear_errors(data, msg):
    with pytest.raises(ProjectError, match=msg):
        project_from_dict(data)


def test_template_round_trips(tmp_path):
    path = write_project_template(tmp_path / "regurg")
    assert path.suffix == ".yaml"
    p = load_project(path)
    assert p.name == "regurg"
    assert p.names == ["Behavior A", "Behavior B"]
    with pytest.raises(ProjectError, match="already exists"):
        write_project_template(path)


def test_bare_name_resolves_in_user_projects_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    proj_dir = tmp_path / "trackviz" / "projects"
    proj_dir.mkdir(parents=True)
    (proj_dir / "flies.yaml").write_text("name: flies\nclasses: [Grooming]\n")
    assert resolve_project_path("flies") == proj_dir / "flies.yaml"
    with pytest.raises(ProjectError, match="no project named"):
        resolve_project_path("missing")
