"""Labeling projects: user-defined behavior classes loaded from a YAML file.

A project file looks like::

    name: fly_regurgitation
    classes:
      - name: Regurgitation          # key/color/hotkey are optional
      - name: Grooming
        key: grooming                # stable id stored in label files
        color: "#009E73"
        hotkey: 2                    # number key 0-9 that selects the class
        model_class: 2               # the model's output index for this class
    export:
      heatmap: true                  # export motion heatmaps (default) or raw frames

Labels are stored by class *key*, never by position, so classes can be
renamed, reordered, added or removed without corrupting existing label files.

Projects are looked up either by path (e.g. a file kept next to the videos
and shared with labmates) or by bare name in the per-user projects directory
(see :func:`user_projects_dir`).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

# Okabe-Ito palette — colorblind-safe; used for classes without an explicit color.
DEFAULT_COLORS: List[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#0072B2",  # blue
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#999999",  # grey
]

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_PROJECT_SUFFIXES = (".yaml", ".yml")
_MISSING = object()


class ProjectError(ValueError):
    """Raised when a project file is missing or invalid."""


@dataclass(frozen=True)
class ClassDef:
    key: str
    name: str
    color: str                      # "#RRGGBB"
    hotkey: Optional[int] = None    # 0-9
    model_class: Optional[int] = None

    @property
    def bgr(self) -> Tuple[int, int, int]:
        r, g, b = (int(self.color[i:i + 2], 16) for i in (1, 3, 5))
        return (b, g, r)


@dataclass
class Project:
    name: str
    classes: List[ClassDef]
    export_heatmap: bool = True
    path: Optional[Path] = None
    _by_key: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_key = {c.key: i for i, c in enumerate(self.classes)}

    @property
    def keys(self) -> List[str]:
        return [c.key for c in self.classes]

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.classes]

    def index_of(self, key: str) -> Optional[int]:
        return self._by_key.get(key)

    def index_for_hotkey(self, digit: int) -> Optional[int]:
        for i, c in enumerate(self.classes):
            if c.hotkey == digit:
                return i
        return None

    def index_for_model_class(self, model_cls: int) -> Optional[int]:
        for i, c in enumerate(self.classes):
            if c.model_class is not None and c.model_class == model_cls:
                return i
        return None

    def labels_path(self, video_path: Path) -> Path:
        """Label file for *video_path*: ``<video_stem>_<project>_labels.json`` next to it."""
        video_path = Path(video_path)
        return video_path.parent / f"{video_path.stem}_{slugify(self.name)}_labels.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Lower-case, underscore-separated identifier derived from *text*."""
    return re.sub(r"[^0-9a-z]+", "_", str(text).strip().lower()).strip("_")


def user_projects_dir() -> Path:
    """Per-user directory searched for projects referenced by bare name."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "trackviz" / "projects"


def resolve_project_path(ref: Union[str, Path]) -> Path:
    """Turn a ``--project`` argument into a file path.

    *ref* may be a path to a YAML file, or the bare name of a project stored
    in :func:`user_projects_dir` (``fly_regurgitation`` →
    ``~/.config/trackviz/projects/fly_regurgitation.yaml``).
    """
    p = Path(ref).expanduser()
    if p.is_file():
        return p
    if p.suffix.lower() not in _PROJECT_SUFFIXES and len(p.parts) == 1:
        for suffix in _PROJECT_SUFFIXES:
            cand = user_projects_dir() / f"{p.name}{suffix}"
            if cand.is_file():
                return cand
        raise ProjectError(
            f"No project file '{ref}' and no project named '{ref}' in {user_projects_dir()}"
        )
    raise ProjectError(f"Project file not found: {ref}")


def load_project(ref: Union[str, Path]) -> Project:
    """Load and validate a project from a path or a bare project name."""
    path = resolve_project_path(ref)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ProjectError(f"{path.name}: invalid YAML — {e}") from e
    project = project_from_dict(data, source=path.name)
    project.path = path
    return project


def project_from_dict(data: object, source: str = "project") -> Project:
    """Build a :class:`Project` from parsed YAML, raising :class:`ProjectError` on bad input."""
    if not isinstance(data, dict):
        raise ProjectError(f"{source}: expected a mapping with 'name' and 'classes'")

    name = data.get("name")
    if not isinstance(name, str) or not slugify(name):
        raise ProjectError(f"{source}: 'name' must be a non-empty string")

    raw_classes = data.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ProjectError(f"{source}: 'classes' must be a non-empty list")

    # First pass: validate each entry and collect explicit values.
    entries = []
    for i, raw in enumerate(raw_classes):
        where = f"{source}: classes[{i}]"
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict):
            raise ProjectError(f"{where}: expected a class name or a mapping")
        unknown = set(raw) - {"name", "key", "color", "hotkey", "model_class"}
        if unknown:
            raise ProjectError(f"{where}: unknown field(s) {sorted(unknown)}")

        cname = raw.get("name")
        if not isinstance(cname, str) or not cname.strip():
            raise ProjectError(f"{where}: 'name' must be a non-empty string")
        cname = cname.strip()

        key = raw.get("key", slugify(cname))
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-z_]+", key):
            raise ProjectError(
                f"{where} ({cname}): 'key' must use only lowercase letters, digits and '_'"
            )

        color = raw.get("color")
        if color is not None and (not isinstance(color, str) or not _HEX_COLOR.match(color)):
            raise ProjectError(f"{where} ({cname}): 'color' must look like \"#RRGGBB\"")

        hotkey = raw.get("hotkey", _MISSING)
        if hotkey is not _MISSING and hotkey is not None and (
            isinstance(hotkey, bool) or not isinstance(hotkey, int) or not 0 <= hotkey <= 9
        ):
            raise ProjectError(f"{where} ({cname}): 'hotkey' must be a number 0-9 or null")

        model_class = raw.get("model_class")
        if model_class is not None and (
            isinstance(model_class, bool) or not isinstance(model_class, int) or model_class < 0
        ):
            raise ProjectError(f"{where} ({cname}): 'model_class' must be a non-negative integer")

        entries.append({"name": cname, "key": key, "color": color,
                        "hotkey": hotkey, "model_class": model_class})

    _check_unique(entries, "key", source)
    _check_unique(entries, "name", source)
    _check_unique([e for e in entries if e["hotkey"] not in (_MISSING, None)], "hotkey", source)
    _check_unique([e for e in entries if e["model_class"] is not None], "model_class", source)

    # Second pass: fill in defaults without colliding with explicit values.
    used_colors = {e["color"].upper() for e in entries if e["color"]}
    free_colors = [c for c in DEFAULT_COLORS if c.upper() not in used_colors]
    used_hotkeys = {e["hotkey"] for e in entries if e["hotkey"] not in (_MISSING, None)}
    free_hotkeys = [d for d in range(10) if d not in used_hotkeys]

    classes: List[ClassDef] = []
    for i, e in enumerate(entries):
        color = e["color"]
        if color is None:
            color = free_colors.pop(0) if free_colors else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        hotkey = e["hotkey"]
        if hotkey is _MISSING:
            # Omitted → next free digit; an explicit null means "no hotkey".
            hotkey = free_hotkeys.pop(0) if free_hotkeys else None
        classes.append(ClassDef(key=e["key"], name=e["name"], color=color.upper(),
                                hotkey=hotkey, model_class=e["model_class"]))

    export = data.get("export") or {}
    if not isinstance(export, dict):
        raise ProjectError(f"{source}: 'export' must be a mapping")
    heatmap = export.get("heatmap", True)
    if not isinstance(heatmap, bool):
        raise ProjectError(f"{source}: 'export.heatmap' must be true or false")

    return Project(name=name.strip(), classes=classes, export_heatmap=heatmap)


def _check_unique(entries: List[dict], field_name: str, source: str) -> None:
    seen: Dict[object, str] = {}
    for e in entries:
        value = e[field_name]
        if value in seen:
            raise ProjectError(
                f"{source}: classes '{seen[value]}' and '{e['name']}' share the same {field_name} ({value!r})"
            )
        seen[value] = e["name"]


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

PROJECT_TEMPLATE = """\
# trackviz labeling project.
# Open it with:  trackviz gui --project {filename}
# or use File → Open Project… in the viewer.

name: {name}

# One entry per behavior. Only 'name' is required; everything else is optional.
#   key:         stable id written to label files (default: name in snake_case).
#                Keep it unchanged once you have labels, even if you rename the class.
#   color:       "#RRGGBB" box color (default: colorblind-safe palette).
#   hotkey:      number key 0-9 that selects the class (default: next free digit;
#                null for none).
#   model_class: the prediction model's class index for this behavior. When set,
#                frames predicted as that class pre-select it in the dropdown.
#                Leave it out for behaviors the model doesn't know.
classes:
  - name: Behavior A
  - name: Behavior B

export:
  heatmap: true   # export training frames as motion heatmaps (false = raw frames)
"""


def write_project_template(path: Path, name: Optional[str] = None) -> Path:
    """Write a commented starter project file; refuses to overwrite."""
    path = Path(path)
    if path.suffix.lower() not in _PROJECT_SUFFIXES:
        path = path.with_suffix(".yaml")
    if path.exists():
        raise ProjectError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        PROJECT_TEMPLATE.format(filename=path.name, name=name or slugify(path.stem)),
        encoding="utf-8",
    )
    return path
