"""Read/write labeling-mode label files (``<video_stem>_<project>_labels.json``).

Format::

    {
      "format": "trackviz-labels",
      "version": 1,
      "project": "fly_regurgitation",
      "video": "fly01.mp4",
      "classes": {"regurgitation": "Regurgitation", ...},   # key → name at save time
      "labels": {"1234": {"class": "regurgitation", "bbox": [x1, y1, x2, y2]}, ...}
    }

Classes are referenced by their stable project *key*; ``classes`` is a
snapshot so a file stays readable even after the project changes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trackviz.io.project import Project, slugify

FORMAT_NAME = "trackviz-labels"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Label:
    cls: str            # project class key
    bbox: List[float]   # [x1, y1, x2, y2] in source-video pixels


@dataclass
class LabelFile:
    labels: Dict[int, Label] = field(default_factory=dict)
    class_names: Dict[str, str] = field(default_factory=dict)   # snapshot: key → name

    def unknown_classes(self, project: Project) -> Dict[str, int]:
        """Class keys used by labels but missing from *project*, with label counts."""
        counts: Dict[str, int] = {}
        for lab in self.labels.values():
            if project.index_of(lab.cls) is None:
                counts[lab.cls] = counts.get(lab.cls, 0) + 1
        return counts

    def reassign(self, mapping: Dict[str, str]) -> None:
        """Rewrite class keys according to *mapping* (old key → new key)."""
        self.labels = {
            f: Label(mapping.get(lab.cls, lab.cls), lab.bbox) for f, lab in self.labels.items()
        }


class LabelFileError(ValueError):
    pass


def read_labels(path: Path) -> LabelFile:
    """Read a label file; a missing file yields an empty :class:`LabelFile`."""
    path = Path(path)
    if not path.exists():
        return LabelFile()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise LabelFileError(f"Could not read {path.name}: {e}") from e
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise LabelFileError(f"{path.name} is not a trackviz label file")
    if int(data.get("version", 0)) > FORMAT_VERSION:
        raise LabelFileError(
            f"{path.name} was written by a newer trackviz (format v{data.get('version')}); please upgrade"
        )

    labels: Dict[int, Label] = {}
    for frame_str, entry in (data.get("labels") or {}).items():
        try:
            bbox = [float(v) for v in entry["bbox"]]
            if len(bbox) != 4:
                raise ValueError("bbox needs 4 values")
            labels[int(frame_str)] = Label(str(entry["class"]), bbox)
        except (KeyError, TypeError, ValueError) as e:
            raise LabelFileError(f"{path.name}: bad label for frame {frame_str!r}: {e}") from e
    class_names = {str(k): str(v) for k, v in (data.get("classes") or {}).items()}
    return LabelFile(labels=labels, class_names=class_names)


def import_default_annotations(
    path: Path, class_names: List[str], project: Project
) -> Tuple[LabelFile, List[int]]:
    """Convert a default-mode ``<video>_annotations.json`` into a :class:`LabelFile`.

    Default-mode entries store the class as an index into *class_names* (the
    built-in fly classes).  Each is matched to a project class **by name**
    (case-insensitive) or by key; unmatched classes keep a derived key so the
    caller's reassignment step can map them.  Entries without a box cannot be
    labels and are returned as skipped frame numbers.  *path* is only read.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise LabelFileError(f"Could not read {Path(path).name}: {e}") from e
    if not isinstance(raw, dict):
        raise LabelFileError(f"{Path(path).name} is not a trackviz annotation file")

    by_name = {c.name.lower(): c.key for c in project.classes}
    labels: Dict[int, Label] = {}
    old_names: Dict[str, str] = {}
    skipped: List[int] = []
    for frame_str, entry in raw.items():
        try:
            frame = int(frame_str)
            cls = int(entry if isinstance(entry, int) else entry["cls"])
            bbox = None if isinstance(entry, int) else entry.get("bbox")
        except (KeyError, TypeError, ValueError, AttributeError):
            skipped.append(int(frame_str) if str(frame_str).isdigit() else -1)
            continue
        if bbox is None or len(bbox) != 4:
            skipped.append(frame)
            continue
        name = class_names[cls] if 0 <= cls < len(class_names) else f"cls_{cls}"
        key = by_name.get(name.lower()) or slugify(name)
        old_names.setdefault(key, name)
        labels[frame] = Label(key, [float(v) for v in bbox])
    return LabelFile(labels=labels, class_names=old_names), sorted(skipped)


def write_labels(path: Path, label_file: LabelFile, project: Project,
                 video_name: Optional[str] = None) -> None:
    """Write *label_file* atomically (temp file + rename) so a crash never truncates it."""
    path = Path(path)
    # Snapshot names for every key in use: current project names win, and keys
    # the project no longer has keep their last known name.
    class_names = {}
    for lab in label_file.labels.values():
        idx = project.index_of(lab.cls)
        class_names[lab.cls] = (
            project.classes[idx].name if idx is not None
            else label_file.class_names.get(lab.cls, lab.cls)
        )
    label_file.class_names = class_names

    data = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "project": project.name,
        "video": video_name,
        "classes": dict(sorted(class_names.items())),
        "labels": {
            str(f): {"class": lab.cls, "bbox": [round(v, 2) for v in lab.bbox]}
            for f, lab in sorted(label_file.labels.items())
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
