"""Export labeling-project labels as a YOLO detection dataset.

For every label file of a project (``<video_stem>_<project>_labels.json``)
this writes one full-resolution image plus one YOLO ``.txt`` per labeled
frame::

    <out>/images/<video_stem>_<ClassName>_<frame>_heatmap.jpg
    <out>/labels/<video_stem>_<ClassName>_<frame>_heatmap.txt   # "<cls> cx cy w h" (normalized)
    <out>/data.yaml                                              # class index → name, project order
    <out>/manifest.csv                                           # one row per exported / skipped frame

Images are 6-frame motion heatmaps by default (``heatmap=False`` → raw frames,
named without ``_heatmap``).  Class indices follow the project's class order.

Adding to an existing dataset
-----------------------------
With ``match_dataset`` pointing at an existing split YOLO dataset
(``train/``, ``val/``, ``test/`` each with ``images/`` + ``labels/``, and a
``data.yaml``), the export instead writes ``<out>/<split>/images|labels``:

* frames from a video already in the dataset go to **that video's split**, so
  no video ends up in two splits;
* frames from videos the dataset has never seen go to ``<out>/unassigned/``
  for you to place;
* frames already in the dataset are skipped — including under a different
  class, which is reported so the conflict can be fixed in the dataset;
* the dataset's class names must match the project's classes index for index
  (the project may add classes after them).

The existing dataset is only read, never modified.

The heatmap is computed exactly like the flyolo dataset builder
(``step2_build_datasets.py``): the newest frame of the window ending at the
labeled frame is the reference, earlier frames are thresholded absolute
differences, and the normalized accumulator is blended 60/40 over the color
reference.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from trackviz.io.labels import read_labels
from trackviz.io.project import Project, slugify

HEATMAP_FRAMES = 6
HEATMAP_THRESHOLD = 10
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
SPLITS = ("train", "val", "test")
UNASSIGNED = "unassigned"

# Frames further apart than this are reached by seeking instead of decoding forward.
_MAX_READ_GAP = 30


# ---------------------------------------------------------------------------
# Image + label primitives
# ---------------------------------------------------------------------------

def motion_heatmap(window_bgr: Sequence[np.ndarray], threshold: int = HEATMAP_THRESHOLD) -> np.ndarray:
    """Heatmap for a window of BGR frames ending at the labeled frame (flyolo-identical)."""
    ref_bgr = window_bgr[-1]
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    acc = np.zeros(ref_gray.shape, dtype=np.float32)
    for frame in window_bgr[:-1]:
        diff = cv2.absdiff(ref_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        acc += mask.astype(np.float32)
    if acc.max() > 0:
        acc = (acc / acc.max() * 255).astype(np.uint8)
    else:
        acc = acc.astype(np.uint8)
    overlay = cv2.applyColorMap(acc, cv2.COLORMAP_HOT)
    return cv2.addWeighted(ref_bgr, 0.6, overlay, 0.4, 0)


def yolo_line(bbox_xyxy: Sequence[float], class_id: int, img_w: int, img_h: int) -> Optional[str]:
    """``"<cls> cx cy w h"`` normalized to the image, or None if the box is unusable."""
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    xn, yn, wn, hn = cx / img_w, cy / img_h, w / img_w, h / img_h
    if not (0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0) or wn <= 0 or hn <= 0:
        return None
    return f"{class_id} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}\n"


def image_stem(video_stem: str, class_name: str, frame: int, heatmap: bool) -> str:
    safe_name = re.sub(r"[^0-9A-Za-z]+", "", class_name) or "class"
    return f"{video_stem}_{safe_name}_{frame}" + ("_heatmap" if heatmap else "")


class _WindowReader:
    """Reads frame windows in increasing order, decoding forward instead of seeking where possible."""

    def __init__(self, video_path: Path) -> None:
        self.cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")
        self._buffer: Dict[int, np.ndarray] = {}
        self._pos = -1  # index of the next frame cap.read() returns

    def window(self, start: int, end: int) -> List[np.ndarray]:
        needed = [i for i in range(start, end + 1) if i not in self._buffer]
        if needed:
            first = needed[0]
            if self._pos < 0 or first < self._pos or first - self._pos > _MAX_READ_GAP:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            else:
                for _ in range(first - self._pos):
                    self.cap.read()
            self._pos = first
            for i in range(first, end + 1):
                ok, frame = self.cap.read()
                self._pos = i + 1
                if ok and frame is not None:
                    self._buffer[i] = frame
        for old in [k for k in self._buffer if k < start]:
            del self._buffer[old]
        return [self._buffer[i] for i in range(start, end + 1) if i in self._buffer]

    def close(self) -> None:
        self.cap.release()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabeledVideo:
    video: Path
    labels: Path


def find_labeled_videos(project: Project, paths: Iterable[Path]) -> Tuple[List[LabeledVideo], List[str]]:
    """Resolve videos / folders to (video, label file) pairs for *project*.

    Folders are searched recursively for the project's label files.  Returns
    the pairs plus human-readable problems (e.g. a label file without video).
    """
    suffix = f"_{slugify(project.name)}_labels.json"
    found: Dict[Path, LabeledVideo] = {}
    problems: List[str] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            lab = project.labels_path(p)
            if lab.exists():
                found[lab.resolve()] = LabeledVideo(p, lab)
            else:
                problems.append(f"{p.name}: no labels for project '{project.name}' ({lab.name})")
        elif p.is_dir():
            for lab in sorted(p.rglob(f"*{suffix}")):
                stem = lab.name[: -len(suffix)]
                videos = [lab.with_name(stem + s) for s in VIDEO_SUFFIXES
                          if lab.with_name(stem + s).exists()]
                if videos:
                    found[lab.resolve()] = LabeledVideo(videos[0], lab)
                else:
                    problems.append(f"{lab}: no matching video next to it")
        else:
            problems.append(f"{p}: not a video file or folder")
    return sorted(found.values(), key=lambda lv: str(lv.video)), problems


# ---------------------------------------------------------------------------
# Existing dataset
# ---------------------------------------------------------------------------

@dataclass
class ExistingDataset:
    root: Path
    names: List[str]
    video_splits: Dict[str, str] = field(default_factory=dict)   # video stem → split
    frames: Dict[Tuple[str, int], str] = field(default_factory=dict)  # (video stem, frame) → class
    heatmap: bool = True   # images are named *_heatmap.jpg

    @classmethod
    def load(cls, root: Path) -> "ExistingDataset":
        root = Path(root)
        data_yaml = root / "data.yaml"
        if not data_yaml.exists():
            raise ValueError(f"No data.yaml in {root}")
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_names = data.get("names")
        if isinstance(raw_names, dict):
            names = [str(raw_names[k]) for k in sorted(raw_names, key=int)]
        elif isinstance(raw_names, list):
            names = [str(n) for n in raw_names]
        else:
            raise ValueError(f"{data_yaml}: missing 'names'")

        ds = cls(root=root, names=names)
        # Image names look like <video_stem>_<Class>_<frame>[_heatmap].jpg
        name_re = re.compile(
            r"^(?P<video>.+)_(?P<cls>" + "|".join(re.escape(n) for n in names)
            + r")_(?P<frame>\d+)(?P<heat>_heatmap)?$"
        )
        n_heat = n_images = 0
        conflicts: Dict[str, set] = {}
        for split in SPLITS:
            img_dir = root / split / "images"
            if not img_dir.is_dir():
                continue
            for img in img_dir.iterdir():
                m = name_re.match(img.stem)
                if m is None:
                    continue
                n_images += 1
                n_heat += bool(m["heat"])
                video = m["video"]
                ds.frames[(video, int(m["frame"]))] = m["cls"]
                ds.video_splits.setdefault(video, split)
                if ds.video_splits[video] != split:
                    conflicts.setdefault(video, {ds.video_splits[video]}).add(split)
        if conflicts:
            detail = "; ".join(f"{v}: {sorted(s)}" for v, s in sorted(conflicts.items()))
            raise ValueError(f"Videos appear in more than one split of {root}: {detail}")
        if not n_images:
            raise ValueError(f"No images named <video>_<Class>_<frame>[_heatmap].jpg under "
                             f"{root}/<train|val|test>/images")
        ds.heatmap = n_heat * 2 > n_images
        return ds

    def check_classes(self, project: Project) -> None:
        """The dataset's classes must be a prefix of the project's, index for index."""
        mismatched = [
            (i, ds_name, project.names[i] if i < len(project.names) else None)
            for i, ds_name in enumerate(self.names)
            if i >= len(project.names) or project.names[i] != ds_name
        ]
        if mismatched:
            lines = "\n".join(f"  {i}: dataset '{d}' vs project '{p}'" for i, d, p in mismatched)
            raise ValueError(
                "Project classes don't line up with the dataset's data.yaml — class indices "
                f"would be wrong:\n{lines}\nReorder the project's classes to match."
            )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    exported: Counter = field(default_factory=Counter)   # (split, class name) → count
    skipped: Counter = field(default_factory=Counter)    # reason → count
    out_dir: Optional[Path] = None

    @property
    def total_exported(self) -> int:
        return sum(self.exported.values())


def export_dataset(
    project: Project,
    videos: Sequence[LabeledVideo],
    out_dir: Path,
    *,
    heatmap: Optional[bool] = None,
    classes: Optional[Sequence[str]] = None,
    match_dataset: Optional[ExistingDataset] = None,
    dry_run: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> ExportResult:
    """Write images + YOLO labels for *videos* into *out_dir* (see module docstring).

    *classes* limits the export to those class names or keys.  *heatmap*
    defaults to the project's ``export.heatmap`` setting.
    """
    heatmap = project.export_heatmap if heatmap is None else heatmap
    out_dir = Path(out_dir)
    say = progress or (lambda msg: None)

    wanted: Optional[set] = None
    if classes:
        wanted = set()
        for c in classes:
            idx = project.index_of(c)
            if idx is None:
                idx = next((i for i, n in enumerate(project.names) if n.lower() == c.lower()), None)
            if idx is None:
                raise ValueError(f"Unknown class '{c}'. Project classes: {', '.join(project.names)}")
            wanted.add(project.classes[idx].key)

    if match_dataset is not None:
        match_dataset.check_classes(project)
        if match_dataset.heatmap != heatmap:
            kind = "heatmap" if match_dataset.heatmap else "raw-frame"
            raise ValueError(f"{match_dataset.root.name} is a {kind} dataset; export with "
                             f"{'--heatmap' if match_dataset.heatmap else '--raw'} to match it.")
        if out_dir.resolve() == match_dataset.root.resolve():
            raise ValueError("Output folder must differ from the existing dataset (it is never modified).")

    if not dry_run:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise ValueError(f"Output folder {out_dir} is not empty.")
        out_dir.mkdir(parents=True, exist_ok=True)

    result = ExportResult(out_dir=out_dir)
    manifest_rows: List[dict] = []

    for lv in videos:
        label_file = read_labels(lv.labels)
        split = ""
        if match_dataset is not None:
            split = match_dataset.video_splits.get(lv.video.stem, UNASSIGNED)

        jobs = []
        for frame, lab in sorted(label_file.labels.items()):
            idx = project.index_of(lab.cls)
            row = {"video": str(lv.video), "frame": frame, "class": lab.cls, "split": split,
                   "image": "", "status": ""}
            manifest_rows.append(row)
            if idx is None:
                row["status"] = "skipped: class not in project"
            elif wanted is not None and lab.cls not in wanted:
                row["status"] = "skipped: class not selected"
            elif heatmap and frame < 1:
                row["status"] = "skipped: no earlier frames for heatmap"
            else:
                stem = image_stem(lv.video.stem, project.classes[idx].name, frame, heatmap)
                row["image"] = f"{stem}.jpg"
                in_dataset = (match_dataset.frames.get((lv.video.stem, frame))
                              if match_dataset is not None else None)
                if in_dataset is None:
                    jobs.append((frame, idx, lab.bbox, stem, row))
                elif in_dataset == project.classes[idx].name:
                    row["status"] = "skipped: already in dataset"
                else:
                    row["status"] = f"skipped: in dataset as {in_dataset}"
            if row["status"]:
                result.skipped[row["status"]] += 1

        if not jobs:
            continue
        say(f"{lv.video.name}: {len(jobs)} frames" + (f" → {split}" if split else ""))
        if dry_run:
            for _, idx, _, _, row in jobs:
                row["status"] = "would export"
                result.exported[(split, project.classes[idx].name)] += 1
            continue

        base = out_dir / split if split else out_dir
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "labels").mkdir(parents=True, exist_ok=True)
        reader = _WindowReader(lv.video)
        try:
            for frame, idx, bbox, stem, row in jobs:
                start = max(0, frame - HEATMAP_FRAMES + 1) if heatmap else frame
                window = reader.window(start, frame)
                if not window or len(window) != frame - start + 1:
                    _skip(row, result, "skipped: frame read failed")
                    continue
                image = motion_heatmap(window) if heatmap else window[-1]
                h, w = image.shape[:2]
                line = yolo_line(bbox, idx, w, h)
                if line is None:
                    _skip(row, result, "skipped: box outside image")
                    continue
                if not cv2.imwrite(str(base / "images" / f"{stem}.jpg"), image):
                    _skip(row, result, "skipped: image write failed")
                    continue
                (base / "labels" / f"{stem}.txt").write_text(line, encoding="utf-8")
                row["status"] = "exported"
                result.exported[(split, project.classes[idx].name)] += 1
        finally:
            reader.close()

    if not dry_run:
        _write_data_yaml(out_dir / "data.yaml", project, split_layout=match_dataset is not None)
        with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video", "frame", "class", "split", "image", "status"])
            writer.writeheader()
            writer.writerows(manifest_rows)
    return result


def _skip(row: dict, result: ExportResult, reason: str) -> None:
    row["status"] = reason
    result.skipped[reason] += 1


def _write_data_yaml(path: Path, project: Project, split_layout: bool) -> None:
    lines = ["# Written by trackviz export-dataset", "path: .",]
    if split_layout:
        lines += ["train: train/images", "val: val/images", "test: test/images"]
    else:
        lines += ["train: images", "val: images"]
    lines += [f"nc: {len(project.classes)}", "names:"]
    lines += [f"  {i}: '{c.name}'" for i, c in enumerate(project.classes)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
