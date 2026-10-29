"""Train a YOLO detector on a dataset written by ``trackviz export-dataset --split``.

Defaults follow the flyolo paper preset ``exp2_native_800_strict_color``
(``flyolo/02_training/run_color_experiments.py``): YOLO11-l, 800 px,
rectangular batches, and **no hue/saturation augmentation**, because the
color of a motion heatmap is part of what the model learns.

Needs the optional ``train`` extra (``pip install "trackviz[train]"``), which
pulls in ultralytics + PyTorch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

DEFAULT_MODEL = "yolo11l.pt"

# flyolo preset exp2_native_800_strict_color (minus epochs/patience, which the
# CLI exposes, and workers/device, which depend on the machine).
PRESET: Dict[str, Any] = {
    "imgsz": 800,
    "rect": True,
    "batch": 16,
    "seed": 0,
    "close_mosaic": 10,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.4,
    "mosaic": 1.0,
    "erasing": 0.4,
    "scale": 0.5,
    "translate": 0.1,
    "fliplr": 0.5,
    "mixup": 0.0,
    "auto_augment": "randaugment",
    "optimizer": "auto",
}


class TrainError(ValueError):
    pass


def check_dataset(dataset: Path) -> Dict[str, Any]:
    """Validate a split dataset folder and return its data.yaml contents."""
    dataset = Path(dataset)
    data_yaml = dataset / "data.yaml" if dataset.is_dir() else dataset
    if not data_yaml.is_file():
        raise TrainError(f"No data.yaml in {dataset}. Point this at the --out folder of "
                         "`trackviz export-dataset --split ...`.")
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = data_yaml.parent
    train, val = data.get("train"), data.get("val")
    if not train or not val:
        raise TrainError(f"{data_yaml} needs 'train' and 'val' entries.")
    if Path(train) == Path(val):
        raise TrainError(
            "train and val are the same folder, so validation scores would be meaningless. "
            "Re-export with a split, e.g.  trackviz export-dataset ... --split 0.8 0.2"
        )
    for key in ("train", "val"):
        folder = root / data[key]
        if not folder.is_dir() or not any(folder.glob("*.jpg")):
            raise TrainError(f"No images in {folder} ({key} split).")
    data["_yaml"] = data_yaml
    return data


def pick_device() -> str:
    """CUDA GPU if present, else Apple Silicon (MPS), else CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    dataset: Path,
    *,
    model: str = DEFAULT_MODEL,
    epochs: int = 200,
    patience: int = 100,
    imgsz: Optional[int] = None,
    batch: Optional[int] = None,
    device: Optional[str] = None,
    workers: int = 8,
    out: Optional[Path] = None,
    name: str = "train",
    say: Callable[[str], None] = print,
) -> Path:
    """Train and return the path to ``best.pt``."""
    data = check_dataset(Path(dataset))
    data_yaml: Path = data.pop("_yaml")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise TrainError(
            "Training needs ultralytics, which isn't installed. Install it with\n"
            '    pip install "trackviz[train]"\n'
            "(or `uv sync --extra train` in a trackviz checkout). On a CUDA machine, install the "
            "CUDA build of PyTorch first: https://pytorch.org/get-started/locally/"
        )

    out = Path(out) if out else data_yaml.parent / "runs"
    out.mkdir(parents=True, exist_ok=True)
    # Absolute dataset root, so ultralytics finds the images whatever the cwd.
    resolved = {k: v for k, v in data.items() if k != "path"}
    resolved["path"] = str(data_yaml.parent.resolve())
    run_yaml = out / f"{name}_data.yaml"
    run_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    device = device or pick_device()
    if device == "cpu":
        say("Warning: no GPU found — training on CPU will be very slow.")
    args = dict(PRESET)
    if imgsz:
        args["imgsz"] = imgsz
    if batch:
        args["batch"] = batch
    if device == "mps":
        args["workers"] = 0   # dataloader workers are unreliable with MPS
    else:
        args["workers"] = workers

    say(f"Training {model} on {data_yaml.parent} ({len(data.get('names', {}))} classes) "
        f"for up to {epochs} epochs on device {device}")
    yolo = YOLO(model)
    yolo.train(
        data=str(run_yaml), epochs=epochs, patience=patience, device=device,
        project=str(out.resolve()), name=name, exist_ok=False, **args,
    )
    save_dir = Path(getattr(yolo.trainer, "save_dir", out / name))
    return save_dir / "weights" / "best.pt"
