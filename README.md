# trackviz

A Python package for **scrubbing, playing, and annotating** videos with model predictions (bboxes + track IDs + confidences) frame-by-frame.

## Install (dev)

### Using Pip
```bash
pip install -e .
```

> **AVI / codec support:** `opencv-python` from PyPI ships without FFmpeg on some platforms (notably macOS), so AVI files may fail to open. If you hit `Couldn't read movie file`, replace the PyPI OpenCV with the conda-forge build:
> ```bash
> pip uninstall opencv-python
> conda install -c conda-forge opencv
> ```

### Using Conda (recommended on macOS)
The conda-forge OpenCV build includes FFmpeg and handles AVI, MKV, and other formats correctly.
```bash
conda create -n trackviz python=3.10 -y
conda activate trackviz
pip install -e .
```

## Quick start

```bash
trackviz view path/to/video.mp4 --preds path/to/preds.npz
```

## GUI (drag & drop)

Open the viewer with no arguments, then drag a video file into the window:

```bash
trackviz gui
```

When you drop `my_video.mp4`, trackviz will look in the same folder for matching files. It supports several workflows:

### 1. Flyloop YOLO Workflow (Single File)
Optimized for behavioral neuroscience experiments using the `flyloop` system.
- Looks for `<run_id>_yolo_fast.pkl` (even if timestamps differ slightly).
- Overlays behavioral class names (e.g., "ProbPumping", "Grooming").
- Supports **Heatmap Mode**: Toggling this shows the real-time 6-frame motion heatmap used by the model during inference.

### 2. Standard NPZ Workflow
- `my_video.npz` or `my_video_preds.npz` containing bboxes, confidences, and track_ids.

### 3. Triplet Workflow (npy/csv)
- `my_video_bboxes.npy` or `my_video_bboxes.csv` (required)
- `my_video_confidences.npy` or `my_video_confidences.csv` (optional)
- `my_video_track_ids.npy` or `my_video_track_ids.csv` (optional)
- `my_video_metadata.npz` (optional)

It also supports the generic fallback names `bboxes.*`, `confidences.*`, `track_ids.*`, `metadata.npz`.

## Annotation System

trackviz includes a full frame-by-frame annotation workflow, including bounding box correction.

### Labeling frames

Each frame can have one annotation: a behavior class plus a bounding box. To label the current frame:

1. Select a behavior class from the **Behavior** dropdown (or press a number key `0`–`7`).
2. Press **Save [S]** or the `S` hotkey.

When saved, the predicted bounding box for that frame is automatically stored alongside the class label.

Annotations are saved to `<video_stem>_annotations.json` next to the video file and reload automatically on the next session.

### Bounding box correction (Edit Mode)

Enable **Edit Mode** to interactively correct bounding boxes:

- **Select** an existing predicted or corrected box by clicking inside it.
- **Move** a selected box by dragging.
- **Resize** a selected box by dragging any of the 8 handles (corners + edge midpoints).
- **Draw** a new box by clicking and dragging on empty space.

After editing, press **Save [S]** to save the annotation. The corrected box renders in **cyan**; predicted boxes render in **green**. Corrected entries appear with a `✓` in the annotation list.

### Annotation list (sidebar)

All annotations are shown in the persistent sidebar list:
- **Double-click** any entry to jump to that frame.
- Select an entry and press **Delete Selected** to remove it.
- Deleting an annotation immediately updates the frame display.

### Hotkeys

| Key | Action |
|-----|--------|
| `0`–`7` | Select behavior class |
| `S` | Save annotation for current frame |
| `Space` | Play / Pause |
| `←` / `→` | Step one frame back / forward |

## Features for Long Videos

`trackviz` is optimized for ultra-long recordings (e.g., 16-hour experiments):
- **Sequential Decoding**: Uses sequential frame reading during playback to avoid high-latency seeks.
- **Sliding Window Cache**: Heatmap generation uses a frame cache to reduce decoding overhead by ~80%.
- **Smart Seek**: Intelligent frame management to maintain responsiveness during scrubbing.

## Supported prediction formats

1) **Dense per-frame arrays** (common when you have <=1 object per frame):

- `bboxes`: (T, 4) float32 in **xyxy** pixel coords
- `confidences`: (T,) float32
- `track_ids`: (T,) int/float
- Missing frames can be `NaN` bboxes/conf/ids.

2) **Ragged detections** (multiple objects per frame):

- `frame_idx`: (N,) int
- `bboxes`: (N, 4)
- `confidences`: (N,)
- `track_ids`: (N,)

## Notes

- Uses OpenCV to read frames and draw overlays; UI is PySide6.
- Random access is implemented via `cv2.CAP_PROP_POS_FRAMES` (fast on many codecs, but not all).


### CSV support

Custom exports can be loaded from CSV as well (bbox CSV with `Frame,x,y,w,h` or `Frame,x1,y1,x2,y2`; optional confidences and track ids).

If your bbox CSV uses YOLO-style `x,y,w,h` with `x/y` as the box center, pass `--yolo`:

```bash
trackviz view path/to/video.mp4 --preds path/to/folder --bboxes path/to/video_bboxes.csv --conf path/to/video_confidences.csv --yolo
```

For `.npy` bboxes, the drag-and-drop GUI auto-detects `xyxy` vs YOLO-style `xywh` and converts as needed.
