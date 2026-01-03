# trackviz

A small Python package that opens a **scrubbable, playable** video window and overlays your model's
predictions (bboxes + track IDs + confidences) frame-by-frame.

## Install (dev)

```bash
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

When you drop `my_video.mp4`, trackviz will look in the same folder for matching files:

- `my_video_bboxes.npy` or `my_video_bboxes.csv` (required)
- `my_video_confidences.npy` or `my_video_confidences.csv` (optional)
- `my_video_track_ids.npy` or `my_video_track_ids.csv` (optional)
- `my_video_metadata.npz` (optional)

It also supports the generic fallback names `bboxes.*`, `confidences.*`, `track_ids.*`, `metadata.npz`.

### Supported prediction formats

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
