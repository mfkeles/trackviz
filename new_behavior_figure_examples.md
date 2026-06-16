# New-behaviour figure examples — fly / frame provenance

Which fly recordings and frames are shown as the **Twitching** and **Defecation**
examples in the novel-behaviour panel, as configured in the `EVENTS` list of
[`fig_new_behavior_confidence.py`](fig_new_behavior_confidence.py) (also the input
to [`fig_new_behavior_psd.py`](fig_new_behavior_psd.py)). This is a human-readable
copy of that config; `EVENTS` remains the source of truth.

**Conventions.** Frame numbers are 0-based video frame indices
(`fi = frame_seq − 1`, from the `*_yolo_fast.pkl` prediction stream); video is
30 fps, so time ≈ frame / 30 s. Each fly is one recording under
`/Volumes/Lab_Files3/mfk/projects/2025_flyvista2/PE_Origins_Project/ClosedLoopArousal/Data/<date>/`.

## Twitching — flies 20260320 and 20260319

### `twitch_20260320` (sequence)
- Fly: `20260320_200000_raw.mp4` · predictions `20260320_175222_yolo_fast.pkl`
- 1-minute confidence trace: frames **1115700–1117500**
- Frame strip (every 4th): **1116599, 1116603, 1116607, 1116611, 1116615**
- Highlighted span: **1116603–1116607** (window-min confidence 0.693 at frame 1116610)
- This 1-minute trace is also the input to the PSD figure (`fig_new_behavior_psd.py`).

### `twitch_20260319` (span)
- Fly: `20260319_200000_raw.mp4` · predictions `20260319_183419_yolo_fast.pkl`
- Span: **831300–832300** (±90 s confidence trace; heatmap snapshot at the lowest-confidence frame in the span)

## Defecation — fly 20260316 only

All from `20260316_200000_raw.mp4` · predictions `20260316_172528_yolo_fast.pkl`.

Single-frame examples (±15 s trace + plain/heatmap snapshot at the frame):

| Event | Frame | Category |
|---|---:|---|
| `defec_pellet_visible` | 71160 | round pellet visible at abdomen tip |
| `defec_while_walking` | 325136 | defecates while walking past |
| `defec_whole_body` | 1523326 | whole-body movement |
| `defec_smeared_window` | 823348 | feces smeared on window |
| `defec_placed_floor` | 7195 | places/drags feces on floor |

### `defec_series_20260316` (sequence)
- 20-second confidence trace: frames **70800–71400**
- Frame strip (every 5th): **71152, 71157, 71162, 71167, 71172, 71177, 71182**
- Highlighted span: **71152–71185** · hypnogram class panel
