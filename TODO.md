# TODO

## Features

- [x] **Scramble button** — jump to a random unlabeled frame to assist with efficient annotation coverage
- [x] **Export video from UI** — dialog wrapping the existing `export_video()` pipeline (quality slider, scale, optional frame range); runs in a QThread with a progress bar so the UI stays responsive. Adds toggles to omit predictions and/or annotations from the export.

## Labeling mode

- [ ] **Range labeling** — mark start/end of an event (e.g. regurgitation) and label the frames in between, sampling every Nth frame for training
- [ ] **Class editor dialog** — add/rename/recolor/reorder/delete classes from the GUI and save back to the project YAML (deleting a class with labels forces reassignment)
- [x] **YOLO dataset export** — `trackviz export-dataset`: full frames + `.txt` boxes + `data.yaml`; motion heatmap by default, raw frames optional; `--match-dataset` adds to an existing split dataset
- [ ] **Train/val/test split for new datasets** — video-grouped split when exporting without `--match-dataset` (today: one flat folder; `unassigned/` for new videos)
- [ ] **Per-class targets** — "Regurgitation: 34/50" progress, and scramble toward under-labeled classes
- [ ] **Carry box forward** — keep the last drawn box on the next frame so it only needs nudging when there are no predictions
- [ ] **`trackviz export` CLI in labeling mode** — `--project` so exported videos show project labels
