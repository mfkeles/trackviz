# TODO

## Features

- [x] **Scramble button** — jump to a random unlabeled frame to assist with efficient annotation coverage
- [x] **Export video from UI** — dialog wrapping the existing `export_video()` pipeline (quality slider, scale, optional frame range); runs in a QThread with a progress bar so the UI stays responsive. Adds toggles to omit predictions and/or annotations from the export.
