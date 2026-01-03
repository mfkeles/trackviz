import numpy as np
from pathlib import Path

# Example: convert dense per-frame triplet into a single NPZ
bboxes = np.load("bboxes.npy")
conf = np.load("confidences.npy")
tids = np.load("track_ids.npy")

out = Path("preds_dense.npz")
np.savez_compressed(
    out,
    total_frames=bboxes.shape[0],
    bboxes=bboxes,
    confidences=conf,
    track_ids=tids,
)
print("Wrote", out)
