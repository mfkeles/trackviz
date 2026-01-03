from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detection/tracked object on a given frame."""
    frame: int
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: Optional[float] = None
    track_id: Optional[int] = None
    cls: Optional[int] = None  # optional class id


class Predictions:
    """Indexable predictions with a uniform `for_frame(i)` API.

    Internally supports:
      - dense arrays per frame (T,4) with possible NaNs
      - ragged arrays with `frame_idx` mapping detections to frames
    """

    def __init__(
        self,
        *,
        total_frames: int,
        dense_bbox_xyxy: Optional[np.ndarray] = None,     # (T,4) or None
        dense_conf: Optional[np.ndarray] = None,         # (T,) or None
        dense_track_id: Optional[np.ndarray] = None,     # (T,) or None
        ragged_frame_idx: Optional[np.ndarray] = None,   # (N,) or None
        ragged_bbox_xyxy: Optional[np.ndarray] = None,   # (N,4) or None
        ragged_conf: Optional[np.ndarray] = None,        # (N,) or None
        ragged_track_id: Optional[np.ndarray] = None,    # (N,) or None
        meta: Optional[Dict] = None,
    ):
        self.total_frames = int(total_frames)
        self.meta = meta or {}

        self._dense_bbox = dense_bbox_xyxy
        self._dense_conf = dense_conf
        self._dense_tid = dense_track_id

        self._ragged_frame_idx = ragged_frame_idx
        self._ragged_bbox = ragged_bbox_xyxy
        self._ragged_conf = ragged_conf
        self._ragged_tid = ragged_track_id

        self._ragged_index: Optional[List[np.ndarray]] = None
        if self._ragged_frame_idx is not None:
            self._build_ragged_index()

    def _build_ragged_index(self) -> None:
        fi = np.asarray(self._ragged_frame_idx, dtype=np.int64)
        buckets: List[List[int]] = [[] for _ in range(self.total_frames)]
        for j, f in enumerate(fi):
            if 0 <= f < self.total_frames:
                buckets[int(f)].append(int(j))
        self._ragged_index = [np.asarray(b, dtype=np.int64) for b in buckets]

    @staticmethod
    def _is_valid_bbox(b: np.ndarray) -> bool:
        return b.shape == (4,) and np.isfinite(b).all()

    def for_frame(self, i: int) -> List[Detection]:
        """Return a list of detections for frame i (0-indexed)."""
        i = int(i)
        if i < 0 or i >= self.total_frames:
            return []

        dets: List[Detection] = []

        if self._dense_bbox is not None:
            b = np.asarray(self._dense_bbox[i])
            if self._is_valid_bbox(b):
                conf = None if self._dense_conf is None else float(self._dense_conf[i])
                if conf is not None and not np.isfinite(conf):
                    conf = None
                tid = None if self._dense_tid is None else self._dense_tid[i]
                tid_int = None
                if tid is not None and np.isfinite(tid):
                    tid_int = int(tid)
                dets.append(
                    Detection(
                        frame=i,
                        bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        confidence=conf,
                        track_id=tid_int,
                    )
                )

        if self._ragged_index is not None and self._ragged_bbox is not None:
            idxs = self._ragged_index[i]
            if idxs.size:
                bboxes = self._ragged_bbox[idxs]
                confs = None if self._ragged_conf is None else self._ragged_conf[idxs]
                tids = None if self._ragged_tid is None else self._ragged_tid[idxs]
                for k in range(len(idxs)):
                    b = np.asarray(bboxes[k])
                    if not self._is_valid_bbox(b):
                        continue
                    conf = None
                    if confs is not None:
                        c = float(confs[k])
                        if np.isfinite(c):
                            conf = c
                    tid_int = None
                    if tids is not None:
                        t = float(tids[k])
                        if np.isfinite(t):
                            tid_int = int(t)
                    dets.append(
                        Detection(
                            frame=i,
                            bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                            confidence=conf,
                            track_id=tid_int,
                        )
                    )

        return dets

    # -------- loaders --------

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "Predictions":
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        # Determine total_frames
        total_frames = None
        if "total_frames" in data.files:
            total_frames = int(np.asarray(data["total_frames"]).item())
        elif "frame_count" in data.files:
            total_frames = int(np.asarray(data["frame_count"]).item())
        elif "bboxes" in data.files:
            total_frames = int(np.asarray(data["bboxes"]).shape[0])
        else:
            raise ValueError(f"Could not infer total_frames from {path.name}")

        if "frame_idx" in data.files:
            # ragged
            return cls(
                total_frames=total_frames,
                ragged_frame_idx=np.asarray(data["frame_idx"]),
                ragged_bbox_xyxy=np.asarray(data["bboxes"]),
                ragged_conf=np.asarray(data["confidences"]) if "confidences" in data.files else None,
                ragged_track_id=np.asarray(data["track_ids"]) if "track_ids" in data.files else None,
                meta={k: data[k].item() if np.asarray(data[k]).shape == () else data[k] for k in data.files},
            )

        # dense
        return cls(
            total_frames=total_frames,
            dense_bbox_xyxy=np.asarray(data["bboxes"]) if "bboxes" in data.files else None,
            dense_conf=np.asarray(data["confidences"]) if "confidences" in data.files else None,
            dense_track_id=np.asarray(data["track_ids"]) if "track_ids" in data.files else None,
            meta={k: data[k].item() if np.asarray(data[k]).shape == () else data[k] for k in data.files},
        )

    @classmethod
    def from_custom_npy_triplet(
        cls,
        *,
        bboxes_npy: Union[str, Path],
        confidences_npy: Optional[Union[str, Path]] = None,
        track_ids_npy: Optional[Union[str, Path]] = None,
        metadata_npz: Optional[Union[str, Path]] = None,
        bbox_format: str = "xyxy",
    ) -> "Predictions":
        bboxes = np.load(bboxes_npy)

        conf = None
        if confidences_npy is not None:
            conf = np.load(confidences_npy)

        tids = None
        if track_ids_npy is not None:
            tids = np.load(track_ids_npy)

        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError("Expected bboxes shape (T,4).")

        bboxes = np.asarray(bboxes, dtype=np.float32)
        fmt = (bbox_format or "xyxy").lower()

        if fmt == "auto":
            finite = np.isfinite(bboxes).all(axis=1)
            if finite.any():
                x1 = bboxes[finite, 0]
                y1 = bboxes[finite, 1]
                x2 = bboxes[finite, 2]
                y2 = bboxes[finite, 3]
                invalid_xyxy = ((x2 <= x1) | (y2 <= y1)).mean()
                # Heuristic: if many rows violate x2>x1/y2>y1, assume xywh (YOLO-style).
                if invalid_xyxy > 0.30:
                    fmt = "xywh_center"
                else:
                    fmt = "xyxy"

        if fmt in {"xywh_center", "yolo"}:
            x, y, w, h = bboxes.T
            bboxes = np.stack([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], axis=1)
        elif fmt in {"xywh_tl", "xywh"}:
            x, y, w, h = bboxes.T
            bboxes = np.stack([x, y, x + w, y + h], axis=1)
        elif fmt != "xyxy":
            raise ValueError(f"Unsupported bbox_format={bbox_format!r}. Use 'xyxy', 'xywh_center', 'xywh_tl', or 'auto'.")
        total_frames = int(bboxes.shape[0])

        if conf is not None and conf.shape[0] != total_frames:
            raise ValueError("Expected confidences shape (T,) matching bboxes length.")

        if tids is not None and tids.shape[0] != total_frames:
            raise ValueError("Expected track_ids shape (T,) matching bboxes length.")

        meta = {}
        if metadata_npz is not None:
            md = np.load(metadata_npz, allow_pickle=True)
            meta = {k: md[k].item() if np.asarray(md[k]).shape == () else md[k] for k in md.files}
        return cls(
            total_frames=total_frames,
            dense_bbox_xyxy=bboxes,
            dense_conf=conf,
            dense_track_id=tids,
            meta=meta,
        )

    @classmethod
    def from_custom_csv_triplet(
        cls,
        *,
        bboxes_csv: Union[str, Path],
        confidences_csv: Optional[Union[str, Path]] = None,
        track_ids_csv: Optional[Union[str, Path]] = None,
        metadata_npz: Optional[Union[str, Path]] = None,
        expected_total_frames: Optional[int] = None,
        xywh_is_center: bool = False,
    ) -> "Predictions":
        """Load custom CSV exports.

        Supported bbox CSV schemas:
          1) columns: Frame, x1, y1, x2, y2  (xyxy)
          2) columns: Frame, x, y, w, h      (xywh; interpretation depends on `xywh_is_center`)

        Confidence CSV schema:
          - columns: Frame, Confidence (or conf/score)

        Track-id CSV schema (optional):
          - columns: Frame, TrackID / track_id / id

        Missing frames are filled with NaNs.
        """

        def _read_csv_table(path: Union[str, Path]) -> np.ndarray:
            path = Path(path)
            arr = np.genfromtxt(
                path,
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            if arr.size == 0:
                raise ValueError(f"CSV appears empty: {path}")
            # genfromtxt returns scalar for single-row files; normalize to 1D
            if arr.shape == ():
                arr = np.asarray([arr])
            return arr

        b = _read_csv_table(bboxes_csv)
        names = {n.lower(): n for n in b.dtype.names}

        if "frame" not in names:
            raise ValueError("BBox CSV must contain a 'Frame' column.")
        frame = np.asarray(b[names["frame"]], dtype=np.int64)

        def _col(*cands: str) -> Optional[np.ndarray]:
            for c in cands:
                if c.lower() in names:
                    return np.asarray(b[names[c.lower()]], dtype=np.float32)
            return None

        x1 = _col("x1")
        y1 = _col("y1")
        x2 = _col("x2")
        y2 = _col("y2")
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            bbox_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        else:
            # Try xywh
            x = _col("x")
            y = _col("y")
            w = _col("w", "width")
            h = _col("h", "height")
            if x is None or y is None or w is None or h is None:
                raise ValueError(
                    "BBox CSV must contain either (Frame,x1,y1,x2,y2) or (Frame,x,y,w,h)."
                )
            if xywh_is_center:
                bbox_xyxy = np.stack([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], axis=1)
            else:
                # x,y are top-left
                bbox_xyxy = np.stack([x, y, x + w, y + h], axis=1)

        if expected_total_frames is not None:
            total_frames = int(expected_total_frames)
        else:
            total_frames = int(frame.max() + 1)

        dense_bbox = np.full((total_frames, 4), np.nan, dtype=np.float32)
        valid = (frame >= 0) & (frame < total_frames)
        dense_bbox[frame[valid]] = bbox_xyxy[valid]

        # confidences (optional)
        dense_conf = None
        if confidences_csv is not None:
            ctab = _read_csv_table(confidences_csv)
            cnames = {n.lower(): n for n in ctab.dtype.names}
            if "frame" not in cnames:
                raise ValueError("Confidence CSV must contain a 'Frame' column.")
            cframe = np.asarray(ctab[cnames["frame"]], dtype=np.int64)
            ccol = None
            for cand in ("confidence", "conf", "score"):
                if cand in cnames:
                    ccol = np.asarray(ctab[cnames[cand]], dtype=np.float32)
                    break
            if ccol is None:
                raise ValueError("Confidence CSV must contain 'Confidence' (or conf/score) column.")
            dense_conf = np.full((total_frames,), np.nan, dtype=np.float32)
            cvalid = (cframe >= 0) & (cframe < total_frames)
            dense_conf[cframe[cvalid]] = ccol[cvalid]

        # track ids (optional)
        dense_tid = None
        if track_ids_csv is not None:
            itab = _read_csv_table(track_ids_csv)
            inames = {n.lower(): n for n in itab.dtype.names}
            if "frame" not in inames:
                raise ValueError("Track-id CSV must contain a 'Frame' column.")
            iframe = np.asarray(itab[inames["frame"]], dtype=np.int64)
            icol = None
            for cand in ("trackid", "track_id", "id"):
                if cand in inames:
                    icol = np.asarray(itab[inames[cand]], dtype=np.float32)
                    break
            if icol is None:
                raise ValueError("Track-id CSV must contain TrackID/track_id/id column.")
            dense_tid = np.full((total_frames,), np.nan, dtype=np.float32)
            ivalid = (iframe >= 0) & (iframe < total_frames)
            dense_tid[iframe[ivalid]] = icol[ivalid]

        meta = {}
        if metadata_npz is not None:
            md = np.load(metadata_npz, allow_pickle=True)
            meta = {k: md[k].item() if np.asarray(md[k]).shape == () else md[k] for k in md.files}

        return cls(
            total_frames=total_frames,
            dense_bbox_xyxy=dense_bbox,
            dense_conf=dense_conf,
            dense_track_id=dense_tid,
            meta=meta,
        )
