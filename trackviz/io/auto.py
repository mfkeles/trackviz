from __future__ import annotations

from pathlib import Path
from typing import Optional

from trackviz.io.predictions import Predictions


def _match_video_pkl(cands: list, stem: str, run_id: str) -> Optional[Path]:
    """Choose the prediction pickle that belongs to a specific video.

    Detection/tracking files are named after their source video with an extra
    run suffix inserted, e.g. ``<video_stem>_ultra_<timestamp>_tracking.pkl``.
    The matching file is therefore the one whose stem *begins with the full
    video stem* (preferred) or the ``_raw``-stripped run id. Matching on the
    whole stem — instead of a single leading token — keeps sibling videos that
    share a prefix (``copy_…``) from colliding. ``cands`` order is preserved, so
    when one video has several prediction files the earlier (higher-priority)
    one wins.
    """
    for prefix in (stem, run_id):
        if not prefix:
            continue
        matches = [p for p in cands if p.stem.startswith(prefix)]
        if matches:
            return matches[0]
    return None


def autoload_predictions(
    video_path: Path,
    total_frames: Optional[int] = None,
    preds_hint: Optional[Path] = None,
) -> Predictions:
    """Locate and load predictions for *video_path* with no user input.

    Args:
        video_path:   Path to the source video file.
        total_frames: Frame count of the video, used to validate/align predictions.
                      Pass ``None`` if unknown.
        preds_hint:   Optional explicit path to a predictions file or directory.

                      * File with ``.pkl`` extension  → loaded as a YOLO/flyloop pickle.
                      * File with ``.npz`` extension  → loaded as a standard NPZ.
                      * File with ``.npy`` extension  → loaded as a results array.
                      * Directory                     → used as the search root instead of
                        ``video_path.parent``.
                      * ``None``                      → search ``video_path.parent``
                        automatically (same behaviour as the GUI drag-and-drop).

    Returns:
        A :class:`~trackviz.io.predictions.Predictions` instance.

    Raises:
        SystemExit: When no supported prediction file can be found.
        RuntimeError: When *preds_hint* points to an unsupported file type.
    """

    # ------------------------------------------------------------------ #
    # Explicit file hint — dispatch by extension                           #
    # ------------------------------------------------------------------ #
    if preds_hint is not None and preds_hint.is_file():
        ext = preds_hint.suffix.lower()
        if ext == ".pkl":
            return Predictions.from_yolo_pickle(
                preds_hint, expected_total_frames=total_frames
            )
        if ext == ".npz":
            return Predictions.from_npz(preds_hint)
        if ext == ".npy":
            return Predictions.from_results_npy(
                preds_hint, expected_total_frames=total_frames
            )
        raise RuntimeError(
            f"Unsupported predictions file type: {preds_hint.suffix!r}. "
            "Expected .pkl, .npz, or .npy."
        )

    # ------------------------------------------------------------------ #
    # Directory search                                                     #
    # ------------------------------------------------------------------ #
    root = (
        preds_hint
        if (preds_hint is not None and preds_hint.is_dir())
        else video_path.parent
    )
    stem = video_path.stem

    def _pick(cands: list) -> Optional[Path]:
        for c in cands:
            if Path(c).exists():
                return Path(c)
        return None

    # Strip common suffix so run_id matches the predictions stem
    run_id = stem[:-4] if stem.endswith("_raw") else stem

    # 1. Flyloop / YOLO pickle  (.pkl)
    yolo_pkl = _pick([
        root / f"{run_id}_yolo_fast.pkl",
        root / f"{stem}_yolo_fast.pkl",
        root / "yolo_fast.pkl",
        root / "results" / f"{run_id}_yolo_fast.pkl",
        root / "results" / f"{stem}_yolo_fast.pkl",
        root / "results" / "yolo_fast.pkl",
        root / ".." / "results" / f"{run_id}_yolo_fast.pkl",
        root / f"{run_id}_tracking.pkl",
        root / f"{stem}_tracking.pkl",
        root / "tracking.pkl",
    ])

    if not yolo_pkl:
        possible_pkls = (
            list(root.glob("*_yolo_fast.pkl"))
            + list(root.glob("*_tracking.pkl"))
        )
        if not possible_pkls:
            possible_pkls = (
                list(root.glob("results/*_yolo_fast.pkl"))
                + list(root.glob("results/*_tracking.pkl"))
            )
        if len(possible_pkls) == 1:
            yolo_pkl = possible_pkls[0]
        elif len(possible_pkls) > 1:
            yolo_pkl = _match_video_pkl(possible_pkls, stem, run_id)

    if yolo_pkl:
        return Predictions.from_yolo_pickle(
            yolo_pkl, expected_total_frames=total_frames
        )

    # 2. Results npy  (_results.npy / results.npy)
    results_npy = _pick([
        root / f"{stem}_results.npy",
        root / "results.npy",
    ])
    if results_npy:
        return Predictions.from_results_npy(
            results_npy, expected_total_frames=total_frames
        )

    # 3. Standard NPZ
    npz_file = _pick([
        root / f"{stem}.npz",
        root / f"{stem}_preds.npz",
        root / "predictions.npz",
        root / "preds.npz",
    ])
    if npz_file:
        return Predictions.from_npz(npz_file)

    # 4. Triplet  (bboxes + optional conf / track_ids)
    bbox = _pick([
        root / f"{stem}_bboxes.npy",
        root / f"{stem}_bboxes.csv",
        root / "bboxes.npy",
        root / "bboxes.csv",
    ])
    if bbox is None:
        raise SystemExit(
            f"Could not find any supported prediction files for: {video_path}\n"
            "Expected a .pkl, .npz, .npy, or bboxes.npy/csv in the same directory."
        )

    conf = _pick([
        root / f"{stem}_confidences.npy",
        root / f"{stem}_confidences.csv",
        root / "confidences.npy",
        root / "confidences.csv",
        root / "conf.npy",
        root / "conf.csv",
    ])
    tids = _pick([
        root / f"{stem}_track_ids.npy",
        root / f"{stem}_track_ids.csv",
        root / "track_ids.npy",
        root / "track_ids.csv",
        root / "ids.npy",
        root / "ids.csv",
    ])
    meta = _pick([
        root / f"{stem}_metadata.npz",
        root / "metadata.npz",
    ])

    if bbox.suffix.lower() == ".npy":
        return Predictions.from_custom_npy_triplet(
            bboxes_npy=bbox,
            confidences_npy=conf if conf and conf.suffix.lower() == ".npy" else None,
            track_ids_npy=tids if tids and tids.suffix.lower() == ".npy" else None,
            metadata_npz=meta,
            bbox_format="auto",
        )

    if bbox.suffix.lower() == ".csv":
        return Predictions.from_custom_csv_triplet(
            bboxes_csv=bbox,
            confidences_csv=conf if conf and conf.suffix.lower() == ".csv" else None,
            track_ids_csv=tids if tids and tids.suffix.lower() == ".csv" else None,
            metadata_npz=meta,
            expected_total_frames=total_frames,
            xywh_is_center=True,
        )

    raise SystemExit(f"Unsupported bbox format: {bbox.suffix!r}")
