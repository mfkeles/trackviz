"""
New-behavior detection panel — confidence drop + motion heatmaps.

Narrative: the 6-class YOLO behaviour classifier (ProbPumping, Moving, Grooming,
Feeding, Quiescent, HaltereSwitch) was never trained on *twitching* or
*defecation*. When the fly performs one of these novel behaviours the model's
top-1 confidence drops. For each curated event this script produces:

  1. <name>_conf_trace.{png,svg}  — top-1 confidence vs time around the event.
  2. <name>_frame<idx>_plain.svg  — raw frame with the prediction box / conf.
  3. <name>_frame<idx>_heatmap.svg — motion-heatmap composite (same technique as
     fig1_example_static_vs_heatmap.py) so the twitch trail / fecal pellet is
     visible at the moment the confidence dips.

Outputs are written to  <date_dir>/fig_new_behavior/ .

─────────────────────────────────────────
Run (from repo root):
    .venv/bin/python fig_new_behavior_confidence.py
─────────────────────────────────────────
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Data root ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path(
    "/Volumes/Lab_Files3/mfk/projects/2025_flyvista2/PE_Origins_Project/"
    "ClosedLoopArousal/Data"
)

# ── Heatmap settings (identical to fig1_example_static_vs_heatmap.py) ───────────
WINDOW_SIZE  = 6     # frames in sliding window (inclusive of target)
THRESHOLD    = 10    # min pixel-intensity change to count as motion (0–255)
MOTION_ALPHA = 0.5   # heatmap weight on motion pixels (plain frame = 1 − this)

# ── Confidence-trace window half-widths (frames; 30 fps) ───────────────────────
HALFWIN_TWITCH = 2700   # ±90 s  → ~3 min total
HALFWIN_DEFEC  = 450    # ±15 s  → ~30 s total
ROLL_FRAMES    = 31     # rolling-mean window (~1 s) for the smoothed overlay
FPS_FALLBACK   = 30.0

# ── Sequence settings ──────────────────────────────────────────────────────────
# For a "sequence" event we plot a confidence trace + predicted-class strip over
# a region and render a fine, frame-by-frame strip (every Nth frame over a short
# span), marking the highlighted span on the trace for editing in Illustrator.

# Behavior colours (Okabe-Ito; see behavior_colors.md), keyed by class id.
CLASS_HEX = {
    0: "#E69F00",  # ProbPumping
    1: "#56B4E9",  # Moving
    2: "#009E73",  # Grooming
    3: "#D55E00",  # Feeding
    4: "#0072B2",  # Quiescent
    5: "#CC79A7",  # HaltereSwitch
}
NODET_HEX = "#DDDDDD"  # frames with no detection

# ── Event configuration ────────────────────────────────────────────────────────
# Twitching: event is a span; the heatmap is rendered at the lowest-confidence
# frame inside the span. Defecation: event is a single annotated frame.
EVENTS: List[dict] = [
    # ---- Twitching --------------------------------------------------------------
    {
        "name": "twitch_20260320",
        "date": "20260320",
        "pkl": "20260320_175222_yolo_fast.pkl",
        "video": "20260320_200000_raw.mp4",
        "kind": "sequence",
        "region": (1115700, 1117500),  # 1 min trace (1800 frames @ 30 fps)
        "seq_start": 1116599,
        "seq_stop": 1116616,
        "seq_step": 4,                 # → 1116599, 1116603, 1116607, 1116611, 1116615
        "highlight": (1116603, 1116607),  # span marked on the trace
        "ylim": (0.65, 1.0),  # window min is 0.693 (frame 1116610); 0.7 clipped it
    },
    {
        "name": "twitch_20260319",
        "date": "20260319",
        "pkl": "20260319_183419_yolo_fast.pkl",
        "video": "20260319_200000_raw.mp4",
        "kind": "twitch",
        "span": (831300, 832300),
    },
    # ---- Defecation (all 20260316) ---------------------------------------------
    {
        "name": "defec_pellet_visible",
        "date": "20260316",
        "pkl": "20260316_172528_yolo_fast.pkl",
        "video": "20260316_200000_raw.mp4",
        "kind": "defec",
        "frame": 71160,
        "category": "round pellet visible at abdomen tip",
    },
    {
        "name": "defec_while_walking",
        "date": "20260316",
        "pkl": "20260316_172528_yolo_fast.pkl",
        "video": "20260316_200000_raw.mp4",
        "kind": "defec",
        "frame": 325136,
        "category": "defecates while walking past",
    },
    {
        "name": "defec_whole_body",
        "date": "20260316",
        "pkl": "20260316_172528_yolo_fast.pkl",
        "video": "20260316_200000_raw.mp4",
        "kind": "defec",
        "frame": 1523326,
        "category": "whole-body movement",
    },
    {
        "name": "defec_smeared_window",
        "date": "20260316",
        "pkl": "20260316_172528_yolo_fast.pkl",
        "video": "20260316_200000_raw.mp4",
        "kind": "defec",
        "frame": 823348,  # DF annotation (user note said 823353)
        "category": "feces smeared on window",
    },
    {
        "name": "defec_placed_floor",
        "date": "20260316",
        "pkl": "20260316_172528_yolo_fast.pkl",
        "video": "20260316_200000_raw.mp4",
        "kind": "defec",
        "frame": 7195,
        "category": "places/drags feces on floor",
    },
    # Defecation as a sequence: 20 s trace + fine frame strip, to show how much
    # the predictions vary during an unknown behaviour.
    {
        "name": "defec_series_20260316",
        "date": "20260316",
        "pkl": "20260316_172528_yolo_fast.pkl",
        "video": "20260316_200000_raw.mp4",
        "kind": "sequence",
        "region": (70800, 71400),       # 20 s @ 30 fps
        "seq_start": 71152,
        "seq_stop": 71185,
        "seq_step": 5,                  # → 71152,71157,71162,71167,71172,71177,71182
        "highlight": (71152, 71185),
        "class_panel": "hypnogram",     # line plot instead of colour strip
        # no ylim → full 0–1 to show the confidence swing
    },
]


# ── Windowed prediction reader ─────────────────────────────────────────────────

def read_conf_window(
    pkl_path: Path, lo: int, hi: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, list]]:
    """Read predictions for frames in [lo, hi] from a yolo_fast pickle stream.

    The stream is ordered by frame, so we skip frames < lo and stop once past hi.
    Returns:
        frames : (M,) int     — every frame in [lo, hi] (gaps filled, dense)
        conf   : (M,) float   — top-1 confidence per frame (NaN if no detection)
        cls    : (M,) int     — top-1 class id per frame (-1 if no detection)
        dets_by_frame : {frame: [(bbox_xyxy, conf, cls), ...]} for all detections
    """
    from trackviz.io.predictions import Detection  # local import (lazy)

    top_conf: Dict[int, float] = {}
    top_cls: Dict[int, int] = {}
    dets_by_frame: Dict[int, List[Detection]] = {}

    with open(pkl_path, "rb") as fh:
        while True:
            try:
                d = pickle.load(fh)
            except EOFError:
                break
            if "frame_seq" in d:
                fi = int(d["frame_seq"]) - 1
            else:
                fi = int(d["frame_idx"])
            if fi < lo:
                continue
            if fi > hi:
                break

            boxes = d["boxes"]
            n = len(boxes)
            if n == 0:
                continue
            confs = np.asarray(d["confs"], dtype=float)
            classes = np.asarray(d["classes"], dtype=float)

            dets: List[Detection] = []
            for i in range(n):
                b = np.asarray(boxes[i], dtype=float)
                dets.append(
                    Detection(
                        frame=fi,
                        bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        confidence=float(confs[i]),
                        cls=int(classes[i]),
                    )
                )
            dets_by_frame[fi] = dets

            k = int(np.argmax(confs))
            top_conf[fi] = float(confs[k])
            top_cls[fi] = int(classes[k])

    frames = np.arange(lo, hi + 1, dtype=int)
    conf = np.array([top_conf.get(f, np.nan) for f in frames], dtype=float)
    cls = np.array([top_cls.get(f, -1) for f in frames], dtype=int)
    return frames, conf, cls, dets_by_frame


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    """NaN-aware centered rolling mean."""
    if win <= 1:
        return x
    half = win // 2
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        seg = x[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


# ── Confidence trace plot ──────────────────────────────────────────────────────

def plot_confidence_trace(
    frames: np.ndarray,
    conf: np.ndarray,
    fps: float,
    center: int,
    event: dict,
    out_stub: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Arial"
    import matplotlib.pyplot as plt

    t = (frames - center) / fps  # seconds relative to event center
    smooth = _rolling_mean(conf, ROLL_FRAMES)

    fig, ax = plt.subplots(figsize=(5.0, 2.2), dpi=150)

    # Event marker(s)
    if event["kind"] == "twitch":
        lo, hi = event["span"]
        ax.axvspan((lo - center) / fps, (hi - center) / fps,
                   color="#D84315", alpha=0.12, lw=0, zorder=0,
                   label="annotated twitching")
    else:
        ax.axvline(0.0, color="#D84315", lw=1.0, alpha=0.7, zorder=1,
                   label="defecation frame")

    # Confidence
    ax.plot(t, conf, color="#9e9e9e", lw=0.6, alpha=0.8, zorder=2,
            label="top-1 confidence")
    ax.plot(t, smooth, color="#1565C0", lw=1.6, zorder=3,
            label=f"rolling mean ({ROLL_FRAMES} fr)")

    # Min-confidence annotation (within the event window for twitch; global otherwise)
    if event["kind"] == "twitch":
        lo, hi = event["span"]
        mask = (frames >= lo) & (frames <= hi)
    else:
        mask = np.ones_like(frames, dtype=bool)
    if np.any(mask & ~np.isnan(conf)):
        sub_idx = np.where(mask & ~np.isnan(conf))[0]
        j = sub_idx[np.nanargmin(conf[sub_idx])]
        ax.scatter([t[j]], [conf[j]], s=18, color="#D84315", zorder=4)
        ax.annotate(f"min {conf[j]:.2f}\nframe {frames[j]}",
                    (t[j], conf[j]), textcoords="offset points", xytext=(6, -2),
                    fontsize=6, color="#D84315", va="top")

    ax.set_xlabel("time relative to event (s)", fontsize=8)
    ax.set_ylabel("top-1 confidence", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    title = event["name"]
    if event.get("category"):
        title += f"  ·  {event['category']}"
    ax.set_title(title, fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()

    for ext in ("png", "svg"):
        fig.savefig(out_stub.with_suffix(f".{ext}"), dpi=150,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ── Heatmap generator (motion-mask blend; copied from fig1 example) ────────────

def generate_heatmap(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    """Return a BGR heatmap composite for *idx* using motion-mask blending.

    Static pixels keep full original brightness; motion pixels are blended
    (MOTION_ALPHA) with the HOT colormap. Identical to the fig1 example so the
    new-behaviour panels match the methods figure.
    """
    gray_frames: List[np.ndarray] = []
    for i in range(max(0, idx - WINDOW_SIZE + 1), idx + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok and frame is not None:
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if not gray_frames:
        return None

    ref_gray = gray_frames[-1]
    h, w = ref_gray.shape
    accumulator = np.zeros((h, w), dtype=np.float32)

    for g in gray_frames[:-1]:
        diff = cv2.absdiff(ref_gray, g)
        _, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        accumulator += mask

    mx = accumulator.max()
    acc_u8 = (
        (accumulator * (255.0 / mx)).astype(np.uint8)
        if mx > 0
        else np.zeros((h, w), dtype=np.uint8)
    )

    heatmap = cv2.applyColorMap(acc_u8, cv2.COLORMAP_HOT)
    ref_bgr = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
    motion_mask = (acc_u8 > 0).astype(np.float32)[:, :, np.newaxis]
    blended = cv2.addWeighted(ref_bgr, 1 - MOTION_ALPHA, heatmap, MOTION_ALPHA, 0)
    return (blended * motion_mask + ref_bgr * (1 - motion_mask)).astype(np.uint8)


# ── Sequence processing (1-min trace + fine frame strip) ───────────────────────

def process_sequence(ev, pkl, out_dir, cap, fps, style, save_snapshot) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Arial"
    import matplotlib.pyplot as plt

    region_lo, region_hi = ev["region"]
    picked = list(range(ev["seq_start"], ev["seq_stop"] + 1, ev["seq_step"]))

    print(f"  reading predictions [{region_lo}, {region_hi}] …")
    frames, conf, cls, dets_by_frame = read_conf_window(pkl, region_lo, region_hi)

    # ---- confidence trace (top) + predicted-class strip (bottom) -------------
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from trackviz.io.class_names import BEHAVIOR_NAMES

    t0 = region_lo
    t = (frames - t0) / fps
    smooth = _rolling_mean(conf, ROLL_FRAMES)
    ylo, yhi = ev.get("ylim", (0.0, 1.02))
    hl0, hl1 = ev["highlight"]
    x0, x1 = (hl0 - t0) / fps, (hl1 - t0) / fps
    xc = 0.5 * (x0 + x1)

    panel = ev.get("class_panel", "strip")
    if panel == "hypnogram":
        fig, (ax, axc) = plt.subplots(
            2, 1, figsize=(7.0, 3.5), dpi=150, sharex=True,
            constrained_layout=True, gridspec_kw=dict(height_ratios=[3, 2]))
    else:
        fig, (ax, axc) = plt.subplots(
            2, 1, figsize=(7.0, 3.1), dpi=150, sharex=True,
            constrained_layout=True, gridspec_kw=dict(height_ratios=[4, 1]))

    # top: confidence
    ax.plot(t, conf, color="#9e9e9e", lw=0.5, alpha=0.8, zorder=2,
            label="top-1 confidence")
    ax.plot(t, smooth, color="#1565C0", lw=1.4, zorder=3,
            label=f"rolling mean ({ROLL_FRAMES} fr)")
    ax.axvspan(x0, x1, color="#D84315", alpha=0.18, lw=0, zorder=1)
    ax.axvline(x0, color="#D84315", lw=0.6, alpha=0.8, zorder=4)
    ax.axvline(x1, color="#D84315", lw=0.6, alpha=0.8, zorder=4)
    ax.set_ylabel("top-1 confidence", fontsize=8)
    ax.set_ylim(ylo, yhi)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=2)

    # class-lability metric within the highlighted span
    hmask = (frames >= hl0) & (frames <= hl1)
    hc = cls[hmask]
    n_sw = int(np.sum(hc[1:] != hc[:-1])) if hc.size > 1 else 0
    uniq = sorted({int(x) for x in hc if x >= 0})
    ax.annotate(f"{hl0}–{hl1}: {n_sw} class switches, {len(uniq)} classes",
                xy=(xc, yhi - 0.02), xytext=(xc, ylo + 0.04), ha="center",
                va="bottom", fontsize=6, color="#D84315",
                arrowprops=dict(arrowstyle="-", color="#D84315", lw=0.6))

    # bottom: predicted-class panel (hypnogram line or colour strip)
    axc.axvline(x0, color="#D84315", lw=0.6, alpha=0.9, zorder=5)
    axc.axvline(x1, color="#D84315", lw=0.6, alpha=0.9, zorder=5)
    axc.set_xlabel(f"time since frame {region_lo} (s)", fontsize=8)
    axc.tick_params(labelsize=7)
    axc.set_xlim(t[0], t[-1])

    if panel == "hypnogram":
        # custom top→bottom order: Quiescent, ProbPumping, Grooming, Moving
        order = ev.get("hypnogram_order", [4, 0, 2, 1])
        ypos = {cid: (len(order) - 1 - i) for i, cid in enumerate(order)}
        missing = sorted({int(c) for c in cls if c >= 0} - set(order))
        if missing:
            print(f"  [warn] classes not in hypnogram_order (hidden): "
                  f"{[BEHAVIOR_NAMES[m] for m in missing]}")
        yv = np.array([ypos.get(int(c), np.nan) for c in cls], dtype=float)
        axc.step(t, yv, where="post", color="#333333", lw=0.9, zorder=2)
        axc.set_yticks([ypos[c] for c in order])
        axc.set_yticklabels([BEHAVIOR_NAMES[c] for c in order], fontsize=6)
        for tick, c in zip(axc.get_yticklabels(), order):
            tick.set_color(CLASS_HEX[c])
        axc.set_ylim(-0.5, len(order) - 0.5)
        axc.spines[["top", "right"]].set_visible(False)
    else:
        # colour-strip ethogram (one column per frame)
        rgb = np.zeros((1, len(cls), 3))
        for k, c in enumerate(cls):
            rgb[0, k] = mcolors.to_rgb(CLASS_HEX.get(int(c), NODET_HEX))
        axc.imshow(rgb, extent=[t[0], t[-1], 0, 1], aspect="auto",
                   interpolation="nearest", zorder=1)
        axc.set_yticks([])
        axc.set_ylabel("pred.\nclass", fontsize=7, rotation=0, ha="right", va="center")
        present = [c for c in range(6) if np.any(cls == c)]
        handles = [mpatches.Patch(color=CLASS_HEX[c], label=BEHAVIOR_NAMES[c])
                   for c in present]
        if np.any(cls < 0):
            handles.append(mpatches.Patch(color=NODET_HEX, label="no detection"))
        axc.legend(handles=handles, fontsize=5.5, frameon=False,
                   ncol=min(len(handles), 4), loc="upper center",
                   bbox_to_anchor=(0.5, -0.6), handlelength=1.0, columnspacing=1.0)

    dur_s = (region_hi - region_lo) / fps
    dur_lbl = f"{dur_s / 60:.0f} min" if dur_s >= 60 and dur_s % 60 == 0 else f"{dur_s:.0f} s"
    ax.set_title(f"{ev['name']}  ·  {dur_lbl}, frame strip ×{len(picked)}",
                 fontsize=8, loc="left")
    stub = out_dir / f"{ev['name']}_conf_trace"
    for ext in ("png", "svg"):
        fig.savefig(stub.with_suffix(f".{ext}"), dpi=150,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- fine frame strip (plain + heatmap per picked frame) ------------------
    print(f"  picked frames: {picked}")
    for fr in picked:
        idx = fr - region_lo
        c = conf[idx] if (0 <= idx < len(conf)) else np.nan
        print(f"    f{fr}  conf={c:.2f}" if not np.isnan(c) else f"    f{fr}  (no det)")
        dets = dets_by_frame.get(fr, [])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, plain = cap.read()
        base = f"{ev['name']}_f{fr:07d}"
        if ok and plain is not None:
            save_snapshot(plain, dets, None, fr, style,
                          out_dir / f"{base}_plain.svg")
        hm = generate_heatmap(cap, fr)
        if hm is not None:
            save_snapshot(hm, dets, None, fr, style,
                          out_dir / f"{base}_heatmap.svg")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from trackviz.io.class_names import resolve_class_names
    from trackviz.render.overlay import OverlayStyle
    from trackviz.render.snapshot import save_snapshot

    style = OverlayStyle(show_confidence=True, show_track_id=False, show_class=True)
    style.class_names = resolve_class_names(None)  # canonical 8-name list

    for ev in EVENTS:
        date_dir = DATA_ROOT / ev["date"]
        pkl = date_dir / ev["pkl"]
        video = date_dir / ev["video"]
        out_dir = date_dir / "fig_new_behavior"
        out_dir.mkdir(exist_ok=True)

        print(f"\n=== {ev['name']} ({ev['kind']}) ===")
        if not pkl.exists() or not video.exists():
            print(f"  [skip] missing pkl or video:\n    {pkl}\n    {video}")
            continue

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"  [skip] cannot open video: {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK

        # Sequence events: 1-min trace + fine frame strip (handled separately)
        if ev["kind"] == "sequence":
            process_sequence(ev, pkl, out_dir, cap, fps, style, save_snapshot)
            cap.release()
            continue

        # Event center + trace window (span/frame events)
        if ev["kind"] == "twitch":
            lo_e, hi_e = ev["span"]
            center = (lo_e + hi_e) // 2
            half = HALFWIN_TWITCH
        else:
            center = ev["frame"]
            half = HALFWIN_DEFEC
        lo = max(0, center - half)
        hi = center + half

        # 1) Confidence trace
        print(f"  reading predictions [{lo}, {hi}] …")
        frames, conf, cls, dets_by_frame = read_conf_window(pkl, lo, hi)
        plot_confidence_trace(frames, conf, fps, center, ev,
                              out_dir / f"{ev['name']}_conf_trace")
        n_valid = int(np.sum(~np.isnan(conf)))
        print(f"  trace: {n_valid}/{len(frames)} frames with a detection")

        # 2) Heatmap target frame
        if ev["kind"] == "twitch":
            mask = (frames >= lo_e) & (frames <= hi_e) & ~np.isnan(conf)
            sub = np.where(mask)[0]
            target = int(frames[sub[np.nanargmin(conf[sub])]]) if sub.size else center
        else:
            target = ev["frame"]

        dets = dets_by_frame.get(target, [])

        # plain frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, plain = cap.read()
        if ok and plain is not None:
            plain_out = out_dir / f"{ev['name']}_frame{target:07d}_plain.svg"
            save_snapshot(plain, dets, None, target, style, plain_out)
            print(f"  saved {plain_out.name}")
        else:
            print(f"  [warn] could not read plain frame {target}")

        # heatmap frame
        hm = generate_heatmap(cap, target)
        if hm is not None:
            hm_out = out_dir / f"{ev['name']}_frame{target:07d}_heatmap.svg"
            save_snapshot(hm, dets, None, target, style, hm_out)
            print(f"  saved {hm_out.name}")
        else:
            print(f"  [warn] could not generate heatmap for frame {target}")

        cap.release()

    print("\nDone.")


if __name__ == "__main__":
    main()
