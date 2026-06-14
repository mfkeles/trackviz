"""
Periodogram (PSD) of the extracted new-behaviour confidence trace.

Replicates the cotrack method (run_cotrack.ipynb, commit ed1cc03):
  - zero-pad each trace to the next power of two,
  - scipy.signal.periodogram(data, fs=30)   (default detrend='constant'),
  - normalize the power by its max (per trace).

Here it is applied to the 1-min top-1 confidence trace of the "sequence"
event(s) in fig_new_behavior_confidence.py (e.g. twitch_20260320, frames
1115700–1117500). Output written next to the other panels in
<date>/fig_new_behavior/ .

Run:  .venv/bin/python fig_new_behavior_psd.py
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

import fig_new_behavior_confidence as M

FS = 30.0          # Hz
XLIM = (0.0, 5.0)  # Hz, display range (power concentrated ~1.5 Hz)


def normalized_periodogram(data: np.ndarray, fs: float = FS):
    """Zero-pad to next power of two, periodogram, normalize by max."""
    data = np.asarray(data, dtype=float)
    nfft = int(2 ** np.ceil(np.log2(len(data))))   # next power of two
    f, p = periodogram(data, fs, nfft=nfft)         # detrend='constant' (removes mean)
    p = p / p.max()
    return f, p


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Arial"
    import matplotlib.pyplot as plt

    for ev in M.EVENTS:
        if ev.get("kind") != "sequence":
            continue
        date_dir = M.DATA_ROOT / ev["date"]
        pkl = date_dir / ev["pkl"]
        out_dir = date_dir / "fig_new_behavior"
        out_dir.mkdir(exist_ok=True)
        lo, hi = ev["region"]

        print(f"\n=== PSD {ev['name']} [{lo},{hi}] ===")
        frames, conf, cls, _ = M.read_conf_window(pkl, lo, hi)
        data = conf.astype(float)
        nan = np.isnan(data)
        if nan.any():
            data[nan] = np.interp(frames[nan], frames[~nan], data[~nan])
            print(f"  interpolated {int(nan.sum())} missing frames")

        f, p = normalized_periodogram(data, FS)
        i_peak = int(np.argmax(p[f > 0.01]) + np.searchsorted(f, 0.01, "right"))
        f_peak = float(f[i_peak])
        print(f"  N={len(data)}  nfft={int(2**np.ceil(np.log2(len(data))))}"
              f"  df={FS/2**np.ceil(np.log2(len(data))):.4f} Hz")
        print(f"  dominant peak: {f_peak:.3f} Hz  (period {1/f_peak:.2f} s)")

        # save the spectrum as CSV
        np.savetxt(out_dir / f"{ev['name']}_psd.csv",
                   np.column_stack([f, p]), delimiter=",",
                   header="frequency_hz,normalized_power", comments="")

        # plot
        fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=150)
        ax.plot(f, p, color="#C2185B", lw=1.2)
        ax.scatter([f_peak], [p[i_peak]], s=18, color="#C2185B", zorder=5)
        ax.annotate(f"{f_peak:.2f} Hz\n({1/f_peak:.1f} s)",
                    (f_peak, p[i_peak]), textcoords="offset points",
                    xytext=(6, -2), fontsize=6, color="#C2185B", va="top")
        ax.set_xlim(*XLIM)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Normalized PSD", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{ev['name']}  ·  top-1 confidence PSD", fontsize=8, loc="left")
        fig.tight_layout()
        stub = out_dir / f"{ev['name']}_psd"
        for ext in ("png", "svg"):
            fig.savefig(stub.with_suffix(f".{ext}"), dpi=150,
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  saved {stub.name}.png/.svg + .csv")


if __name__ == "__main__":
    main()
