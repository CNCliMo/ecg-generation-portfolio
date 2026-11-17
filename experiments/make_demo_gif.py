"""Create a short animated demo GIF + a still frame for the project README.

Frames:
  1        title / one-liner
  2..K     progressive drawing of Lead II: real vs generated (A / c_full_v2 / c_mask_v2)
  K..end   metric bars (MMD, PSD, QT Wasserstein) for 5 routes, growing
  last     hold frame (saved as demo_still.png)

Outputs:
  results/figures/demo.gif          (~5 s, 8 fps)
  results/figures/demo_still.png

Usage: python experiments/make_demo_gif.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config
from ecg_common.logging_utils import get_logger

LOG = get_logger("make_demo_gif")
ROUTES = [("A", "crimson", "A mechanistic"), ("c_full_v2", "tab:blue", "c_full_v2"),
          ("c_mask_v2", "tab:green", "c_mask_v2")]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cfg = load_config()
    fs = cfg["fs"]
    lead = cfg["lead_ii_index"]
    t = np.arange(cfg["n_samples"]) / fs

    real = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)["signals"][0]
    gens = {}
    for name, color, label in ROUTES:
        z = np.load(os.path.join(paths.GENERATED, f"{name}_cond.npz"), allow_pickle=False)
        gens[name] = (z["signals"][0], color, label)

    # metric bars (values from the evaluation tables, hard-coded to keep gif generation fast)
    route_names = ["A", "B_cond", "c_full_v2", "c_mask_v2", "c_aux05_v2"]
    mmd = [0.00262, 0.00058, 0.0005, 0.0008, 0.0008]
    psd = [0.03451, 0.01267, 0.0117, 0.0099, 0.0101]
    qtw = [14.88, 96.39, 85.15, 91.85, 85.66]
    metrics = [("MMD (lower better)", mmd, "{:.4f}"),
               ("PSD (lower better)", psd, "{:.4f}"),
               ("QT Wasserstein (ms, lower better)", qtw, "{:.1f}")]

    n_wave_frames = 26
    n_bar_frames = 16
    frames = []

    def render(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=95, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    # frame 1: title
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.axis("off")
    ax.text(0.5, 0.62, "ECG Generation & Diagnosis System (PTB-XL)", ha="center",
            fontsize=18, weight="bold")
    ax.text(0.5, 0.5, "4 generation routes  ·  3-layer evaluation  ·  Streamlit demo",
            ha="center", fontsize=12)
    ax.text(0.5, 0.36, "single 8GB laptop  ·  10 s · 12 leads · 100 Hz", ha="center",
            fontsize=11, style="italic")
    ax.text(0.5, 0.18, "results/figures/demo.gif", ha="center", fontsize=9, color="gray")
    frames.append(render(fig))

    # wave frames
    for k in range(1, n_wave_frames + 1):
        frac = k / n_wave_frames
        n_show = max(2, int(frac * len(t)))
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        ax.plot(t[:n_show], real[lead, :n_show], color="black", lw=1.2, label="real PTB-XL")
        for name, color, label in ROUTES:
            sig = gens[name][0]
            ax.plot(t[:n_show], sig[lead, :n_show], color=color, lw=1.0, alpha=0.9,
                    label=label)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Lead II (z-score)")
        ax.set_title("Lead II: real vs generated (synthetic signal drawn over time)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        frames.append(render(fig))

    # bar frames
    for k in range(1, n_bar_frames + 1):
        frac = k / n_bar_frames
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 5.4))
        for ax, (title, vals, fmt) in zip(axes, metrics):
            vmax = max(vals) if max(vals) > 0 else 1.0
            heights = [v * frac for v in vals]
            ax.bar(range(len(route_names)), heights, color="tab:blue")
            ax.set_xticks(range(len(route_names)))
            ax.set_xticklabels(route_names, rotation=35, ha="right", fontsize=7)
            ax.set_title(title, fontsize=9)
            for i, v in enumerate(vals):
                ax.text(i, heights[i] + 0.02 * vmax, fmt.format(v * frac), ha="center",
                        fontsize=7)
            ax.set_ylim(0, vmax * 1.18)
        fig.suptitle("Generated ECG quality (v1 + v2 + W2 routes)", fontsize=11)
        fig.tight_layout()
        frames.append(render(fig))

    # final still + hold frames
    still = frames[-1]
    still.save(os.path.join(paths.FIGURES, "demo_still.png"))
    for _ in range(8):
        frames.append(still)

    gif_path = os.path.join(paths.FIGURES, "demo.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=110, loop=0, optimize=True)
    LOG.info("saved %s (%.1f MB, %d frames)", gif_path,
             os.path.getsize(gif_path) / 1e6, len(frames))
    LOG.info("saved demo_still.png")


if __name__ == "__main__":
    main()
