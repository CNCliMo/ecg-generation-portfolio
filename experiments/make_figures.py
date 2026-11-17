"""Generate every comparison figure required by task book section 8/9.

Figures (results/figures/):
  fig_waveform_comparison.png   lead II of each route vs real reference (same condition)
  fig_12lead_grid.png           12-lead sample per route
  fig_fft_comparison.png        Lead II FFT spectra, all routes vs real
  fig_morphology_bars.png       MMD/DTW/PSD/PRD per route (ecgen-eval)
  fig_regularity_dist.png       QT/PR/QRS distributions per route vs references
  fig_crossdomain_delta.png     Delta_Wasserstein heat map (route x QT/PR/QRS/RR_cv)
  fig_class_distribution.png    predicted 5-class prevalence per route vs real test
  fig_ablation.png              route-C ablation comparison (regularity + functional)
  fig_classifier_training.png   classifier learning curve (written by train_classifier)
  fig_training_curves.png       diffusion/GAN training losses

Usage: python experiments/make_figures.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_common import paths
from ecg_common.config import load_config
from ecg_common.logging_utils import get_logger

LOG = get_logger("make_figures")
cfg = load_config()
CLASSES = cfg["superclasses"]
COLORS = plt.cm.tab10.colors
ROUTES_ORDER = ["A", "B_uncond", "B_cond", "c_full", "c_no_label", "c_no_hr",
                "c_label_noise", "c_label_shuffle", "d_seed42", "d_seed43", "d_seed44",
                "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"]


def _routes():
    found = set()
    for pattern in ("*_cond.npz", "*_uncond.npz"):
        for f in glob.glob(os.path.join(paths.GENERATED, pattern)):
            found.add(os.path.basename(f).replace("_cond.npz", "").replace("_uncond.npz", ""))
    return sorted(found, key=lambda r: ROUTES_ORDER.index(r) if r in ROUTES_ORDER else 99)


def _load(route):
    for suffix in ("_cond.npz", "_uncond.npz"):
        p = os.path.join(paths.GENERATED, route + suffix)
        if os.path.exists(p):
            return np.load(p, allow_pickle=False)
    return None


def fig_waveforms(routes):
    real = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)["signals"]
    picks = [r for r in routes]
    if not picks:
        return
    n = len(picks) + 1
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True)
    t = np.arange(cfg["n_samples"]) / cfg["fs"]
    axes[0].plot(t, real[0, cfg["lead_ii_index"]], color="black", lw=1.0)
    axes[0].set_ylabel("REAL", rotation=0, labelpad=25)
    for i, r in enumerate(picks, 1):
        z = _load(r)
        axes[i].plot(t, z["signals"][0, cfg["lead_ii_index"]], color=COLORS[i % 10], lw=1.0)
        axes[i].set_ylabel(r, rotation=0, labelpad=25, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Lead II: real reference vs generated (first sample of each route)")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = os.path.join(paths.FIGURES, "fig_waveform_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("saved %s", out)


def fig_12lead(routes):
    routes = routes[:6]
    real = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)["signals"]
    items = [("REAL", real[0])] + [(r, _load(r)["signals"][0]) for r in routes]
    ncol = 3
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow))
    t = np.arange(cfg["n_samples"]) / cfg["fs"]
    for ax, (name, sig) in zip(np.ravel(axes), items):
        for li in range(12):
            ax.plot(t, sig[li] + li * 6, lw=0.5, color="#111111")
        ax.set_yticks(np.arange(12) * 6)
        ax.set_yticklabels(cfg["lead_names"], fontsize=6)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.15)
    for ax in np.ravel(axes)[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(paths.FIGURES, "fig_12lead_grid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("saved %s", out)


def fig_fft(routes):
    real = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)["signals"]
    fig, ax = plt.subplots(figsize=(10, 4))

    def spec(x):
        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        s = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        f = np.fft.rfftfreq(len(x), 1 / cfg["fs"])
        return f, s / (s.max() + 1e-9)

    f, s = spec(real[0, cfg["lead_ii_index"]])
    ax.plot(f, s, color="black", lw=1.5, label="REAL")
    for i, r in enumerate(routes):
        f, s = spec(_load(r)["signals"][0, cfg["lead_ii_index"]])
        ax.plot(f, s, lw=0.8, alpha=0.8, label=r, color=COLORS[i % 10])
    ax.set_xlim(0, 40)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised |FFT|")
    ax.set_title("Lead II spectrum: real vs generated")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out = os.path.join(paths.FIGURES, "fig_fft_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("saved %s", out)


def fig_morphology():
    p = os.path.join(paths.RESULTS, "metrics_morphology.csv")
    if not os.path.exists(p):
        LOG.warning("skip morphology: %s missing", p)
        return
    df = pd.read_csv(p)
    metrics = ["mmd", "dtw", "psd", "prd"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, m in zip(axes.ravel(), metrics):
        sub = df[df["metric"] == m]
        if sub.empty:
            ax.axis("off")
            continue
        ax.bar(sub["route"], sub["value"], color=COLORS[:len(sub)])
        ax.set_title(f"{m} (lower is better)" if m != "dtw" else "dtw (lower is better)")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    out = os.path.join(paths.FIGURES, "fig_morphology_bars.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("saved %s", out)


def fig_regularity():
    p = os.path.join(paths.RESULTS, "metrics_regularity.csv")
    if not os.path.exists(p):
        LOG.warning("skip regularity: %s missing", p)
        return
    df = pd.read_csv(p)
    params = ["QT_ms", "PR_ms", "QRS_ms"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, param in zip(axes, params):
        sub = df[(df["param"] == param) & (df["metric"] == "wasserstein")]
        for dom, marker in (("in_domain", "o"), ("cross_domain", "s")):
            s2 = sub[sub["domain"] == dom]
            ax.plot(s2["route"], s2["value"], marker=marker, label=dom, lw=1)
        ax.set_title(f"Wasserstein distance -- {param}")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(paths.FIGURES, "fig_regularity_dist.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("saved %s", out)

    # delta heat map
    d = df[df["metric"] == "delta_wasserstein"]
    if not d.empty:
        piv = d.pivot_table(index="route", columns="param", values="value")
        fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(piv))))
        im = ax.imshow(piv.values, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(piv.columns)), piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
        ax.set_title("Delta_Wasserstein = cross_domain - in_domain (ms)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        out = os.path.join(paths.FIGURES, "fig_crossdomain_delta.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        LOG.info("saved %s", out)


def fig_class_distribution():
    p = os.path.join(paths.RESULTS, "class_distribution.csv")
    if not os.path.exists(p):
        LOG.warning("skip class distribution: %s missing", p)
        return
    df = pd.read_csv(p)
    real = df[df["route"] == "REAL_PTBXL_TEST"].set_index("class")["fraction"]
    gen = df[df["route"] != "REAL_PTBXL_TEST"]
    fig, ax = plt.subplots(figsize=(12, 4))
    wide = gen.pivot_table(index="route", columns="class", values="fraction").reindex(columns=CLASSES)
    x = np.arange(len(wide))
    w = 0.8 / (len(CLASSES) + 1)
    for i, c in enumerate(CLASSES):
        ax.bar(x + i * w, wide[c].values, width=w, label=c, color=COLORS[i])
    ax.scatter(x + (len(CLASSES) * w) / 2, [1] * len(x), alpha=0)  # spacing
    for i, c in enumerate(CLASSES):
        ax.axhline(real[c], color=COLORS[i], ls="--", lw=0.8, alpha=0.7)
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(wide.index, rotation=60, fontsize=7)
    ax.set_ylabel("predicted class prevalence")
    ax.set_title("Unconditional 5-class distribution: generated routes (bars) vs real test (dashed)")
    ax.legend(fontsize=7, ncol=5)
    ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    out = os.path.join(paths.FIGURES, "fig_class_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("saved %s", out)


def fig_ablation():
    p = os.path.join(paths.RESULTS, "metrics.csv")
    if not os.path.exists(p):
        LOG.warning("skip ablation: metrics.csv missing")
        return
    df = pd.read_csv(p)
    abl = df[df["route"].str.startswith("C_")]
    if abl.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # (1) regularity: mean Wasserstein across params
    reg = abl[(abl["metric_group"] == "regularity") & (abl["metric"] == "wasserstein") &
              (abl["domain"] == "in_domain")]
    if not reg.empty:
        mean = reg.groupby("route")["value"].mean().sort_values()
        axes[0].bar(mean.index, mean.values, color=COLORS[:len(mean)])
        axes[0].set_title("Route-C ablations: mean in-domain Wasserstein (QT/PR/QRS/RR)")
        axes[0].tick_params(axis="x", rotation=60, labelsize=7)
    # (2) morphology mmd
    mor = abl[(abl["metric_group"] == "morphology") & (abl["metric"] == "mmd")]
    if not mor.empty:
        axes[1].bar(mor["route"], mor["value"], color=COLORS[:len(mor)])
        axes[1].set_title("Route-C ablations: MMD")
        axes[1].tick_params(axis="x", rotation=60, labelsize=7)
    # (3) functional KL
    fn = abl[(abl["metric"] == "unconditional_class_KL")]
    if not fn.empty:
        axes[2].bar(fn["route"], fn["value"], color=COLORS[:len(fn)])
        axes[2].set_title("Route-C ablations: unconditional class KL")
        axes[2].tick_params(axis="x", rotation=60, labelsize=7)
    for ax in axes:
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    out = os.path.join(paths.FIGURES, "fig_ablation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("saved %s", out)


def fig_training_curves():
    files = sorted(glob.glob(os.path.join(paths.LOGS, "*_train.csv")))
    if not files:
        LOG.warning("skip training curves: no *_train.csv logs")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        if "train_loss" in df:
            ax.plot(df["epoch"], df["train_loss"], lw=1, label=os.path.basename(f).replace("_train.csv", ""))
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("diffusion training losses")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(paths.FIGURES, "fig_training_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("saved %s", out)


def main():
    routes = _routes()
    LOG.info("routes: %s", routes)
    if routes:
        fig_waveforms(routes)
        fig_12lead(routes)
        fig_fft(routes)
    fig_morphology()
    fig_regularity()
    fig_class_distribution()
    fig_ablation()
    fig_training_curves()


if __name__ == "__main__":
    main()
