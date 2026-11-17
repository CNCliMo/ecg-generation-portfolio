"""W2 experiment P-E6: TSTR scaling -- how much does more synthetic data help?

For each route we have a large conditional synthetic set ({route}_cond_8k.npz).
At synthetic-sample sizes {500, 2000, 8000} we train a small 1D ResNet from
scratch on the synthetic data and evaluate it on the real PTB-XL test split
(TSTR). The control is the same architecture trained on the same number of
REAL training samples (CONTROL). We report macro-F1 / AUROC (mean over seeds)
and the TSTR/control ratio.

Usage:
  python experiments/eval_tstr_scaling.py [--routes c_full_v2 c_mask_v2 B_cond]
      [--sizes 500 2000 8000] [--seeds 3] [--epochs 30]

Outputs:
  results/metrics_tstr_scaling.csv
  results/logs/eval_tstr_scaling.json
  results/figures/fig_tstr_scaling.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.data import load_ptbxl_split
from ecg_common.logging_utils import get_logger, save_json
from models.classifier import build_classifier

LOG = get_logger("eval_tstr_scaling")
DEFAULT_ROUTES = ["c_full_v2", "c_mask_v2", "B_cond"]


def train_small(x_tr, y_tr, x_te, y_te, device, epochs, base, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_classifier(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = np.random.default_rng(seed)
    n = len(x_tr)
    bs = 128
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.as_tensor(x_tr[idx], device=device)
            yb = torch.as_tensor(y_tr[idx].astype(np.float32), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x_te), 256):
            xb = torch.as_tensor(x_te[i:i + 256], device=device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    probs = np.concatenate(probs, 0)
    pred = (probs >= 0.5).astype(int)
    y = y_te.astype(int)
    res = {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
           "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0))}
    try:
        res["macro_auroc"] = float(roc_auc_score(y, probs, average="macro"))
    except ValueError:
        res["macro_auroc"] = float("nan")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="*", default=DEFAULT_ROUTES)
    ap.add_argument("--sizes", nargs="*", type=int, default=[500, 2000, 8000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    te = load_ptbxl_split("test")
    tr = load_ptbxl_split("train")
    x_te, y_te = te["signals"], te["labels"]
    LOG.info("device=%s test=%s train=%s", device, x_te.shape, tr["signals"].shape)

    rows = []
    t0 = time.time()
    for route in args.routes:
        candidates = [os.path.join(paths.RESULTS, "generated_8k", f"{route}_8k_cond.npz"),
                      os.path.join(paths.RESULTS, "generated_8k", f"{route}_cond_8k.npz"),
                      os.path.join(paths.GENERATED, f"{route}_cond_8k.npz"),
                      os.path.join(paths.GENERATED, f"{route}_8k_cond.npz")]
        large = next((q for q in candidates if os.path.exists(q)), None)
        base_p = os.path.join(paths.GENERATED, f"{route}_cond.npz")
        if large:
            z = np.load(large, allow_pickle=False)
            xs, ys = z["signals"], z["labels"].astype(np.float32)
        elif os.path.exists(base_p):
            z = np.load(base_p, allow_pickle=False)
            xs, ys = z["signals"], z["labels"].astype(np.float32)
            LOG.warning("route %s: 8k set missing, using %d samples only", route, len(xs))
        else:
            LOG.warning("route %s: no conditional set, skipped", route)
            continue
        LOG.info("route %s: synthetic pool = %d", route, len(xs))
        for size in args.sizes:
            size_eff = min(size, len(xs))
            tstr_f1, ctrl_f1, tstr_auc, ctrl_auc = [], [], [], []
            for s in range(args.seeds):
                rng = np.random.default_rng(1000 + s)
                sel = rng.choice(len(xs), size=size_eff, replace=False)
                r1 = train_small(xs[sel], ys[sel], x_te, y_te, device,
                                 args.epochs, args.base, cfg["seed"] + s)
                ridx = rng.choice(len(tr["signals"]), size=size_eff, replace=False)
                r2 = train_small(tr["signals"][ridx], tr["labels"][ridx].astype(np.float32),
                                 x_te, y_te, device, args.epochs, args.base, cfg["seed"] + 100 + s)
                tstr_f1.append(r1["macro_f1"]); ctrl_f1.append(r2["macro_f1"])
                tstr_auc.append(r1["macro_auroc"]); ctrl_auc.append(r2["macro_auroc"])
                LOG.info("  %s size=%d seed=%d TSTR f1=%.4f | control f1=%.4f",
                         route, size_eff, s, r1["macro_f1"], r2["macro_f1"])
            for name, vals in (("tstr_macro_f1", tstr_f1), ("control_macro_f1", ctrl_f1),
                               ("tstr_macro_auroc", tstr_auc), ("control_macro_auroc", ctrl_auc)):
                rows.append({"route": route, "n_synth": size_eff, "metric": name,
                             "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                             "n_seeds": args.seeds})
            ratio = float(np.mean(tstr_f1) / max(np.mean(ctrl_f1), 1e-9))
            rows.append({"route": route, "n_synth": size_eff,
                         "metric": "tstr_control_ratio", "mean": ratio, "std": float("nan"),
                         "n_seeds": args.seeds})

    df = pd.DataFrame(rows)
    out = os.path.join(paths.RESULTS, "metrics_tstr_scaling.csv")
    df.to_csv(out, index=False)
    save_json(os.path.join(paths.LOGS, "eval_tstr_scaling.json"),
              {"routes": args.routes, "sizes": args.sizes, "seeds": args.seeds,
               "epochs": args.epochs, "elapsed_s": time.time() - t0})
    LOG.info("saved %s (%d rows, %.0fs)", out, len(df), time.time() - t0)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for route in args.routes:
            sub = df[(df["route"] == route) & (df["metric"] == "tstr_macro_f1")]
            if sub.empty:
                continue
            axes[0].errorbar(sub["n_synth"], sub["mean"], yerr=sub["std"], marker="o",
                             capsize=3, label=f"{route} TSTR")
            sub = df[(df["route"] == route) & (df["metric"] == "control_macro_f1")]
            axes[0].errorbar(sub["n_synth"], sub["mean"], yerr=sub["std"], marker="s",
                             ls="--", capsize=3, label=f"{route} control")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("synthetic training samples")
        axes[0].set_ylabel("macro F1 on real test")
        axes[0].set_title("TSTR scaling")
        axes[0].legend(fontsize=7)
        for route in args.routes:
            sub = df[(df["route"] == route) & (df["metric"] == "tstr_control_ratio")]
            if sub.empty:
                continue
            axes[1].plot(sub["n_synth"], sub["mean"], marker="o", label=route)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("synthetic training samples")
        axes[1].set_ylabel("TSTR / control")
        axes[1].set_title("utility ratio vs real-data training")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(paths.FIGURES, "fig_tstr_scaling.png"), dpi=150)
        plt.close(fig)
        LOG.info("saved fig_tstr_scaling.png")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("figure failed: %s", exc)


if __name__ == "__main__":
    main()
