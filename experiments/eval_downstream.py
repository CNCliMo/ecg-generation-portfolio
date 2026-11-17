"""Downstream-utility evaluation of the generated ECG (TSTR / TRTS), v2.

Motivation (SSSD-ECG / SSSD-ECG-nle): distribution metrics (MMD/DTW/PSD/PRD) do
not tell whether synthetic data is *useful*. This script measures two standard
downstream transfer protocols:

  TSTR  "train on synthetic, test on real": train a fresh classifier on the
        conditional generations of a route (labels = the conditioning labels),
        evaluate it on the real PTB-XL test split.
  CONTROL  same architecture / epochs / seed but trained on a same-size random
        subset of the real training split -> the reference level for TSTR.
  TRTS  "train on real, test on synthetic": the trained diagnostic classifier
        (classifier_best_v2.pt if present, else classifier_best.pt) predicts
        labels of the conditional generations.

Only label-conditioned routes can be evaluated (the others have no label
supervision for TSTR / TRTS); all routes are listed in the CSV for auditing.

Outputs:
  results/metrics_downstream.csv
  results/logs/eval_downstream.json
  results/figures/fig_downstream.png

Usage: python experiments/eval_downstream.py [--routes ...] [--max-train 2000] [--epochs 30]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.data import load_ptbxl_split
from ecg_common.logging_utils import get_logger, save_json
from models.classifier import build_classifier

LOG = get_logger("eval_downstream")
ROUTES_ORDER = ["A", "B_uncond", "B_cond", "c_full", "c_no_label", "c_no_hr",
                "c_label_noise", "c_label_shuffle", "d_seed42", "d_seed43", "d_seed44",
                "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"]
LABEL_CONDITIONED = {"B_cond", "c_full", "c_no_hr", "c_label_noise", "c_label_shuffle",
                     "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"}


def discover_routes(only=None):
    found = set()
    for pattern in ("*_cond.npz",):
        for f in glob.glob(os.path.join(paths.GENERATED, pattern)):
            found.add(os.path.basename(f).replace("_cond.npz", ""))
    routes = sorted(found, key=lambda r: ROUTES_ORDER.index(r) if r in ROUTES_ORDER else 99)
    return [r for r in routes if not only or r in only]


def train_eval_model(x_tr, y_tr, x_te, y_te, base, epochs, seed, device, bs=128, lr=1e-3):
    """Train a small 1D ResNet from scratch and evaluate on the real test split."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_classifier(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = np.random.default_rng(seed)
    n = len(x_tr)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.as_tensor(x_tr[idx], device=device)
            yb = torch.as_tensor(y_tr[idx].astype(np.float32), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
    model.eval()
    probs, losses = [], []
    with torch.no_grad():
        for i in range(0, len(x_te), 256):
            xb = torch.as_tensor(x_te[i:i + 256], device=device)
            out = model(xb)
            probs.append(torch.sigmoid(out).cpu().numpy())
    probs = np.concatenate(probs, 0)
    pred = (probs >= 0.5).astype(int)
    y = y_te.astype(int)
    res = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
    }
    try:
        res["macro_auroc"] = float(roc_auc_score(y, probs, average="macro"))
        res["macro_ap"] = float(average_precision_score(y, probs, average="micro"))
    except ValueError:
        res["macro_auroc"] = float("nan")
        res["macro_ap"] = float("nan")
    return res


def evaluate_pretrained(probs, y):
    pred = (probs >= 0.5).astype(int)
    y = y.astype(int)
    res = {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
           "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0))}
    try:
        res["macro_auroc"] = float(roc_auc_score(y, probs, average="macro"))
        res["macro_ap"] = float(average_precision_score(y, probs, average="micro"))
    except ValueError:
        res["macro_auroc"] = float("nan")
        res["macro_ap"] = float("nan")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="*", default=None)
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    classes = cfg["superclasses"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = load_ptbxl_split("train")
    te = load_ptbxl_split("test")
    x_te, y_te = te["signals"], te["labels"]
    LOG.info("real train=%s real test=%s device=%s", tr["signals"].shape, x_te.shape, device)

    routes = discover_routes(args.routes)
    LOG.info("routes: %s", routes)

    # any pretrained classifier for TRTS (classifier v2 preferred)
    ckpt_v2 = os.path.join(paths.CKPT, "classifier_best_v2.pt")
    ckpt_v1 = os.path.join(paths.CKPT, "classifier_best.pt")
    eval_ckpt = ckpt_v2 if os.path.exists(ckpt_v2) else ckpt_v1
    ckpt = torch.load(eval_ckpt, map_location=device, weights_only=False)
    eval_model = build_classifier(cfg, base=int(ckpt.get("base", 64))).to(device)
    eval_model.load_state_dict(ckpt["model"])
    eval_model.eval()
    LOG.info("TRTS evaluator: %s (base=%s, kind=%s)", os.path.basename(eval_ckpt),
             ckpt.get("base"), ckpt.get("kind"))

    rows = []
    t_all = time.time()
    rng = np.random.default_rng(cfg["seed"])
    for route in routes:
        if route not in LABEL_CONDITIONED:
            rows.append({"route": route, "metric_group": "downstream", "metric": "tstr_macro_f1",
                         "value": float("nan"), "n": 0,
                         "note": "not_applicable: route is not label-conditioned"})
            continue
        cond_p = os.path.join(paths.GENERATED, f"{route}_cond.npz")
        if not os.path.exists(cond_p):
            continue
        z = np.load(cond_p, allow_pickle=False)
        xs, ys = z["signals"], z["labels"].astype(np.float32)

        # ---------------- TSTR: train on synthetic conditional set, test on real
        n_use = min(len(xs), args.max_train)
        idx = rng.choice(len(xs), size=n_use, replace=False)
        tstr = train_eval_model(xs[idx], ys[idx], x_te, y_te, args.base, args.epochs,
                                cfg["seed"], device)
        # ---------------- CONTROL: same budget, real training subset
        ridx = rng.choice(len(tr["signals"]), size=n_use, replace=False)
        ctrl = train_eval_model(tr["signals"][ridx], tr["labels"][ridx].astype(np.float32),
                                x_te, y_te, args.base, args.epochs, cfg["seed"] + 7, device)
        # ---------------- TRTS: real-trained classifier on synthetic data
        with torch.no_grad():
            probs = []
            for i in range(0, len(xs), 256):
                xb = torch.as_tensor(xs[i:i + 256], device=device)
                probs.append(torch.sigmoid(eval_model(xb)).cpu().numpy())
        trts = evaluate_pretrained(np.concatenate(probs, 0), ys)
        for name, res in (("tstr", {f"tstr_{k}": v for k, v in tstr.items()}),
                          ("control", {f"control_{k}": v for k, v in ctrl.items()}),
                          ("trts", {f"trts_{k}": v for k, v in trts.items()})):
            for k, v in res.items():
                rows.append({"route": route, "metric_group": "downstream", "metric": k,
                             "value": float(v), "n": int(len(xs)),
                             "note": f"TSTR/control: base={args.base}, epochs={args.epochs}, "
                                     f"n_train={n_use}; TRTS evaluator={os.path.basename(eval_ckpt)}"})
        LOG.info("route %s: TSTR f1=%.4f | control f1=%.4f | TRTS f1=%.4f (%.0fs)", route,
                 tstr["macro_f1"], ctrl["macro_f1"], trts["macro_f1"], time.time() - t_all)

    df = pd.DataFrame(rows)
    out = os.path.join(paths.RESULTS, "metrics_downstream.csv")
    df.to_csv(out, index=False)
    save_json(os.path.join(paths.LOGS, "eval_downstream.json"),
              {"routes": routes, "eval_ckpt": os.path.basename(eval_ckpt),
               "max_train": args.max_train, "epochs": args.epochs, "base": args.base,
               "elapsed_s": time.time() - t_all})
    LOG.info("saved %s (%d rows, %.0fs)", out, len(df), time.time() - t_all)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sub = df[df["metric"].isin(["tstr_macro_f1", "control_macro_f1", "trts_macro_f1"])]
        piv = sub.pivot_table(index="route", columns="metric", values="value")
        if len(piv):
            piv = piv.reindex([r for r in ROUTES_ORDER if r in piv.index])
            ax = piv.plot(kind="bar", figsize=(10, 4.5),
                          color=["#4c72b0", "#dd8452", "#55a868"])
            ax.set_ylabel("macro F1")
            ax.set_title("Downstream utility: TSTR / control / TRTS")
            ax.set_xticklabels(piv.index, rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(paths.FIGURES, "fig_downstream.png"), dpi=150)
            plt.close()
            LOG.info("saved fig_downstream.png")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("figure failed: %s", exc)


if __name__ == "__main__":
    main()
