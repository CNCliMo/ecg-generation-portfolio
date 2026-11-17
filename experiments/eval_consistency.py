"""Consistency / fidelity diagnostics for generated 12-lead ECGs (v2 evaluation layer).

Complements the task-book layers (morphology / regularity / functional) with
metrics that answer "is the generated signal physiologically and distributionally
consistent with the conditioning and with the real data?":

  HR consistency        : for heart-rate-conditioned routes, detect the HR of each
                          generated signal and compare with the requested HR
                          (MAE / bias / Pearson r / fraction within 5 and 10 bpm).
  label sensitivity     : for label-conditioned routes, per-class TPR and TNR of the
                          diagnostic classifier on the conditional generations
                          (macro averages; complements the exact-match consistency).
  lead coherence        : the four dependent limb leads (III, aVR, aVL, aVF) are
                          affine functions of leads I and II. We fit that affine map
                          per record (intercept included, because each lead is
                          z-scored) and measure the residual relative to the real
                          test-set residual distribution (mean + Wasserstein).
  coverage / precision  : k-NN based coverage of the real manifold (recall) and
                          fraction of synthetic samples inside the real manifold
                          (precision), on 12x100 resampled signals.
  classifier-FID        : Fréchet distance between real and generated distributions
                          in the penultimate feature space of the diagnostic
                          classifier (ECG-FID style, no external weights needed).

Outputs:
  results/metrics_consistency.csv
  results/logs/eval_consistency.json
  results/figures/fig_consistency.png
  results/figures/fig_downstream_coverage.png

Usage: python experiments/eval_consistency.py [--routes ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.logging_utils import get_logger, save_json
from models.classifier import build_classifier, predict_proba
from ecg_common.config import detect_rpeaks, estimate_hr

LOG = get_logger("eval_consistency")
ROUTES_ORDER = ["A", "B_uncond", "B_cond", "c_full", "c_no_label", "c_no_hr",
                "c_label_noise", "c_label_shuffle", "d_seed42", "d_seed43", "d_seed44",
                "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"]
LABEL_CONDITIONED = {"B_cond", "c_full", "c_no_hr", "c_label_noise", "c_label_shuffle",
                     "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"}
HR_CONDITIONED = {"A", "c_full", "c_no_label", "c_label_noise", "c_label_shuffle",
                  "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"}

# lead indices: I=0, II=1, III=2, aVR=3, aVL=4, aVF=5
LEAD_I, LEAD_II = 0, 1
DEPENDENT = [2, 3, 4, 5]


def discover_routes(only=None):
    found = set()
    for pattern in ("*_cond.npz", "*_uncond.npz"):
        for f in glob.glob(os.path.join(paths.GENERATED, pattern)):
            found.add(os.path.basename(f).replace("_cond.npz", "").replace("_uncond.npz", ""))
    routes = sorted(found, key=lambda r: ROUTES_ORDER.index(r) if r in ROUTES_ORDER else 99)
    return [r for r in routes if not only or r in only]


def load_route(route: str, max_n: int = 1000):
    """Concatenate conditional + unconditional generations (as arrays)."""
    xs, labels, hrs = [], [], []
    cp = os.path.join(paths.GENERATED, f"{route}_cond.npz")
    up = os.path.join(paths.GENERATED, f"{route}_uncond.npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        n = min(len(z["signals"]), max_n // 2)
        xs.append(z["signals"][:n])
        labels.append(z["labels"][:n])
        if "hr" in z:
            hrs.append(z["hr"][:n])
    if os.path.exists(up):
        z = np.load(up, allow_pickle=False)
        n = min(len(z["signals"]), max_n // 2)
        xs.append(z["signals"][:n])
        labels.append(np.full((n, 5), np.nan, dtype=np.float32))
        if "hr" in z:
            hrs.append(z["hr"][:n])
    if not xs:
        return None
    return {"signals": np.concatenate(xs, 0), "labels": np.concatenate(labels, 0),
            "hr": np.concatenate(hrs, 0) if hrs else None}


def down_feats(signals: np.ndarray, n_points: int = 100) -> np.ndarray:
    """(N,12,1000) -> (N,12*n_points) resampled, for k-NN coverage."""
    N, L, T = signals.shape
    idx = np.linspace(0, T - 1, n_points)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = (idx - i0)[None, None, :]
    out = signals[:, :, i0] * (1 - w) + signals[:, :, i1] * w
    return out.reshape(N, -1).astype(np.float64)


def lead_residual(signals: np.ndarray) -> np.ndarray:
    """Per-record relative residual of the affine map from (I, II) to the four
    dependent limb leads; returns (N,) mean relative RMS residual over the 4 leads."""
    N = signals.shape[0]
    out = np.zeros(N, dtype=np.float64)
    for n in range(N):
        x = signals[n].astype(np.float64)
        A = np.stack([x[LEAD_I], x[LEAD_II], np.ones_like(x[LEAD_I])], axis=1)  # (T,3)
        ATA = A.T @ A + 1e-6 * np.eye(3)
        res = []
        for li in DEPENDENT:
            y = x[li]
            coef = np.linalg.solve(ATA, A.T @ y)
            r = np.sqrt(np.mean((A @ coef - y) ** 2))
            res.append(r / (np.std(y) + 1e-8))
        out[n] = float(np.mean(res))
    return out


def frechet_distance(feats_real: np.ndarray, feats_synth: np.ndarray, eps: float = 1e-6) -> float:
    mu_r, mu_s = feats_real.mean(0), feats_synth.mean(0)
    sr = np.cov(feats_real, rowvar=False) + eps * np.eye(feats_real.shape[1])
    ss = np.cov(feats_synth, rowvar=False) + eps * np.eye(feats_synth.shape[1])
    covmean = sqrtm(sr @ ss)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu_r - mu_s) ** 2) + np.trace(sr + ss - 2.0 * covmean))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="*", default=None)
    ap.add_argument("--max-n", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    fs = cfg["fs"]
    classes = cfg["superclasses"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    real = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)["signals"]
    LOG.info("real test: %s", real.shape)

    ckpt_path = os.path.join(paths.CKPT, "classifier_best_v2.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(paths.CKPT, "classifier_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    clf = build_classifier(cfg, base=int(ckpt.get("base", 64))).to(device)
    clf.load_state_dict(ckpt["model"])
    clf.eval()
    feat_dim = int(getattr(clf, "feature_dim", 512))
    LOG.info("classifier: %s (base=%s, feat_dim=%d)", os.path.basename(ckpt_path),
             ckpt.get("base"), feat_dim)

    @torch.no_grad()
    def feats(x: np.ndarray, bs: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(x), bs):
            xb = torch.as_tensor(x[i:i + bs], device=device)
            out.append(clf.features(xb).cpu().numpy())
        return np.concatenate(out, 0)

    # real reference statistics (computed once)
    t0 = time.time()
    real_lead_res = lead_residual(real[: min(len(real), 800)])
    real_feats = feats(real)
    real_dn = down_feats(real)
    mu, sd = real_dn.mean(0), real_dn.std(0)
    sd[sd < 1e-8] = 1.0
    real_dn = (real_dn - mu) / sd
    nn_real = NearestNeighbors(n_neighbors=4).fit(real_dn)
    real_nn = nn_real.kneighbors(real_dn, return_distance=True)[0][:, -1]  # 3rd NN (self excluded)
    r_real_95 = float(np.percentile(real_nn, 95))
    LOG.info("real reference ready (lead residual mean=%.4f, kNN radius95=%.3f, %.0fs)",
             float(real_lead_res.mean()), r_real_95, time.time() - t0)

    rows = []
    rows.append({"route": "REAL_PTBXL_TEST", "metric_group": "consistency",
                 "metric": "lead_residual_mean", "value": float(real_lead_res.mean()),
                 "n": len(real_lead_res), "note": "real test reference"})

    routes = discover_routes(args.routes)
    LOG.info("routes: %s", routes)
    t_all = time.time()
    for route in routes:
        data = load_route(route, max_n=args.max_n)
        if data is None:
            continue
        xs = data["signals"]
        note = f"evaluator={os.path.basename(ckpt_path)}"

        # ------------------------------------------------ HR consistency
        if route in HR_CONDITIONED and data["hr"] is not None:
            cp = os.path.join(paths.GENERATED, f"{route}_cond.npz")
            if os.path.exists(cp):
                z = np.load(cp, allow_pickle=False)
                sig_c, hr_t = z["signals"][:500], z["hr"][:500]
                def _hr_value(x):
                    r = estimate_hr(x, fs)
                    if isinstance(r, dict):
                        return float(r.get(hr, np.nan))
                    if isinstance(r, (tuple, list)):
                        return float(r[0])
                    return float(r)
                hr_e = np.array([_hr_value(ls) for ls in sig_c[:, LEAD_II, :]])
                valid = np.isfinite(hr_e) & np.isfinite(hr_t)
                if valid.sum() >= 20:
                    err = hr_e[valid] - hr_t[valid]
                    rows += [
                        {"route": route, "metric_group": "consistency", "metric": "hr_mae",
                         "value": float(np.mean(np.abs(err))), "n": int(valid.sum()),
                         "note": "detected HR vs requested HR (bpm)"},
                        {"route": route, "metric_group": "consistency", "metric": "hr_bias",
                         "value": float(np.mean(err)), "n": int(valid.sum()), "note": ""},
                        {"route": route, "metric_group": "consistency", "metric": "hr_pearson",
                         "value": float(np.corrcoef(hr_e[valid], hr_t[valid])[0, 1]),
                         "n": int(valid.sum()), "note": ""},
                        {"route": route, "metric_group": "consistency", "metric": "hr_within_5bpm",
                         "value": float(np.mean(np.abs(err) <= 5)), "n": int(valid.sum()),
                         "note": ""},
                        {"route": route, "metric_group": "consistency", "metric": "hr_within_10bpm",
                         "value": float(np.mean(np.abs(err) <= 10)), "n": int(valid.sum()),
                         "note": ""},
                    ]
                    LOG.info("route %s HR: MAE=%.2f bias=%.2f r=%.3f within10=%.3f", route,
                             float(np.mean(np.abs(err))), float(np.mean(err)),
                             float(np.corrcoef(hr_e[valid], hr_t[valid])[0, 1]),
                             float(np.mean(np.abs(err) <= 10)))

        # ------------------------------------------------ label sensitivity
        if route in LABEL_CONDITIONED:
            cp = os.path.join(paths.GENERATED, f"{route}_cond.npz")
            if os.path.exists(cp):
                z = np.load(cp, allow_pickle=False)
                sig_c, lab = z["signals"], z["labels"].astype(int)
                probs = predict_proba(clf, sig_c, device=device)
                pred = (probs >= 0.5).astype(int)
                tpr, tnr = [], []
                for ci, cname in enumerate(classes):
                    mask = lab[:, ci] == 1
                    tpr.append(float(pred[mask, ci].mean()) if mask.any() else float("nan"))
                    mask0 = lab[:, ci] == 0
                    tnr.append(float(1 - pred[mask0, ci].mean()) if mask0.any() else float("nan"))
                rows += [
                    {"route": route, "metric_group": "consistency", "metric": "label_tpr_macro",
                     "value": float(np.nanmean(tpr)), "n": int(len(sig_c)),
                     "note": "per-class TPR at 0.5 (macro): " + str(
                         {c: round(v, 3) for c, v in zip(classes, tpr) if np.isfinite(v)})},
                    {"route": route, "metric_group": "consistency", "metric": "label_tnr_macro",
                     "value": float(np.nanmean(tnr)), "n": int(len(sig_c)), "note": ""},
                ]

        # ------------------------------------------------ lead coherence
        res = lead_residual(xs[: min(len(xs), 500)])
        rows += [
            {"route": route, "metric_group": "consistency", "metric": "lead_residual_mean",
             "value": float(res.mean()), "n": int(len(res)),
             "note": "affine residual of III/aVR/aVL/aVF given I,II"},
            {"route": route, "metric_group": "consistency",
             "metric": "lead_residual_wasserstein",
             "value": float(wasserstein_distance(real_lead_res, res)), "n": int(len(res)),
             "note": "vs real test residual distribution"},
        ]

        # ------------------------------------------------ coverage / precision
        dn = (down_feats(xs) - mu) / sd
        d_synth_to_real = nn_real.kneighbors(dn, n_neighbors=1, return_distance=True)[0][:, 0]
        nn_synth = NearestNeighbors(n_neighbors=4).fit(dn)
        r_synth = nn_synth.kneighbors(dn, return_distance=True)[0][:, -1]
        r_synth_95 = float(np.percentile(r_synth, 95))
        precision = float(np.mean(d_synth_to_real <= r_real_95))
        # coverage of real set: fraction of real samples with a synth neighbour closer than the
        # synth-manifold radius (symmetric k-NN formulation)
        nn_from_real_to_synth = nn_synth.kneighbors(real_dn, n_neighbors=1,
                                                    return_distance=True)[0][:, 0]
        coverage = float(np.mean(nn_from_real_to_synth <= r_synth_95))
        rows += [
            {"route": route, "metric_group": "consistency", "metric": "coverage",
             "value": coverage, "n": int(len(xs)), "note": "k-NN coverage of real manifold"},
            {"route": route, "metric_group": "consistency", "metric": "precision_knn",
             "value": precision, "n": int(len(xs)), "note": "fraction inside real manifold (95th pct)"},
            {"route": route, "metric_group": "consistency", "metric": "nn_dist_synth_real",
             "value": float(d_synth_to_real.mean()), "n": int(len(xs)), "note": ""},
            {"route": route, "metric_group": "consistency", "metric": "nn_dist_real_real",
             "value": float(real_nn.mean()), "n": int(len(real_dn)), "note": "reference"},
        ]

        # ------------------------------------------------ classifier-FID
        route_feats = feats(xs)
        fid = frechet_distance(real_feats, route_feats)
        rows.append({"route": route, "metric_group": "consistency", "metric": "classifier_fid",
                     "value": fid, "n": int(len(xs)),
                     "note": f"Fréchet distance in {feat_dim}-d classifier features (lower better)"})
        LOG.info("route %s: lead_res=%.4f W=%.4f cover=%.3f prec=%.3f fid=%.1f (%.0fs)",
                 route, float(res.mean()),
                 float(wasserstein_distance(real_lead_res, res)), coverage, precision, fid,
                 time.time() - t_all)

    df = pd.DataFrame(rows)
    out = os.path.join(paths.RESULTS, "metrics_consistency.csv")
    df.to_csv(out, index=False)
    save_json(os.path.join(paths.LOGS, "eval_consistency.json"),
              {"routes": routes, "real_n": len(real), "max_n": args.max_n,
               "elapsed_s": time.time() - t_all})
    LOG.info("saved %s (%d rows, %.0fs)", out, len(df), time.time() - t_all)

    # ------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        ax = axes[0, 0]
        sub = df[df.metric == "classifier_fid"].set_index("route")["value"]
        sub = sub.drop(index=["REAL_PTBXL_TEST"], errors="ignore")
        sub = sub.reindex([r for r in ROUTES_ORDER if r in sub.index])
        ax.bar(range(len(sub)), sub.values, color="#4c72b0")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=45, ha="right")
        ax.set_title("classifier-FID (lower better)")
        ax = axes[0, 1]
        sub = df[df.metric == "lead_residual_mean"].set_index("route")["value"]
        sub = sub.reindex([r for r in ROUTES_ORDER if r in sub.index])
        ax.bar(range(len(sub)), sub.values, color="#55a868")
        ax.axhline(float(real_lead_res.mean()), color="k", ls="--", label="real")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=45, ha="right")
        ax.set_title("lead coherence residual (lower better)")
        ax.legend()
        ax = axes[1, 0]
        sub = df[df.metric.isin(["precision_knn", "coverage"])]
        piv = sub.pivot_table(index="route", columns="metric", values="value")
        piv = piv.reindex([r for r in ROUTES_ORDER if r in piv.index])
        piv.plot(kind="bar", ax=ax, color=["#dd8452", "#4c72b0"], width=0.8)
        ax.set_title("k-NN precision / coverage (higher better)")
        ax.tick_params(axis="x", rotation=45)
        ax = axes[1, 1]
        sub = df[df.metric == "hr_mae"].set_index("route")["value"]
        sub = sub.reindex([r for r in ROUTES_ORDER if r in sub.index])
        ax.bar(range(len(sub)), sub.values, color="#c44e52")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=45, ha="right")
        ax.set_title("HR consistency MAE (bpm, lower better)")
        fig.suptitle("v2 consistency diagnostics")
        fig.tight_layout()
        fig.savefig(os.path.join(paths.FIGURES, "fig_consistency.png"), dpi=150)
        plt.close(fig)
        LOG.info("saved fig_consistency.png")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("figure failed: %s", exc)


if __name__ == "__main__":
    main()
