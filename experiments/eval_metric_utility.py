"""W2 experiment P-B: which generation metrics actually predict utility?

We have 15+ routes (A/B/C/D/v2) each evaluated with 10 morphology metrics,
4 interval Wasserstein distances, functional consistency, classifier-FID,
coverage, lead residuals and TSTR downstream utility. This script asks:

  "Do low-level generation metrics (MMD/DTW/PSD/...) correlate with the things
   we actually care about (TSTR utility, conditional consistency, HR control,
   lead coherence)?"

Spearman correlations (route-level, n<=15) with p-values; also the correlation
between the morphology metrics themselves (redundancy check).

Usage: python experiments/eval_metric_utility.py
Outputs:
  results/metrics_metric_utility.csv
  results/logs/eval_metric_utility.json
  results/figures/fig_metric_utility_heatmap.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.logging_utils import get_logger, save_json

LOG = get_logger("eval_metric_utility")
MORPH = ["mmd", "dtw", "psd", "prd", "fd", "swd", "nn_dist", "pr_dist",
         "heart_rate", "spectral_entropy"]
REG_PARAMS = ["QT_ms", "PR_ms", "QRS_ms", "rr_cv"]
UTILITY = {
    # name: (source, sign) sign=+1 higher-better, -1 lower-better
    "tstr_macro_f1": ("downstream", +1),
    "conditional_consistency": ("functional", +1),
    "coverage": ("consistency", +1),
    "precision_knn": ("consistency", +1),
    "unconditional_class_KL": ("functional", -1),
    "classifier_fid": ("consistency", -1),
    "lead_residual": ("consistency", -1),
    "hr_mae": ("consistency", -1),
}
SKIP_ROUTES = {"classifier", "classifier_v2", "REAL_PTBXL_TEST", "overall"}


def read(name):
    p = os.path.join(paths.RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def build_table():
    """route -> {feature/utility columns} from the individual metric tables."""
    rows = {}
    mor = read("metrics_morphology.csv")
    if mor is not None:
        for r, g in mor[mor["metric"].isin(MORPH)].groupby("route"):
            rows.setdefault(r, {})
            for _, x in g.iterrows():
                rows[r][f"morph_{x['metric']}"] = float(x["value"])
    reg = read("metrics_regularity.csv")
    if reg is not None:
        g = reg[(reg["metric"] == "wasserstein") & (reg["domain"] == "in_domain")]
        for r, gg in g.groupby("route"):
            rows.setdefault(r, {})
            for _, x in gg.iterrows():
                rows[r][f"w_{x['param']}"] = float(x["value"])
    fun = read("metrics_functional.csv")
    if fun is not None:
        for r, g in fun.groupby("route"):
            rows.setdefault(r, {})
            for metric, col in (("conditional_consistency_macro", "conditional_consistency"),
                                ("unconditional_class_KL", "unconditional_class_KL")):
                sub = g[g["metric"] == metric]
                if not sub.empty and np.isfinite(sub["value"].iloc[0]):
                    rows[r][col] = float(sub["value"].iloc[0])
    cons = read("metrics_consistency.csv")
    if cons is not None:
        for r, g in cons.groupby("route"):
            rows.setdefault(r, {})
            for metric, col in (("coverage", "coverage"), ("precision_knn", "precision_knn"),
                                ("classifier_fid", "classifier_fid"),
                                ("lead_residual_mean", "lead_residual"),
                                ("hr_mae", "hr_mae")):
                sub = g[g["metric"] == metric]
                if not sub.empty and np.isfinite(sub["value"].iloc[0]):
                    rows[r][col] = float(sub["value"].iloc[0])
    down = read("metrics_downstream.csv")
    if down is not None:
        for r, g in down.groupby("route"):
            rows.setdefault(r, {})
            sub = g[g["metric"] == "tstr_macro_f1"]
            if not sub.empty and np.isfinite(sub["value"].iloc[0]):
                rows[r]["tstr_macro_f1"] = float(sub["value"].iloc[0])
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.drop(index=[r for r in df.index if r in SKIP_ROUTES], errors="ignore")
    return df.sort_index()


def main() -> None:
    t0 = time.time()
    df = build_table()
    LOG.info("routes with data: %d (%s)", len(df), ", ".join(df.index))
    feature_cols = [c for c in df.columns if c.startswith(("morph_", "w_"))]
    utility_cols = [c for c in df.columns if c in UTILITY]
    LOG.info("features: %d | utility columns: %d", len(feature_cols), len(utility_cols))
    df.to_csv(os.path.join(paths.RESULTS, "metrics_route_feature_table.csv"))
    if len(df) < 5:
        raise SystemExit("too few routes for correlation analysis")

    # ---------------------------------------------- feature vs utility table
    rows = []
    for f in feature_cols:
        for u in utility_cols:
            sub = df[[f, u]].dropna()
            if len(sub) < 5:
                continue
            rho, p = spearmanr(sub[f], sub[u])
            sign = UTILITY[u][1]
            rows.append({"feature": f, "utility": u, "n_routes": len(sub),
                         "spearman_rho": float(rho), "p_value": float(p),
                         "sign_adjusted_rho": float(rho) * sign})
    res = pd.DataFrame(rows)
    out = os.path.join(paths.RESULTS, "metrics_metric_utility.csv")
    res.to_csv(out, index=False)
    LOG.info("saved %s (%d rows, %.0fs)", out, len(res), time.time() - t0)

    # top correlations (sign-adjusted: positive = the metric moves with utility)
    top = (res.dropna(subset=["sign_adjusted_rho"])
             .sort_values("sign_adjusted_rho", ascending=False))
    LOG.info("=== top positive correlations (metric moves with utility) ===")
    for _, x in top.head(12).iterrows():
        LOG.info("  %-22s vs %-26s rho=%+.3f (p=%.3g, n=%d)",
                 x["feature"], x["utility"], x["sign_adjusted_rho"], x["p_value"], x["n_routes"])
    neg = top[top["sign_adjusted_rho"] < 0]
    if not neg.empty:
        LOG.info("=== notable negative correlations ===")
        for _, x in neg.tail(6).iterrows():
            LOG.info("  %-22s vs %-26s rho=%+.3f (p=%.3g, n=%d)",
                     x["feature"], x["utility"], x["sign_adjusted_rho"], x["p_value"], x["n_routes"])
    save_json(os.path.join(paths.LOGS, "eval_metric_utility.json"),
              {"n_routes": int(len(df)), "routes": list(df.index),
               "n_feature_utility_pairs": int(len(res)), "elapsed_s": time.time() - t0})

    # ---------------------------------------------------------- heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        piv = res.pivot_table(index="feature", columns="utility",
                              values="sign_adjusted_rho")
        order_f = [f for f in feature_cols if f in piv.index]
        order_u = [u for u in utility_cols if u in piv.columns]
        piv = piv.reindex(index=order_f, columns=order_u)
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(piv.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(order_u)))
        ax.set_xticklabels(order_u, rotation=35, ha="right")
        ax.set_yticks(range(len(order_f)))
        ax.set_yticklabels(order_f)
        for i in range(len(order_f)):
            for j in range(len(order_u)):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.55 else "black")
        ax.set_title("Spearman correlation: generation metrics vs utility\n"
                     "(sign-adjusted: +1 = metric moves with utility)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(paths.FIGURES, "fig_metric_utility_heatmap.png"), dpi=150)
        plt.close(fig)
        LOG.info("saved fig_metric_utility_heatmap.png")

        # metric-vs-metric redundancy heatmap (pairwise scipy Spearman)
        mcols = [c for c in feature_cols if c in df.columns and df[c].notna().sum() >= 8]
        mcols = list(dict.fromkeys(mcols))
        corr = pd.DataFrame(np.nan, index=mcols, columns=mcols, dtype=float)
        for ai, a in enumerate(mcols):
            for b in mcols:
                sub = df[[a, b]].dropna()
                if len(sub) >= 5:
                    try:
                        val = float(spearmanr(sub[a].values, sub[b].values).statistic)
                    except Exception:
                        val = np.nan
                    corr.loc[a, b] = val
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(corr.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mcols)))
        ax.set_xticklabels(mcols, rotation=45, ha="right")
        ax.set_yticks(range(len(mcols)))
        ax.set_yticklabels(mcols)
        for i in range(len(mcols)):
            for j in range(len(mcols)):
                v = corr.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.55 else "black")
        ax.set_title("Metric redundancy (Spearman between generation metrics)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(paths.FIGURES, "fig_metric_redundancy.png"), dpi=150)
        plt.close(fig)
        LOG.info("saved fig_metric_redundancy.png")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("figure failed: %s", exc)


if __name__ == "__main__":
    main()
