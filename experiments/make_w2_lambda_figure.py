"""W2 lambda-scan figure (short, robust)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.logging_utils import get_logger

LOG = get_logger("make_w2_lambda_figure")
ROUTES = [("c_nle_v2", "0"), ("c_full_v2", "0.1"), ("c_aux05_v2", "0.5")]


def main() -> None:
    def read(name):
        p = os.path.join(paths.RESULTS, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    mor, reg = read("metrics_morphology.csv"), read("metrics_regularity.csv")
    fun, con, down = (read("metrics_functional.csv"), read("metrics_consistency.csv"),
                      read("metrics_downstream.csv"))

    def pick(df, route, metric, group=None, domain=None, param=None):
        x = df[df["route"] == route]
        if group is not None and "metric_group" in x:
            x = x[x["metric_group"] == group]
        if "metric" in x:
            x = x[x["metric"] == metric]
        if domain is not None and "domain" in x:
            x = x[x["domain"] == domain]
        if param is not None and "param" in x:
            x = x[x["param"] == param]
        return float(x["value"].iloc[0]) if len(x) else np.nan

    rows = []
    for route, lam in ROUTES:
        rows.append({
            "lambda": lam,
            "QT_W": pick(reg, route, "wasserstein", domain="in_domain", param="QT_ms"),
            "RR_W": pick(reg, route, "wasserstein", domain="in_domain", param="rr_cv"),
            "PSD": pick(mor, route, "psd"),
            "cons": pick(fun, route, "conditional_consistency_macro"),
            "HR_MAE": pick(con, route, "hr_mae"),
            "TSTR": pick(down, route, "tstr_macro_f1"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(paths.RESULTS, "metrics_w2_lambda_scan.csv"), index=False)
    LOG.info("wrote metrics_w2_lambda_scan.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    x = np.arange(len(df))
    axes[0].bar(x, df["QT_W"], color="tab:red")
    axes[0].set_title("QT Wasserstein (ms) - lower better")
    axes[1].bar(x, df["PSD"], color="tab:blue")
    axes[1].set_title("PSD - lower better")
    axes[2].bar(x, df["cons"], color="tab:orange")
    axes[2].set_title("conditional consistency - higher better")
    axes[3].bar(x, df["TSTR"], color="tab:green")
    axes[3].set_title("TSTR macro-F1 - higher better")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"lambda={v}" for v in df["lambda"]])
    fig.suptitle("W2 E2: interval-auxiliary-loss lambda scan (same budget, nle+HR, 40ep/base48)")
    fig.tight_layout()
    fig.savefig(os.path.join(paths.FIGURES, "fig_w2_lambda_scan.png"), dpi=150)
    plt.close(fig)
    LOG.info("saved fig_w2_lambda_scan.png")


if __name__ == "__main__":
    main()
