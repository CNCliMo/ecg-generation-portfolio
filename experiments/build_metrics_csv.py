"""Merge every metric table into the single deliverable `results/metrics.csv`.

Task book 8/9: one table, all routes x all metrics, including in-domain /
cross-domain / Delta, with no missing cells for applicable metrics.
N/A cells are explicit rows with a `note` explaining why (never silently absent).

Sources (all produced by independent, re-runnable scripts):
  results/metrics_morphology.csv   (5.1, ecgen-eval)
  results/metrics_regularity.csv   (5.2, Wasserstein/KS + Delta)
  results/metrics_functional.csv   (5.3, classifier-based)
  results/classifier_test_report.csv (section 4, classifier baseline)

Output columns:
  route, metric_group, metric, param, domain, value, ks_stat, ks_p, n, note

Usage: python experiments/build_metrics_csv.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.logging_utils import get_logger, save_json

LOG = get_logger("build_metrics_csv")
ROUTES_ORDER = ["A", "B_uncond", "B_cond", "c_full", "c_no_label", "c_no_hr",
                "c_label_noise", "c_label_shuffle", "d_seed42", "d_seed43", "d_seed44",
                "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"]
REQUIRED = {
    "morphology": ["mmd", "dtw", "psd", "prd"],
    "regularity": ["QT_ms", "PR_ms", "QRS_ms", "rr_cv"],
    "functional": ["unconditional_class_KL"],
}


def _read(name: str) -> pd.DataFrame | None:
    p = os.path.join(paths.RESULTS, name)
    if not os.path.exists(p):
        LOG.warning("missing source: %s", p)
        return None
    return pd.read_csv(p)


def main() -> None:
    frames = []

    morph = _read("metrics_morphology.csv")
    if morph is not None:
        frames.append(morph.assign(param="", ks_stat=np.nan, ks_p=np.nan,
                                   n=morph["n_synth"], note=""))

    reg = _read("metrics_regularity.csv")
    if reg is not None:
        frames.append(reg.assign(n=reg["n_route"]))

    func = _read("metrics_functional.csv")
    if func is not None:
        frames.append(func.assign(domain="in_domain", ks_stat=np.nan, ks_p=np.nan))

    cls = _read("classifier_test_report.csv")
    if cls is not None:
        c = pd.DataFrame({
            "route": "classifier",
            "metric_group": "classifier",
            "metric": cls["metric"],
            "param": cls["scope"].astype(str).str.replace("class:", "", regex=False),
            "domain": "in_domain",
            "value": cls["value"],
            "ks_stat": np.nan,
            "ks_p": np.nan,
            "n": cls["n"],
            "note": "test fold 10",
        })
        frames.append(c)

    down = _read("metrics_downstream.csv")
    if down is not None:
        frames.append(down.assign(domain="in_domain", ks_stat=np.nan, ks_p=np.nan))

    cons = _read("metrics_consistency.csv")
    if cons is not None:
        frames.append(cons.assign(domain="in_domain", ks_stat=np.nan, ks_p=np.nan))

    cls2 = _read("classifier_test_report_v2.csv")
    if cls2 is not None:
        c2 = pd.DataFrame({
            "route": "classifier_v2",
            "metric_group": "classifier",
            "metric": cls2["metric"],
            "param": cls2["scope"].astype(str).str.replace("class:", "", regex=False),
            "domain": "in_domain",
            "value": cls2["value"],
            "ks_stat": np.nan,
            "ks_p": np.nan,
            "n": cls2["n"],
            "note": "test fold 10 (v2: base96 + label smoothing + EMA + threshold tuning)",
        })
        frames.append(c2)

    if not frames:
        raise SystemExit("no metric tables found -- run the eval scripts first")
    cols = ["route", "metric_group", "metric", "param", "domain", "value",
            "ks_stat", "ks_p", "n", "note"]
    out = pd.concat([f[[c for c in cols if c in f.columns]].reindex(columns=cols)
                     for f in frames], ignore_index=True)

    def rank(r):
        return ROUTES_ORDER.index(r) if r in ROUTES_ORDER else (0 if r == "overall" else 99)
    out["_rank"] = out["route"].map(rank)
    out = out.sort_values(["_rank", "metric_group", "metric", "param", "domain"]).drop(columns="_rank")
    out = out.reset_index(drop=True)
    path = os.path.join(paths.RESULTS, "metrics.csv")
    out.to_csv(path, index=False)
    LOG.info("wrote %s (%d rows, %d routes)", path, len(out), out["route"].nunique())

    # -------------------------------------------------- completeness report
    routes = [r for r in ROUTES_ORDER if r in set(out["route"])]
    problems = []
    for r in routes:
        sub = out[out["route"] == r]
        for grp, metrics in REQUIRED.items():
            for m in metrics:
                cell = sub[(sub["metric_group"] == grp) &
                           ((sub["metric"] == m) | (sub["param"] == m))]
                if cell.empty:
                    problems.append(f"{r}: missing {grp}/{m}")
                elif np.isnan(cell["value"]).all() and not str(cell["note"].fillna("")).str.contains("not_applicable|missing").any():
                    problems.append(f"{r}: {grp}/{m} all NaN without explanation")
        if not ((sub["metric"] == "conditional_consistency").any()):
            problems.append(f"{r}: missing functional conditional_consistency row")
    if "A" not in routes:
        problems.append("route A missing entirely")
    if not any(r.lower().startswith("d_seed") for r in routes):
        problems.append("route D missing entirely")

    summary = {"n_rows": int(len(out)), "routes": routes, "problems": problems}
    save_json(os.path.join(paths.RESULTS, "metrics_completeness.json"), summary)
    if problems:
        LOG.warning("completeness problems (%d):", len(problems))
        for p in problems:
            LOG.warning("  - %s", p)
    else:
        LOG.info("completeness check OK: all applicable metrics present")


if __name__ == "__main__":
    main()
