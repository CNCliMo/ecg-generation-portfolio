"""Regularity-layer evaluation (task book 5.2) -- the core metric layer.

Parameters: QT, PR, QRS intervals + R-R interval variability (CV over 10 s windows).
Extraction: NeuroKit2 ecg_delineate on Lead II (synthetic) / MLII (MIT-BIH);
R peaks via the hard-coded scipy detector used for HR labels.

Two reference distributions:
  in_domain   : PTB-XL test split (fold 10), from preprocess/extract_reference_params.py
                + RR-CV from all 2196 test signals
  cross_domain: MIT-BIH normal ('N') beats, from preprocess/extract_reference_params.py
                + RR-CV from sliding 10 s windows over the filtered MIT-BIH records

Metrics: Wasserstein distance (primary) + two-sample KS test (statistic, p-value).
Reported: per route x {QT, PR, QRS, RR_cv} x {in_domain, cross_domain} plus
          Delta_Wasserstein = cross_domain - in_domain (task book requirement).

Every delineation failure is recorded with a reason; counts are included in the
output table (`n_failed`) -- nothing is silently skipped.

Outputs:
  results/metrics_regularity.csv
  results/logs/failures_regularity_<route>.csv
  results/logs/eval_regularity.json

Usage: python experiments/eval_regularity.py [--routes A_cond B_cond ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import intervals, paths
from ecg_common.config import load_config, seed_everything
from ecg_common.logging_utils import FailureRecorder, get_logger, save_json

LOG = get_logger("eval_regularity")
REF_CACHE = os.path.join(paths.CACHE, "regularity_refs.npz")
PARAMS = ("QT_ms", "PR_ms", "QRS_ms", "rr_cv")
ROUTES_ORDER = ["A", "B_uncond", "B_cond", "c_full", "c_no_label", "c_no_hr",
                "c_label_noise", "c_label_shuffle", "d_seed42", "d_seed43", "d_seed44",
                "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"]


def discover_routes(only=None) -> list[str]:
    found = set()
    for pattern in ("*_cond.npz", "*_uncond.npz"):
        for f in glob.glob(os.path.join(paths.GENERATED, pattern)):
            found.add(os.path.basename(f).replace("_cond.npz", "").replace("_uncond.npz", ""))
    routes = sorted(found, key=lambda r: ROUTES_ORDER.index(r) if r in ROUTES_ORDER else 99)
    return [r for r in routes if not only or r in only]


def load_route_signals(route: str, max_n: int = 500):
    for suffix in ("_cond.npz", "_uncond.npz"):
        p = os.path.join(paths.GENERATED, route + suffix)
        if os.path.exists(p):
            sigs = np.load(p, allow_pickle=False)["signals"]
            if max_n and len(sigs) > max_n:
                rng = np.random.default_rng(seed_everything())
                idx = np.sort(rng.choice(len(sigs), size=max_n, replace=False))
                LOG.info("route %s: subsampled %d -> %d signals for distribution estimation",
                         route, len(sigs), max_n)
                sigs = sigs[idx]
            return sigs
    raise FileNotFoundError(route)


def rr_cv_in_domain() -> np.ndarray:
    test = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)["signals"]
    vals = []
    for sig in test:
        v = intervals.rr_variability(np.asarray(sig[1], dtype=np.float64), 100.0)
        vals.append(v["rr_cv"])
    return np.asarray(vals, dtype=np.float64)


def rr_cv_cross_domain(cfg: dict, n_sample: int) -> np.ndarray:
    """RR-CV on sliding 10 s windows of the filtered MIT-BIH records (fs=360)."""
    fs = cfg["preprocess"]["mitbih_fs"]
    win = int(10 * fs)
    step = int(5 * fs)
    files = sorted(glob.glob(os.path.join(paths.CACHE, "mitbih_records", "*.npy")))
    vals = []
    for f in files:
        sig = np.load(f).astype(np.float64)
        for start in range(0, max(0, len(sig) - win), step):
            v = intervals.rr_variability(sig[start:start + win], fs)
            if np.isfinite(v["rr_cv"]):
                vals.append(v["rr_cv"])
    vals = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed_everything())
    if n_sample and len(vals) > n_sample:
        vals = vals[rng.choice(len(vals), n_sample, replace=False)]
    return vals


def load_ref(cfg: dict):
    if os.path.exists(REF_CACHE):
        z = np.load(REF_CACHE, allow_pickle=False)
        ref = {}
        for domain in ("in_domain", "cross_domain"):
            ref[domain] = {p: z[f"{domain}__{p}"] for p in PARAMS if f"{domain}__{p}" in z.files}
        LOG.info("reference distributions loaded from cache: %s", REF_CACHE)
        for dom in ref:
            LOG.info("reference[%s]: %s", dom, {k: len(v) for k, v in ref[dom].items()})
        return ref
    pp = cfg["preprocess"]
    n_ref = pp["reference_beats_per_domain"]
    df_in = pd.read_csv(os.path.join(paths.CACHE, "ref_params_ptbxl_test.csv"))
    df_cross = pd.read_csv(os.path.join(paths.CACHE, "ref_params_mitbih.csv"))
    ref = {
        "in_domain": {"QT_ms": df_in["QT_ms"].dropna().values,
                      "PR_ms": df_in["PR_ms"].dropna().values,
                      "QRS_ms": df_in["QRS_ms"].dropna().values,
                      "rr_cv": rr_cv_in_domain()},
        "cross_domain": {"QT_ms": df_cross["QT_ms"].dropna().values,
                         "PR_ms": df_cross["PR_ms"].dropna().values,
                         "QRS_ms": df_cross["QRS_ms"].dropna().values,
                         "rr_cv": rr_cv_cross_domain(cfg, n_ref)},
    }
    for dom in ref:
        LOG.info("reference[%s]: %s", dom, {k: len(v) for k, v in ref[dom].items()})
    np.savez(REF_CACHE, **{f"{dom}__{p}": np.asarray(v, dtype=float)
                           for dom, d in ref.items() for p, v in d.items()})
    LOG.info("reference distributions cached -> %s", REF_CACHE)
    return ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    n_ref = cfg["preprocess"]["reference_beats_per_domain"]
    routes = discover_routes(args.routes)
    LOG.info("routes: %s", routes)
    if not routes:
        raise SystemExit("no generated signals found")
    ref = load_ref(cfg)

    rows = []
    t_all = time.time()
    for route in routes:
        sigs = load_route_signals(route, cfg["generation"]["n_conditional"])
        fr = FailureRecorder(f"regularity_{route}")
        t0 = time.time()
        interval_rows, fails = intervals.extract_from_signals(
            sigs, cfg["fs"], cfg["lead_ii_index"])
        for f in fails:
            fr.record(str(f["item"]), str(f["reason"]))
        fr.flush()
        df = pd.DataFrame(interval_rows)
        LOG.info("route %s: n=%d, interval rows=%d, failures=%d (%.0fs)", route, len(sigs),
                 len(df), len(fails), time.time() - t0)
        for param in PARAMS:
            gen = df[param].values if param in df else np.array([])
            gen = np.asarray(gen, dtype=float)
            gen = gen[np.isfinite(gen)]
            if len(gen) < 30:
                LOG.warning("  %s: only %d valid values, distribution unreliable", param, len(gen))
            for domain in ("in_domain", "cross_domain"):
                refv = np.asarray(ref[domain][param], dtype=float)
                refv = refv[np.isfinite(refv)]
                if len(gen) < 2 or len(refv) < 2:
                    w = float("nan")
                    ks_stat, ks_p = float("nan"), float("nan")
                else:
                    w = float(wasserstein_distance(gen, refv))
                    r = ks_2samp(gen, refv)
                    ks_stat, ks_p = float(r.statistic), float(r.pvalue)
                rows.append({
                    "route": route, "metric_group": "regularity", "param": param,
                    "domain": domain, "metric": "wasserstein", "value": w,
                    "ks_stat": ks_stat, "ks_p": ks_p,
                    "n_route": int(len(gen)), "n_ref": int(len(refv)),
                    "n_failed_delineation": int(len(fails)),
                })
                LOG.info("  %-6s %-12s Wasserstein=%.2f KS=%.3f (p=%.2g) n=%d/%d",
                         param, domain, w, ks_stat, ks_p, len(gen), len(refv))
        # Delta_Wasserstein rows
        for param in PARAMS:
            sub = [r for r in rows if r["route"] == route and r["param"] == param]
            win = next((r["value"] for r in sub if r["domain"] == "in_domain"), float("nan"))
            wcross = next((r["value"] for r in sub if r["domain"] == "cross_domain"), float("nan"))
            rows.append({
                "route": route, "metric_group": "regularity", "param": param,
                "domain": "delta", "metric": "delta_wasserstein",
                "value": float(wcross - win) if np.isfinite(win) and np.isfinite(wcross) else float("nan"),
                "ks_stat": float("nan"), "ks_p": float("nan"),
                "n_route": int(len(df)), "n_ref": int(n_ref),
                "n_failed_delineation": int(len(fails)),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(paths.RESULTS, "metrics_regularity.csv"), index=False)
    save_json(os.path.join(paths.LOGS, "eval_regularity.json"), {
        "routes": routes,
        "reference_sizes": {d: {k: int(len(v)) for k, v in ref[d].items()} for d in ref},
        "elapsed_s": time.time() - t_all,
    })
    LOG.info("saved results/metrics_regularity.csv (%d rows, %.0fs)", len(out), time.time() - t_all)


if __name__ == "__main__":
    main()
