"""Morphology-layer evaluation (task book 5.1) using the `ecgen-eval` library.

All four routes are scored with the SAME library and the SAME config against the
SAME real reference (PTB-XL test split, fold 10, 100 Hz, 12 leads). This script
must never be changed after numbers are produced.

Metrics (ecgen-eval 0.8.0):
  mmd : MMDMetric()            distribution distance in waveform space
  dtw : DTWMetric()            200 sampled real/synth pairs, per-lead DTW
  psd : PSDMetric(fs=100)      spectral distance
  prd : PRDMetric()            percent root-mean-square difference
Note PSD fs is set to the project sampling rate (100 Hz) explicitly.

Outputs:
  results/metrics_morphology.csv   long format: route, metric, value, n_real, n_synth
  results/logs/eval_morphology.json

Usage: python experiments/eval_morphology.py [--routes A_cond B_uncond ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config
from ecg_common.logging_utils import get_logger, save_json

LOG = get_logger("eval_morphology")

ROUTES_ORDER = ["A", "B_uncond", "B_cond", "c_full", "c_no_label", "c_no_hr",
                "c_label_noise", "c_label_shuffle", "d_seed42", "d_seed43", "d_seed44",
                "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"]


def discover_routes(only=None) -> list[str]:
    found = []
    for pattern in ("A_cond.npz", "*_cond.npz", "*_uncond.npz"):
        for f in glob.glob(os.path.join(paths.GENERATED, pattern)):
            name = os.path.basename(f)
            route = name.replace("_cond.npz", "").replace("_uncond.npz", "")
            if route not in found:
                found.append(route)
    def key(r):
        return ROUTES_ORDER.index(r) if r in ROUTES_ORDER else 99
    found.sort(key=key)
    return [r for r in found if not only or r in only]


def load_route_signals(route: str) -> np.ndarray:
    for suffix in ("_cond.npz", "_uncond.npz"):
        p = os.path.join(paths.GENERATED, route + suffix)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            return z["signals"]
    raise FileNotFoundError(route)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config()
    leads = cfg["lead_names"]
    real = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)["signals"]
    LOG.info("real reference: PTB-XL test signals %s", real.shape)

    from ecgen_eval.metrics.mmd import MMDMetric
    from ecgen_eval.metrics.dtw import DTWMetric
    from ecgen_eval.metrics.psd import PSDMetric
    from ecgen_eval.metrics.prd import PRDMetric
    from ecgen_eval.metrics.fd import FDMetric
    from ecgen_eval.metrics.swd import SWDMetric
    from ecgen_eval.metrics.nn_dist import NNDistMetric
    from ecgen_eval.metrics.pr_dist import PRDistMetric
    from ecgen_eval.metrics.heart_rate import HeartRateMetric
    from ecgen_eval.metrics.spectral_entropy import SpectralEntropyMetric

    metrics = {
        # v1 core 4
        "mmd": MMDMetric(),
        "dtw": DTWMetric(seed=cfg["seed"]),
        "psd": PSDMetric(fs=float(cfg["fs"])),
        "prd": PRDMetric(seed=cfg["seed"]),
        # v2 additions (all from ecgen-eval, no pretrained weights needed)
        "fd": FDMetric(n_subsample=2000),
        "swd": SWDMetric(seed=cfg["seed"]),
        "nn_dist": NNDistMetric(seed=cfg["seed"]),
        "pr_dist": PRDistMetric(seed=cfg["seed"]),
        "heart_rate": HeartRateMetric(fs=float(cfg["fs"])),
        "spectral_entropy": SpectralEntropyMetric(fs=float(cfg["fs"])),
    }
    routes = discover_routes(args.routes)
    LOG.info("routes: %s", routes)
    if not routes:
        raise SystemExit("no generated signals found under results/generated/")

    rows, details = [], {}
    t_all = time.time()
    for route in routes:
        synth = load_route_signals(route)
        LOG.info("route %s: %s", route, synth.shape)
        details[route] = {}
        for mname, metric in metrics.items():
            t0 = time.time()
            res = metric.compute(real, synth, lead_names=leads)
            rows.append({
                "route": route, "metric_group": "morphology", "metric": mname,
                "domain": "in_domain", "value": float(res.score), "n_real": int(len(real)),
                "n_synth": int(len(synth)), "higher_is_better": bool(res.higher_is_better),
                "runtime_s": round(time.time() - t0, 1),
            })
            details[route][mname] = {"score": float(res.score),
                                     "per_lead": {k: float(v) for k, v in
                                                  (res.per_lead_scores or {}).items()}}
            LOG.info("  %-5s = %10.4f (%.1fs)", mname, res.score, time.time() - t0)
    df = pd.DataFrame(rows)
    out = os.path.join(paths.RESULTS, "metrics_morphology.csv")
    df.to_csv(out, index=False)
    save_json(os.path.join(paths.LOGS, "eval_morphology.json"), details)
    LOG.info("saved %s (%d rows, %.0fs)", out, len(df), time.time() - t_all)


if __name__ == "__main__":
    main()
