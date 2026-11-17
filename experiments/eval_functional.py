"""Functional-layer evaluation (task book 5.3) with the trained classifier.

Metrics:
  1. conditional consistency: for routes that use diagnostic-label conditioning,
     generate with condition class c and measure the fraction of samples the
     classifier predicts as c (per class + macro mean).
     Routes without label conditioning (A, B_uncond, D_seed*) are reported as
     not_applicable with the reason -- never fabricated.
  2. unconditional class distribution: predict 1000 unconditional samples per
     route, compare the 5-class predicted prevalence with the real PTB-XL test
     class prevalence -> KL divergence (Laplace-smoothed) + saved distributions.

Route names are lowercase for the C variants (c_full, ...); D routes are
d_seed42/43/44.

Outputs:
  results/metrics_functional.csv
  results/class_distribution.csv
  results/logs/eval_functional.json

Usage: python experiments/eval_functional.py [--routes ...]
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

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.logging_utils import get_logger, save_json
from models.classifier import load_classifier, predict_proba

LOG = get_logger("eval_functional")
ROUTES_ORDER = ["A", "B_uncond", "B_cond", "c_full", "c_no_label", "c_no_hr",
                "c_label_noise", "c_label_shuffle", "d_seed42", "d_seed43", "d_seed44",
                "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"]
# routes whose training included the diagnostic label as an input condition
LABEL_CONDITIONED = {"B_cond", "c_full", "c_no_hr", "c_label_noise", "c_label_shuffle",
                     "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"}


def discover_routes(only=None) -> list[str]:
    found = set()
    for pattern in ("*_cond.npz", "*_uncond.npz"):
        for f in glob.glob(os.path.join(paths.GENERATED, pattern)):
            found.add(os.path.basename(f).replace("_cond.npz", "").replace("_uncond.npz", ""))
    routes = sorted(found, key=lambda r: ROUTES_ORDER.index(r) if r in ROUTES_ORDER else 99)
    return [r for r in routes if not only or r in only]


def kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    classes = cfg["superclasses"]
    model, cfg, device = load_classifier()
    LOG.info("classifier loaded on %s", device)

    test = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)
    real_prev = test["labels"].astype(float).mean(0)
    LOG.info("real test prevalence: %s", dict(zip(classes, real_prev.round(3))))

    routes = discover_routes(args.routes)
    LOG.info("routes: %s", routes)
    rows, dist_rows = [], []
    for cname, p_real in zip(classes, real_prev):
        dist_rows.append({"route": "REAL_PTBXL_TEST", "class": cname, "fraction": p_real,
                          "source": "true label prevalence (n=%d)" % len(test["labels"])})

    t_all = time.time()
    for route in routes:
        cond_p = os.path.join(paths.GENERATED, f"{route}_cond.npz")
        uncond_p = os.path.join(paths.GENERATED, f"{route}_uncond.npz")
        label_cond = route in LABEL_CONDITIONED

        # ------------------------------------------------ conditional set
        if label_cond:
            if os.path.exists(cond_p):
                z = np.load(cond_p, allow_pickle=False)
                sigs, labels = z["signals"], z["labels"].astype(int)
                probs = predict_proba(model, sigs, device=device)
                pred = (probs >= 0.5).astype(int)
                per_class = {}
                for ci, cname in enumerate(classes):
                    mask = labels[:, ci] == 1
                    per_class[cname] = float(pred[mask, ci].mean()) if mask.sum() else float("nan")
                macro = float(np.nanmean(list(per_class.values())))
                for cname, v in per_class.items():
                    rows.append({"route": route, "metric_group": "functional",
                                 "metric": "conditional_consistency", "param": cname,
                                 "value": v, "n": int(labels[:, classes.index(cname)].sum()),
                                 "note": ""})
                rows.append({"route": route, "metric_group": "functional",
                             "metric": "conditional_consistency_macro", "param": "mean",
                             "value": macro, "n": int(len(sigs)), "note": ""})
                LOG.info("route %s conditional consistency macro=%.3f %s", route, macro,
                         {k: round(v, 3) for k, v in per_class.items() if np.isfinite(v)})
            else:
                rows.append({"route": route, "metric_group": "functional",
                             "metric": "conditional_consistency_macro", "param": "mean",
                             "value": float("nan"), "n": 0,
                             "note": "missing conditional generation set"})
        else:
            rows.append({"route": route, "metric_group": "functional",
                         "metric": "conditional_consistency", "param": "N/A",
                         "value": float("nan"), "n": 0,
                         "note": "not_applicable: route has no diagnostic-label conditioning"})

        # ---------------------------------------------- unconditional set
        if os.path.exists(uncond_p):
            sigs_u = np.load(uncond_p, allow_pickle=False)["signals"]
            probs_u = predict_proba(model, sigs_u, device=device)
            gen_prev = (probs_u >= 0.5).astype(int).mean(0)
            d_kl = kl(real_prev, gen_prev)
            rows.append({"route": route, "metric_group": "functional",
                         "metric": "unconditional_class_KL", "param": "all5",
                         "value": float(d_kl), "n": int(len(sigs_u)), "note": ""})
            for ci, cname in enumerate(classes):
                dist_rows.append({"route": route, "class": cname,
                                  "fraction": float(gen_prev[ci]),
                                  "source": "classifier-predicted prevalence (n=%d)" % len(sigs_u)})
            LOG.info("route %s uncond KL=%.4f gen_prev=%s", route, d_kl,
                     dict(zip(classes, gen_prev.round(3))))
        else:
            rows.append({"route": route, "metric_group": "functional",
                         "metric": "unconditional_class_KL", "param": "all5",
                         "value": float("nan"), "n": 0,
                         "note": "missing unconditional generation set"})
            LOG.warning("route %s: no unconditional set, KL missing", route)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(paths.RESULTS, "metrics_functional.csv"), index=False)
    pd.DataFrame(dist_rows).to_csv(os.path.join(paths.RESULTS, "class_distribution.csv"),
                                   index=False)
    save_json(os.path.join(paths.LOGS, "eval_functional.json"),
              {"routes": routes, "label_conditioned": sorted(LABEL_CONDITIONED),
               "elapsed_s": time.time() - t_all})
    LOG.info("saved results/metrics_functional.csv + class_distribution.csv (%.0fs)",
             time.time() - t_all)


if __name__ == "__main__":
    main()
