"""Extract the reference interval distributions (QT/PR/QRS/RR-variability) for the
in-domain (PTB-XL test, fold 10) and cross-domain (MIT-BIH 'N' beats) evaluations.

Task book 5.2:
  - in-domain reference: PTB-XL test split real signals (100 Hz, 12 leads)
  - cross-domain reference: MIT-BIH normal ('N') beats (MLII, 360 Hz)
  - Lead II / MLII; NeuroKit2 ecg_delineate on >= 4 s signals; failures recorded
    with reasons (nothing silently skipped)

Reference size: `preprocess.reference_beats_per_domain` (default 1000 beats; the
per-route generated-signal sets are 500 per task requirement, 1000 reference
beats are ample for Wasserstein/KS estimation while keeping runtime sane).

Outputs (preprocess/cache/):
  ref_params_ptbxl_test.csv   columns: item, rpeak, QRS_ms, PR_ms, QT_ms, rr_cv, rmssd_ms, n_rr, failure
  ref_params_mitbih.csv
  ref_params_report.md / .json

Usage: python preprocess/extract_reference_params.py [--max-ptbxl N] [--max-mitbih N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import intervals, paths
from ecg_common.config import load_config, seed_everything
from ecg_common.logging_utils import get_logger, save_json

LOG = get_logger("extract_reference_params")


def summarize(rows: list[dict], failures: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {"n_beats": int(len(df)), "failures": len(failures), "failure_reasons": {}}
    for f in failures:
        key = f["reason"].split(":")[0]
        out["failure_reasons"][key] = out["failure_reasons"].get(key, 0) + 1
    for k in ("QRS_ms", "PR_ms", "QT_ms", "rr_cv"):
        if k in df.columns:
            out[f"n_valid_{k}"] = int(np.isfinite(df[k]).sum())
            out[f"mean_{k}"] = float(np.nanmean(df[k])) if np.isfinite(df[k]).any() else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ptbxl", type=int, default=None)
    ap.add_argument("--max-mitbih", type=int, default=None)
    ap.add_argument("--beats-per-sample", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    n_ref = cfg["preprocess"]["reference_beats_per_domain"]
    lead_ii = cfg["lead_ii_index"]
    fs = cfg["fs"]
    t0 = time.time()

    # ----------------------------------------------------------- in-domain ---
    z = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)
    test_signals = z["signals"]
    n = args.max_ptbxl or n_ref
    rng = np.random.default_rng(cfg["seed"])
    if n < len(test_signals):
        sel = np.sort(rng.choice(len(test_signals), size=n, replace=False))
        test_signals = test_signals[sel]
    LOG.info("PTB-XL test signals used for in-domain reference: %d", len(test_signals))
    rows, fails = intervals.extract_from_signals(
        test_signals, fs, lead_ii, max_beats_per_sample=args.beats_per_sample)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(paths.CACHE, "ref_params_ptbxl_test.csv"), index=False)
    ptbxl_summary = summarize(rows, fails)
    LOG.info("PTB-XL ref: %s", ptbxl_summary)

    # --------------------------------------------------------- cross-domain --
    mit = np.load(os.path.join(paths.CACHE, "mitbih.npz"), allow_pickle=False)
    record_ids = mit["record_ids"]
    rpeak_abs = mit["rpeak_abs"]
    mit_fs = int(mit["fs"][0])
    n2 = args.max_mitbih or n_ref
    if n2 < len(record_ids):
        sel2 = np.sort(rng.choice(len(record_ids), size=n2, replace=False))
        record_ids, rpeak_abs = record_ids[sel2], rpeak_abs[sel2]
    LOG.info("MIT-BIH reference beats used: %d (fs=%d)", len(record_ids), mit_fs)
    rows2, fails2 = intervals.extract_from_mitbih(
        record_ids, rpeak_abs, os.path.join(paths.CACHE, "mitbih_records"), mit_fs)
    df2 = pd.DataFrame(rows2)
    df2.to_csv(os.path.join(paths.CACHE, "ref_params_mitbih.csv"), index=False)
    mit_summary = summarize(rows2, fails2)
    LOG.info("MIT-BIH ref: %s", mit_summary)

    # ---------------------------------------------------------------- report
    rep = ["# Reference interval distributions (QT / PR / QRS)\n",
           f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
           "- delineation: NeuroKit2 ecg_delineate (dwt) on >= 4 s signals; "
           "Lead II (PTB-XL) / MLII (MIT-BIH)\n",
           "- R peaks: project-wide Pan-Tompkins-style detector (same as HR labels)\n",
           f"- beats per signal sampled (PTB-XL): {args.beats_per_sample}\n\n",
           "## in-domain reference: PTB-XL test (fold 10)\n",
           f"- beats delineated: {ptbxl_summary['n_beats']}, failures: {ptbxl_summary['failures']}\n",
           f"- failure reasons: {ptbxl_summary['failure_reasons']}\n",
           f"- valid counts: QRS={ptbxl_summary.get('n_valid_QRS_ms')}, "
           f"PR={ptbxl_summary.get('n_valid_PR_ms')}, QT={ptbxl_summary.get('n_valid_QT_ms')}, "
           f"RR-CV={ptbxl_summary.get('n_valid_rr_cv')}\n",
           f"- means: QRS={ptbxl_summary.get('mean_QRS_ms'):.1f} ms, "
           f"PR={ptbxl_summary.get('mean_PR_ms'):.1f} ms, "
           f"QT={ptbxl_summary.get('mean_QT_ms'):.1f} ms, "
           f"RR-CV={ptbxl_summary.get('mean_rr_cv'):.3f}\n\n",
           "## cross-domain reference: MIT-BIH normal beats\n",
           f"- beats delineated: {mit_summary['n_beats']}, failures: {mit_summary['failures']}\n",
           f"- failure reasons: {mit_summary['failure_reasons']}\n",
           f"- valid counts: QRS={mit_summary.get('n_valid_QRS_ms')}, "
           f"PR={mit_summary.get('n_valid_PR_ms')}, QT={mit_summary.get('n_valid_QT_ms')}\n",
           f"- means: QRS={mit_summary.get('mean_QRS_ms'):.1f} ms, "
           f"PR={mit_summary.get('mean_PR_ms'):.1f} ms, QT={mit_summary.get('mean_QT_ms'):.1f} ms\n"]
    with open(os.path.join(paths.CACHE, "ref_params_report.md"), "w", encoding="utf-8") as f:
        f.write("".join(rep))
    save_json(os.path.join(paths.CACHE, "ref_params_report.json"),
              {"ptbxl_test": ptbxl_summary, "mitbih": mit_summary,
               "elapsed_s": time.time() - t0})
    LOG.info("report -> preprocess/cache/ref_params_report.md")
    LOG.info("DONE in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
