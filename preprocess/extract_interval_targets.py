"""Extract per-record interval targets (QT/PR/QRS/RR-CV) for the training and
validation splits -- used by the v2 diffusion models as an auxiliary "regularity
injection" signal (auxiliary regression head) and for interval-conditioning.

Method: same hard-coded pipeline as evaluation (`ecg_common.intervals`):
scipy/PT-style R peaks on Lead II + NeuroKit2 ecg_delineate (dwt) on the full 10 s
signal; one beat per record (closest to the middle); failures recorded with reason.

Outputs (preprocess/cache/):
  interval_targets_train.npz / interval_targets_val.npz
      qt, pr, qrs, rr_cv, valid masks  (float32, NaN where failed)
  interval_targets_report.md/.json

Runtime: ~0.25-0.35 s/record -> ~80-100 min for the full train+val splits (CPU).

Usage: python preprocess/extract_interval_targets.py [--max-records N]
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
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("neurokit2").setLevel(logging.ERROR)

from ecg_common.config import load_config, seed_everything
from ecg_common.logging_utils import get_logger, save_json

LOG = get_logger("extract_interval_targets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    seed_everything()
    fs = cfg["fs"]
    lead = cfg["lead_ii_index"]
    t0 = time.time()

    summary = {}
    for split in ("train", "val"):
        z = np.load(os.path.join(paths.CACHE, f"ptbxl_{split}.npz"), allow_pickle=False)
        sigs = z["signals"]
        if args.max_records:
            sigs = sigs[: args.max_records]
        n = len(sigs)
        rows, fails = [], []
        part_path = os.path.join(paths.CACHE, f"interval_targets_{split}.npz")
        if os.path.exists(part_path):
            z0 = np.load(part_path, allow_pickle=False)
            if "done" in z0 and int(z0["done"]) < n:
                done = int(z0["done"])
                LOG.info("%s: resuming after %d/%d records", split, done, n)
                partial = {k: z0[k].copy() for k in ("qt", "pr", "qrs", "rr_cv")}
                for i in range(done):
                    rows.append({"i": i, "QT_ms": partial["qt"][i], "PR_ms": partial["pr"][i],
                                 "QRS_ms": partial["qrs"][i], "rr_cv": partial["rr_cv"][i]})
                start_i = done
            else:
                start_i = 0
                done = 0
        else:
            start_i, done = 0, 0
        LOG.info("%s: extracting intervals for %d records (from %d)", split, n, start_i)
        for i in range(start_i, n):
            lead_sig = np.asarray(sigs[i, lead], dtype=np.float64)
            peaks = intervals.detect_rpeaks(lead_sig, fs)
            tv = intervals.rr_variability(lead_sig, fs)
            row = {"i": i, "rr_cv": tv["rr_cv"]}
            if len(peaks) < 2:
                fails.append({"item": i, "reason": "too_few_rpeaks"})
                row.update({"QT_ms": np.nan, "PR_ms": np.nan, "QRS_ms": np.nan})
            else:
                d = intervals.delineate_signal(lead_sig, fs, peaks)
                if d is None:
                    fails.append({"item": i, "reason": "delineate_error"})
                    row.update({"QT_ms": np.nan, "PR_ms": np.nan, "QRS_ms": np.nan})
                else:
                    # use the beat closest to the middle of the record
                    mid = len(lead_sig) // 2
                    bi = int(np.argmin(np.abs(peaks - mid)))
                    iv = intervals.beat_intervals(d, bi, fs)
                    row.update(iv)
                    row.pop("failure", None)
            rows.append(row)
            if (i + 1) % 1000 == 0:
                LOG.info("  %s %d/%d (%.0fs)", split, i + 1, n, time.time() - t0)
                part = pd.DataFrame(rows).set_index("i")
                np.savez(part_path,
                         qt=part.get("QT_ms", pd.Series(np.nan, index=part.index)).astype(np.float32).values,
                         pr=part.get("PR_ms", pd.Series(np.nan, index=part.index)).astype(np.float32).values,
                         qrs=part.get("QRS_ms", pd.Series(np.nan, index=part.index)).astype(np.float32).values,
                         rr_cv=part.get("rr_cv", pd.Series(np.nan, index=part.index)).astype(np.float32).values,
                         done=np.int64(i + 1))
        df = pd.DataFrame(rows).set_index("i")
        qt = df["QT_ms"].astype(np.float32).values if "QT_ms" in df else np.full(n, np.nan, np.float32)
        pr = df["PR_ms"].astype(np.float32).values if "PR_ms" in df else np.full(n, np.nan, np.float32)
        qrs = df["QRS_ms"].astype(np.float32).values if "QRS_ms" in df else np.full(n, np.nan, np.float32)
        rrcv = df["rr_cv"].astype(np.float32).values if "rr_cv" in df else np.full(n, np.nan, np.float32)
        out = os.path.join(paths.CACHE, f"interval_targets_{split}.npz")
        np.savez(out, qt=qt, pr=pr, qrs=qrs, rr_cv=rrcv)
        reason_counts = {}
        for f in fails:
            k = f["reason"].split(":")[0]
            reason_counts[k] = reason_counts.get(k, 0) + 1
        summary[split] = {
            "n": n,
            "valid_qt": int(np.isfinite(qt).sum()),
            "valid_pr": int(np.isfinite(pr).sum()),
            "valid_qrs": int(np.isfinite(qrs).sum()),
            "valid_rr_cv": int(np.isfinite(rrcv).sum()),
            "failures": reason_counts,
            "median_qt": float(np.nanmedian(qt)),
            "median_pr": float(np.nanmedian(pr)),
            "median_qrs": float(np.nanmedian(qrs)),
        }
        LOG.info("%s done: %s -> %s", split, summary[split], out)

    rep = ["# Interval targets for v2 auxiliary regularity supervision\n",
           f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
           "- pipeline: Lead II R peaks (PT-style) + NeuroKit2 ecg_delineate (dwt) on full 10 s; "
           "beat closest to record middle; failures recorded\n\n"]
    for split, s in summary.items():
        rep.append(f"## {split}\n")
        rep.append(f"- n={s['n']}, valid QT={s['valid_qt']}, PR={s['valid_pr']}, "
                   f"QRS={s['valid_qrs']}, RR-CV={s['valid_rr_cv']}\n")
        rep.append(f"- median QT={s['median_qt']:.1f} ms, PR={s['median_pr']:.1f} ms, "
                   f"QRS={s['median_qrs']:.1f} ms\n")
        rep.append(f"- failure reasons: {s['failures']}\n\n")
    with open(os.path.join(paths.CACHE, "interval_targets_report.md"), "w", encoding="utf-8") as f:
        f.write("".join(rep))
    save_json(os.path.join(paths.CACHE, "interval_targets_report.json"),
              {"summary": summary, "elapsed_s": time.time() - t0})
    LOG.info("ALL DONE in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
