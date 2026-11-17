"""Recompute the cached heart-rate labels with the project-wide R-peak detector.

Why: the first PTB-XL cache pass used a plain band-pass + prominence peak picker
that occasionally counted T waves as R peaks (mean HR ~108 bpm instead of the
plausible ~75 bpm). The detector was replaced by a Pan-Tompkins-style one
(`ecg_common.config.detect_rpeaks`, validated against NeuroKit2) and this script
patches the existing caches in place instead of re-reading 21k WFDB records.

Patches:
  preprocess/cache/ptbxl_{train,val,test}.npz   -> 'hr' array replaced
  preprocess/cache/ptbxl_meta.csv               -> 'hr_bpm' column replaced
  preprocess/cache/data_report.md               -> HR summary refreshed

Usage: python preprocess/recompute_hr.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import estimate_hr, load_config, seed_everything
from ecg_common.logging_utils import FailureRecorder, get_logger

LOG = get_logger("recompute_hr")


def main() -> None:
    cfg = load_config()
    seed_everything()
    fs = cfg["fs"]
    lead = cfg["lead_ii_index"]
    t0 = time.time()
    summary = {}
    meta = pd.read_csv(os.path.join(paths.CACHE, "ptbxl_meta.csv"))
    fail = FailureRecorder("recompute_hr")

    for split in ("train", "val", "test"):
        p = os.path.join(paths.CACHE, f"ptbxl_{split}.npz")
        z = dict(np.load(p, allow_pickle=False))
        sigs = z["signals"]
        hrs = np.full(len(sigs), np.nan, dtype=np.float32)
        for i in range(len(sigs)):
            hr, n_peaks = estimate_hr(sigs[i, lead].astype(np.float64), fs, cfg)
            hrs[i] = hr
            if not np.isfinite(hr):
                fail.record(f"{split}:{i}", f"hr_failed_after_recompute: n_peaks={n_peaks}")
        z["hr"] = hrs
        np.savez(p, **z)
        summary[split] = {"n": int(len(hrs)), "n_missing": int(np.isnan(hrs).sum()),
                          "mean": float(np.nanmean(hrs)), "std": float(np.nanstd(hrs))}
        LOG.info("%s: n=%d mean HR=%.1f std=%.1f missing=%d", split, len(hrs),
                 np.nanmean(hrs), np.nanstd(hrs), np.isnan(hrs).sum())
        # patch metadata rows for this split
        mask = meta["split"] == split
        if mask.sum() == len(hrs):
            meta.loc[mask, "hr_bpm"] = hrs
        else:
            LOG.warning("metadata/split length mismatch for %s (%d vs %d), skipping meta patch",
                        split, mask.sum(), len(hrs))
    meta.to_csv(os.path.join(paths.CACHE, "ptbxl_meta.csv"), index=False)
    fail_path = fail.flush()
    LOG.info("HR failures: %s", fail.summary())
    LOG.info("updated %s", fail_path)

    # refresh the data-report HR section
    rep_path = os.path.join(paths.CACHE, "data_report.md")
    if os.path.exists(rep_path):
        with open(rep_path, "a", encoding="utf-8") as f:
            f.write("\n## HR recompute (Pan-Tompkins detector, after first cache pass)\n\n")
            f.write(f"- time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for k, v in summary.items():
                f.write(f"- {k}: n={v['n']}, mean HR={v['mean']:.1f} bpm, "
                        f"std={v['std']:.1f}, missing={v['n_missing']}\n")
            f.write(f"- detector: 5-15 Hz band-pass -> derivative -> square -> 80 ms moving average "
                    f"-> peaks (0.3 s refractory, median-based threshold)\n")
            f.write(f"- failures: {fail.summary()}\n")
    LOG.info("ALL DONE in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
