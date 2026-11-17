"""Build the PTB-XL 100 Hz cache used by every model.

Steps (hard-coded per task book):
  1. labels: ptbxl_database.csv -> scp_codes -> 5 diagnostic superclasses
     (NORM/MI/STTC/CD/HYP) using scp_statements.csv (`diagnostic == 1`).
     Records without any diagnostic superclass are dropped (count reported).
  2. signals: WFDB records100 (100 Hz, 1000 samples, 12 leads) ->
     0.5-40 Hz Butterworth band-pass (order 3, zero-phase) ->
     50 Hz notch (only when fs allows; PTB-XL 100 Hz == Nyquist -> skipped,
     documented in the data report) -> per-lead z-score.
  3. target heart rate: scipy R-peak detector on Lead II (hard-coded spec).
  4. split by official strat_fold: 1-8 train, 9 val, 10 test.

Outputs (preprocess/cache/):
  ptbxl_train.npz, ptbxl_val.npz, ptbxl_test.npz   (signals/labels/hr/ids/fold)
  ptbxl_meta.csv                                  (per-record metadata incl. hr)
  data_report.md / data_report.csv                (counts, distributions, failures)

Usage: python preprocess/make_ptbxl_cache.py [--limit N]
"""
from __future__ import annotations

import argparse
import ast
import collections
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import data as ecg_data
from ecg_common import paths
from ecg_common.config import load_config, preprocess_record, estimate_hr, seed_everything
from ecg_common.logging_utils import FailureRecorder, get_logger, save_json

LOG = get_logger("make_ptbxl_cache")


def aggregate_labels(db: pd.DataFrame, scp: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    agg = scp[scp["diagnostic"] == 1]
    mapping = agg["diagnostic_class"].to_dict()

    def to_labels(code_dict: dict) -> list[str]:
        labs = {mapping[k] for k in code_dict if k in mapping}
        return [c for c in classes if c in labs]

    db = db.copy()
    db["scp_codes_dict"] = db["scp_codes"].apply(lambda s: ast.literal_eval(s))
    db["labels"] = db["scp_codes_dict"].apply(to_labels)
    db["n_labels"] = db["labels"].apply(len)
    return db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: only process first N records")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    classes = cfg["superclasses"]
    fs = cfg["fs"]
    n_samples = cfg["n_samples"]
    folds = cfg["folds"]
    t_start = time.time()

    db = pd.read_csv(os.path.join(paths.DATA_PTBXL, "ptbxl_database.csv"), index_col="ecg_id")
    scp = pd.read_csv(os.path.join(paths.DATA_PTBXL, "scp_statements.csv"), index_col=0)
    LOG.info("ptbxl_database rows=%d, scp_statements rows=%d", len(db), len(scp))

    db = aggregate_labels(db, scp, classes)
    n_total = len(db)
    db = db[db["n_labels"] > 0].copy()
    LOG.info("records with >=1 diagnostic superclass: %d / %d (dropped %d)",
             len(db), n_total, n_total - len(db))
    if args.limit:
        db = db.head(args.limit).copy()
        LOG.warning("DEBUG --limit %d", args.limit)

    folders = {"train": folds["train"], "val": folds["val"], "test": folds["test"]}
    splits = {k: db[db["strat_fold"].isin(v)] for k, v in folders.items()}
    LOG.info("split sizes: %s", {k: len(v) for k, v in splits.items()})

    failures = FailureRecorder("ptbxl_preprocess")
    meta_rows = []
    out = {}
    rec_root = os.path.join(paths.DATA_PTBXL)

    import wfdb

    for split, part in splits.items():
        sigs, labels, hrs, ecg_ids, patient_ids, fold_ids = [], [], [], [], [], []
        t0 = time.time()
        for i, (ecg_id, row) in enumerate(part.iterrows(), 1):
            rec_path = os.path.join(rec_root, str(row["filename_lr"]))
            try:
                rec = wfdb.rdrecord(rec_path)
                sig = np.asarray(rec.p_signal, dtype=np.float64)  # (1000, 12)
                if sig.shape != (n_samples, cfg["n_leads"]):
                    raise ValueError(f"shape {sig.shape}")
                lead_order = [n.upper() for n in rec.sig_name]
                if lead_order != [n.upper() for n in cfg["lead_names"]]:
                    raise ValueError(f"lead order {rec.sig_name}")
                sig_p, info = preprocess_record(sig, fs, cfg)
                if not info["zscore_ok"]:
                    raise ValueError("non-finite after preprocessing")
                hr, n_peaks = estimate_hr(sig_p[:, cfg["lead_ii_index"]], fs, cfg)
                if not np.isfinite(hr):
                    failures.record(f"ecg_id={ecg_id}", f"hr_failed: n_peaks={n_peaks}")
            except Exception as exc:  # noqa: BLE001 -- must log, never skip silently
                failures.record(f"ecg_id={ecg_id}", f"record_failed: {type(exc).__name__}: {exc}")
                continue
            sigs.append(sig_p.T.copy())  # (12, 1000)
            lab = np.zeros(len(classes), dtype=np.uint8)
            for c in row["labels"]:
                lab[classes.index(c)] = 1
            labels.append(lab)
            hrs.append(hr)
            ecg_ids.append(int(ecg_id))
            patient_ids.append(int(row["patient_id"]) if not pd.isna(row["patient_id"]) else -1)
            fold_ids.append(int(row["strat_fold"]))
            meta_rows.append({
                "ecg_id": int(ecg_id), "split": split, "strat_fold": int(row["strat_fold"]),
                "patient_id": row["patient_id"], "age": row["age"], "sex": row["sex"],
                "hr_bpm": hr,
                "n_labels": int(row["n_labels"]),
                "labels": "|".join(row["labels"]),
                "scp_codes": row["scp_codes"],
            })
            if i % 1000 == 0:
                LOG.info("  %s %d/%d (%.0fs)", split, i, len(part), time.time() - t0)

        arr = {
            "signals": np.stack(sigs).astype(np.float32),
            "labels": np.stack(labels).astype(np.uint8),
            "hr": np.asarray(hrs, dtype=np.float32),
            "ecg_id": np.asarray(ecg_ids, dtype=np.int32),
            "patient_id": np.asarray(patient_ids, dtype=np.int32),
            "fold": np.asarray(fold_ids, dtype=np.int8),
        }
        out[split] = arr
        p = os.path.join(paths.CACHE, f"ptbxl_{split}.npz")
        np.savez(p, **arr)
        LOG.info("%s saved %s signals=%s (%.0fs)", split, p, arr["signals"].shape, time.time() - t0)

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(os.path.join(paths.CACHE, "ptbxl_meta.csv"), index=False)
    LOG.info("metadata -> preprocess/cache/ptbxl_meta.csv (%d rows)", len(meta))

    # ------------------------------------------------------------------ report
    rep = []
    rep.append("# PTB-XL 100 Hz cache report\n")
    rep.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    rep.append(f"- records in ptbxl_database.csv: {n_total}\n")
    rep.append(f"- kept (>=1 diagnostic superclass): {n_total - (n_total - len(db) if not args.limit else 0)}\n")
    rep.append(f"- dropped (no diagnostic superclass): {n_total - len(db) if not args.limit else 'n/a (limit)'}\n")
    rep.append("- preprocessing: 0.5-40 Hz Butterworth band-pass order 3 (zero-phase), "
               "50 Hz notch **skipped at 100 Hz** (50 Hz == Nyquist; PTB-XL records100 "
               "is already anti-alias filtered), per-lead z-score\n")
    rep.append(f"- heart rate: scipy R-peak detector on Lead II (index {cfg['lead_ii_index']})\n\n")
    rep.append("## split sizes / label distribution\n\n")
    rep.append("| split | n | " + " | ".join(classes) + " | mean HR | HR missing |\n")
    rep.append("|---|---|" + "---|" * len(classes) + "---|---|\n")
    for split, arr in out.items():
        lab = arr["labels"].astype(int)
        rep.append(f"| {split} | {len(lab)} | " + " | ".join(str(lab[:, j].sum()) for j in range(len(classes)))
                   + f" | {np.nanmean(arr['hr']):.1f} | {int(np.isnan(arr['hr']).sum())} |\n")
    rep.append("\n## per-fold counts\n\n")
    rep.append("| fold | n |\n|---|---|\n")
    for f, c in collections.Counter(meta["strat_fold"]).items():
        rep.append(f"| {f} | {c} |\n")
    fail_path = failures.flush()
    rep.append(f"\n## failures\n\n- {failures.summary()}\n- detail: `{fail_path}`\n")
    with open(os.path.join(paths.CACHE, "data_report.md"), "w", encoding="utf-8") as f:
        f.write("".join(rep))
    save_json(os.path.join(paths.CACHE, "data_report.json"), {
        "n_total": n_total, "n_kept": int(sum(len(a["signals"]) for a in out.values())),
        "splits": {k: len(v["signals"]) for k, v in out.items()},
        "failures": failures.summary(),
        "elapsed_s": time.time() - t_start,
    })
    LOG.info("report -> preprocess/cache/data_report.md")
    LOG.info("ALL DONE in %.0fs", time.time() - t_start)


if __name__ == "__main__":
    main()
