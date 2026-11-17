"""Build the MIT-BIH (MIT-BIH Arrhythmia Database) cross-domain reference cache.

Task book 2/5.2:
  - MLII lead only, keep native 360 Hz, NO resampling
  - same preprocessing as PTB-XL: 0.5-40 Hz band-pass + 50 Hz notch
    (fs=360 permits the notch here, unlike PTB-XL at 100 Hz), per-record z-score
  - reference distribution = beats annotated as 'N' (normal) only
  - MIT-BIH is used ONLY for cross-domain evaluation: no training, no tuning

Data source (see preprocess/download_data.py): the HF mirror copy of the official
MIT-BIH 1.0.0 release, stored as a single parquet with 48 rows:
  signals (T,2) @ 360 Hz in mV, annotation_sample, annotation_symbol, header JSON.
A WFDB directory (data/mit-bih/*.hea) is also supported if present.

Outputs (preprocess/cache/):
  mitbih.npz               sampled normal-beat windows + metadata
  mitbih_records/<id>.npy  filtered full MLII record (native 360 Hz)
  mitbih_report.md / .json

Usage: python preprocess/make_mitbih_cache.py [--max-beats N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config, preprocess_record, seed_everything
from ecg_common.logging_utils import FailureRecorder, get_logger, save_json

LOG = get_logger("make_mitbih_cache")


def load_records_from_parquet(path: str) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    cols = {name: t.column(name).to_pylist() for name in t.column_names}
    n = t.num_rows
    LOG.info("parquet rows (records): %d", n)
    records = []
    for i in range(n):
        header = json.loads(cols["header"][i]) if isinstance(cols["header"][i], str) \
            else cols["header"][i]
        records.append({
            "record": str(header.get("record_name", f"rec{i:03d}")),
            "fs": int(header.get("fs", 360)),
            "sig_name": list(header.get("sig_name", ["MLII", "V5"])),
            "signals": np.asarray(cols["signals"][i], dtype=np.float64),
            "ann_sample": np.asarray(cols["annotation_sample"][i], dtype=np.int64),
            "ann_symbol": np.asarray(cols["annotation_symbol"][i], dtype=object),
        })
    return records


def load_records_from_wfdb(dir_: str) -> list[dict]:
    import wfdb
    records = []
    for hea in sorted(glob.glob(os.path.join(dir_, "*.hea"))):
        name = os.path.splitext(os.path.basename(hea))[0]
        rec = wfdb.rdrecord(os.path.join(dir_, name))
        ann = wfdb.rdann(os.path.join(dir_, name), "atr")
        records.append({
            "record": name, "fs": int(rec.fs), "sig_name": list(rec.sig_name),
            "signals": np.asarray(rec.p_signal, dtype=np.float64),
            "ann_sample": np.asarray(ann.sample, dtype=np.int64),
            "ann_symbol": np.asarray(ann.symbol, dtype=object),
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-beats", type=int, default=0, help="cap total sampled N beats")
    args = ap.parse_args()

    cfg = load_config()
    pp = cfg["preprocess"]
    seed = seed_everything()
    pre_n = int(round(pp["mitbih_beat_pre_s"] * pp["mitbih_fs"]))
    post_n = int(round(pp["mitbih_beat_post_s"] * pp["mitbih_fs"]))
    win = pre_n + post_n
    n_ref = args.max_beats or pp["reference_beats_per_domain"]

    parquet = os.path.join(paths.DATA_MITBIH, "mitbih.parquet")
    if os.path.exists(parquet):
        records = load_records_from_parquet(parquet)
        source = "hf-mirror: epr-labs/mit-bih-arrhythmia-database (official mitdb 1.0.0 content)"
    elif glob.glob(os.path.join(paths.DATA_MITBIH, "*.hea")):
        records = load_records_from_wfdb(paths.DATA_MITBIH)
        source = "local WFDB files"
    else:
        raise SystemExit(f"no MIT-BIH data in {paths.DATA_MITBIH}; run preprocess/download_data.py")
    LOG.info("MIT-BIH records: %d (source: %s)", len(records), source)

    failures = FailureRecorder("mitbih_preprocess")
    rec_dir = os.path.join(paths.CACHE, "mitbih_records")
    os.makedirs(rec_dir, exist_ok=True)

    per_record = []
    beat_pool = []
    t0 = time.time()
    for rec in records:
        rec_name = rec["record"]
        fs = rec["fs"]
        if fs != pp["mitbih_fs"]:
            failures.record(f"record={rec_name}", f"unexpected_fs:{fs}")
            continue
        try:
            names = [n.upper() for n in rec["sig_name"]]
            ch = names.index("MLII") if "MLII" in names else 0
            if "MLII" not in names:
                failures.record(f"record={rec_name}",
                                f"mlii_missing: using channel {rec['sig_name'][0]}")
            raw = np.asarray(rec["signals"][:, ch], dtype=np.float64)
            filt, info = preprocess_record(raw[:, None], fs, cfg)
            sig = filt[:, 0]
            if not np.isfinite(sig).all():
                raise ValueError("non-finite after preprocessing")
            np.save(os.path.join(rec_dir, f"{rec_name}.npy"), sig.astype(np.float32))

            sym = np.asarray(rec["ann_symbol"])
            samp = np.asarray(rec["ann_sample"], dtype=np.int64)
            normal = sym == pp["mitbih_normal_symbol"]
            rpeaks = samp[normal]
            rpeaks = rpeaks[(rpeaks >= pre_n) & (rpeaks + post_n <= len(sig))]
            for r in rpeaks:
                beat_pool.append((rec_name, int(r), sig[r - pre_n:r + post_n].copy()))
            hr = float("nan")
            if len(rpeaks) >= 3:
                rr = np.diff(rpeaks) / fs
                rr = rr[(rr > 0.2) & (rr < 2.5)]
                if len(rr) > 1:
                    hr = float(60.0 / np.median(rr))
            per_record.append({
                "record": rec_name, "n_samples": int(len(sig)), "fs": int(fs),
                "n_beats_total": int(len(samp)), "n_normal_N": int(normal.sum()),
                "usable_N_windows": int(len(rpeaks)), "hr_bpm": hr,
                "notch_applied": bool(info["notch_applied"]),
            })
            LOG.info("  %s: len=%d N=%d usable=%d hr=%.1f", rec_name, len(sig),
                     int(normal.sum()), len(rpeaks), hr)
        except Exception as exc:  # noqa: BLE001
            failures.record(f"record={rec_name}", f"record_failed: {type(exc).__name__}: {exc}")
            LOG.error("  %s FAILED: %s", rec_name, exc)

    LOG.info("total usable N-beat windows: %d in %.0fs", len(beat_pool), time.time() - t0)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(beat_pool), size=min(n_ref, len(beat_pool)), replace=False))
    beats = np.stack([beat_pool[i][2] for i in idx]).astype(np.float32)
    beat_record = np.array([beat_pool[i][0] for i in idx])
    beat_rpeak = np.array([beat_pool[i][1] for i in idx], dtype=np.int32)

    np.savez(os.path.join(paths.CACHE, "mitbih.npz"),
             beats=beats[:, None, :],
             rpeak_idx=np.full(len(beats), pre_n, dtype=np.int32),
             record_ids=beat_record,
             rpeak_abs=beat_rpeak,
             fs=np.array([pp["mitbih_fs"]], dtype=np.int32),
             pre_n=np.array([pre_n], dtype=np.int32),
             win=np.array([win], dtype=np.int32))
    LOG.info("saved preprocess/cache/mitbih.npz beats=%s", beats.shape)

    rep = ["# MIT-BIH cross-domain reference cache\n",
           f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
           f"- source: {source}\n",
           f"- records: {len(per_record)} / {len(records)}\n",
           "- lead: MLII (native 360 Hz, **not resampled**), 0.5-40 Hz band-pass + "
           "50 Hz notch, per-record z-score\n",
           f"- normal beats ('N' annotation) usable windows: {len(beat_pool)}\n",
           f"- reference distribution: fixed-seed sample of {len(beats)} N-beat windows "
           f"({pre_n} samples before R + {post_n} after at {pp['mitbih_fs']} Hz = {win} samples)\n\n",
           "| record | n_normal_N | usable | hr_bpm |\n|---|---|---|---|\n"]
    for r in per_record:
        rep.append(f"| {r['record']} | {r['n_normal_N']} | {r['usable_N_windows']} | {r['hr_bpm']:.1f} |\n")
    fail_path = failures.flush()
    rep.append(f"\n## failures\n\n- {failures.summary()}\n- detail: `{fail_path}`\n")
    with open(os.path.join(paths.CACHE, "mitbih_report.md"), "w", encoding="utf-8") as f:
        f.write("".join(rep))
    save_json(os.path.join(paths.CACHE, "mitbih_report.json"), {
        "n_records": len(per_record), "n_normal_windows": len(beat_pool),
        "n_reference_beats": int(len(beats)), "source": source,
        "failures": failures.summary(), "elapsed_s": time.time() - t0,
    })
    LOG.info("report -> preprocess/cache/mitbih_report.md")
    LOG.info("ALL DONE")


if __name__ == "__main__":
    main()
