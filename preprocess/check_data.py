"""Verify that the public datasets and preprocessing caches are present and valid.

The repository does NOT ship the datasets (size + licensing); each user downloads
them once with `preprocess/download_data.py` (see docs/DATA.md). This script tells
you exactly what is missing and which command to run next.

Usage:
  python preprocess/check_data.py
Exit code: 0 = ready for training/evaluation, 1 = something essential is missing.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "preprocess", "cache")

fail = 0
warn = 0


def row(status, name, detail, fix=""):
    global fail, warn
    print(f"[{status:4s}] {name:38s} {detail}" + (f"  ->  {fix}" if fix else ""))
    if status == "FAIL":
        fail += 1
    elif status == "WARN":
        warn += 1


def size_mb(p):
    return os.path.getsize(p) / 1024 / 1024 if os.path.exists(p) else 0


print("== raw datasets (download once, not shipped in the repo) ==")
ptbxl_zip = os.path.join(DATA, "ptb-xl", "ptb-xl-data.zip")
ptbxl_records = os.path.join(DATA, "ptb-xl", "records100")
mit_parquet = os.path.join(DATA, "mit-bih", "mitbih.parquet")
mit_wfdb = os.path.join(DATA, "mit-bih")
ptbxl_csv = os.path.join(DATA, "ptb-xl", "ptbxl_database.csv")

n_records = 0
if os.path.isdir(ptbxl_records):
    n_records = len([f for f in os.listdir(ptbxl_records) if f.endswith(".hea")]) if any(
        x.endswith(".hea") for x in os.listdir(ptbxl_records)) else sum(
        1 for sub in os.listdir(ptbxl_records)
        for f in os.listdir(os.path.join(ptbxl_records, sub))
        if f.endswith(".hea"))
n_mit = 0
if os.path.isdir(mit_wfdb):
    n_mit = sum(1 for f in os.listdir(mit_wfdb) if f.endswith(".hea"))

if os.path.exists(ptbxl_zip):
    row("OK", "PTB-XL archive", f"{size_mb(ptbxl_zip):7.0f} MB")
elif n_records > 1000:
    row("OK", "PTB-XL extracted records", f"{n_records} records100 headers (zip optional)")
else:
    row("FAIL", "PTB-XL archive / records", "missing",
        "python preprocess/download_data.py  (see docs/DATA.md)")

if os.path.isdir(ptbxl_records) and n_records > 20000:
    row("OK", "PTB-XL records100", f"{n_records} records")
else:
    row("WARN", "PTB-XL records100", f"{n_records} records",
        "python preprocess/extract_ptbxl.py")

if os.path.exists(ptbxl_csv):
    row("OK", "PTB-XL metadata", "ptbxl_database.csv / scp_statements.csv")
else:
    row("FAIL", "PTB-XL metadata", "missing", "python preprocess/download_data.py")

if os.path.exists(mit_parquet):
    row("OK", "MIT-BIH parquet", f"{size_mb(mit_parquet):7.1f} MB")
elif n_mit >= 40:
    row("OK", "MIT-BIH WFDB", f"{n_mit} records")
else:
    row("FAIL", "MIT-BIH", "missing",
        "python preprocess/download_data.py --skip-ptbxl  (see docs/DATA.md)")

print("\n== preprocessing caches (generated once from the raw data) ==")
expected = [
    ("ptbxl_train.npz", "PTB-XL train cache"),
    ("ptbxl_val.npz", "PTB-XL val cache"),
    ("ptbxl_test.npz", "PTB-XL test cache"),
    ("mitbih.npz", "MIT-BIH normal-beat cache"),
    ("ref_params_ptbxl_test.csv", "interval reference (PTB-XL)"),
    ("ref_params_mitbih.csv", "interval reference (MIT-BIH)"),
]
for name, label in expected:
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        row("OK", label, f"{size_mb(p):7.1f} MB")
    else:
        row("FAIL", label, "missing", "run the preprocess pipeline (docs/DATA.md)")

optional = [
    ("regularity_refs.npz", "regularity reference cache"),
    ("interval_targets_train.npz", "v2 interval targets (train)"),
    ("interval_targets_val.npz", "v2 interval targets (val)"),
]
for name, label in optional:
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        row("OK", label, f"{size_mb(p):7.1f} MB")
    else:
        row("WARN", label, "not present (optional; needed for v2 aux loss / faster eval)",
            "python preprocess/extract_interval_targets.py")

print("\n== quick sanity check of the main caches ==")
try:
    import numpy as np
    for split in ("train", "val", "test"):
        p = os.path.join(CACHE, f"ptbxl_{split}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        sig, lab = z["signals"], z["labels"]
        row("OK", f"ptbxl_{split}.npz shape",
            f"signals={sig.shape} labels={lab.shape} finite={bool(np.isfinite(sig).all())}")
        if sig.ndim != 3 or sig.shape[1] != 12:
            row("FAIL", f"ptbxl_{split}.npz format", "expected (N, 12, T)")
except Exception as exc:  # noqa: BLE001
    row("FAIL", "cache sanity", f"{exc}")

print("\n== next steps ==")
if fail == 0:
    print("All essential data present. You can run evaluation/training, e.g.:")
    print("  python tools/run_eval_all.py        (or powershell -File tools/run_eval_all.ps1)")
else:
    print("Some essential data is missing. Follow docs/DATA.md, then rerun this check:")
    print("  python preprocess/check_data.py")
print(f"\nRESULT: {'READY' if fail == 0 else 'MISSING DATA'} "
      f"(fatal={fail}, warnings={warn})")
sys.exit(0 if fail == 0 else 1)
