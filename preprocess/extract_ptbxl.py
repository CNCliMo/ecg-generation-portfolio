"""Extract records100/ + metadata from the official PTB-XL 1.0.3 archive.

The archive at data/ptb-xl/ptb-xl-data.zip is the official PhysioNet release
layout: records100/, records500/, ptbxl_database.csv, scp_statements.csv.
We only extract records100 (100 Hz, the version required by the task book) and
the two metadata CSVs; records500 is left inside the archive (unused).

Usage: python preprocess/extract_ptbxl.py
"""
from __future__ import annotations

import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.logging_utils import get_logger

ZIP = os.path.join(paths.DATA_PTBXL, "ptb-xl-data.zip")
LOG = get_logger("extract_ptbxl")


def main() -> None:
    if not os.path.exists(ZIP):
        raise SystemExit(f"missing {ZIP}; run preprocess/download_data.py first")
    LOG.info("zip size %.1f MB", os.path.getsize(ZIP) / 1024 / 1024)
    out = paths.DATA_PTBXL
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        wanted = [n for n in names
                  if n.startswith("records100/") or n in
                  ("ptbxl_database.csv", "scp_statements.csv", "RECORDS", "LICENSE.txt")]
        LOG.info("archive entries=%d, extracting=%d", len(names), len(wanted))
        t0 = time.time()
        for i, n in enumerate(wanted, 1):
            dest = os.path.join(out, n)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(n) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if i % 5000 == 0:
                LOG.info("  extracted %d/%d (%.0fs)", i, len(wanted), time.time() - t0)
    # sanity
    rec_dir = os.path.join(out, "records100")
    n_dat = sum(1 for r, _, fs in os.walk(rec_dir) for f in fs if f.endswith(".dat"))
    n_hea = sum(1 for r, _, fs in os.walk(rec_dir) for f in fs if f.endswith(".hea"))
    LOG.info("done in %.0fs: records100 .dat=%d .hea=%d", time.time() - t0, n_dat, n_hea)
    for f in ("ptbxl_database.csv", "scp_statements.csv"):
        p = os.path.join(out, f)
        LOG.info("%s exists=%s size=%s", f, os.path.exists(p),
                 os.path.getsize(p) if os.path.exists(p) else 0)
    if n_dat == 0:
        raise SystemExit("no records extracted -- archive layout unexpected")


if __name__ == "__main__":
    main()
