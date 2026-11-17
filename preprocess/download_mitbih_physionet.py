"""Fallback downloader: MIT-BIH 1.0.0 archive from PhysioNet static endpoint.

physionet.org direct throughput is ~0.1 MB/s aggregate from this network, but the
77 MB archive is still feasible (~10-15 min) as a fallback when the HuggingFace
mirror copies are unavailable (the public HF copies turned out to be private
repos -- HTTP 403).

Usage: python preprocess/download_mitbih_physionet.py
Output: data/mit-bih/mitdb-1.0.0.zip, extracted to data/mit-bih/ (WFDB files)
"""
from __future__ import annotations

import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.logging_utils import get_logger
from preprocess.download_data import download_ranges, log

LOG = get_logger("download_mitbih_physionet")
URL = "https://physionet.org/static/published-projects/mitdb/mitdb-1.0.0.zip"


def main() -> None:
    dest_zip = os.path.join(paths.DATA_MITBIH, "mitdb-1.0.0.zip")
    log(f"fallback MIT-BIH download: {URL}")
    download_ranges(URL, dest_zip, workers=12, chunk=1024 * 1024, attempts=6)
    with zipfile.ZipFile(dest_zip) as z:
        names = z.namelist()
        LOG.info("archive entries: %d", len(names))
        want = [n for n in names if n.endswith((".dat", ".hea", ".atr"))]
        for n in want:
            out = os.path.join(paths.DATA_MITBIH, os.path.basename(n))
            if not os.path.exists(out):
                with z.open(n) as src, open(out, "wb") as dst:
                    dst.write(src.read())
    n_hea = len([f for f in os.listdir(paths.DATA_MITBIH) if f.endswith(".hea")])
    LOG.info("extracted WFDB records: %d .hea files", n_hea)
    if n_hea == 0:
        raise SystemExit("no .hea extracted")


if __name__ == "__main__":
    main()
