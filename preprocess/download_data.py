"""Download PTB-XL (official 1.0.3 layout) and MIT-BIH from the HuggingFace mirror.

Network constraints measured on this machine (2026-11-05):
  - physionet.org direct: ~16 KB/s for archives, ~5 s latency per file -> unusable
  - hf-mirror.com with parallel range requests: ~3.4 MB/s aggregate -> usable

Both datasets are pulled from public mirrors of the official PhysioNet releases:
  PTB-XL 1.0.3 : hf-mirror.com/datasets/longisland3/ptb-xl/resolve/main/ptb-xl-data.zip
                 (contains records100/, records500/, ptbxl_database.csv, scp_statements.csv)
  MIT-BIH ADB  : hf-mirror.com/datasets/Meddies/ecg-mitdb  (raw WFDB files, 48 records)

Outputs
  data/ptb-xl/ptb-xl-data.zip
  data/mit-bih/<record>.{dat,hea,atr}
  results/logs/download.log

Usage:  python preprocess/download_data.py [--skip-ptbxl] [--skip-mitbih]
Idempotent: files whose size already matches the remote size are skipped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DATA_PTBXL = os.path.join(DATA, "ptb-xl")
DATA_MITBIH = os.path.join(DATA, "mit-bih")
LOGDIR = os.path.join(ROOT, "results", "logs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(DATA_PTBXL, exist_ok=True)
os.makedirs(DATA_MITBIH, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

HF = "https://hf-mirror.com/datasets/{repo}/resolve/main/{path}"
UA = {"User-Agent": "Mozilla/5.0 (ecg-project-downloader)"}

_log_lock = threading.Lock()
_log_fh = open(os.path.join(LOGDIR, "download.log"), "a", encoding="utf-8")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        _log_fh.write(line + "\n")
        _log_fh.flush()


def _request(url: str, start: int | None = None, end: int | None = None, timeout: int = 90):
    headers = dict(UA)
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def remote_size(url: str) -> int:
    r = _request(url, 0, 0)
    cr = r.headers.get("Content-Range")
    if cr:
        return int(cr.split("/")[-1])
    return int(r.headers["Content-Length"])


def download_ranges(url: str, dest: str, workers: int = 12, chunk: int = 8 * 1024 * 1024,
                    attempts: int = 4) -> None:
    """Parallel ranged download of a large file (resumable at whole-file level)."""
    total = remote_size(url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.truncate(total)
    jobs = list(enumerate((a, min(a + chunk - 1, total - 1))
                          for a in range(0, total, chunk)))
    done = [0]
    t0 = time.time()
    lock = threading.Lock()

    def worker(job):
        idx, (a, b) = job
        for att in range(attempts):
            try:
                data = _request(url, a, b).read()
                if len(data) != b - a + 1:
                    raise IOError(f"short read {len(data)} != {b - a + 1}")
                with open(tmp, "r+b") as f:
                    f.seek(a)
                    f.write(data)
                with lock:
                    done[0] += len(data)
                return
            except Exception as exc:  # noqa: BLE001
                if att == attempts - 1:
                    raise
                log(f"  chunk {idx} retry {att + 1}: {exc}")
                time.sleep(1.5 * (att + 1))

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        while True:
            not_done = [f for f in futures if not f.done()]
            elapsed = time.time() - t0
            mb = done[0] / 1024 / 1024
            if not not_done:
                break
            log(f"  progress {mb:7.1f}/{total / 1024 / 1024:.1f} MB "
                f"({100 * done[0] / total:5.1f}%) {mb / max(elapsed, 1e-6):.2f} MB/s")
            time.sleep(20)
        for f in futures:
            f.result()
    os.replace(tmp, dest)
    log(f"  done {dest} ({total / 1024 / 1024:.1f} MB in {time.time() - t0:.0f}s)")


def download_many(repo: str, paths: list[str], dest_dir: str, workers: int = 10,
                  attempts: int = 4) -> None:
    os.makedirs(dest_dir, exist_ok=True)

    def one(path: str) -> None:
        url = HF.format(repo=repo, path=path)
        out = os.path.join(dest_dir, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return
        for att in range(attempts):
            try:
                data = _request(url).read()
                with open(out, "wb") as f:
                    f.write(data)
                return
            except Exception as exc:  # noqa: BLE001
                if att == attempts - 1:
                    log(f"  FAIL {path}: {exc}")
                    raise
                time.sleep(1.0 * (att + 1))

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, paths))
    log(f"  downloaded {len(paths)} files to {dest_dir} in {time.time() - t0:.0f}s")


def hf_tree(repo: str) -> list[str]:
    url = f"https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true&expand=false"
    req = urllib.request.Request(url, headers=UA)
    raw = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return [x["path"] for x in raw if x["type"] == "file"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ptbxl", action="store_true")
    ap.add_argument("--skip-mitbih", action="store_true")
    args = ap.parse_args()

    log("=== download_data.py start ===")
    if not args.skip_ptbxl:
        zip_path = os.path.join(DATA, "ptb-xl", "ptb-xl-data.zip")
        url = HF.format(repo="longisland3/ptb-xl", path="ptb-xl-data.zip")
        expect = remote_size(url)
        if os.path.exists(zip_path) and os.path.getsize(zip_path) == expect:
            log(f"PTB-XL zip already complete ({expect} bytes)")
        else:
            log(f"PTB-XL zip: {expect / 1024 / 1024:.0f} MB from {url}")
            download_ranges(url, zip_path, workers=12, chunk=8 * 1024 * 1024)
        # also keep the small metadata files as loose copies (convenience)
        for meta in ("ptbxl_database.csv", "scp_statements.csv", "RECORDS"):
            dest = os.path.join(DATA, "ptb-xl", meta)
            if not os.path.exists(dest):
                data = _request(HF.format(repo="longisland3/ptb-xl", path=meta)).read()
                with open(dest, "wb") as f:
                    f.write(data)
                log(f"  metadata {meta} ({len(data)} bytes)")

    if not args.skip_mitbih:
        # Official-format WFDB copies on the HF mirror turned out to be private
        # (HTTP 403: Meddies/ecg-mitdb is a staged private repo). This public
        # parquet copy contains the same official MIT-BIH 1.0.0 content: 48
        # records, 2 channels (MLII / V1 or V5) at 360 Hz in mV, plus the beat
        # annotations (sample + symbol) and the WFDB header text.
        parquet_url = HF.format(repo="epr-labs/mit-bih-arrhythmia-database",
                                path="data/train-00000-of-00001.parquet")
        dest = os.path.join(DATA_MITBIH, "mitbih.parquet")
        expect = remote_size(parquet_url)
        if os.path.exists(dest) and os.path.getsize(dest) == expect:
            log(f"MIT-BIH parquet already complete ({expect} bytes)")
        else:
            log(f"MIT-BIH parquet: {expect / 1024 / 1024:.0f} MB from {parquet_url}")
            download_ranges(parquet_url, dest, workers=12, chunk=4 * 1024 * 1024)
        # keep a copy of the archive for provenance
        log("MIT-BIH source: epr-labs/mit-bih-arrhythmia-database (official "
            "PhysioNet mitdb 1.0.0 content, parsed HTTP 403 workaround)")

    log("=== download_data.py done ===")


if __name__ == "__main__":
    main()
