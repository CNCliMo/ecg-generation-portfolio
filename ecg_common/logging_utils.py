"""Logging + experiment bookkeeping utilities.

Every run must produce a log under results/logs/ and record *failures with
reasons* in CSV/JSONL form -- the task book explicitly forbids silently
skipping failed samples/steps.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

from ecg_common import paths


def get_logger(name: str, logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is None:
        logfile = os.path.join(paths.LOGS, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("logging to %s", logfile)
    return logger


class FailureRecorder:
    """Append per-sample / per-step failures to a CSV so nothing is skipped silently."""

    def __init__(self, name: str):
        self.path = os.path.join(paths.LOGS, f"failures_{name}.csv")
        self.rows: list[dict] = []
        self.name = name

    def record(self, item: str, reason: str, **extra) -> None:
        row = {"item": item, "reason": reason, **extra}
        self.rows.append(row)

    def flush(self) -> str:
        import csv
        fields = ["item", "reason"]
        for row in self.rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.rows)
        return self.path

    def summary(self) -> dict:
        import collections
        c = collections.Counter(r["reason"].split(":")[0] for r in self.rows)
        return {"n_failed": len(self.rows), "by_reason": dict(c), "file": self.path}


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
