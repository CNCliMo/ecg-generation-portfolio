"""Run route D three times with fixed seeds (task book: GAN 至少训练 3 次, 报告均值与方差).

Seeds 42/43/44, identical data split / architecture / epoch budget. Each run saves
its own checkpoint (models/checkpoints/d_seed{42,43,44}.pt), then generates the
conditional + unconditional sample sets, then the summary script aggregates
mean/std across seeds.

Usage:
  python experiments/run_route_d.py [--epochs 100]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config
from ecg_common.logging_utils import get_logger

LOG = get_logger("run_route_d")


def run(cmd: list[str], log_name: str) -> None:
    log_path = os.path.join(paths.LOGS, f"{log_name}.out")
    LOG.info("RUN %s", " ".join(cmd))
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    LOG.info("  exit=%d in %.0fs", rc, time.time() - t0)
    if rc != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)} (see {log_path})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--gen-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    epochs = args.epochs or cfg["route_d"]["epochs"]
    py = sys.executable
    seeds = cfg["route_d"]["seeds"]

    for seed in seeds:
        name = f"d_seed{seed}"
        ckpt = os.path.join(paths.CKPT, f"{name}.pt")
        if not args.gen_only and not os.path.exists(ckpt):
            run([py, os.path.join(paths.EXPERIMENTS, "train_gan.py"),
                 "--seed", str(seed), "--epochs", str(epochs)], f"train_{name}")
        run([py, os.path.join(paths.EXPERIMENTS, "generate_gan.py"),
             "--ckpt", ckpt, "--out", name], f"gen_{name}")
    LOG.info("DONE route D (3 seeds)")


if __name__ == "__main__":
    main()
