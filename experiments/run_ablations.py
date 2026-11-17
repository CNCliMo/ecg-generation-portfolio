"""Run the five route-C ablation configurations (task book section 6).

All five use the same backbone, data split, seed, epoch budget and sampling
protocol; only the conditioning mechanism changes:

  c_full         label + HR conditioning (full C)
  c_no_label     HR only
  c_no_hr        label only
  c_label_noise  labels randomly flipped with 10% probability
  c_label_shuffle random labels replace the true ones (does conditioning matter?)

Each run trains via experiments/train_diffusion.py (checkpoints are atomic and
resumable), then samples 500 conditional + 1000 unconditional signals via
experiments/generate_diffusion.py, and finally the shared eval scripts score it.

Usage:
  python experiments/run_ablations.py --epochs 90            # train missing configs
  python experiments/run_ablations.py --gen-only             # only sample + skip existing
  python experiments/run_ablations.py --eval-only            # only evaluation
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

LOG = get_logger("run_ablations")

CONFIGS = {
    "c_full":        {"args": ["--cond-mode", "both"]},
    "c_no_label":    {"args": ["--cond-mode", "hr"]},
    "c_no_hr":       {"args": ["--cond-mode", "label"]},
    "c_label_noise": {"args": ["--cond-mode", "both", "--label-noise-prob", "0.1"]},
    "c_label_shuffle": {"args": ["--cond-mode", "both", "--label-shuffle"]},
}


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
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    epochs = args.epochs or cfg["route_c"]["epochs"]
    py = sys.executable

    if not args.eval_only and not args.gen_only:
        for name, spec in CONFIGS.items():
            ckpt = os.path.join(paths.CKPT, f"{name}.pt")
            if os.path.exists(ckpt):
                LOG.info("skip training %s (checkpoint exists)", name)
                continue
            run([py, os.path.join(paths.EXPERIMENTS, "train_diffusion.py"),
                 "--name", name, "--epochs", str(epochs), *spec["args"]],
                f"train_{name}")

    if not args.eval_only:
        for name in CONFIGS:
            run([py, os.path.join(paths.EXPERIMENTS, "generate_diffusion.py"),
                 "--ckpt", os.path.join(paths.CKPT, f"{name}.pt"), "--out", name], f"gen_{name}")

    if not args.gen_only:
        for script in ("eval_morphology.py", "eval_regularity.py", "eval_functional.py",
                       "build_metrics_csv.py"):
            run([py, os.path.join(paths.EXPERIMENTS, script)], f"eval_{script}")
    LOG.info("DONE")


if __name__ == "__main__":
    main()
