"""Route A generation: mechanistic model, no training.

Conditions:
  - conditional set (n=500): target HR sampled from the real PTB-XL test HR
    distribution; the diagnostic label is *not* a model input for route A, so
    labels are attached only for bookkeeping (this is reported as N/A for the
    functional conditional-consistency metric).
  - unconditional set (n=1000): HR sampled from the same real distribution.

Outputs (results/generated/):
  A_cond.npz       signals (500,12,1000), hr (500,), labels (500,5), conditions
  A_uncond.npz     signals (1000,12,1000), hr (1000,)
  A_meta.json      generation config / seed / HR sampling rule

Usage: python experiments/generate_a.py [--n-cond 500] [--n-uncond 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.logging_utils import get_logger, save_json
from models.mechanistic import generate_batch

LOG = get_logger("generate_a")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cond", type=int, default=None)
    ap.add_argument("--n-uncond", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    gen = cfg["generation"]
    n_cond = args.n_cond or gen["n_conditional"]
    n_uncond = args.n_uncond or gen["n_unconditional"]
    classes = cfg["superclasses"]

    test = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)
    real_hr = test["hr"]
    real_labels = test["labels"]
    valid_hr = real_hr[np.isfinite(real_hr)]
    valid_idx = np.where(np.isfinite(real_hr))[0]
    LOG.info("real test HR: n=%d mean=%.1f std=%.1f (missing=%d)",
             len(valid_hr), valid_hr.mean(), valid_hr.std(), len(real_hr) - len(valid_hr))

    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()

    # ---- conditional set: real (hr, label) pairs sampled from the test split
    pick = rng.choice(valid_idx, size=n_cond, replace=True)
    hrs_c = real_hr[pick].astype(np.float32)
    labels_c = real_labels[pick].astype(np.uint8)
    sig_c = generate_batch(hrs_c, seed=int(rng.integers(1 << 30)), fs=cfg["fs"],
                           n_samples=cfg["n_samples"])
    np.savez(os.path.join(paths.GENERATED, "A_cond.npz"),
             signals=sig_c, hr=hrs_c, labels=labels_c,
             label_names=np.array(classes))
    LOG.info("A_cond.npz signals=%s (%.0fs)", sig_c.shape, time.time() - t0)

    # ---- unconditional set
    hrs_u = valid_hr[rng.choice(len(valid_hr), size=n_uncond, replace=True)].astype(np.float32)
    sig_u = generate_batch(hrs_u, seed=int(rng.integers(1 << 30)), fs=cfg["fs"],
                           n_samples=cfg["n_samples"])
    np.savez(os.path.join(paths.GENERATED, "A_uncond.npz"), signals=sig_u, hr=hrs_u)
    LOG.info("A_uncond.npz signals=%s", sig_u.shape)

    save_json(os.path.join(paths.GENERATED, "A_meta.json"), {
        "route": "A", "method": "McSharry three-coupled ODE + 12-lead projection matrix",
        "seed": cfg["seed"], "n_conditional": int(n_cond), "n_unconditional": int(n_uncond),
        "hr_sampling": "empirical PTB-XL test-split HR distribution",
        "conditional_label_note": "route A has no label conditioning; labels copied from "
                                  "the sampled real test record for bookkeeping only",
        "output_spec": "float32 (N,12,1000), 100 Hz, per-lead z-score, Lead II index 1",
        "elapsed_s": time.time() - t0,
    })
    LOG.info("DONE in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
