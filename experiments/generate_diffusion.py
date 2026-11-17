"""Sample from a trained diffusion route (B or C) and save .npy/.npz outputs.

Conditional set (n=500 by default): (target HR, diagnostic label) pairs sampled
from the real PTB-XL test distribution. Unconditional set (n=1000): samples drawn
from the model's null condition (trained with condition dropout) so the
functional-layer class distribution can be compared with the real test split.

Outputs (results/generated/):
  {out}_cond.npz    signals (n,12,1000), hr, labels
  {out}_uncond.npz  signals (n_uncond,12,1000), hr (sampled but ignored by model)
  {out}_meta.json

Usage:
  python experiments/generate_diffusion.py --ckpt models/checkpoints/c_full.pt --out C_full
  python experiments/generate_diffusion.py --ckpt models/checkpoints/b_uncond.pt --out B_uncond --only uncond
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.logging_utils import get_logger, save_json
from models.diffusion import load_checkpoint

LOG = get_logger("generate_diffusion")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output name prefix in results/generated/")
    ap.add_argument("--n-cond", type=int, default=None)
    ap.add_argument("--n-uncond", type=int, default=None)
    ap.add_argument("--ddim-steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--only", choices=["both", "cond", "uncond"], default="both")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    gen = cfg["generation"]
    n_cond = args.n_cond or gen["n_conditional"]
    n_uncond = args.n_uncond or gen["n_unconditional"]
    steps = args.ddim_steps or gen["ddim_steps"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = cfg["superclasses"]

    model, meta = load_checkpoint(args.ckpt, cfg, device=device)
    cond_mode = meta["cond_mode"]
    cond_style = meta.get("cond_style", "mask")
    hr_mean, hr_std = meta["hr_mean"], meta["hr_std"]
    LOG.info("loaded %s cond_mode=%s cond_style=%s epoch=%s steps=%d ema=%s", args.ckpt,
             cond_mode, cond_style, meta.get("epoch"), steps, meta.get("used_ema"))

    test = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)
    real_hr = test["hr"]
    valid_idx = np.where(np.isfinite(real_hr))[0]
    rng = np.random.default_rng(cfg["seed"] + 999)
    t0 = time.time()

    def sample_chunks(n: int, labels_np: np.ndarray | None, hr_np: np.ndarray | None,
                      null_condition: bool = False) -> np.ndarray:
        outs = []
        done = 0
        while done < n:
            b = min(args.batch, n - done)
            lb = None if labels_np is None else torch.as_tensor(
                labels_np[done:done + b].astype(np.float32), device=device)
            hb = None if hr_np is None else torch.as_tensor(
                ((hr_np[done:done + b] - hr_mean) / hr_std).astype(np.float32), device=device)
            outs.append(model.sample(b, cfg["n_samples"], device=device, ddim_steps=steps,
                                     labels=lb, hr=hb, guidance_scale=args.guidance,
                                     cond_mode=cond_mode, n_classes=len(classes),
                                     null_condition=null_condition))
            done += b
            LOG.info("  sampled %d/%d (%.0fs)", done, n, time.time() - t0)
        return np.concatenate(outs, 0)

    meta_out = {"route": args.out, "ckpt": os.path.basename(args.ckpt), "cond_mode": cond_mode,
                "cond_style": cond_style, "used_ema": bool(meta.get("used_ema", False)),
                "ckpt_epoch": meta.get("epoch"),
                "ddim_steps": steps, "guidance_scale": args.guidance, "seed": cfg["seed"],
                "output_spec": "float32 (N,12,1000), 100 Hz, per-lead z-score, Lead II index 1"}

    if args.only in ("both", "cond") and cond_mode != "none":
        pick = rng.choice(valid_idx, size=n_cond, replace=True)
        hr_c = real_hr[pick].astype(np.float32)
        labels_c = test["labels"][pick].astype(np.uint8)
        lab_in = labels_c if cond_mode in ("label", "both") else None
        hr_in = hr_c if cond_mode in ("hr", "both") else None
        sig_c = sample_chunks(n_cond, lab_in, hr_in)
        np.savez(os.path.join(paths.GENERATED, f"{args.out}_cond.npz"),
                 signals=sig_c, hr=hr_c, labels=labels_c, label_names=np.array(classes))
        LOG.info("%s_cond.npz signals=%s", args.out, sig_c.shape)
        meta_out["n_conditional"] = int(n_cond)

    if args.only in ("both", "uncond"):
        hr_u = real_hr[rng.choice(len(valid_idx), size=n_uncond, replace=True)].astype(np.float32)
        if cond_mode == "none":
            sig_u = sample_chunks(n_uncond, None, None)
        else:  # null condition (nle null token / v1 zero condition)
            zero_lab = np.zeros((n_uncond, len(classes)), dtype=np.float32)
            zero_hr = np.zeros(n_uncond, dtype=np.float32)
            lab_in = zero_lab if cond_mode in ("label", "both") else None
            hr_in = zero_hr if cond_mode in ("hr", "both") else None
            sig_u = sample_chunks(n_uncond, lab_in, hr_in, null_condition=(cond_style == "nle"))
        np.savez(os.path.join(paths.GENERATED, f"{args.out}_uncond.npz"),
                 signals=sig_u, hr=hr_u)
        LOG.info("%s_uncond.npz signals=%s", args.out, sig_u.shape)
        meta_out["n_unconditional"] = int(n_uncond)

    save_json(os.path.join(paths.GENERATED, f"{args.out}_meta.json"), meta_out)
    LOG.info("DONE in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
