"""Sample from a trained route-D GAN checkpoint (conditional + unconditional sets).

Outputs mirror generate_diffusion.py so the shared evaluation scripts can score
route D identically:
  results/generated/{out}_cond.npz    (n=500; condition pairs sampled from PTB-XL test)
  results/generated/{out}_uncond.npz  (n=1000; pure noise samples)

For an unconditional GAN (cond_mode=none) no condition vector is used; the
"conditional" file is still produced by pairing random test conditions with
unconditional samples, and the functional layer marks its conditional
consistency as not_applicable.

Usage:
  python experiments/generate_gan.py --ckpt models/checkpoints/d_seed42.pt --out D_seed42
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.logging_utils import get_logger, save_json
from models.gan import Generator1D

LOG = get_logger("generate_gan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cond", type=int, default=None)
    ap.add_argument("--n-uncond", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    gen = cfg["generation"]
    n_cond = args.n_cond or gen["n_conditional"]
    n_uncond = args.n_uncond or gen["n_unconditional"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = cfg["superclasses"]

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    G = Generator1D(z_dim=meta["z_dim"], cond_dim=meta["cond_dim"],
                    n_leads=cfg["n_leads"], n_samples=cfg["n_samples"]).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    cond_mode = meta["cond_mode"]
    LOG.info("loaded %s cond_mode=%s epoch=%s", args.ckpt, cond_mode, meta.get("epoch"))

    test = np.load(os.path.join(paths.CACHE, "ptbxl_test.npz"), allow_pickle=False)
    real_hr = test["hr"]
    valid_idx = np.where(np.isfinite(real_hr))[0]
    rng = np.random.default_rng(cfg["seed"] + 777)

    def build_cond(labels_np, hr_np):
        if cond_mode == "none":
            return None
        parts = []
        if cond_mode in ("label", "both"):
            parts.append(torch.as_tensor(labels_np.astype(np.float32), device=device))
        if cond_mode in ("hr", "both"):
            parts.append(torch.as_tensor(((hr_np - meta["hr_mean"]) / meta["hr_std"]).astype(np.float32), device=device)[:, None])
        c = torch.cat(parts, 1)
        return c

    def sample(n, labels_np, hr_np):
        outs, done = [], 0
        t0 = time.time()
        while done < n:
            b = min(args.batch, n - done)
            z = torch.randn(b, meta["z_dim"], device=device)
            cond = build_cond(labels_np[done:done + b], hr_np[done:done + b]) if labels_np is not None else None
            with torch.no_grad():
                outs.append(G(z, cond).cpu().numpy().astype(np.float32))
            done += b
        LOG.info("  sampled %d (%.0fs)", n, time.time() - t0)
        return np.concatenate(outs, 0)

    pick = rng.choice(valid_idx, size=n_cond, replace=True)
    hr_c = real_hr[pick].astype(np.float32)
    labels_c = test["labels"][pick].astype(np.uint8)
    if cond_mode == "none":
        sig_c = sample(n_cond, None, None)
    else:
        sig_c = sample(n_cond, labels_c, hr_c)
    np.savez(os.path.join(paths.GENERATED, f"{args.out}_cond.npz"),
             signals=sig_c, hr=hr_c, labels=labels_c, label_names=np.array(classes))
    LOG.info("%s_cond.npz signals=%s", args.out, sig_c.shape)

    hr_u = real_hr[rng.choice(len(valid_idx), size=n_uncond, replace=True)].astype(np.float32)
    if cond_mode == "none":
        sig_u = sample(n_uncond, None, None)
    else:
        zeros_lab = np.zeros((n_uncond, len(classes)), dtype=np.float32)
        zeros_hr = np.zeros(n_uncond, dtype=np.float32)
        sig_u = sample(n_uncond, zeros_lab, zeros_hr)
    np.savez(os.path.join(paths.GENERATED, f"{args.out}_uncond.npz"), signals=sig_u, hr=hr_u)
    LOG.info("%s_uncond.npz signals=%s", args.out, sig_u.shape)

    save_json(os.path.join(paths.GENERATED, f"{args.out}_meta.json"), {
        "route": args.out, "ckpt": os.path.basename(args.ckpt), "cond_mode": cond_mode,
        "seed": meta.get("seed"), "epoch": meta.get("epoch"),
        "output_spec": "float32 (N,12,1000), 100 Hz, per-lead z-score, Lead II index 1"})


if __name__ == "__main__":
    main()
