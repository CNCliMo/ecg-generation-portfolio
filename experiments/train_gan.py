"""Train the route-D GAN baseline (Pulse2Pulse-style, one run per seed).

The task book requires >=3 full runs with fixed seeds and reported mean/variance,
so this script trains exactly one run; experiments/run_route_d.py runs the three
seeds (42/43/44) sequentially.

Usage:
  python experiments/train_gan.py --seed 42 --epochs 100
  python experiments/train_gan.py --seed 42 --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.data import load_ptbxl_split
from ecg_common.logging_utils import get_logger, save_json
from models.gan import GAN, hinge_d_loss, hinge_g_loss, spectral_norm_discriminator

LOG = get_logger("train_gan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond-mode", default="none", choices=["none", "label", "hr", "both"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(args.seed)
    dcfg = cfg["route_d"]
    epochs = args.epochs or dcfg["epochs"]
    bs = args.batch_size or dcfg["batch_size"]
    name = f"d_seed{args.seed}" if args.cond_mode == "none" else f"d_cond_seed{args.seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load_ptbxl_split("train")
    hr_valid = tr["hr"][np.isfinite(tr["hr"])]
    hr_mean, hr_std = float(hr_valid.mean()), float(hr_valid.std())
    X = torch.as_tensor(tr["signals"], device=device)
    labels_all = torch.as_tensor(tr["labels"].astype(np.float32), device=device)
    hr_np = np.asarray(tr["hr"], dtype=np.float32).copy()
    hr_np[~np.isfinite(hr_np)] = hr_mean
    hr_all = torch.as_tensor((hr_np - hr_mean) / hr_std, device=device)

    model = GAN(cfg, cond_mode=args.cond_mode).to(device)
    model = spectral_norm_discriminator(model)
    nG = sum(p.numel() for p in model.G.parameters())
    nD = sum(p.numel() for p in model.D.parameters())
    LOG.info("name=%s cond=%s device=%s G=%.2fM D=%.2fM epochs=%d bs=%d",
             name, args.cond_mode, device, nG / 1e6, nD / 1e6, epochs, bs)
    optG = torch.optim.AdamW(model.G.parameters(), lr=dcfg["lr_g"], betas=(0.5, 0.999))
    optD = torch.optim.AdamW(model.D.parameters(), lr=dcfg["lr_d"], betas=(0.5, 0.999))
    scalerG = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    scalerD = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = np.random.default_rng(args.seed)

    ckpt_path = os.path.join(paths.CKPT, f"{name}.pt")
    hist_path = os.path.join(paths.LOGS, f"{name}_train.csv")
    history = []
    t0 = time.time()
    n = len(tr["signals"])
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        ep_d, ep_g, seen = 0.0, 0.0, 0
        for start in range(0, n, bs):
            idx = torch.as_tensor(perm[start:start + bs], device=device)
            real = X[idx]
            b = len(idx)
            z = torch.randn(b, model.z_dim, device=device)
            labels = labels_all[idx] if args.cond_mode in ("label", "both") else None
            hrs = hr_all[idx] if args.cond_mode in ("hr", "both") else None
            cond = model.cond_vector(labels, hrs)
            # --- discriminator step
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                fake = model.G(z, cond)
                d_real = model.D(real, cond)
                d_fake = model.D(fake.detach(), cond)
                d_loss = hinge_d_loss(d_real, d_fake)
            optD.zero_grad(set_to_none=True)
            scalerD.scale(d_loss).backward()
            scalerD.step(optD)
            scalerD.update()
            # --- generator step
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                fake = model.G(z, cond)
                g_loss = hinge_g_loss(model.D(fake, cond))
            optG.zero_grad(set_to_none=True)
            scalerG.scale(g_loss).backward()
            scalerG.step(optG)
            scalerG.update()
            ep_d += d_loss.item() * b
            ep_g += g_loss.item() * b
            seen += b
            if args.smoke and seen >= 512:
                break
        row = {"epoch": epoch, "d_loss": ep_d / seen, "g_loss": ep_g / seen,
               "elapsed_s": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        tmp = ckpt_path + ".tmp"
        torch.save({"G": model.G.state_dict(), "D": model.D.state_dict(),
                    "meta": {"name": name, "seed": args.seed, "cond_mode": args.cond_mode,
                             "hr_mean": hr_mean, "hr_std": hr_std, "cfg": cfg,
                             "epoch": epoch, "z_dim": model.z_dim,
                             "cond_dim": model.cond_dim, "n_samples": cfg["n_samples"]}},
                   tmp)
        os.replace(tmp, ckpt_path)
        LOG.info("epoch %d/%d d_loss=%.4f g_loss=%.4f (%.0fs)", epoch, epochs,
                 row["d_loss"], row["g_loss"], time.time() - t0)

    save_json(os.path.join(paths.LOGS, f"{name}_summary.json"), {
        "name": name, "seed": args.seed, "cond_mode": args.cond_mode,
        "epochs": epochs, "elapsed_s": time.time() - t0,
        "params_G_m": nG / 1e6, "params_D_m": nD / 1e6,
        "final_d_loss": history[-1]["d_loss"], "final_g_loss": history[-1]["g_loss"]})
    LOG.info("DONE %s in %.0fs -> %s", name, time.time() - t0, ckpt_path)


if __name__ == "__main__":
    main()
