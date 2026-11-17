"""Train the diffusion routes (B: standard DDPM, C: regularity-injected DDPM + ablations).

v1 recipes (default):
  Route B variants :  b_uncond (cond_mode=none), b_cond (cond_mode=label)
  Route C configs  :  c_full (both), c_no_label (hr), c_no_hr (label),
                      c_label_noise (both + 10% label flip), c_label_shuffle (random labels)

v2 recipe (new flags, all optional / backward compatible):
  --cond-style nle     SSSD-ECG-nle style per-label present/absent embeddings + null token
  --ema-decay 0.999    keep an EMA of the U-Net weights; saved in the checkpoint and
                       used by `generate_diffusion.py --use-ema` (default on)
  --aux-intervals      add the regularity auxiliary loss (predict QT/PR/QRS/RR-CV from
                       the bottleneck; targets from preprocess/cache/interval_targets_*.npz)
  --aux-weight 0.1     weight of the auxiliary loss
  --base 64            channel width (v1: 32)

Training contract: same data split / seed / epochs / batch for every variant of a
given recipe; atomic per-epoch checkpoints (EMA included); losses -> CSV.

Usage examples:
  python experiments/train_diffusion.py --name b_uncond --cond-mode none --epochs 100
  python experiments/train_diffusion.py --name c_full   --cond-mode both --epochs 120
  python experiments/train_diffusion.py --name c_full_v2 --cond-mode both --epochs 120 \
      --base 64 --cond-style nle --ema-decay 0.999 --aux-intervals --aux-weight 0.1
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
from models.diffusion import EMA, build_model, save_checkpoint

LOG = get_logger("train_diffusion")
AUX_KEYS = ("qt", "pr", "qrs", "rr_cv")


def prepare_conditions(split: dict, hr_mean: float, hr_std: float, device: str):
    labels = torch.as_tensor(split["labels"].astype(np.float32), device=device)
    hr = np.asarray(split["hr"], dtype=np.float32).copy()
    hr[~np.isfinite(hr)] = hr_mean
    hr = torch.as_tensor((hr - hr_mean) / hr_std, device=device)
    return labels, hr


def load_aux_targets(split_name: str, n_expected: int, scaler: dict | None = None,
                     device: str = "cuda"):
    """Load interval targets and normalise; returns (targets tensor or None, scaler)."""
    p = os.path.join(paths.CACHE, f"interval_targets_{split_name}.npz")
    if not os.path.exists(p):
        LOG.warning("aux targets missing: %s", p)
        return None, scaler
    z = np.load(p, allow_pickle=False)
    arr = np.stack([z[k] for k in AUX_KEYS], axis=1).astype(np.float32)  # (N, 4)
    if scaler is None:
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        scaler = {"mean": mu.tolist(), "std": sd.tolist(), "keys": list(AUX_KEYS)}
    mu = np.asarray(scaler["mean"], dtype=np.float32)
    sd = np.asarray(scaler["std"], dtype=np.float32)
    norm = (arr - mu) / sd
    norm[~np.isfinite(arr)] = np.nan
    if len(norm) < n_expected:
        pad = np.full((n_expected - len(norm), 4), np.nan, dtype=np.float32)
        norm = np.concatenate([norm, pad], axis=0)
    return torch.as_tensor(norm[:n_expected], device=device), scaler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="checkpoint/run name, e.g. b_uncond, c_full_v2")
    ap.add_argument("--cond-mode", default="none", choices=["none", "label", "hr", "both"])
    ap.add_argument("--cond-style", default="mask", choices=["mask", "nle"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--base", type=int, default=None, help="base channel width")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA")
    ap.add_argument("--aux-intervals", action="store_true",
                    help="add the interval auxiliary regression loss (regularity injection)")
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--label-noise-prob", type=float, default=0.0,
                    help="route-C ablation '条件加噪': flip labels with this probability")
    ap.add_argument("--label-shuffle", action="store_true",
                    help="route-C ablation '条件替换': replace labels by random labels")
    ap.add_argument("--cond-dropout", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 epochs x 200 batches")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    rcfg = cfg["route_c"] if args.name.startswith("c_") else cfg["route_b"]
    epochs = args.epochs or rcfg["epochs"]
    bs = args.batch_size or rcfg["batch_size"]
    lr = args.lr or rcfg["lr"]
    cond_dropout = (rcfg.get("cond_dropout_prob", 0.15) if args.cond_dropout is None
                    else args.cond_dropout)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load_ptbxl_split("train")
    va = load_ptbxl_split("val")
    hr_valid = tr["hr"][np.isfinite(tr["hr"])]
    hr_mean, hr_std = float(np.mean(hr_valid)), float(np.std(hr_valid))
    LOG.info("name=%s cond_mode=%s cond_style=%s epochs=%d bs=%d lr=%g base=%s ema=%.3f "
             "aux=%s(%.2f) device=%s", args.name, args.cond_mode, args.cond_style, epochs,
             bs, lr, args.base or rcfg.get("base_channels") or cfg["route_b"]["base_channels"],
             args.ema_decay, args.aux_intervals, args.aux_weight, device)
    LOG.info("train=%s val=%s | hr mean=%.1f std=%.1f", tr["signals"].shape,
             va["signals"].shape, hr_mean, hr_std)

    labels_tr, hr_tr = prepare_conditions(tr, hr_mean, hr_std, device)
    labels_va, hr_va = prepare_conditions(va, hr_mean, hr_std, device)
    x_va = torch.as_tensor(va["signals"], device=device)
    aux_tr, aux_scaler = (None, None)
    aux_va = None
    if args.aux_intervals:
        aux_tr, aux_scaler = load_aux_targets("train", len(tr["signals"]), device=device)
        aux_va, _ = load_aux_targets("val", len(va["signals"]), scaler=aux_scaler, device=device)

    model = build_model(cfg, args.cond_mode, base=args.base,
                        cond_style=args.cond_style, aux=args.aux_intervals).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("model params %.2f M", n_params / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    ckpt_path = os.path.join(paths.CKPT, f"{args.name}.pt")
    meta = {"name": args.name, "cond_mode": args.cond_mode, "cond_style": args.cond_style,
            "base_channels": args.base or cfg["route_b"]["base_channels"],
            "cfg": cfg, "hr_mean": hr_mean, "hr_std": hr_std,
            "label_noise_prob": args.label_noise_prob, "label_shuffle": args.label_shuffle,
            "cond_dropout": cond_dropout, "n_steps": cfg["generation"]["ddpm_train_steps"],
            "epochs": epochs, "batch_size": bs, "lr": lr, "seed": cfg["seed"],
            "ema_decay": args.ema_decay, "aux": args.aux_intervals,
            "aux_weight": args.aux_weight, "aux_scaler": aux_scaler}
    start_epoch = 1
    history = []
    hist_path = os.path.join(paths.LOGS, f"{args.name}_train.csv")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.shadow = {k: v.detach().cpu().clone() for k, v in ckpt["ema"].items()}
        start_epoch = int(ckpt["meta"].get("epoch", 0)) + 1
        LOG.info("resumed from %s at epoch %d", ckpt_path, start_epoch - 1)
        if os.path.exists(hist_path):
            history = pd.read_csv(hist_path).to_dict("records")

    t0 = time.time()
    n_train = len(tr["signals"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    use_label = args.cond_mode in ("label", "both")
    use_hr = args.cond_mode in ("hr", "both")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        ep_diff, ep_aux, seen = 0.0, 0.0, 0
        for bi, start in enumerate(range(0, n_train, bs)):
            if args.smoke and bi >= 200:
                break
            idx_np = perm[start:start + bs]
            idx = torch.as_tensor(idx_np, device=device)
            x0 = torch.as_tensor(tr["signals"][idx_np], device=device)
            labels = labels_tr[idx] if use_label else None
            hr = hr_tr[idx] if use_hr else None
            targets = aux_tr[idx] if (args.aux_intervals and aux_tr is not None) else None
            if labels is not None:
                if args.label_shuffle:  # C-条件替换: random labels (condition mechanism test)
                    labels = labels[torch.randperm(len(labels), device=device)]
                elif args.label_noise_prob > 0:  # C-条件加噪: label noise
                    flip = torch.rand(len(labels), device=device) < args.label_noise_prob
                    if flip.any():
                        multi = (labels[flip].sum(1) > 0)
                        new_lab = torch.zeros_like(labels[flip])
                        new_lab[torch.arange(int(flip.sum()), device=device),
                                torch.randint(0, labels.shape[1], (int(flip.sum()),), device=device)] = 1.0
                        labels[flip] = torch.where(multi[:, None], new_lab, labels[flip])
            null_mask = None
            if cond_dropout > 0 and args.cond_mode != "none":
                done = torch.rand(len(x0), device=device) < cond_dropout
                if done.any():
                    if args.cond_style == "nle":
                        null_mask = done.float()
                    else:
                        if labels is not None:
                            labels = labels.clone()
                            labels[done] = 0.0
                        if hr is not None:
                            hr = hr.clone()
                            hr[done] = 0.0
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                losses = model.loss(x0, labels, hr, null_mask, targets, args.aux_weight)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_diff += losses["diffusion"].item() * len(x0)
            ep_aux += (losses["aux"].item() if torch.is_tensor(losses["aux"]) else 0.0) * len(x0)
            seen += len(x0)
        sched.step()
        if ema is not None:
            steps_per_epoch = max(1, n_train // bs)
            ema.update(model, decay=float(args.ema_decay) ** steps_per_epoch)
        row = {"epoch": epoch, "train_loss": ep_diff / max(seen, 1),
               "aux_loss": ep_aux / max(seen, 1), "lr": sched.get_last_lr()[0]}
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
                n_va = min(len(x_va), 512)
                vl = model.loss(x_va[:n_va],
                                labels_va[:n_va] if use_label else None,
                                hr_va[:n_va] if use_hr else None,
                                None,
                                aux_va[:n_va] if (aux_va is not None) else None,
                                0.0)["diffusion"].item()
            row["val_loss"] = vl
        history.append(row)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        save_checkpoint(ckpt_path, model, {**meta, "epoch": epoch,
                                           "train_loss": row["train_loss"]},
                        ema_state=ema.state() if ema is not None else None)
        LOG.info("epoch %d/%d train_loss=%.4f aux=%.4f%s (%.0fs)", epoch, epochs,
                 row["train_loss"], row["aux_loss"],
                 f" val_loss={row.get('val_loss', float('nan')):.4f}", time.time() - t0)

    save_json(os.path.join(paths.LOGS, f"{args.name}_summary.json"), {
        "name": args.name, "params_m": n_params / 1e6, "epochs": epochs,
        "train_loss_last": history[-1]["train_loss"], "elapsed_s": time.time() - t0,
        "meta": {k: v for k, v in meta.items() if k != "cfg"},
    })
    LOG.info("DONE %s in %.0fs -> %s", args.name, time.time() - t0, ckpt_path)


if __name__ == "__main__":
    main()
