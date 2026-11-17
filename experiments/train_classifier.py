"""Train the PTB-XL 5-class multi-label diagnostic classifier.

v1 (default): flat BCE, 40 epochs, base=64, no smoothing/EMA  -> classifier_best.pt
v2 (--tag v2 --base 96 --epochs 70 --label-smoothing 0.05 --ema-decay 0.999 --threshold-tune):
   stronger baseline used by the v2 evaluation stack.

Usage:
  python experiments/train_classifier.py [--tag v2] [--base 96] [--epochs 70]
      [--label-smoothing 0.05] [--ema-decay 0.999] [--threshold-tune] [--smoke]

Outputs:
  models/checkpoints/classifier_best[_tag].pt      best-by-val-macro-F1 checkpoint
  results/classifier_history[_tag].csv            per-epoch train/val loss + metrics
  results/classifier_test_report[_tag].csv/.md    test metrics (fixed 0.5 and tuned thresholds)
  results/figures/classifier_training[_tag].png   training curve
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from ecg_common import paths
from ecg_common.config import load_config, seed_everything
from ecg_common.data import load_ptbxl_split
from ecg_common.logging_utils import get_logger, save_json
from models.classifier import build_classifier

LOG = get_logger("train_classifier")


def iter_batches(signals, labels, batch_size, shuffle, rng=None):
    n = len(signals)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield torch.as_tensor(signals[j]), torch.as_tensor(labels[j].astype(np.float32))


@torch.no_grad()
def evaluate(model, signals, labels, batch_size=256, device="cuda"):
    model.eval()
    logits, losses = [], []
    crit = nn.BCEWithLogitsLoss()
    for xb, yb in iter_batches(signals, labels, batch_size, False):
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        losses.append(crit(out, yb).item() * len(xb))
        logits.append(out.cpu())
    logits = torch.cat(logits)
    probs = torch.sigmoid(logits).numpy()
    y = labels.astype(int)
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "loss": float(np.sum(losses) / len(labels)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "exact_match": float((pred == y).all(1).mean()),
    }
    try:
        metrics["macro_auroc"] = float(roc_auc_score(y, probs, average="macro"))
        metrics["macro_ap"] = float(average_precision_score(y, probs, average="micro"))
    except ValueError:
        metrics["macro_auroc"] = float("nan")
        metrics["macro_ap"] = float("nan")
    return metrics, probs


def tune_thresholds(y: np.ndarray, probs: np.ndarray) -> list[float]:
    """Per-class threshold maximising F1 on the validation split."""
    ths = []
    grid = np.arange(0.05, 0.96, 0.05)
    for ci in range(y.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y[:, ci], (probs[:, ci] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        ths.append(best_t)
    return ths


class EMA:
    """Exponential moving average of model weights (CPU shadow copy).

    Updated once per epoch (or every `ema_every` steps) because a per-step CPU
    update of an 8M-parameter model forces a CUDA sync per step and dominated the
    runtime. The effective decay for an epoch containing n steps is
    decay ** n, so the EMA horizon stays comparable to per-step EMA.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float | None = None):
        d = self.decay if decay is None else decay
        for k, v in model.state_dict().items():
            v = v.detach().cpu()
            if not torch.is_floating_point(self.shadow[k]):
                self.shadow[k].copy_(v)
                continue
            self.shadow[k].mul_(d).add_(v, alpha=1.0 - d)

    def state(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. v2")
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA")
    ap.add_argument("--threshold-tune", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything()
    ccfg = cfg["classifier"]
    epochs = args.epochs or ccfg["epochs"]
    bs = ccfg["batch_size"]
    classes = cfg["superclasses"]
    base = args.base or ccfg["base_channels"]
    tag = f"_{args.tag}" if args.tag else ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("device=%s torch=%s | tag=%s base=%d epochs=%d smoothing=%.2f ema=%.3f tune=%s",
             device, torch.__version__, args.tag or "(v1)", base, epochs,
             args.label_smoothing, args.ema_decay, args.threshold_tune)

    tr = load_ptbxl_split("train")
    va = load_ptbxl_split("val")
    te = load_ptbxl_split("test")
    if args.smoke:
        tr = {k: v[:512] for k, v in tr.items()}
        va = {k: v[:128] for k, v in va.items()}
        te = {k: v[:128] for k, v in te.items()}
        epochs = 2
    LOG.info("data: train=%s val=%s test=%s", len(tr["signals"]), len(va["signals"]),
             len(te["signals"]))

    model = build_classifier(cfg, base=base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("model params %.2f M", n_params / 1e6)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=ccfg["lr"], weight_decay=ccfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    ema_model = build_classifier(cfg, base=base).to(device) if ema else None

    rng = np.random.default_rng(cfg["seed"])
    best = {"score": -1.0, "epoch": -1, "state": None, "kind": "raw", "val_metrics": {}}
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, n_seen = 0.0, 0
        for xb, yb in iter_batches(tr["signals"], tr["labels"], bs, True, rng):
            xb, yb = xb.to(device), yb.to(device)
            if args.label_smoothing > 0:
                yb = yb * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item() * len(xb)
            n_seen += len(xb)
        sched.step()
        if ema is not None:
            steps_per_epoch = max(1, (len(tr["signals"]) + bs - 1) // bs)
            ema.update(model, decay=float(args.ema_decay) ** steps_per_epoch)

        val_raw, _ = evaluate(model, va["signals"], va["labels"], device=device)
        row = {"epoch": epoch, "train_loss": ep_loss / n_seen,
               **{f"val_{k}": v for k, v in val_raw.items()}, "lr": sched.get_last_lr()[0]}
        candidates = [("raw", val_raw["macro_f1"],
                       {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       val_raw)]
        if ema is not None:
            ema_model.load_state_dict(ema.state())
            val_ema, _ = evaluate(ema_model, va["signals"], va["labels"], device=device)
            row.update({f"val_ema_{k}": v for k, v in val_ema.items()})
            candidates.append(("ema", val_ema["macro_f1"], ema.state(), val_ema))
        kind, score, state, vm = max(candidates, key=lambda c: c[1])
        history.append(row)
        LOG.info("epoch %d/%d train_loss=%.4f val_f1(raw)=%.4f%s (%.0fs)", epoch, epochs,
                 row["train_loss"], val_raw["macro_f1"],
                 f" ema={row.get('val_ema_macro_f1', float('nan')):.4f}" if ema else "",
                 time.time() - t0)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "state": state, "kind": kind,
                    "val_metrics": vm}
            torch.save({"model": best["state"], "epoch": epoch, "cfg": cfg, "base": base,
                        "val_metrics": vm, "classes": classes, "kind": kind},
                       os.path.join(paths.CKPT, f"classifier_best{tag}.pt"))
            LOG.info("  new best (%s) -> classifier_best%s.pt", kind, tag)
    pd.DataFrame(history).to_csv(os.path.join(paths.RESULTS, f"classifier_history{tag}.csv"),
                                 index=False)

    # ------------------------------------------------------------- test report
    model.load_state_dict(best["state"])
    test_metrics, probs = evaluate(model, te["signals"], te["labels"], device=device)
    y = te["labels"].astype(int)
    pred_fixed = (probs >= 0.5).astype(int)
    metrics_list = [(f"fixed0.5_{k}", v) for k, v in test_metrics.items()]
    ths = [0.5] * len(classes)
    if args.threshold_tune:
        _, val_probs = evaluate(model, va["signals"], va["labels"], device=device)
        ths = tune_thresholds(va["labels"].astype(int), val_probs)
        pred_t = (probs >= np.asarray(ths)[None, :]).astype(int)
        m_tuned = {
            "macro_f1": float(f1_score(y, pred_t, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y, pred_t, average="micro", zero_division=0)),
            "exact_match": float((pred_t == y).all(1).mean()),
        }
        metrics_list += [(f"tuned_{k}", v) for k, v in m_tuned.items()]
        LOG.info("thresholds=%s | tuned test macro-F1=%.4f (fixed=%.4f)", ths,
                 m_tuned["macro_f1"], test_metrics["macro_f1"])

    p, r, f, s = precision_recall_fscore_support(y, pred_fixed, labels=range(len(classes)),
                                                 zero_division=0)
    aurocs, aps = [], []
    for ci in range(len(classes)):
        aurocs.append(float(roc_auc_score(y[:, ci], probs[:, ci]))
                      if len(np.unique(y[:, ci])) > 1 else float("nan"))
        aps.append(float(average_precision_score(y[:, ci], probs[:, ci])))
    torch.save({"model": best["state"], "epoch": best["epoch"], "cfg": cfg, "base": base,
                "val_metrics": best["val_metrics"], "test_metrics": test_metrics,
                "thresholds": ths, "classes": classes, "kind": best["kind"]},
               os.path.join(paths.CKPT, f"classifier_best{tag}.pt"))

    rows = [{"scope": "overall", "metric": k, "value": v, "n": len(y)} for k, v in metrics_list]
    for ci, cname in enumerate(classes):
        per = {"precision": float(p[ci]), "recall": float(r[ci]), "f1": float(f[ci]),
               "support": int(s[ci]), "auroc": aurocs[ci], "ap": aps[ci],
               "threshold": float(ths[ci])}
        for k, v in per.items():
            rows.append({"scope": f"class:{cname}", "metric": k, "value": v, "n": int(s[ci])})
    pd.DataFrame(rows).to_csv(os.path.join(paths.RESULTS, f"classifier_test_report{tag}.csv"),
                              index=False)

    md = [f"# Diagnostic classifier -- test report (fold 10) {tag}\n\n",
          f"- best epoch: {best['epoch']} ({best['kind']}, val macro-F1 {best['score']:.4f})\n",
          f"- params: {n_params / 1e6:.2f} M (base={base}), device: {device}, "
          f"label_smoothing={args.label_smoothing}, ema_decay={args.ema_decay}\n",
          "- thresholds: fixed 0.5" +
          (f" / tuned {np.round(ths, 2).tolist()}" if args.threshold_tune else "") + "\n\n",
          "## overall\n\n| metric | value |\n|---|---|\n"]
    for k, v in metrics_list:
        md.append(f"| {k} | {v:.4f} |\n")
    md.append("\n## per class (fixed 0.5)\n\n"
              "| class | precision | recall | F1 | AUROC | threshold | support |\n"
              "|---|---|---|---|---|---|---|\n")
    for ci, cname in enumerate(classes):
        md.append(f"| {cname} | {p[ci]:.4f} | {r[ci]:.4f} | {f[ci]:.4f} | "
                  f"{aurocs[ci]:.4f} | {ths[ci]:.2f} | {int(s[ci])} |\n")
    with open(os.path.join(paths.RESULTS, f"classifier_test_report{tag}.md"), "w",
              encoding="utf-8") as fh:
        fh.write("".join(md))
    save_json(os.path.join(paths.RESULTS, f"classifier_test_report{tag}.json"),
              {"fixed": test_metrics, "thresholds": ths, "base": base,
               "best_kind": best["kind"], "epoch": best["epoch"]})
    LOG.info("test metrics: %s", {k: round(v, 4) for k, v in test_metrics.items()})
    LOG.info("wrote classifier_test_report%s.{csv,md}", tag)
    LOG.info("DONE in %.0fs", time.time() - t0)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = pd.DataFrame(history)
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(h["epoch"], h["train_loss"], label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss")
        ax2 = ax1.twinx()
        ax2.plot(h["epoch"], h["val_macro_f1"], color="tab:orange", label="val macro-F1")
        if "val_ema_macro_f1" in h:
            ax2.plot(h["epoch"], h["val_ema_macro_f1"], color="tab:green",
                     label="val EMA macro-F1")
        ax2.set_ylabel("val macro-F1")
        fig.suptitle("classifier training" + (f" ({args.tag})" if args.tag else ""))
        fig.tight_layout()
        fig.savefig(os.path.join(paths.FIGURES, f"classifier_training{tag}.png"), dpi=150)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("figure failed: %s", exc)


if __name__ == "__main__":
    main()
