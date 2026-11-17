"""1D ResNet classifier for PTB-XL 5-class multi-label diagnostic superclasses.

Architecture: conv stem (k=7, stride 2) + 4 residual stages (base/2/4/8x base,
2 basic blocks each, stride 2 at the first block) + global average pooling +
5 sigmoid logits. v1: base=64 (~0.97M params). v2: base=96 (~2.2M params).

Used by:
  - experiments/train_classifier.py  (training / test report)
  - experiments/eval_functional.py   (conditional consistency, class distribution)
  - experiments/eval_downstream.py   (TSTR/TRTS with synthetic data)
  - experiments/eval_consistency.py  (classifier-FID / coverage features)
  - app/streamlit_app.py             (uploaded-signal diagnosis, generated-signal checks)
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from ecg_common import paths
from ecg_common.config import load_config


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch))

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5, base: int = 64,
                 n_blocks: tuple[int, ...] = (2, 2, 2, 2)):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        stages = []
        in_ch = base
        for si, (w, nb) in enumerate(zip(widths, n_blocks)):
            for bi in range(nb):
                stages.append(BasicBlock1D(in_ch, w, stride=2 if bi == 0 else 1))
                in_ch = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(widths[-1], n_classes)
        self.feature_dim = widths[-1]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate 512-d (or 4*base) representation, used for FID/coverage."""
        return self.pool(self.stages(self.stem(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_classifier(cfg: dict | None = None, base: int | None = None) -> ResNet1D:
    cfg = cfg or load_config()
    c = cfg["classifier"]
    return ResNet1D(n_leads=cfg["n_leads"], n_classes=len(cfg["superclasses"]),
                    base=base or c["base_channels"])


def load_classifier(ckpt_path: str | None = None, device: str | None = None):
    """Return (model in eval mode, cfg, device). Reads `base` from the checkpoint meta."""
    cfg = load_config()
    if ckpt_path is None:
        v2 = os.path.join(paths.CKPT, "classifier_best_v2.pt")
        ckpt_path = v2 if os.path.exists(v2) else os.path.join(paths.CKPT, "classifier_best.pt")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base = int(ckpt.get("base", cfg["classifier"]["base_channels"]))
    model = build_classifier(cfg, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.thresholds_ = ckpt.get("thresholds", None)
    return model, cfg, device


@torch.no_grad()
def predict_proba(model, signals: np.ndarray, device: str | None = None,
                  batch_size: int = 256) -> np.ndarray:
    """signals: (N, 12, T) float32 z-scored -> (N, 5) probabilities."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(signals), batch_size):
        x = torch.as_tensor(signals[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def extract_features(model, signals: np.ndarray, device: str | None = None,
                     batch_size: int = 256) -> np.ndarray:
    """Penultimate features (N, feature_dim) for FID / coverage metrics."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(signals), batch_size):
        x = torch.as_tensor(signals[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(model.features(x).cpu().numpy())
    return np.concatenate(out, 0)
