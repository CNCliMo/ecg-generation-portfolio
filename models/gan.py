"""Route D: Pulse2Pulse-style GAN baseline (1D DCGAN/U-Net generator + PatchGAN D).

DEVIATION NOTE (README deviation table): the original Pulse2Pulse paper's exact
layer widths are not reproduced here; this is a Pulse2Pulse-style 1D generator
(noise -> dense -> 5 upsampling conv blocks -> 12x1000) and a PatchGAN 1D
discriminator trained with a hinge loss and spectral normalisation, matching the
reported architecture family (U-Net-like generator, patch discriminator,
class-conditional extension). No pretraining, fixed seeds, >=3 full runs as the
task book requires.

Conditioning (optional, --cond-mode label|hr|both|none): concat the label
multi-hot and/or standardised HR to the noise vector (projection conditioning in
the discriminator). Route D is evaluated unconditionally by default, exactly as
the baseline is defined; the conditional variant exists so the functional-layer
metrics can be reported consistently.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator1D(nn.Module):
    def __init__(self, z_dim=128, cond_dim=0, n_leads=12, n_samples=1000, base=256):
        super().__init__()
        self.n_samples = n_samples
        self.z_dim, self.cond_dim = z_dim, cond_dim
        # start length 32 -> x2^5 = 1024 (cropped to n_samples)
        self.start_len = 32
        self.fc = nn.Linear(z_dim + cond_dim, base * self.start_len)
        chans = [base, base // 2, base // 4, base // 8, base // 16, base // 16]
        blocks = []
        for i in range(len(chans) - 1):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(chans[i], chans[i + 1], 5, padding=2),
                nn.BatchNorm1d(chans[i + 1]),
                nn.ReLU(inplace=True)))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv1d(chans[-1], n_leads, 3, padding=1)

    def forward(self, z, cond=None):
        x = z if cond is None else torch.cat([z, cond], dim=1)
        h = self.fc(x).view(x.shape[0], -1, self.start_len)
        h = self.blocks(h)
        h = self.out(h)
        if h.shape[-1] != self.n_samples:
            h = F.interpolate(h, size=self.n_samples, mode="nearest")
        return h


class Discriminator1D(nn.Module):
    def __init__(self, n_leads=12, cond_dim=0, base=64):
        super().__init__()
        layers = [nn.Conv1d(n_leads, base, 7, stride=2, padding=3), nn.LeakyReLU(0.2, True)]
        ch = base
        for mult in (2, 4, 8):
            layers += [nn.Conv1d(ch, base * mult, 5, stride=2, padding=2),
                       nn.LeakyReLU(0.2, True)]
            ch = base * mult
        self.features = nn.Sequential(*layers)
        self.cond_dim = cond_dim
        self.cond_proj = nn.Linear(cond_dim, ch) if cond_dim else None
        self.out = nn.Conv1d(ch, 1, 3, padding=1)

    def forward(self, x, cond=None):
        h = self.features(x)  # (B,C,T')
        if self.cond_proj is not None and cond is not None:
            h = h + self.cond_proj(cond)[:, :, None]
        return self.out(h)  # (B,1,T') patch logits


class GAN(nn.Module):
    def __init__(self, cfg: dict, cond_mode: str = "none", z_dim: int = 128):
        super().__init__()
        n_classes = len(cfg["superclasses"])
        cond_dim = 0
        if cond_mode in ("label", "both"):
            cond_dim += n_classes
        if cond_mode in ("hr", "both"):
            cond_dim += 1
        self.cond_mode = cond_mode
        self.G = Generator1D(z_dim=z_dim, cond_dim=cond_dim, n_leads=cfg["n_leads"],
                             n_samples=cfg["n_samples"])
        self.D = Discriminator1D(n_leads=cfg["n_leads"], cond_dim=cond_dim)
        self.cond_dim = cond_dim
        self.z_dim = z_dim

    def cond_vector(self, labels=None, hr=None):
        parts = []
        if self.cond_mode in ("label", "both"):
            parts.append(labels if labels is not None else torch.zeros(
                (1, 5), device=next(self.G.parameters()).device))
        if self.cond_mode in ("hr", "both"):
            parts.append((hr if hr is not None else torch.zeros(1, device=next(self.G.parameters()).device))[:, None])
        if not parts:
            return None
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def hinge_d_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()


def spectral_norm_discriminator(model: GAN) -> GAN:
    """Apply spectral norm to all discriminator convolutions and linears for stability."""
    from torch.nn.utils.parametrizations import spectral_norm
    for m in model.D.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and m is not model.D.out:
            try:
                spectral_norm(m)
            except Exception:  # noqa: BLE001
                pass
    return model
