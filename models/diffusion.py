"""Conditional / unconditional 1D diffusion model for 12-lead ECG (routes B, C and v2).

Architecture: SSSD-ECG-style 1D U-Net with residual blocks, timestep embedding,
self-attention at the low-resolution level and FiLM conditioning.

DEVIATION NOTE (documented in README): the original SSSD-ECG backbone uses S4 /
structured-state-space layers; this implementation uses an equivalent residual
convolution + self-attention backbone. The diffusion formulation (DDPM
epsilon-prediction, linear beta schedule) and the conditioning interface are
implemented as specified. Swapping in S4 layers is a self-contained future step.

Conditioning modes:
  none    : unconditional
  label   : 5-class diagnostic multi-hot embedding
  hr      : target heart rate (standardised scalar)
  both    : label + hr  (route C)
Conditioning styles (v2 addition):
  mask    : v1 style, one linear embedding of the multi-hot vector
  nle     : SSSD-ECG-nle style: one embedding per (label, present/absent) pair,
            summed over labels, plus a learned null token for classifier-free
            guidance and a per-sample `null_mask` for condition dropout.
Auxiliary regularity supervision (v2 addition, `aux=True`):
  the U-Net has an extra regression head on the pooled bottleneck features that
  predicts (QT, PR, QRS, RR-CV); the training loss adds
  aux_weight * masked-MSE. This is the "regularity injection" for route C-v2.

V2 models also use an EMA of the weights; see `EMA` below and the generation
script (`generate_diffusion.py --use-ema`).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COND_MODES = ("none", "label", "hr", "both")
COND_STYLES = ("mask", "nle")


# ------------------------------------------------------------------ pieces ---
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None].chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class SelfAttention1D(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        B, C, T = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, self.heads, C // self.heads, T).transpose(-1, -2)  # B,H,T,d
        k = k.view(B, self.heads, C // self.heads, T).transpose(-1, -2)  # B,H,T,d
        v = v.view(B, self.heads, C // self.heads, T).transpose(-1, -2)  # B,H,T,d
        h = F.scaled_dot_product_attention(q, k, v)  # B,H,T,d
        h = h.transpose(-1, -2).reshape(B, C, T)
        return x + self.proj(h)


class CondEncoder(nn.Module):
    """v1 style: label multi-hot + standardised HR -> flat conditioning vector."""

    def __init__(self, n_classes: int, emb_dim: int, mode: str):
        super().__init__()
        assert mode in COND_MODES, mode
        self.mode = mode
        self.n_classes = n_classes
        if mode in ("label", "both"):
            self.label_emb = nn.Sequential(nn.Linear(n_classes, emb_dim), nn.SiLU(),
                                           nn.Linear(emb_dim, emb_dim))
        if mode in ("hr", "both"):
            self.hr_emb = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim))
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, labels: torch.Tensor | None, hr: torch.Tensor | None,
                null_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "none":
            raise RuntimeError("unconditional model has no condition encoder")
        parts = []
        if self.mode in ("label", "both"):
            if labels is None:
                labels = torch.zeros(1, self.n_classes, device=next(self.parameters()).device)
            lab = self.label_emb(labels)
            if null_mask is not None:  # conditions dropped -> v1 compatible zero vector
                lab = lab * (1.0 - null_mask[:, None])
            parts.append(lab)
        if self.mode in ("hr", "both"):
            if hr is None:
                hr = torch.zeros(1, device=next(self.parameters()).device)
            h = self.hr_emb(hr[:, None])
            if null_mask is not None:
                h = h * (1.0 - null_mask[:, None])
            parts.append(h)
        return self.proj(sum(parts))


class CondEncoderNLE(nn.Module):
    """SSSD-ECG-nle style label embedding.

    For each diagnostic label i there are two embeddings (present / absent); the
    sample embedding is the sum over labels of the corresponding vector, plus an
    optional learned null token for classifier-free guidance / condition dropout.
    Heart-rate conditioning (if enabled) is added on top.
    """

    def __init__(self, n_classes: int, emb_dim: int, mode: str):
        super().__init__()
        assert mode in COND_MODES, mode
        self.mode = mode
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        if mode in ("label", "both"):
            self.label_table = nn.Parameter(torch.randn(n_classes, 2, emb_dim) * 0.02)
            self.null_label = nn.Parameter(torch.randn(emb_dim) * 0.02)
        if mode in ("hr", "both"):
            self.hr_emb = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim))
            self.null_hr = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, labels: torch.Tensor | None, hr: torch.Tensor | None,
                null_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "none":
            raise RuntimeError("unconditional model has no condition encoder")
        B = None
        for t in (labels, hr):
            if t is not None:
                B = t.shape[0]
        if B is None:
            B = 1
        parts = []
        if self.mode in ("label", "both"):
            if labels is None:
                lab = self.null_label.expand(B, -1)
            else:
                y = labels.clamp(0.0, 1.0)
                # (B, n_classes, 2, emb): select the present/absent embedding per label
                sel = torch.stack([1.0 - y, y], dim=-1)  # (B, C, 2)
                lab = torch.einsum("bcd,bcde->be", sel,
                                   self.label_table.unsqueeze(0).expand(len(y), -1, -1, -1))
            if null_mask is not None:
                lab = torch.where(null_mask[:, None] > 0, self.null_label.expand_as(lab), lab)
            parts.append(lab)
        if self.mode in ("hr", "both"):
            if hr is None:
                h = self.null_hr.expand(B, -1)
            else:
                h = self.hr_emb(hr[:, None])
                if null_mask is not None:
                    h = torch.where(null_mask[:, None] > 0, self.null_hr.expand_as(h), h)
            parts.append(h)
        return self.proj(sum(parts))


def build_cond_encoder(n_classes: int, emb_dim: int, mode: str, style: str):
    if style == "nle":
        return CondEncoderNLE(n_classes, emb_dim, mode)
    return CondEncoder(n_classes, emb_dim, mode)


class UNet1D(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5, base: int = 64,
                 n_blocks: int = 2, cond_mode: str = "none", emb_dim: int = 128,
                 attn: bool = True, cond_style: str = "mask", aux: bool = False):
        super().__init__()
        assert cond_style in COND_STYLES, cond_style
        self.cond_mode = cond_mode
        self.cond_style = cond_style
        self.emb_dim = emb_dim
        self.aux = aux
        chans = [base, base * 2, base * 4]
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        if cond_mode != "none":
            self.cond_enc = build_cond_encoder(n_classes, emb_dim, cond_mode, cond_style)
        else:
            self.cond_enc = None
        self.in_conv = nn.Conv1d(n_leads, base, 7, padding=3)
        self.downs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        cur = base
        for i, ch in enumerate(chans):
            blocks = nn.ModuleList([ResBlock(cur if j == 0 else ch, ch, emb_dim)
                                    for j in range(n_blocks)])
            self.downs.append(blocks)
            cur = ch
            if i < len(chans) - 1:
                self.down_samples.append(Downsample(ch))
        self.mid1 = ResBlock(cur, cur, emb_dim)
        self.attn = SelfAttention1D(cur) if attn else nn.Identity()
        self.mid2 = ResBlock(cur, cur, emb_dim)
        self.mid_channels = cur
        if aux:
            self.aux_head = nn.Sequential(nn.SiLU(), nn.Linear(cur, 4))
        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, ch in enumerate(reversed(chans)):
            prev = cur if i == 0 else list(reversed(chans))[i - 1]
            self.up_samples.append(Upsample(prev))
            blocks = nn.ModuleList([ResBlock(prev + ch if j == 0 else ch, ch, emb_dim)
                                    for j in range(n_blocks)])
            self.ups.append(blocks)
            cur = ch
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv1d(base, n_leads, 3, padding=1)

    def forward(self, x, t, labels=None, hr=None, null_mask=None):
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim))
        if self.cond_mode != "none":
            emb = emb + self.cond_enc(labels, hr, null_mask)
        h = self.in_conv(x)
        skips = []
        for blocks, ds in zip(self.downs, list(self.down_samples) + [None]):
            for blk in blocks:
                h = blk(h, emb)
            skips.append(h)
            if ds is not None:
                h = ds(h)
        h = self.mid2(self.attn(self.mid1(h, emb)), emb)
        aux_pred = self.aux_head(h.mean(dim=-1)) if self.aux else None
        for i, (us, blocks) in enumerate(zip(self.up_samples, self.ups)):
            h = us(h)
            skip = skips[-(i + 1)]
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, emb)
        out = self.out_conv(F.silu(self.out_norm(h)))
        if self.aux:
            return out, aux_pred
        return out


# ------------------------------------------------------------------- EMA -----
class EMA:
    """Exponential moving average of the U-Net weights (CPU shadow copy)."""

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


# ------------------------------------------------------------------- DDPM ---
class DDPM(nn.Module):
    def __init__(self, unet: UNet1D, n_steps: int = 1000, schedule: str = "linear"):
        super().__init__()
        self.unet = unet
        self.n_steps = n_steps
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, n_steps, dtype=torch.float64)
        else:  # cosine (Nichol & Dhariwal)
            s = 0.008
            steps = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
            f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cum = f / f[0]
            betas = torch.clamp(1 - alphas_cum[1:] / alphas_cum[:-1], 1e-5, 0.999)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cum", alphas_cum.float())
        self.register_buffer("sqrt_ac", alphas_cum.sqrt().float())
        self.register_buffer("sqrt_1mac", (1 - alphas_cum).float().sqrt())

    def q_sample(self, x0, t, noise):
        return self.sqrt_ac[t][:, None, None] * x0 + self.sqrt_1mac[t][:, None, None] * noise

    def loss(self, x0, labels=None, hr=None, null_mask=None,
             aux_targets=None, aux_weight: float = 0.0):
        """Returns dict with diffusion/aux/total losses (v1 callers can use
        `["diffusion"]` or the float value of `["loss"]`)."""
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        out = self.unet(x_t, t, labels, hr, null_mask)
        if isinstance(out, tuple):
            eps, aux_pred = out
        else:
            eps, aux_pred = out, None
        diff = F.mse_loss(eps, noise)
        aux = torch.zeros((), device=x0.device)
        if self.unet.aux and aux_pred is not None and aux_targets is not None and aux_weight > 0:
            mask = torch.isfinite(aux_targets).all(dim=1)
            if mask.any():
                aux = F.mse_loss(aux_pred[mask], aux_targets[mask])
        return {"diffusion": diff, "aux": aux, "loss": diff + aux_weight * aux}

    @torch.no_grad()
    def sample(self, n: int, n_samples: int, device="cuda", ddim_steps: int = 50,
               labels=None, hr=None, guidance_scale: float = 1.0, cond_mode="none",
               n_classes: int = 5, null_condition: bool = False) -> np.ndarray:
        """DDIM sampling (eta=0) with optional classifier-free guidance.

        null_condition=True samples from the learned null condition (nle style) or
        the zero condition (mask style), i.e. the "unconditional" branch.
        """
        self.unet.eval()
        use_nle = getattr(self.unet, "cond_style", "mask") == "nle"
        if null_condition and self.unet.cond_mode != "none":
            if use_nle:
                labels, hr = None, None
            else:
                labels = torch.zeros(n, len(COND_MODES) * 0 + (labels.shape[1] if labels is not None else 5),
                                     device=device) if labels is not None else None
                hr = torch.zeros(n, device=device) if hr is not None else None
        x = torch.randn(n, 12, n_samples, device=device)
        ts = torch.linspace(self.n_steps - 1, 0, ddim_steps, dtype=torch.long, device=device)
        for i in range(ddim_steps):
            t = ts[i]
            tb = torch.full((n,), int(t), device=device, dtype=torch.long)
            eps = self.unet(x, tb, labels, hr)
            if isinstance(eps, tuple):
                eps = eps[0]
            if guidance_scale != 1.0 and cond_mode != "none":
                if use_nle:
                    null_mask = torch.ones(n, device=device)
                    eps_u = self.unet(x, tb, labels, hr, null_mask)
                else:
                    eps_u = self.unet(x, tb, torch.zeros_like(labels) if labels is not None else None,
                                      torch.zeros_like(hr) if hr is not None else None)
                if isinstance(eps_u, tuple):
                    eps_u = eps_u[0]
                eps = eps_u + guidance_scale * (eps - eps_u)
            ac_t = self.sqrt_ac[t]
            x0 = (x - self.sqrt_1mac[t] * eps) / ac_t
            x0 = x0.clamp(-4, 4)
            if i < ddim_steps - 1:
                ac_prev = self.sqrt_ac[ts[i + 1]]
                x = ac_prev * x0 + (1 - ac_prev ** 2).sqrt() * eps
            else:
                x = x0
        return x.cpu().numpy().astype(np.float32)


def build_model(cfg: dict, cond_mode: str, base: int | None = None,
                cond_style: str = "mask", aux: bool = False) -> DDPM:
    unet = UNet1D(n_leads=cfg["n_leads"], n_classes=len(cfg["superclasses"]),
                  base=base or cfg["route_b"]["base_channels"],
                  cond_mode=cond_mode, attn=True, cond_style=cond_style, aux=aux)
    return DDPM(unet, n_steps=cfg["generation"]["ddpm_train_steps"], schedule="linear")


def save_checkpoint(path: str, model: DDPM, meta: dict, ema_state: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    payload = {"model": model.state_dict(), "meta": meta}
    if ema_state is not None:
        payload["ema"] = ema_state
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, cfg: dict, device="cuda", use_ema: bool = True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = build_model(cfg, meta["cond_mode"], base=meta.get("base_channels"),
                        cond_style=meta.get("cond_style", "mask"),
                        aux=meta.get("aux", False))
    state = ckpt.get("ema") if (use_ema and ckpt.get("ema") is not None) else ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    meta["used_ema"] = bool(use_ema and ckpt.get("ema") is not None)
    return model, meta
