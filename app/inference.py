"""Inference helpers shared by the Streamlit app (and by manual CLI checks).

Every route is exposed through one uniform interface:

    available_routes()                     -> dict route -> availability + reason
    generate(route, labels, hr, n=1)       -> (n,12,1000) float32 signals
    measure(signals, fs=100)               -> interval rows (QT/PR/QRS/RR)
    classify(signals)                      -> (n,5) probabilities
    load_real_sample(index=None)           -> (12,1000), meta dict  (PTB-XL test)
    load_reference_signal()                -> (12,1000) real reference for comparison

Route identifiers used across checkpoints/generated data:
    A          mechanistic (no checkpoint; HR only)
    B_uncond   diffusion unconditional (models/checkpoints/b_uncond.pt)
    B_cond     diffusion label-conditioned (models/checkpoints/b_cond.pt)
    C_full     diffusion label+HR (models/checkpoints/c_full.pt)
    D_seed42/43/44  GAN baseline seeds
"""
from __future__ import annotations

import os

import numpy as np

from ecg_common import intervals, paths
from ecg_common.config import load_config

ROUTE_SPECS = {
    "A": {"kind": "mechanistic", "cond": "hr only",
          "note": "McSharry ODE; diagnostic label is not an input"},
    "B_uncond": {"kind": "diffusion", "ckpt": "b_uncond.pt", "cond": "none"},
    "B_cond": {"kind": "diffusion", "ckpt": "b_cond.pt", "cond": "label"},
    "C_full": {"kind": "diffusion", "ckpt": "c_full.pt", "cond": "label+hr"},
    "C_no_label": {"kind": "diffusion", "ckpt": "c_no_label.pt", "cond": "hr"},
    "C_no_hr": {"kind": "diffusion", "ckpt": "c_no_hr.pt", "cond": "label"},
    "C_label_noise": {"kind": "diffusion", "ckpt": "c_label_noise.pt", "cond": "label+hr(10% noise)"},
    "C_label_shuffle": {"kind": "diffusion", "ckpt": "c_label_shuffle.pt", "cond": "label+hr(shuffled)"},
    "D_seed42": {"kind": "gan", "ckpt": "d_seed42.pt", "cond": "none"},
    "D_seed43": {"kind": "gan", "ckpt": "d_seed43.pt", "cond": "none"},
    "D_seed44": {"kind": "gan", "ckpt": "d_seed44.pt", "cond": "none"},
    # ---- v2 routes (literature-driven improvements; base=64, 120 epochs, EMA) ----
    "B_cond_v2": {"kind": "diffusion", "ckpt": "b_cond_v2.pt",
                  "cond": "label (nle embedding)"},
    "C_full_v2": {"kind": "diffusion", "ckpt": "c_full_v2.pt",
                  "cond": "label+hr (nle) + interval-aux (regularity injection)"},
    "C_nle_v2": {"kind": "diffusion", "ckpt": "c_nle_v2.pt",
                 "cond": "label+hr (nle, no interval-aux ablation)"},
    "C_mask_v2": {"kind": "diffusion", "ckpt": "c_mask_v2.pt",
                  "cond": "label+hr (v1 mask embedding + interval-aux ablation)"},
}
MATURE_ROUTES = ["A", "B_uncond", "B_cond", "C_full", "C_full_v2", "D_seed42"]


def available_routes() -> dict[str, dict]:
    out = {}
    for name, spec in ROUTE_SPECS.items():
        kind = spec["kind"]
        if kind == "mechanistic":
            out[name] = {"available": True, "kind": kind, "cond": spec["cond"],
                         "note": spec["note"], "path": None}
        else:
            p = os.path.join(paths.CKPT, spec["ckpt"])
            out[name] = {"available": os.path.exists(p), "kind": kind,
                         "cond": spec["cond"], "note": "",
                         "path": p if os.path.exists(p) else None,
                         "missing_reason": None if os.path.exists(p)
                         else f"checkpoint not trained yet: {spec['ckpt']}"}
    return out


# --------------------------------------------------------------- generation ---
def _cond_tensors(cfg, labels: np.ndarray, hr: np.ndarray, cond_mode: str,
                  hr_mean: float, hr_std: float, device):
    import torch
    lab = None
    h = None
    if cond_mode in ("label", "both", "label+hr"):
        lab = torch.as_tensor(np.asarray(labels, dtype=np.float32), device=device)
    if cond_mode in ("hr", "both", "label+hr"):
        h = torch.as_tensor((np.asarray(hr, dtype=np.float32) - hr_mean) / hr_std,
                            device=device)
    return lab, h


def generate(route: str, labels: np.ndarray | None = None, hr: float = 70.0,
             n: int = 1, seed: int = 42) -> np.ndarray:
    """Generate n signals for a route under one condition. Returns (n,12,1000)."""
    cfg = load_config()
    routes = available_routes()
    if route not in routes:
        raise ValueError(f"unknown route {route}")
    if not routes[route]["available"]:
        raise RuntimeError(routes[route]["missing_reason"])
    n_leads, T = cfg["n_leads"], cfg["n_samples"]
    labels = np.zeros(len(cfg["superclasses"])) if labels is None else np.asarray(labels, float)
    hr_arr = np.full(n, float(hr), dtype=np.float32)
    lab_arr = np.tile(labels[None], (n, 1))
    rng = np.random.default_rng(seed)
    kind = routes[route]["kind"]

    if kind == "mechanistic":
        from models.mechanistic import generate_batch
        return generate_batch(hr_arr, seed=int(rng.integers(1 << 30)), fs=cfg["fs"],
                              n_samples=T)
    if kind == "diffusion":
        import torch
        from models.diffusion import load_checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, meta = load_checkpoint(routes[route]["path"], cfg, device=device)
        cond_mode = meta["cond_mode"]
        lab, h = _cond_tensors(cfg, lab_arr, hr_arr, cond_mode,
                               meta["hr_mean"], meta["hr_std"], device)
        return model.sample(n, T, device=device, ddim_steps=cfg["generation"]["ddim_steps"],
                            labels=lab, hr=h, cond_mode=cond_mode,
                            n_classes=len(cfg["superclasses"]))
    if kind == "gan":
        import torch
        from models.gan import Generator1D
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(routes[route]["path"], map_location=device, weights_only=False)
        meta = ckpt["meta"]
        G = Generator1D(z_dim=meta["z_dim"], cond_dim=meta["cond_dim"],
                        n_leads=n_leads, n_samples=T).to(device)
        G.load_state_dict(ckpt["G"])
        G.eval()
        cond = None
        if meta["cond_mode"] != "none":
            lab, h = _cond_tensors(cfg, lab_arr, hr_arr, meta["cond_mode"],
                                   meta["hr_mean"], meta["hr_std"], device)
            cond = lab if h is None else (lab if lab is not None else None)
            if lab is not None and h is not None:
                cond = torch.cat([lab, h[:, None]], 1)
            elif h is not None:
                cond = h[:, None]
        z = torch.randn(n, meta["z_dim"], device=device,
                        generator=torch.Generator(device=device).manual_seed(seed))
        with torch.no_grad():
            return G(z, cond).cpu().numpy().astype(np.float32)
    raise ValueError(kind)


# ------------------------------------------------------------- measurements ---
def measure(signals: np.ndarray, fs: int = 100) -> list[dict]:
    cfg = load_config()
    rows, failures = intervals.extract_from_signals(signals, fs, cfg["lead_ii_index"])
    for r, f in zip(rows, failures):
        r.setdefault("failure", f["reason"])
    return rows


def classify(signals: np.ndarray):
    from models.classifier import load_classifier, predict_proba
    model, cfg, device = load_classifier()
    return predict_proba(model, signals, device=device)


# --------------------------------------------------------------- real data ---
def load_real_sample(index: int | None = None, seed: int = 42):
    p = os.path.join(paths.CACHE, "ptbxl_test.npz")
    if not os.path.exists(p):
        raise FileNotFoundError("PTB-XL test cache missing -- run preprocessing first")
    z = np.load(p, allow_pickle=False)
    n = len(z["signals"])
    if index is None:
        index = int(np.random.default_rng(seed).integers(0, n))
    index = int(index) % n
    meta = {
        "index": index,
        "ecg_id": int(z["ecg_id"][index]),
        "hr": float(z["hr"][index]),
        "labels": z["labels"][index].astype(int).tolist(),
    }
    return z["signals"][index], meta


PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


def plot_12lead(sig: np.ndarray, fs: int = 100, title: str = ""):
    """Standard 12-lead ECG paper-style plot; sig (12, T)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = load_config()
    leads = cfg["lead_names"]
    T = sig.shape[-1]
    t = np.arange(T) / fs
    fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t, sig[i], lw=0.8, color="#111111")
        ax.set_ylabel(leads[i], fontsize=8, rotation=0, labelpad=14)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=7)
        if i < 10:
            ax.set_xticklabels([])
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_lead_ii_compare(signals: dict[str, np.ndarray], fs: int = 100):
    """Overlay lead II of several routes (dict name -> (12,T))."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (name, sig) in enumerate(signals.items()):
        ax.plot(np.arange(sig.shape[-1]) / fs, sig[1], lw=1.0, label=name,
                color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Lead II (z-score)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def plot_fft(signals: dict[str, np.ndarray], fs: int = 100):
    """FFT magnitude of Lead II for all routes (same condition)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (name, sig) in enumerate(signals.items()):
        x = np.asarray(sig[1], dtype=float)
        x = x - x.mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / fs)
        ax.plot(freqs, spec, lw=1.0, label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlim(0, 40)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|FFT| (a.u.)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig
