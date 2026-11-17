"""Route A: mechanistic ECG synconda environment (no training).

1. McSharry et al. (2003) three-coupled-ODE ECG model, the same dynamical
   system used by PhysioNet's ECGSYN: phase theta advances at the instantaneous
   heart rate and a single state variable z is driven by five Gaussian PQRST
   attractors (parameters below are the ECGSYN defaults, hard-coded).

2. Lead projection: the scalar waveform is projected onto approximate 12-lead
   "lead vectors" as
        lead_i(t) = polarity_i * z(t) + 0.12 * cos(theta(t) - phi_i) * z(t) * gain_i
   where polarity_i encodes the R-wave progression (V1 negative -> V6 positive,
   aVR inverted) and the second term adds lead-dependent phase/morphology
   variation. This is the "lead projection matrix extension" required by the
   task book; it is a heuristic geometric projection, documented as a deviation
   in README section 7.

3. Physiological variability for a generative baseline: HR drawn per sample,
   R-R variability (slow respiratory sinusoid + OU noise), P/T amplitude jitter,
   small phase jitter and noise.

Output convention matches every other route: float32 (12, 1000) at 100 Hz,
per-lead z-scored, Lead II at index 1.
"""
from __future__ import annotations

import numpy as np

from ecg_common.config import load_config

# ECGSYN / McSharry default PQRST attractor parameters
A = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
B = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
THETA = np.array([-np.pi / 3, -np.pi / 12, 0.0, np.pi / 12, np.pi / 2])

# approximate lead vectors: frontal-plane angles (deg) and precordial angles,
# plus polarity/gain approximating R-wave progression V1(neg) -> V6(pos)
LEAD_ANGLES_DEG = np.array([0.0, 60.0, 120.0, -150.0, -30.0, 90.0,
                            115.0, 90.0, 60.0, 30.0, 0.0, -15.0])
LEAD_GAINS = np.array([1.0, 1.2, 0.6, 0.9, 0.5, 0.85,
                       0.8, 1.0, 1.15, 1.25, 1.15, 1.0])
# polarity reflects standard 12-lead morphology (aVR inverted; V1-V2 mostly
# negative R progression; V4-V6 positive)
LEAD_POLARITY = np.array([1.0, 1.2, 0.6, -1.0, 0.45, 0.9,
                          -0.7, -0.4, 0.05, 0.6, 0.9, 0.8])


def synthesize_ecg(hr_bpm: float, fs: float = 100, n_samples: int = 1000,
                   rng: np.random.Generator | None = None, rr_cv: float = 0.03,
                   noise: float = 0.0008, amp_jitter: float = 0.12,
                   fs_int: int = 500) -> np.ndarray:
    """Generate one 12-lead, 10 s ECG at the target heart rate. Returns (12, n_samples).

    The ODE is integrated at fs_int=500 Hz (the McSharry R attractor is ~40 ms
    wide, too narrow for 100 Hz Euler integration) and then decimated to fs with
    a polyphase anti-alias filter.
    """
    from scipy.signal import resample_poly

    cfg = load_config()
    rng = rng or np.random.default_rng(cfg["seed"])
    assert fs_int % fs == 0, "fs_int must be an integer multiple of fs"
    dt = 1.0 / fs_int
    n_int = n_samples * (fs_int // fs)
    n = n_int + int(2.0 * fs_int)          # extra lead-in to settle the ODE
    t = np.arange(n) * dt

    # instantaneous heart rate: slow respiratory sinusoid + OU-like jitter
    resp = 1.0 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
    ou = np.zeros(n)
    for k in range(1, n):
        ou[k] = 0.98 * ou[k - 1] + rng.normal(0, 0.35)
    hr_inst = np.clip(hr_bpm * (1.0 + rr_cv * (resp + ou)), 25.0, 230.0)

    theta = np.full(n, -np.pi)
    z = np.zeros(n)
    a = A * (1.0 + amp_jitter * rng.normal(0, 1, size=5))
    b0 = np.clip(B * (1.0 + 0.1 * rng.normal(0, 1, size=5)), 0.05, 0.6)
    for k in range(1, n):
        theta[k] = (theta[k - 1] + 2 * np.pi * hr_inst[k] / 60.0 * dt + np.pi) % (2 * np.pi) - np.pi
        # ECGSYN heart-rate rescaling: h=sqrt(bpm/60), widths *= h,
        # angles of P,Q,S,T *= sqrt(h), h, h, sqrt(h) respectively
        h = np.sqrt(hr_inst[k] / 60.0)
        angles = THETA * np.array([np.sqrt(h), h, 1.0, h, np.sqrt(h)])
        b = np.clip(b0 * h, 0.05, 0.7)
        d = theta[k] - angles
        d = (d + np.pi) % (2 * np.pi) - np.pi   # ECGSYN wraps into (-pi, pi]
        dz = a * d * np.exp(-(d ** 2) / (2 * b ** 2))
        z[k] = z[k - 1] + dt * (-dz.sum() - z[k - 1])
    start = int(2.0 * fs_int)
    z = z[start:start + n_int]
    theta_seg = theta[start:start + n_int]

    # lead projection: polarity (R-wave progression) + lead-dependent phase term
    ang = np.deg2rad(LEAD_ANGLES_DEG + rng.normal(0, 4.0, size=12))
    gain = LEAD_GAINS * (1.0 + 0.05 * rng.normal(0, 1, size=12))
    leads = (LEAD_POLARITY[:, None] * gain[:, None] * z[None, :] +
             0.12 * gain[:, None] * np.cos(theta_seg[None, :] - ang[:, None]) * z[None, :])

    # anti-aliased decimation to the project sampling rate, then measurement noise
    leads = resample_poly(leads, fs, fs_int, axis=1)[:, :n_samples]
    leads = leads + rng.normal(0, noise, size=leads.shape)

    x = leads.astype(np.float64)
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def generate_batch(hrs: np.ndarray, seed: int = 42, fs: int = 100,
                   n_samples: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((len(hrs), 12, n_samples), dtype=np.float32)
    for i, hr in enumerate(hrs):
        out[i] = synthesize_ecg(float(hr), fs=fs, n_samples=n_samples, rng=rng)
    return out


