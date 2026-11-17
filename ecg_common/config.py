"""Config loading + global seeding + signal preprocessing primitives.

All preprocessing rules are hard-coded here on purpose (task book 11: 导联映射
规则、正常心搏筛选规则、参数提取配置均写死在脚本中).
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

from ecg_common import paths

_CFG = None


def load_config(path: str | None = None) -> dict:
    global _CFG
    if _CFG is not None and path is None:
        return _CFG
    path = path or os.path.join(paths.CONFIGS, "base.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if _CFG is None:
        _CFG = cfg
    return cfg


def seed_everything(seed: int | None = None) -> int:
    cfg = load_config()
    seed = int(cfg["seed"] if seed is None else seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # torch optional for preprocessing-only scripts
        pass
    return seed


# ---------------------------------------------------------------- filters ---
def bandpass(sig: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0. sig: (T,) or (T, L)."""
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    high = min(high, 0.95 * nyq)
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, sig, axis=0)


def notch(sig: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """50 Hz notch. Skipped (returns input) when freq is at/above Nyquist.

    IMPORTANT: PTB-XL `records100` is sampled at 100 Hz, so 50 Hz == Nyquist and
    a notch is mathematically undefined. The 50 Hz notch is therefore only
    applied where fs permits (MIT-BIH at 360 Hz). This is documented in
    results/data_report.md and in the README deviation table.
    """
    from scipy.signal import iirnotch, filtfilt
    if fs <= 2.0 * freq + 5.0:
        return sig
    b, a = iirnotch(freq / (fs / 2.0), q)
    return filtfilt(b, a, sig, axis=0)


def preprocess_record(sig: np.ndarray, fs: float, cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """Band-pass + notch + per-lead z-score. sig: (T, L) float. Returns (T, L), info."""
    cfg = cfg or load_config()
    pp = cfg["preprocess"]
    info = {"notch_applied": False, "zscore_ok": True}
    x = np.nan_to_num(np.asarray(sig, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = bandpass(x, fs, pp["bandpass_hz"][0], pp["bandpass_hz"][1], pp["bandpass_order"])
    if fs > pp["notch_min_fs"]:
        x = notch(x, fs, pp["notch_hz"], pp["notch_q"])
        info["notch_applied"] = True
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    x = (x - mu) / sd
    info["flat_leads"] = int((x.std(axis=0) < 1e-8).sum())
    info["zscore_ok"] = bool(np.isfinite(x).all())
    return x.astype(np.float32), info


# ------------------------------------------------------------ heart rate ---
def detect_rpeaks(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    """R-peak detector used everywhere in the project (HR labels, reference
    R-R variability, generated-signal measurements, delineation windows).

    Pan-Tompkins-style: 5-15 Hz band-pass -> derivative -> square -> 80 ms moving
    average -> peak picking with 0.3 s refractory and a median-relative
    threshold. This variant was validated against NeuroKit2 on PTB-XL test
    records (the plain band-pass + prominence detector over-counted T waves as R
    peaks and inflated the mean HR from ~75 to ~108 bpm).
    """
    from scipy.signal import find_peaks
    x = bandpass(np.asarray(signal_1d, dtype=np.float64), fs, 5.0, 15.0, order=3)
    d = np.diff(x, prepend=x[0])
    win = max(3, int(0.08 * fs))
    integ = np.convolve(d * d, np.ones(win) / win, mode="same")
    med = float(np.median(integ))
    p995 = float(np.percentile(integ, 99.5))
    thr = med + 0.25 * (p995 - med)
    if not np.isfinite(thr) or thr <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(integ, distance=int(0.30 * fs), height=thr)
    return peaks


def estimate_hr(lead_ii: np.ndarray, fs: float, cfg: dict | None = None) -> tuple[float, int]:
    """R-peak based heart rate (bpm).

    Returns (bpm, n_peaks); bpm=nan when fewer than 3 R peaks are found.
    """
    cfg = cfg or load_config()
    pp = cfg["preprocess"]
    peaks = detect_rpeaks(lead_ii, fs)
    n = len(peaks)
    if n < 3:
        return float("nan"), n
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.2) & (rr < 2.5)]
    if len(rr) < 2:
        return float("nan"), n
    bpm = 60.0 / float(np.median(rr))
    if not (pp["hr_min_bpm"] <= bpm <= pp["hr_max_bpm"]):
        return float("nan"), n
    return float(bpm), n
