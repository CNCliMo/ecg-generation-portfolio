"""ECG interval extraction shared by the reference-distribution builder and the
regularity-layer evaluation (task book 5.2).

Rules are hard-coded on purpose:
  - signal used: Lead II (index 1) for 12-lead synthetic/PTB-XL data; MLII for MIT-BIH
  - R peaks: `ecg_common.config.detect_rpeaks` (Pan-Tompkins-style), the single
    source of truth also used for the heart-rate labels
  - delineation: NeuroKit2 `ecg_delineate`, method="dwt", on the full signal with
    the detected R peaks (NeuroKit2 requires >= 4 s of signal, so short beat
    windows are NOT used)
  - intervals (ms): QRS = R_Offset - R_Onset, PR = R_Onset - P_Onset,
    QT = T_Offset - R_Onset
  - failures are counted and reported with the missing landmark name, never
    silently dropped (task book 11)
"""
from __future__ import annotations

import numpy as np

# minimum signal duration NeuroKit2's ecg_delineate can segment
NK_MIN_SECONDS = 4.0
PRE_S = 0.35
POST_S = 0.75


def detect_rpeaks(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    """Single source of truth: the project-wide Pan-Tompkins-style detector
    (see ecg_common.config.detect_rpeaks)."""
    from ecg_common.config import detect_rpeaks as _detect
    return _detect(signal_1d, fs)


def rr_variability(signal_1d: np.ndarray, fs: float) -> dict:
    """R-R interval variability: CV (=SD/mean), RMSSD and n_RR."""
    peaks = detect_rpeaks(signal_1d, fs)
    if len(peaks) < 3:
        return {"rr_cv": np.nan, "rmssd_ms": np.nan, "n_rr": 0}
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.2) & (rr < 2.5)]
    if len(rr) < 2:
        return {"rr_cv": np.nan, "rmssd_ms": np.nan, "n_rr": int(len(rr))}
    return {
        "rr_cv": float(np.std(rr) / np.mean(rr)),
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000),
        "n_rr": int(len(rr)),
    }


def _first_valid(arr) -> float:
    if arr is None:
        return float("nan")
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[arr > 0]
    return float(arr[0]) if len(arr) else float("nan")


def delineate_signal(signal: np.ndarray, fs: float, rpeaks: np.ndarray) -> dict | None:
    """Run NeuroKit2 delineation on a >= 4 s signal. Returns per-beat landmark dict
    (lists, one entry per R peak) or None on failure.
    """
    import neurokit2 as nk
    try:
        res = nk.ecg_delineate(np.asarray(signal, dtype=np.float64),
                               rpeaks=np.asarray(rpeaks), sampling_rate=fs,
                               method="dwt", show=False)
        return res[1] if isinstance(res, tuple) else res
    except Exception:  # noqa: BLE001
        return None


def beat_intervals(d: dict, beat_idx: int, fs: float) -> dict:
    """Intervals (ms) for one beat of a NeuroKit2 delineation dict."""
    out = {"QRS_ms": np.nan, "PR_ms": np.nan, "QT_ms": np.nan, "failure": ""}
    try:
        r_on = _first_valid(d["ECG_R_Onsets"][beat_idx])
        r_off = _first_valid(d["ECG_R_Offsets"][beat_idx])
        p_on = _first_valid(d["ECG_P_Onsets"][beat_idx])
        t_off = _first_valid(d["ECG_T_Offsets"][beat_idx])
        if np.isfinite(r_on) and np.isfinite(r_off) and r_off > r_on:
            out["QRS_ms"] = (r_off - r_on) / fs * 1000.0
        if np.isfinite(p_on) and np.isfinite(r_on) and r_on > p_on:
            out["PR_ms"] = (r_on - p_on) / fs * 1000.0
        if np.isfinite(t_off) and np.isfinite(r_on) and t_off > r_on:
            out["QT_ms"] = (t_off - r_on) / fs * 1000.0
        if not all(np.isfinite(out[k]) for k in ("QRS_ms", "PR_ms", "QT_ms")):
            missing = [k for k in ("QRS_ms", "PR_ms", "QT_ms") if not np.isfinite(out[k])]
            out["failure"] = "missing_landmark:" + "|".join(missing)
    except Exception as exc:  # noqa: BLE001
        out["failure"] = f"delineate_parse_error:{type(exc).__name__}"
    return out


def extract_from_signals(signals: np.ndarray, fs: float, lead_index: int,
                         max_beats_per_sample: int = 1) -> tuple[list[dict], list[dict]]:
    """Extract interval rows + failure rows from (N, leads, T) signals (>= 4 s each).

    One row per delineated beat (beats chosen near the middle of the record).
    """
    interval_rows, failures = [], []
    for i, sig in enumerate(signals):
        lead = np.asarray(sig[lead_index], dtype=np.float64)
        peaks = detect_rpeaks(lead, fs)
        tv = rr_variability(lead, fs)
        if len(peaks) < 2:
            failures.append({"item": i, "reason": "too_few_rpeaks"})
            continue
        if len(lead) < fs * NK_MIN_SECONDS:
            failures.append({"item": i, "reason": "signal_too_short_for_delineate"})
            continue
        d = delineate_signal(lead, fs, peaks)
        if d is None:
            failures.append({"item": i, "reason": "delineate_error"})
            continue
        mid = len(lead) // 2
        order = np.argsort(np.abs(peaks - mid))
        chosen = sorted(order[:max_beats_per_sample].tolist())
        for bi in chosen:
            row = beat_intervals(d, bi, fs)
            row.update({"item": i, "rpeak": int(peaks[bi]), **tv})
            interval_rows.append(row)
            if row["failure"]:
                failures.append({"item": i, "reason": row["failure"]})
    return interval_rows, failures


def extract_from_mitbih(record_ids, rpeak_abs, records_dir: str, fs: float,
                        nk_min_seconds: float = NK_MIN_SECONDS) -> tuple[list[dict], list[dict]]:
    """Extract intervals for sampled MIT-BIH normal beats by delineating a
    >= 4 s window around each target beat (NeuroKit2's minimum requirement)."""
    interval_rows, failures = [], []
    cache = {}
    half = int(nk_min_seconds * fs / 2)
    for i in range(len(record_ids)):
        rec = str(record_ids[i])
        target = int(rpeak_abs[i])
        try:
            if rec not in cache:
                cache[rec] = np.load(f"{records_dir}/{rec}.npy").astype(np.float64)
            full = cache[rec]
        except Exception as exc:  # noqa: BLE001
            failures.append({"item": i, "reason": f"record_load_error:{type(exc).__name__}"})
            continue
        start = max(0, target - half)
        end = min(len(full), target + half)
        if end - start < fs * nk_min_seconds:
            failures.append({"item": i, "reason": "window_too_short"})
            continue
        seg = full[start:end]
        peaks = detect_rpeaks(seg, fs)
        if len(peaks) < 2:
            failures.append({"item": i, "reason": "too_few_rpeaks"})
            continue
        k = int(np.argmin(np.abs(peaks - (target - start))))
        d = delineate_signal(seg, fs, peaks)
        if d is None:
            failures.append({"item": i, "reason": "delineate_error"})
            continue
        row = beat_intervals(d, k, fs)
        row.update({"item": i})
        interval_rows.append(row)
        if row["failure"]:
            failures.append({"item": i, "reason": row["failure"]})
    return interval_rows, failures
