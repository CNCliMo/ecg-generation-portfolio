"""Dataset loaders / preprocessing entry points (PTB-XL 100 Hz + MIT-BIH)."""
from __future__ import annotations

import os

import numpy as np

from ecg_common import paths


# ------------------------------------------------------------------ PTB-XL ---
def load_ptbxl_split(split: str):
    """Load cached PTB-XL split. split in {train,val,test}.

    Returns dict with keys: signals (N,12,1000) float32, labels (N,5) uint8,
    hr (N,) float32, ecg_id, patient_id, fold.
    """
    p = os.path.join(paths.CACHE, f"ptbxl_{split}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found -- run `python preprocess/extract_ptbxl.py` and "
            f"`python preprocess/make_ptbxl_cache.py` first")
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_ptbxl_meta():
    import pandas as pd
    return pd.read_csv(os.path.join(paths.CACHE, "ptbxl_meta.csv"))


# ---------------------------------------------------------------- MIT-BIH ---
def load_mitbih():
    """Returns dict: records (list of ids), signals filtered per record (dict),
    beats (N,1,win) float32, rpeak_idx (N,), record ids."""
    p = os.path.join(paths.CACHE, "mitbih.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found -- run `python preprocess/make_mitbih_cache.py`")
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def concatenate_lead(signals: np.ndarray, lead_index: int) -> np.ndarray:
    """Pick one lead from (N,12,T) -> (N,T)."""
    return signals[:, lead_index, :]
