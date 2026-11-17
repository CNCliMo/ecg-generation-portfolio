"""Shared paths for the ECG generation & diagnosis project.

The whole project lives under the workspace root (<REPO_ROOT>);
nothing is written outside of it (开发环境 sandbox only allows the workspace).
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(ROOT, "data")
DATA_PTBXL = os.path.join(DATA, "ptb-xl")
DATA_MITBIH = os.path.join(DATA, "mit-bih")

PREPROCESS = os.path.join(ROOT, "preprocess")
CACHE = os.path.join(PREPROCESS, "cache")

MODELS = os.path.join(ROOT, "models")
CKPT = os.path.join(MODELS, "checkpoints")

EXPERIMENTS = os.path.join(ROOT, "experiments")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
GENERATED = os.path.join(RESULTS, "generated")
LOGS = os.path.join(RESULTS, "logs")

APP = os.path.join(ROOT, "app")
CONFIGS = os.path.join(ROOT, "configs")

for _d in (DATA, DATA_PTBXL, DATA_MITBIH, CACHE, CKPT, RESULTS, FIGURES,
           GENERATED, LOGS):
    os.makedirs(_d, exist_ok=True)
