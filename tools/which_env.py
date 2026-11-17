# -*- coding: utf-8 -*-
"""Show which Python interpreter and package locations are in use.

Absolute paths are masked (<repo>, <home>) so the output can be pasted into
issues / demo videos without leaking machine-specific paths.
"""
import importlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mask(text: str) -> str:
    text = str(text).replace(REPO, "<repo>")
    text = text.replace(REPO.replace("\\", "/"), "<repo>")
    # mask any remaining Windows user-profile path (drive:\\Users\\<name>)
    text = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+", "<home>", text)
    return text


print("executable :", mask(sys.executable))
print("version    :", sys.version.split()[0])
print("sys.prefix :", mask(sys.prefix))
print("env name   :", mask(sys.prefix).replace("\\", "/").rstrip("/").split("/")[-1])
print("PYTHONPATH :", [mask(p) for p in sys.path if REPO in p or "ECG" in p])

mods = ["torch", "numpy", "scipy", "sklearn", "matplotlib", "pandas", "wfdb",
        "pywt", "neurokit2", "streamlit", "ecgen_eval", "pyarrow"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        loc = mask(getattr(mod, "__file__", "?") or "?")
        print("%-12s %-12s <- %s" % (m, getattr(mod, "__version__", "?"), loc))
    except Exception as exc:  # noqa: BLE001
        print("%-12s FAIL %s" % (m, mask(exc)))

import torch
print("torch cuda :", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
