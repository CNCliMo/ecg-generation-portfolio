"""One-shot environment self-check for reproducibility.

Usage: powershell -File tools\run.ps1 tools\env_check.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    print("python", sys.version.split()[0], "| executable", sys.executable)
    print("cwd", os.getcwd())
    print("PYTHONPATH", os.environ.get("PYTHONPATH"))
    import importlib
    for m in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "wfdb",
              "pywt", "neurokit2", "streamlit", "ecgen_eval", "torch"):
        try:
            mod = importlib.import_module(m)
            print(f"OK   {m:12s} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {m:12s} {type(exc).__name__}: {exc}")
    try:
        import torch
        print("torch cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0),
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    except Exception as exc:  # noqa: BLE001
        print("torch check failed:", exc)
    from ecg_common import paths
    print("workspace: <repo root>")


if __name__ == "__main__":
    main()
