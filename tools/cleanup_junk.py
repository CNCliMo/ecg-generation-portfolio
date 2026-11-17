# -*- coding: utf-8 -*-
"""Safe project cleanup: removes only clearly-unnecessary leftovers.

Never touches: data archives needed for reproducibility, preprocess caches,
checkpoints, results (metrics/figures/logs/generated), venvs, source code.

Usage: powershell -File tools\run.ps1 tools\cleanup_junk.py [--apply]
Without --apply it only reports what would be removed.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLY = "--apply" in sys.argv

removed = []
failed = []


def rm(path, kind="file"):
    rel = os.path.relpath(path, ROOT)
    if APPLY:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed.append((rel, kind))
        except OSError as exc:
            failed.append((rel, str(exc)[:80]))
    else:
        removed.append((rel, kind))


# 1) partial download from the abandoned PhysioNet fallback
p = os.path.join(ROOT, "data", "mit-bih", "mitdb-1.0.0.zip.part")
if os.path.exists(p):
    rm(p, "partial download")

# 2) leftover ACL experiment directories at repo root (0-byte, from the sandbox era)
for name in ("mkd-h_v3f8co", "mt-makedirs700", "mt-mode700"):
    p = os.path.join(ROOT, name)
    if os.path.isdir(p):
        rm(p, "sandbox ACL test dir")

# 3) stray pip temp dirs created at repo root
for name in os.listdir(ROOT):
    if name.startswith(("pip-build-tracker-", "pip-ephem-wheel-cache-", "pip-install-",
                        "pip-unpack-", "pip-download-", "pip-target-")):
        rm(os.path.join(ROOT, name), "stray pip temp dir")

# 4) __pycache__ inside project code only (never inside .deps/.venv*)
code_dirs = ["ecg_common", "experiments", "models", "preprocess", "app", "tools", "configs"]
for d in code_dirs + ["."]:
    base = os.path.join(ROOT, d) if d != "." else ROOT
    if not os.path.isdir(base):
        continue
    for cur, dirs, files in os.walk(base):
        relcur = os.path.relpath(cur, ROOT)
        if relcur.startswith((".deps", ".venv", ".venv_iso", "data", "results", "preprocess\\cache")):
            dirs[:] = []
            continue
        if "__pycache__" in dirs:
            rm(os.path.join(cur, "__pycache__"), "__pycache__")
            dirs.remove("__pycache__")
        if d == ".":  # root: do not recurse into big folders
            dirs[:] = [x for x in dirs if x in code_dirs]

# 5) transient files inside .tmp (keep the directory itself)
tmp = os.path.join(ROOT, ".tmp")
if os.path.isdir(tmp):
    for name in os.listdir(tmp):
        rm(os.path.join(tmp, name), ".tmp leftover")

if APPLY:
    print("[cleanup] removed %d items, %d failed" % (len(removed), len(failed)))
    for rel, _ in removed[:80]:
        print("  -", rel)
    if len(removed) > 80:
        print("  ... and", len(removed) - 80, "more")
    for rel, err in failed:
        print("  FAILED:", rel, err)
else:
    print("[cleanup] DRY RUN: %d items would be removed" % len(removed))
    for rel, kind in removed[:100]:
        print("  -", rel, "|", kind)
    if len(removed) > 100:
        print("  ... and", len(removed) - 100, "more")
