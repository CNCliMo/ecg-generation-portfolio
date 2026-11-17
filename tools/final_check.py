# -*- coding: utf-8 -*-
"""Deliverable completeness check (run before handoff / after the power cut).

Checks that every promised artifact exists with sane content and writes a
human-readable report to results/final_check.md.

Usage: powershell -File tools\run.ps1 tools\final_check.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "preprocess", "cache")
CKPT = os.path.join(ROOT, "models", "checkpoints")
GEN = os.path.join(RESULTS, "generated")
FIG = os.path.join(RESULTS, "figures")
LOGS = os.path.join(RESULTS, "logs")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

lines = []
ok_count = 0
bad_count = 0


def check(name, cond, detail=""):
    global ok_count, bad_count
    if cond:
        ok_count += 1
        lines.append(f"| ✅ | {name} | {detail} |\n")
    else:
        bad_count += 1
        lines.append(f"| ❌ | {name} | {detail} |\n")


def file_ok(path, min_bytes=1):
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


lines.append("# 交付物完整性检查\n")
lines.append("\n| 状态 | 项目 | 详情 |\n|---|---|---|\n")

# ---------------------------------------------------------------- data ----
for f in ("ptbxl_train.npz", "ptbxl_val.npz", "ptbxl_test.npz", "mitbih.npz",
          "ref_params_ptbxl_test.csv", "ref_params_mitbih.csv"):
    check(f"缓存 {f}", file_ok(os.path.join(CACHE, f)))

# ------------------------------------------------------------ checkpoints --
expected_ckpts = {"classifier_best.pt": "分类器", "b_uncond.pt": "路线B无条件",
                  "b_cond.pt": "路线B条件", "c_full.pt": "路线C完整",
                  "c_no_label.pt": "C-去标签", "c_no_hr.pt": "C-去心率",
                  "c_label_noise.pt": "C-条件加噪", "c_label_shuffle.pt": "C-条件替换",
                  "d_seed42.pt": "路线D seed42", "d_seed43.pt": "D seed43",
                  "d_seed44.pt": "D seed44"}
for f, label in expected_ckpts.items():
    p = os.path.join(CKPT, f)
    if file_ok(p):
        try:
            import torch
            ck = torch.load(p, map_location="cpu", weights_only=False)
            epoch = ck.get("meta", {}).get("epoch", "?")
            check(f"checkpoint {f}（{label}）", True, f"{os.path.getsize(p)/1e6:.1f}MB, epoch={epoch}")
        except Exception as exc:  # noqa: BLE001
            check(f"checkpoint {f}（{label}）", False, f"无法加载: {exc}")
    else:
        check(f"checkpoint {f}（{label}）", False, "缺失（明天训练）")

# ------------------------------------------------------------- generated ---
for prefix in ("A", "B_uncond", "B_cond", "C_full", "C_no_label", "C_no_hr",
               "C_label_noise", "C_label_shuffle", "D_seed42", "D_seed43", "D_seed44"):
    cond = os.path.join(GEN, f"{prefix}_cond.npz")
    uncond = os.path.join(GEN, f"{prefix}_uncond.npz")
    present, detail = False, "缺失"
    for p in (cond, uncond):
        if file_ok(p):
            z = np.load(p, allow_pickle=False)
            sig = z["signals"]
            present = True
            detail = f"{os.path.basename(p)} {sig.shape}, finite={bool(np.isfinite(sig).all())}"
            break
    check(f"生成信号 {prefix}", present, detail)

# --------------------------------------------------------------- metrics ---
for f in ("metrics.csv", "metrics_morphology.csv", "metrics_regularity.csv",
          "metrics_functional.csv", "class_distribution.csv", "summary.md",
          "classifier_test_report.md", "classifier_test_report.csv"):
    check(f"指标 {f}", file_ok(os.path.join(RESULTS, f)))

if file_ok(os.path.join(RESULTS, "metrics.csv")):
    import pandas as pd
    df = pd.read_csv(os.path.join(RESULTS, "metrics.csv"))
    routes = sorted(set(df["route"].astype(str)) - {"overall"})
    check("metrics.csv 路线覆盖", True, f"{len(df)} 行, routes={routes}")

# ---------------------------------------------------------------- figures --
n_fig = len([f for f in os.listdir(FIG) if f.endswith(".png")]) if os.path.isdir(FIG) else 0
check("图表", n_fig >= 8, f"{n_fig} 张 PNG")

# ------------------------------------------------------------------ logs ---
n_fail = len([f for f in os.listdir(LOGS) if f.startswith("failures_")]) if os.path.isdir(LOGS) else 0
check("失败记录 CSV", n_fail >= 4, f"{n_fail} 个 failures_*.csv")

# ------------------------------------------------------------------ docs ---
for f in ("README.md", "PROJECT_LAYOUT.md", "requirements.txt", "docs/DATA.md", "docs/CHECKPOINTS.md", "docs/METHOD.md"):
    check(f"文档 {f}", file_ok(os.path.join(ROOT, f), 300))
check("数据下载/检查脚本", file_ok(os.path.join(ROOT, "preprocess", "check_data.py"), 500))

lines.append(f"\n**汇总：{ok_count} 项通过，{bad_count} 项待补/缺失。**")
lines.append("\n> 缺失项在明天补训练/生成/评估后重跑本脚本即可。\n")

out = os.path.join(RESULTS, "final_check.md")
open(out, "w", encoding="utf-8").write("".join(lines))
print(f"[final_check] OK={ok_count} MISSING={bad_count} (details -> {out})")
