"""v2 comparison report: v1 baseline vs v2 improved routes + ablation table.

Reads (produced by the pipeline):
  results/metrics.csv
  results/metrics_downstream.csv
  results/metrics_consistency.csv
  results/metrics_functional.csv

Writes:
  results/summary_v2.md
  results/figures/fig_v1_v2_comparison.png

Usage: python experiments/make_v2_comparison.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.logging_utils import get_logger, save_json

LOG = get_logger("make_v2_comparison")
V1_ROUTES = ["B_cond", "c_full", "c_no_label", "c_no_hr", "c_label_noise", "c_label_shuffle"]
V2_ROUTES = ["b_cond_v2", "c_nle_v2", "c_full_v2", "c_mask_v2", "c_aux05_v2"]


def read(name):
    p = os.path.join(paths.RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main() -> None:
    met = read("metrics.csv")
    down = read("metrics_downstream.csv")
    cons = read("metrics_consistency.csv")
    if met is None:
        raise SystemExit("metrics.csv missing -- run the evaluation pipeline first")

    def value(df, route, metric, group=None, domain=None, param=None):
        if df is None:
            return np.nan
        sub = df[df["route"] == route]
        if group is not None and "metric_group" in sub:
            sub = sub[sub["metric_group"] == group]
        if "metric" in sub:
            sub = sub[sub["metric"] == metric]
        if domain is not None and "domain" in sub:
            sub = sub[sub["domain"] == domain]
        if param is not None and "param" in sub:
            sub = sub[sub["param"] == param]
        if sub.empty:
            return np.nan
        return float(sub["value"].iloc[0])

    def regularity_w(route, param):
        if met is None:
            return np.nan
        sub = met[(met["route"] == route) & (met["metric_group"] == "regularity") &
                  (met["metric"] == "wasserstein") & (met["domain"] == "in_domain") &
                  (met["param"] == param)]
        return float(sub["value"].iloc[0]) if not sub.empty else np.nan

    rows = []
    for route in V1_ROUTES + V2_ROUTES:
        rows.append({
            "route": route,
            "family": "v1" if route in V1_ROUTES else "v2",
            "MMD↓": value(met, route, "mmd", "morphology"),
            "PSD↓": value(met, route, "psd", "morphology"),
            "FD↓": value(met, route, "fd", "morphology"),
            "QT_W↓": regularity_w(route, "QT_ms"),
            "RR_CV_W↓": regularity_w(route, "rr_cv"),
            "cond_consistency↑": value(met, route, "conditional_consistency_macro", "functional"),
            "class_KL↓": value(met, route, "unconditional_class_KL", "functional"),
            "HR_MAE↓": value(cons, route, "hr_mae"),
            "lead_residual↓": value(cons, route, "lead_residual_mean"),
            "coverage↑": value(cons, route, "coverage"),
            "classifier_FID↓": value(cons, route, "classifier_fid"),
            "TSTR_F1↑": value(down, route, "tstr_macro_f1"),
            "control_F1": value(down, route, "control_macro_f1"),
            "TRTS_F1": value(down, route, "trts_macro_f1"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(paths.RESULTS, "metrics_v1_vs_v2.csv"), index=False)
    save_json(os.path.join(paths.LOGS, "make_v2_comparison.json"),
              {"v1": V1_ROUTES, "v2": V2_ROUTES, "n": len(df)})

    md = ["# v2 对比报告（自动生成）\n\n",
          "> v1 基线 = 100 epochs / base 32 / 无 EMA；v2 = 40 epochs / base 48 / EMA / nle 标签嵌入；"
          "`c_full_v2` 另加间期辅助损失（规律注入）。\n\n",
          "## 1. v1 vs v2 关键指标\n\n",
          "| route | family | MMD↓ | PSD↓ | QT W↓ | RR-CV W↓ | 条件一致率↑ | HR MAE↓ | 导联残差↓ | coverage↑ | classifier-FID↓ | TSTR F1↑ | control F1 |\n",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in df.iterrows():
        def f(x, fmt="{:.4f}"):
            return fmt.format(x) if np.isfinite(x) else "N/A"
        md.append(f"| {r['route']} | {r['family']} | {f(r['MMD↓'])} | {f(r['PSD↓'])} | "
                  f"{f(r['QT_W↓'], '{:.2f}')} | {f(r['RR_CV_W↓'], '{:.4f}')} | "
                  f"{f(r['cond_consistency↑'], '{:.3f}')} | {f(r['HR_MAE↓'], '{:.2f}')} | "
                  f"{f(r['lead_residual↓'], '{:.4f}')} | {f(r['coverage↑'], '{:.3f}')} | "
                  f"{f(r['classifier_FID↓'], '{:.1f}')} | {f(r['TSTR_F1↑'], '{:.3f}')} | "
                  f"{f(r['control_F1'], '{:.3f}')} |\n")

    md += ["\n## 2. v2 消融（2×2 对照）\n\n",
           "| 配置 | 标签嵌入 | 间期辅助损失 | 说明 |\n|---|---|---|---|\n",
           "| `c_full_v2` | nle | ✅ | 主方法（nle + 规律注入） |\n",
           "| `c_mask_v2` | mask(v1) | ✅ | 消融：去掉 nle 标签嵌入 |\n",
           "| `c_nle_v2` | nle | ❌ | 消融：去掉规律注入 |\n",
           "| `b_cond_v2` | nle | ❌ | 改进版路线 B（仅标签条件） |\n\n",
           "解读：把 `c_full_v2` 与 `c_mask_v2` 比 → nle 标签嵌入的贡献；"
           "与 `c_nle_v2` 比 → 间期辅助损失（规律注入）的贡献。\n\n",
           "## 3. 备注\n\n",
           "- v2 预算（40 epochs / base 48）小于文献级设置，目的是在截止前得到完整可评估的改进版；\n",
           "- TSTR 使用 500–2000 条合成样本从头训练小分类器，绝对值低于 `control`（真实数据同预算）是预期的，"
           "重点是 **v2 vs v1 的相对变化**；\n",
           "- `classifier-FID` 使用本项目分类器特征（768 维），不是外部医学基础模型特征，仅作同口径比较。\n"]
    out = os.path.join(paths.RESULTS, "summary_v2.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(md))
    LOG.info("saved %s", out)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_routes = [r for r in V2_ROUTES + ["c_full", "B_cond"] if r in set(df["route"])]
        sub = df[df["route"].isin(plot_routes)].set_index("route").reindex(plot_routes)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [("MMD↓", "MMD (lower better)"), ("PSD↓", "PSD (lower better)"),
                  ("cond_consistency↑", "conditional consistency"), ("HR_MAE↓", "HR MAE (bpm)")]
        for ax, (col, title) in zip(axes, panels):
            vals = sub[col].values.astype(float)
            colors = ["#dd8452" if r in V1_ROUTES else "#4c72b0" for r in plot_routes]
            ax.bar(range(len(plot_routes)), vals, color=colors)
            ax.set_xticks(range(len(plot_routes)))
            ax.set_xticklabels(plot_routes, rotation=45, ha="right")
            ax.set_title(title)
        fig.suptitle("v1 (orange) vs v2 (blue) key metrics")
        fig.tight_layout()
        fig.savefig(os.path.join(paths.FIGURES, "fig_v1_v2_comparison.png"), dpi=150)
        plt.close(fig)
        LOG.info("saved fig_v1_v2_comparison.png")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("figure failed: %s", exc)


if __name__ == "__main__":
    main()
