"""Build the human-readable results summary `results/summary.md` from the metric tables.

This is what goes into the README result section / final report. It never invents
numbers: every cell is read from the CSV outputs of the evaluation scripts, and
missing/N-A values are shown explicitly.

Usage: python experiments/summarize_results.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg_common import paths
from ecg_common.config import load_config
from ecg_common.logging_utils import get_logger

LOG = get_logger("summarize_results")
ROUTES_ORDER = ["A", "B_uncond", "B_cond", "c_full", "c_no_label", "c_no_hr",
                "c_label_noise", "c_label_shuffle", "d_seed42", "d_seed43", "d_seed44",
                "b_cond_v2", "c_full_v2", "c_nle_v2", "c_mask_v2", "c_aux05_v2"]


def _read(name: str) -> pd.DataFrame | None:
    p = os.path.join(paths.RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _route_key(r: str):
    return ROUTES_ORDER.index(r) if r in ROUTES_ORDER else 99


def _routes(*dfs) -> list[str]:
    found = set()
    for df in dfs:
        if df is None:
            continue
        col = "route"
        if col in df:
            found.update(str(x) for x in df[col].unique())
    found.discard("REAL_PTBXL_TEST")
    found.discard("overall")
    found = {r for r in found if not r.startswith("class:")}
    return sorted(found, key=_route_key)


def main() -> None:
    cfg = load_config()
    mor = _read("metrics_morphology.csv")
    reg = _read("metrics_regularity.csv")
    fun = _read("metrics_functional.csv")
    cls = _read("classifier_test_report.csv")
    clo = _read("class_distribution.csv")
    down = _read("metrics_downstream.csv")
    cons = _read("metrics_consistency.csv")
    routes = _routes(mor, reg, fun, down, cons)
    lines = []
    A = lines.append

    A("# 结果摘要（自动生成）\n")
    A(f"\n- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"\n- seed: {cfg['seed']}; 采样率 {cfg['fs']} Hz; 12 导联 1000 点; Lead II = 索引 {cfg['lead_ii_index']}")
    A(f"\n- 生成数量要求: 条件 {cfg['generation']['n_conditional']} / 无条件 {cfg['generation']['n_unconditional']}; "
      f"扩散采样 DDIM {cfg['generation']['ddim_steps']} 步")
    A(f"\n- 本次结果包含路线: {', '.join(routes) if routes else '（无）'}\n")

    # ------------------------------------------------------------ classifier
    A("\n## 1. 诊断分类器（PTB-XL fold 10 测试集）\n\n")
    if cls is not None:
        overall = cls[cls["scope"] == "overall"].set_index("metric")["value"]
        A("| metric | value |\n|---|---|\n")
        for k in ("macro_f1", "micro_f1", "macro_auroc", "macro_ap", "exact_match"):
            if k in overall:
                A(f"| {k} | {float(overall[k]):.4f} |\n")
        A("\n| class | precision | recall | F1 | AUROC | support |\n|---|---|---|---|---|---|\n")
        for _, row in cls[cls["scope"].str.startswith("class:")].iterrows():
            pass
        per = cls[cls["scope"].str.startswith("class:")]
        if not per.empty:
            piv = per.pivot_table(index="scope", columns="metric", values="value")
            for scope, row in piv.iterrows():
                A(f"| {scope.replace('class:', '')} | {row.get('precision', float('nan')):.4f} | "
                  f"{row.get('recall', float('nan')):.4f} | {row.get('f1', float('nan')):.4f} | "
                  f"{row.get('auroc', float('nan')):.4f} | {int(row.get('support', 0))} |\n")
    else:
        A("（classifier_test_report.csv 缺失）\n")

    # ------------------------------------------------------------ morphology
    A("\n## 2. 形态层（ecgen-eval，越低越好，参考=PTB-XL 测试集）\n")
    if mor is not None:
        piv = mor.pivot_table(index="route", columns="metric", values="value")
        piv = piv.reindex([r for r in routes if r in piv.index])
        A("\n| route | MMD | DTW | PSD | PRD | n_synth |\n|---|---|---|---|---|---|\n")
        for r, row in piv.iterrows():
            ns = mor[(mor["route"] == r)]["n_synth"].iloc[0]
            A(f"| {r} | {row.get('mmd', float('nan')):.5f} | {row.get('dtw', float('nan')):.3f} | "
              f"{row.get('psd', float('nan')):.5f} | {row.get('prd', float('nan')):.2f} | {int(ns)} |\n")
    else:
        A("（metrics_morphology.csv 缺失）\n")

    # ------------------------------------------------------------ regularity
    A("\n## 3. 规律层（Wasserstein；ΔW = 跨域 − 域内，越低越稳）\n")
    if reg is not None:
        for param in ("QT_ms", "PR_ms", "QRS_ms", "rr_cv"):
            sub = reg[reg["param"] == param]
            if sub.empty:
                continue
            A(f"\n### {param}\n\n| route | 域内 W | 跨域 W | ΔW | 域内 KS(p) | 跨域 KS(p) | n_route/有效 |\n|---|---|---|---|---|---|---|\n")
            for r in [x for x in routes if x in set(sub["route"])]:
                g = sub[sub["route"] == r]
                try:
                    win = g[g["domain"] == "in_domain"].iloc[0]
                    wcr = g[g["domain"] == "cross_domain"].iloc[0]
                    wd = g[g["domain"] == "delta"].iloc[0]
                except IndexError:
                    continue
                A(f"| {r} | {win['value']:.3f} | {wcr['value']:.3f} | {wd['value']:.3f} | "
                  f"{win['ks_stat']:.3f}({win['ks_p']:.2g}) | {wcr['ks_stat']:.3f}({wcr['ks_p']:.2g}) | "
                  f"{int(win['n_route'])}/{int(win['n_ref'])} |\n")
        # failure summary
        fail = reg[["route", "n_failed_delineation"]].drop_duplicates()
        if not fail.empty:
            A("\n**delineate 失败数（已记录，不静默丢弃）**：" +
              ", ".join(f"{r}={int(n)}" for r, n in zip(fail["route"], fail["n_failed_delineation"])) + "\n")
    else:
        A("（metrics_regularity.csv 缺失）\n")

    # ------------------------------------------------------------ functional
    A("\n## 4. 功能层（分类器评估生成信号）\n\n")
    if fun is not None:
        A("| route | 条件一致率(macro) | 无条件 5 类分布 KL |\n|---|---|---|\n")
        for r in routes:
            g = fun[fun["route"] == r]
            if g.empty:
                continue
            cons = g[g["metric"] == "conditional_consistency_macro"]
            kl = g[g["metric"] == "unconditional_class_KL"]
            cons_v = f"{cons['value'].iloc[0]:.3f}" if not cons.empty and np.isfinite(cons["value"].iloc[0]) else "N/A"
            kl_v = f"{kl['value'].iloc[0]:.3f}" if not kl.empty and np.isfinite(kl["value"].iloc[0]) else "N/A"
            A(f"| {r} | {cons_v} | {kl_v} |\n")
        na = fun[fun["note"].astype(str).str.contains("not_applicable", na=False)]
        if not na.empty:
            A("\n*N/A 原因*：" + "；".join(f"{r}: 无标签条件" for r in na["route"].unique()) + "\n")
    else:
        A("（metrics_functional.csv 缺失）\n")

    # ------------------------------------------------------------ ablation
    abl = [r for r in routes if r.startswith(("c_", "C_")) and not r.endswith("_v2")]
    if abl:
        A("\n## 7. 路线 C 消融（5 配置对比）\n")
        A("\n| 配置 | 说明 |\n|---|---|\n")
        A("| C_full | 标签 + 心率条件 |\n| C_no_label | 仅心率 |\n| C_no_hr | 仅标签 |\n"
          "| C_label_noise | 标签 10% 随机翻转 |\n| C_label_shuffle | 随机标签替换 |\n")
        if reg is not None:
            piv = reg[(reg["metric"] == "wasserstein") & (reg["domain"] == "in_domain")]
            pv = piv.pivot_table(index="route", columns="param", values="value")
            pv = pv.reindex([r for r in abl if r in pv.index])
            A("\n| 配置 | QT W | PR W | QRS W | RR-CV W |\n|---|---|---|---|---|\n")
            for r, row in pv.iterrows():
                A(f"| {r} | {row.get('QT_ms', float('nan')):.2f} | {row.get('PR_ms', float('nan')):.2f} | "
                  f"{row.get('QRS_ms', float('nan')):.2f} | {row.get('rr_cv', float('nan')):.4f} |\n")

    # ---------------------------------------------------------- downstream (v2)
    if down is not None:
        A("\n## 5. 下游效用（TSTR / TRTS，v2 新增）\n\n")
        A("> TSTR = 训练于生成数据、测试于真实测试集；control = 同等预算训练于真实数据；"
          "TRTS = 训练于真实数据、测试于生成数据。\n\n")
        A("| route | TSTR macro-F1 | control macro-F1 | TRTS macro-F1 |\n|---|---|---|---|\n")
        for r in routes:
            g = down[down["route"] == r]
            if g.empty:
                continue
            def _v(m, g=g):
                x = g[g["metric"] == m]
                return (f"{x['value'].iloc[0]:.4f}"
                        if not x.empty and np.isfinite(x["value"].iloc[0]) else "N/A")
            A(f"| {r} | {_v('tstr_macro_f1')} | {_v('control_macro_f1')} | {_v('trts_macro_f1')} |\n")
        A("\n*N/A 原因*：该路线不是诊断标签条件生成，无法定义 TSTR/TRTS 标签。\n")

    # --------------------------------------------------------- consistency (v2)
    if cons is not None:
        A("\n## 6. 一致性 / 保真度诊断（v2 新增）\n\n")
        cols = [("classifier_fid", "classifier-FID↓"), ("lead_residual_mean", "导联残差↓"),
                ("coverage", "coverage↑"), ("precision_knn", "precision↑"),
                ("hr_mae", "HR MAE↓")]
        A("| route | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n")
        for r in routes:
            g = cons[cons["route"] == r]
            if g.empty:
                continue
            vals = []
            for m, _ in cols:
                x = g[g["metric"] == m]
                vals.append(f"{x['value'].iloc[0]:.3f}"
                            if not x.empty and np.isfinite(x["value"].iloc[0]) else "N/A")
            A(f"| {r} | " + " | ".join(vals) + " |\n")
        real = cons[(cons["route"] == "REAL_PTBXL_TEST") & (cons["metric"] == "lead_residual_mean")]
        if not real.empty:
            A(f"\n*真实测试集导联残差参考值*：{real['value'].iloc[0]:.4f}（越低越接近真实生理约束）\n")

    A("\n---\n\n> 说明：所有指标由 `experiments/` 下脚本从 `results/*.csv` 直接汇总，未做任何人工美化；"
      "失败样本数与原因见 `results/logs/failures_*.csv`，与 README §7 偏差表。\n")

    out = os.path.join(paths.RESULTS, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    LOG.info("saved %s", out)


if __name__ == "__main__":
    main()
