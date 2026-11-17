# 实验清单（Experiments）

> 记录已完成的实验、协议与结论，以及下一步实验（作品集 W2 的 E2/E6/B）。
> 指标总表：`results/metrics.csv`（762 行、17 条路由）；摘要：`results/summary.md` / `summary_v2.md`。

---

## 0. 实验设计原则

1. **同数据划分**：train = fold 1–8，val = fold 9，test = fold 10（test 只用于最终评估）；
2. **同评估口径**：所有路线用同一套 `experiments/eval_*.py` 与同一真实参考；
3. **同预算对照**：同一组消融使用完全相同的 epochs/batch/base/seed；
4. **多 seed**：GAN 3 seed（42/43/44）；v2 主方法后续补 3 seed；
5. **失败必录**：`results/logs/failures_*.csv` + metrics 中的失败计数列。

---

## 1. 已完成实验（v1 基线，2026-11-06）

| # | 实验 | 配置 | 预算 | 关键结果 |
|---|---|---|---|---|
| E0 | 诊断分类器 v1 | 1D ResNet base64 | 40 ep / bs64 | macro-F1 **0.6838** / AUROC 0.8975 |
| E1 | 路线 A 机理模型 | McSharry ODE | 无需训练 | MMD 0.00262 / QT W 14.9 / ΔW(QT) +22.1 |
| E2 | 路线 B 扩散 | base32, 100 ep | 无/标签条件 | MMD 0.00055 / 0.00058 |
| E3 | 路线 C 消融 ×5 | base32, 100 ep | c_full / no_label / no_hr / label_noise / label_shuffle | c_no_label MMD 0.00013（最低）；条件一致率 0.211–0.223 |
| E4 | 路线 D GAN ×3 seed | base32, 100 ep | seed 42/43/44 | MMD 0.0168±0.0221（方差大） |
| E5 | 三层评估 + 汇总 | 全部路线 | — | metrics.csv 259 行；完整性检查通过 |

## 2. 已完成实验（v2 改进，2026-11-06 晚）

| # | 实验 | 配置 | 预算 | 关键结果 |
|---|---|---|---|---|
| E6 | 分类器 v2 | base96 + smoothing + EMA + 阈值校准 | 70 ep | fixed 0.6868 / tuned 0.6970 |
| E7 | 间期目标提取 | NeuroKit2 delineate | train 17,084 + val 2,146 | 中位 QT/PR/QRS = 390/120/110 ms |
| E8 | nle 标签嵌入 + 规律注入 | `c_full_v2` base48 | 40 ep + EMA | MMD 0.0005、QT W 85.2、HR MAE 17.1、TSTR 0.339 |
| E9 | 消融：mask 嵌入（`c_mask_v2`） | 同 E8 但 mask | 40 ep | PSD 0.0099、TSTR 0.407、classifier-FID 162（混合结果） |
| E10 | 消融：无辅助损失（`c_nle_v2`） | 同 E8 但无 aux | 40 ep | QT W 104.3、HR MAE 21.6、TSTR 0.166 → **规律注入有效** |
| E11 | nle 版路线 B（`b_cond_v2`） | label only | 40 ep | MMD 0.0005，TSTR 0.133（未提升） |
| E12 | 下游效用 TSTR/TRTS | 全部标签条件路线 | — | TSTR 0.13–0.41 vs control 0.49–0.53（合成数据效用有限） |
| E13 | 一致性诊断 | 全部路线 | — | HR MAE 17–22 bpm（A 机理 0.57）、导联残差 0.35 vs 真实 0.005、coverage 0.48–0.86 |
| E14 | 形态层扩展 | 全部路线 | — | 4 → 10 指标（fd/swd/nn_dist/pr_dist/heart_rate/spectral_entropy） |

## 3. W2 实验（已完成，2026-11-07）

| # | 实验 | 配置 | 关键结果 |
|---|---|---|---|
| E15 | **规律注入 λ 扫描** | λ ∈ {0, 0.1, 0.5}，其余完全相同（nle + label+HR，base48/40ep，EMA） | QT W **104.3 → 85.2/85.7**；PSD 0.0143→0.0117/0.0101；TSTR 0.166→0.339/0.377 → **规律注入有效，λ=0.1 最均衡** |
| E16 | **TSTR 放大** | 合成样本 {500, 2000, 8000}，3 条路线，3 seeds；对照真实数据同量 | 500→2000 提升（+0.05~0.10），**2000→8000 停滞**；control 持续上升（0.53→0.60→0.65）→ **合成数据"数量饱和"，质量是瓶颈** |
| E17 | **指标-效用相关性** | 16 条路线 × 14 生成指标 × 8 效用指标，Spearman | **PSD/QT W 与 TSTR/条件一致率 |ρ|≈0.72–0.92（p<0.05）**；classifier-FID/coverage 与 TSTR 弱相关；形态指标高度冗余 |

> 详细报告：`results/w2_report.md`；图表：`fig_w2_lambda_scan.png`、`fig_tstr_scaling.png`、`fig_metric_utility_heatmap.png`、`fig_metric_redundancy.png`。
> 复现：`tools/run_w2.ps1 -SkipTraining`。

## 4. 实验记录规范

- 每次实验记录：日期、配置（name/cond-mode/style/base/epochs/seed）、指标表、结论（含负面）；
- 训练日志：`results/logs/<name>_train.csv`；生成：`results/generated/<name>_{cond,uncond}.npz`；
- 失败：`results/logs/failures_*.csv`（原因/样本索引）；
- 评估输出：`results/metrics_{morphology,regularity,functional,downstream,consistency}.csv`，
  汇总进 `metrics.csv`（含完整性检查）。

## 5. 已知失败与偏差（节选）

| 项 | 情况 | 处理 |
|---|---|---|
| 心率检测 v1 | 把 T 波误判为 R 波（HR 虚高 ~108 bpm） | 改 Pan-Tompkins 式，重算 73.9 bpm，失败 262 条记录 |
| NeuroKit2 delineate | 信号过短报错（100Hz 下 10s 可，心拍窗口不可） | 整段 10s 提取，每段取 2 拍；失败率记录 |
| PTB-XL 50Hz 陷波 | 100Hz 下 Nyquist 不可定义 | 跳过并记录偏差（README §6） |
| S4 骨干 | 未实现（1D U-Net 替代） | README 偏差表第 2 条 |
| 路线 D | Pulse2Pulse 未逐层复刻 | 定位为 GAN 基线，README 偏差表第 3 条 |
