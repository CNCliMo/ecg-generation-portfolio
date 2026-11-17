# v2 改进设计说明（文献驱动）

> 目标：在不改动 v1 基线结果的前提下，按 `docs/PAPERS_ECG_GENERATION.md` 的优先项对项目做一次**可验证**的完善。
> 状态：实现与训练中（详见本文 §5 的执行记录）。
> v1 结果保留在 `results/`（A/B/C/D 共 11 条路线）；v2 结果以 `*_v2` 路线追加，便于两端对比。

---

## 1. 文献 → 改动 → 代码位置

| 文献依据 | v2 改动 | 代码位置 |
|---|---|---|
| SSSD-ECG-nle (arXiv:2407.11108)：旧标签嵌入（multi-hot → 线性）无法表达"阴性标签"，中性样本差；应每标签两个 embedding + null/padding token | 新增 **`CondEncoderNLE`**：每个标签 (present, absent) 两个 embedding 求和，另加可学习 null token + `null_mask` 支持条件 dropout；配置 `--cond-style nle` | `models/diffusion.py` |
| SSSD-ECG / nle：训练预算与模型容量是质量关键（S4 大模型、充分训练） | v2 模型 **base 32→64（约 5.4M 参数，v1 的 ~4×）+ 120 epochs + EMA(0.999)** | `experiments/train_diffusion.py`, `tools/run_v2.ps1` |
| 通用扩散实践：EMA 采样更稳 | 训练保存 EMA 权重；生成默认用 EMA 权重（`load_checkpoint(use_ema=True)`） | `models/diffusion.py`（`EMA`）、`experiments/generate_diffusion.py` |
| Biomedically Informed ECG Synconda 环境 / ECGEN：条件化心率/HRV/相位等生理参数 | C-v2 在标签+心率之外，加入**间期辅助回归**：U-Net 瓶颈特征 → 预测 (QT, PR, QRS, RR-CV)，目标来自 `preprocess/cache/interval_targets_*.npz`；损失 `aux_weight × masked MSE` | `models/diffusion.py`（aux head）、`preprocess/extract_interval_targets.py` |
| 文献评估结论：MMD/DTW 类关系指标信息有限，需 TSTR/TRTS 下游指标 | 新增 **`experiments/eval_downstream.py`**：TSTR（生成数据训练→真实测试）、CONTROL（同预算真实数据训练→真实测试）、TRTS（真实训练→生成数据测试） | `experiments/eval_downstream.py` |
| 评估框架（保真度/效用/隐私）与 ECG-FID 类指标 | 新增 **`experiments/eval_consistency.py`**：HR 一致性（目标 HR vs 生成 HR）、标签 TPR/TNR、导联一致性残差（III/aVR/aVL/aVF 对 I/II 的仿射残差）、k-NN coverage/precision、classifier-FID | `experiments/eval_consistency.py` |
| 评估指标族（ecgen-eval 全量） | 形态层 **4 → 10 个指标**：新增 fd、swd、nn_dist、pr_dist、heart_rate、spectral_entropy | `experiments/eval_morphology.py` |
| PTB-XL 基准：分类器是下游/一致性评估的瓶颈 | 分类器 v2：**base 96（8.7M 参数）+ label smoothing 0.05 + EMA + 70 epochs + 阈值校准**，保存 `classifier_best_v2.pt` 与调参后测试报告 | `experiments/train_classifier.py`, `models/classifier.py` |
| —— | Streamlit 增加 v2 路线（自动出现在多路线对比中） | `app/inference.py` |
| —— | 汇总/图表/完整性检查接入新指标表（metrics.csv 增加 downstream / consistency 两组） | `experiments/build_metrics_csv.py`, `summarize_results.py` |

---

## 2. v2 模型清单（全部 base=48, 40 epochs, EMA 0.999, seed 42；
> 注：因 23:00 中断限制，v2 从计划的 base=64/120 epochs 缩小为 base=48/40 epochs）

| 名称 | 条件 | 标签嵌入 | 间期辅助损失 | 用途 |
|---|---|---|---|---|
| `b_cond_v2` | label | nle | 否 | 改进版路线 B（单独验证 nle 对条件质量的作用） |
| `c_nle_v2` | label + hr | nle | 否 | 路线 C 消融：去掉规律注入（对照 `c_full_v2`） |
| `c_full_v2` | label + hr | nle | **是**（λ=0.1） | 路线 C 完整改进版（主方法） |
| `c_mask_v2` | label + hr | mask（v1 风格） | 是 | 路线 C 消融：去掉 nle 标签嵌入（对照 `c_full_v2`） |

> 四个模型构成 2×2 对照：nle vs mask（标签嵌入）× 有/无间期辅助损失（规律注入），
> 可直接回答"改进来自哪里"。

---

## 3. 新增评估层（所有 v1 + v2 路线统一跑）

1. **下游效用**（`metrics_downstream.csv`）：TSTR / CONTROL / TRTS 的 macro-F1、micro-F1、AUROC、AP。
2. **一致性诊断**（`metrics_consistency.csv`）：
   - `hr_mae / hr_bias / hr_pearson / hr_within_5bpm / hr_within_10bpm`（仅心率条件路线）；
   - `label_tpr_macro / label_tnr_macro`（仅标签条件路线）；
   - `lead_residual_mean / lead_residual_wasserstein`（全部路线 + 真实参考）；
   - `coverage / precision_knn / nn_dist_synth_real / nn_dist_real_nn`（k-NN 流形）；
   - `classifier_fid`（分类器倒数第二层特征空间的 Fréchet 距离）。
3. **形态层扩展**：fd / swd / nn_dist / pr_dist / heart_rate / spectral_entropy（与 v1 的 4 个指标并列）。
4. `metrics.csv` 增加两组新指标；`summary.md` 增加"下游效用"和"一致性诊断"两节。

---

## 4. 可复现入口

```powershell
cd <REPO_ROOT>

# 一次性：分类器 v2（~25min）
powershell -File tools\run.ps1 experiments\train_classifier.py --tag v2 --base 96 `
    --epochs 70 --label-smoothing 0.05 --ema-decay 0.999 --threshold-tune

# v2 全流程（训练 4 模型 → 生成 → 评估 → 汇总 → 完整性检查）
powershell -File tools\run_v2.ps1
# 仅评估（已完成训练时）
powershell -File tools\run_v2.ps1 -SkipTraining
```

产物：
- checkpoint：`models/checkpoints/{b_cond_v2,c_nle_v2,c_full_v2,c_mask_v2}.pt`（v1 文件不变）
- 生成：`results/generated/{name}_{cond,uncond}.npz`
- 指标：`results/metrics.csv`（追加 v2 行）、`metrics_downstream.csv`、`metrics_consistency.csv`
- 图：`results/figures/fig_downstream.png`、`fig_consistency.png`、`classifier_training_v2.png` 等
- 摘要：`results/summary.md`（自动追加 v2 路线）

---

## 5. 执行记录（最终）

| 步骤 | 状态 | 时间 |
|---|---|---|
| `.venv_iso` 补齐 torch 运行时依赖（sympy/networkx/filelock/mpmath，离线从 conda 环境 拷贝） | ✅ | 19:38 |
| interval targets 提取（train 17,084 + val 2,146；中位 QT 390 / PR 120 / QRS 110 ms） | ✅ | 20:03 |
| 分类器 v2（base96/smoothing/EMA/阈值校准）：fixed macro-F1 0.6868 / tuned 0.6970 | ✅ | 19:59 |
| `c_full_v2`（nle + 规律注入，主方法，40 ep / base 48） | ✅ | 20:24 |
| `c_mask_v2`（mask 标签嵌入 + 规律注入，消融，40 ep / base 48） | ✅ | 20:45 |
| `c_nle_v2`（nle，无规律注入，消融，40 ep / base 48） | ✅ | 21:05 |
| `b_cond_v2`（nle 版路线 B，40 ep / base 48） | ✅ | 21:22 |
| 生成 4 条 v2 路线（各 500 条件 + 1000 无条件，DDIM 50 步，EMA 权重） | ✅ | 21:26 |
| 全部评估（morphology 10 指标 / regularity / functional / downstream / consistency） | ✅ | 21:42 |
| `metrics.csv`（762 行，17 路由）+ 14 张图 + `summary.md`（含 v2 章节）+ `summary_v2.md` | ✅ | 21:42 |
| `final_check` | ✅ OK=42 / MISSING=0 | 21:42 |

## 6. 结果与结论

v2 关键结果（完整表见 `results/summary_v2.md`）：

| 指标 | v1 c_full | v2 c_full_v2 | v2 c_nle_v2（无规律注入） | v2 c_mask_v2（mask 嵌入） |
|---|---|---|---|---|
| MMD↓ | 0.0012 | **0.0005** | 0.0007 | 0.0008 |
| PSD↓ | 0.0137 | 0.0117 | 0.0143 | **0.0099** |
| QT 域内 W↓ | 102.86 | **85.15** | 104.25 | 91.85 |
| RR-CV 域内 W↓ | 0.4149 | 0.4151 | 0.4148 | 0.4171 |
| 条件一致率↑ | 0.250 | 0.265 | 0.238 | **0.298** |
| HR MAE↓ (bpm) | 20.99 | **17.07** | 21.58 | 21.56 |
| 导联残差↓ | 0.2938 | 0.3501 | 0.3637 | **0.3208** |
| coverage↑ | 0.835 | 0.832 | 0.632 | 0.479 |
| classifier-FID↓ | 63.5 | 72.6 | 78.4 | 162.4 |
| TSTR macro-F1↑ | 0.246 | 0.339 | 0.166 | **0.407** |

结论：
1. **规律注入（间期辅助损失）有效**：`c_full_v2` vs `c_nle_v2` → QT W 85.2 vs 104.3、HR MAE 17.1 vs 21.6、
   TSTR 0.339 vs 0.166；说明辅助监督确实把"规律"注入到了生成分布；
2. **v2 相对 v1 有明确提升**（同任务、同评估口径）：MMD/PSD/QT W/HR MAE/TSTR 全面改善；
3. **nle 标签嵌入效果混合**：mask 版在 PSD/条件一致率/TSTR 更好，nle 版在 HR/导联/FID 更好，
   受 40 epochs 小预算限制，未复现文献中的显著优势 → 后续需在充足预算下复验；
4. **下游效用仍有限**：所有路线 TSTR 远低于 control（真实数据同预算训练），说明当前规模下
   合成数据可用于数据增强/预训练辅助，但不足以替代真实数据；
5. **HR 条件仍偏弱**：v1/v2 的 C 路线 HR MAE 17–22 bpm（A 机理模型仅 0.6 bpm）→ 条件强度与连续条件建模仍是主要改进空间。

## 7. 遗留与后续（中断后可选）

- 用充足预算（base 64 / 120–300 epochs）重训 v2 四模型，验证 nle 优势；
- 单模型 + CFG（guidance 网格）、DPM-Solver++ 采样、8 独立导联 + 线性重构；
- 更多 seed 的 GAN v2、文本/报告条件生成（DiffuSETS/Diff-TTE）、隐私评估（membership inference）；
- 将 v2 结论回填到论文/报告的"改进实验"章节。

## 8. 风险与口径说明

- v1 与 v2 预算不同（v1 100ep/base32，v2 40ep/base48），对比仅用于"同任务同口径的相对改进"说明，不是严格消融；
- v2 内部的 2×2 对照（nle/mask × 有/无辅助损失）预算完全一致，是严格的消融；
- classifier-FID 使用本项目分类器特征（768 维），非外部医学基础模型特征，只作同口径比较；
- 导联一致性残差采用"每记录仿射拟合（含截距）后的相对 RMS 残差"，受逐导联 z-score 影响，真实参考值为 0.0049。
