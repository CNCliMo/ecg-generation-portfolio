# 心电生成（ECG Generation）文献综述与项目改进建议

> 检索日期：2026-11-06；检索源：arXiv / ScienceDirect / IEEE Xplore / Nature / PubMed Central / Google Scholar / ACM DL。
> 用途：为本项目（PTB-XL 四路线心电生成 + 诊断 + 三层评估）提供**文献依据**与**可落地的改进路线**。
> 说明：外部检索结果仅为参考；下方"对本项目的启发"是我们结合本项目实测结果（`results/summary.md`）的分析。

---

## 一、论文清单（按主题分组，共 20+ 篇）

### A. 扩散模型（Diffusion）方向

| # | 论文 | 年/出处 | 链接 | 核心内容（与我们的关系） |
|---|---|---|---|---|
| A1 | **SSSD-ECG**: Diffusion-based Conditional ECG Generation with Structured State Space Models — Lopez Alcaraz & Strodthoff | 2023, Computers in Biology and Medicine 163:107115 | [arXiv:2301.08227](https://arxiv.org/abs/2301.08227) · [DOI](https://doi.org/10.1016/j.compbiomed.2023.107115) · [代码](https://github.com/AI4HealthUOL/SSSD-ECG) | ⭐ 我们路线 B/C 的"标准答案"：S4 状态空间 + 条件扩散生成 10s 12 导联，条件 >70 个 ECG statement；提出用分类器评估 + 临床 Turing test；**明确对比 GAN 基线**。我们的实现只复刻了"扩散 + 标签条件"的思想，骨干用 1D U-Net 近似 |
| A2 | **SSSD-ECG-nle**: New Label Embeddings with Structured State-Space Models for ECG Generation — Skorik & Avetisyan | 2024, IEEE (arXiv:2407.11108) | [arXiv](https://arxiv.org/abs/2407.11108) · [IEEE](https://ieeexplore.ieee.org/document/10659391) · [代码 EcgLib](https://github.com/ispras/EcgLib) | ⭐⭐ 指出 SSSD-ECG **标签嵌入机制的缺陷**：`y×E` 让"阴性标签"变成 0 向量，模型学不会"没有该病"的中性样本；改为**每个标签 2 个 embedding（有/无）+ padding index 支持 CFG**。用 **TSTR/TRTS** 下游指标评估（并称 MMD/DTW 类关系指标在其实验中"不 informative"）。**我们的功能层结果（B/C 生成的 NORM 预测比例≈0.001，MI 0.6）正是该缺陷的复现** |
| A3 | **ECGTwin**: Personalized ECG Generation Using Controllable Diffusion Model | 2025, arXiv | [arXiv:2508.02720](https://arxiv.org/abs/2508.02720) | 用自适应归一化（AdaGN）注入患者基础心拍向量 + 时间嵌入，做**个体化可控生成**；启发：把"规律/患者特征"作为条件注入（我们路线 C 可借鉴） |
| A4 | **DiffuSETS**: 12-lead ECG Generation Conditioned on Clinical Text Reports and Patient-Specific Information | 2025, arXiv | [arXiv:2501.05932](https://arxiv.org/abs/2501.05932) | 文本报告 + 患者信息条件生成，用 **FID** 评估信号分布相似度；启发：更临床的"报告→心电图"条件，可成为路线 C 的升级方向 |
| A5 | **Diff-TTE**: A Novel Diffusion Model for Text-to-Electrocardiogram | 2025, IEEE | [IEEE](https://ieeexplore.ieee.org/document/11331862) | 文本→ECG 扩散生成；同上，文本条件化方向 |
| A6 | **Biomedically Informed ECG Synconda 环境: Customizing Cardiac Cycle Phases with Diffusion Model** | 2025, IEEE | [IEEE](https://xplorestaging.ieee.org/document/10821878) | ⭐ 按**心动周期相位**（P/QRS/T 时序）条件化生成；直接对应我们的规律层指标（QT/PR/QRS）——把"规律注入"做成"相位/间期条件"而不是只靠心率 |
| A7 | **ECGEN**: Exploring latent diffusion models for ECG generation on the minute scale | 2025, ScienceDirect (Artificial Intelligence in Medicine?) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0169260725005541) | 潜空间扩散 + 分钟级长程生成，条件含 **HR 与 SdNN（心率变异性特征）**；启发：RR 变异是可条件的生理参数，我们目前只用平均 HR |
| A8 | Electrocardiogram Synconda 环境 Using Denoising Diffusion Probabilistic Models and Bidirectional State-Space Models | 2024/25, UW-Madison eScholarship | [链接](https://search.library.wisc.edu/article/cdi_cdl_escholarship_oai_escholarship_org_ark_13030_qt4gb3d31q/staff) | DDPM + 双向 SSM；与 SSSD-ECG 同一思路的变体，可作复现对照 |

### B. GAN 方向（含 Pulse2Pulse 基线）

| # | 论文 | 年/出处 | 链接 | 核心内容 |
|---|---|---|---|---|
| B1 | **Synconda 环境 of Realistic ECG using Generative Adversarial Networks**（ECGGAN 系） | 2019, arXiv | [arXiv:1909.09150](https://arxiv.org/abs/1909.09150) | 早期 ECG GAN 代表工作；按心搏分段生成、判别器结构设计 |
| B2 | **DeepFake electrocardiograms using generative adversarial networks are the beginning of the end for privacy issues in medicine** | 2021, Scientific Reports 11:21896 | [Nature](https://www.nature.com/articles/s41598-021-01295-2) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8578227/) | ⭐ 提出/推广了本文评估用的 **MMD、DTW、PSD、PRD** 指标族（也是 `ecgen-eval` 指标来源之一）；指出 12 导联只有 8 个线性独立导联 |
| B3 | **Pulse2Pulse**（公开资料中常见的 GAN 基线） | — | 公开检索未找到稳定原文出处 | 我们按公开描述的"1D 生成器 + PatchGAN"风格实现（`models/gan.py`）；若需要严格复刻，需要先拿到原文/代码，**这是当前项目的一个文献风险点** |
| B4 | **HQ-DCGAN**: Hybrid Quantum Deep Convolutional GAN for ECG generation | 2024, Knowledge-Based Systems | [X-MOL](https://www.x-mol.com/paper/1821781461287632896/t) | GAN + 量子电路混合；说明 GAN 方向仍在演化（创新点参考） |
| B5 | Synthetic ECG Generation for Data Augmentation and Transfer Learning in Arrhythmia Classification | 2024, arXiv | [arXiv:2411.18456](https://arxiv.org/abs/2411.18456) | 生成数据用于下游数据增强/迁移学习；直接支撑"TSTR + 数据增强"评估设计 |

### C. 评估方法与综述（本项目评估层的改进依据）

| # | 论文 | 年/出处 | 链接 | 核心内容 |
|---|---|---|---|---|
| C1 | **Advancing electrocardiogram synconda 环境: Analyzing key metrics for enhanced evaluation** | 2025, Computers in Biology and Medicine 196 | [DOI](https://doi.org/10.1016/j.compbiomed.2025.110879) · [ACM](https://dl.acm.org/doi/abs/10.1016/j.compbiomed.2025.110879) | ⭐ 系统梳理 ECG 合成评估指标（保真度/分布/下游效用），指出单一指标不足；我们已用 `ecgen-eval` 的 4 指标，可据此补充 FID/下游指标 |
| C2 | **Deep generative models for physiological signals: A systematic literature review** | 2025, Artificial Intelligence in Medicine 165 | [DOI](https://doi.org/10.1016/j.artmed.2025.103127) | 生理信号生成模型的系统综述（方法、数据、评估、隐私） |
| C3 | **A Comprehensive Evaluation Framework for Synthetic ECG: Assessing Fidelity, Utility, and Privacy** | 2024/25, UWaterloo | [PDF](https://uwspace.uwaterloo.ca/bitstreams/2c35ab52-959e-48f6-81a9-6e772d18fbbc/download) | ⭐ 提出"**保真度 + 效用 + 隐私**"三维评估框架；我们目前只覆盖前两项的代理指标，隐私评估缺失（可加分项） |
| C4 | Introducing a Comprehensive Framework for Synthetic ECG Validation: Application to Brugada Syndrome | 2024, Zenodo | [Zenodo](https://zenodo.org/records/11934905) | 合成 ECG 验证框架（含临床一致性） |
| C5 | `ecgen-eval` 工具库（MMD/DTW/PSD/PRD + ECG-FM/ECGFounder/HuBERT-ECG 编码器） | 2024-25, GitHub/PyPI | [GitHub: vlbthambawita/ECGEN-Eval](https://github.com/vlbthambawita/ECGEN-Eval) | 我们已用其 4 个基础指标；库内还有**预训练 ECG 编码器**可算 FID 类指标（尚未启用） |

### D. 数据集 / 下游任务

| # | 论文 | 年/出处 | 链接 |
|---|---|---|---|
| D1 | **PTB-XL**: a large-scale publicly available electrocardiography dataset — Wagner et al. | 2020, Scientific Data 7:154 | [Nature Sci Data](https://www.nature.com/articles/s41597-020-0495-6) |
| D2 | **Deep learning for ECG analysis: Benchmarks and insights from PTB-XL** — Strodthoff et al. | 2021, IEEE JBHI | [arXiv:2004.13701](https://arxiv.org/abs/2004.13701) |

---

## 二、文献结论 vs 我们项目的实测差距

| 文献结论 | 我们项目现状（`results/summary.md`） | 差距/可改进点 |
|---|---|---|
| SSSD-ECG 用 S4 状态空间 + 大模型充分训练（数小时级），才能同时保证形态与节律 | 我们 base=32、1.36M 参数、100 epochs（约 25–50min/模型） | 模型容量与训练预算明显不足：B/C 的 QT Wasserstein ≈ **94ms**（A 仅 15ms）、RR-CV ≈ **0.41**（A 0.04）——长程节律没学好 |
| SSSD-ECG-nle：标签嵌入必须表达"阳性/阴性"两种状态，否则中性样本差 | 我们 B_cond/C 无条件生成的 NORM 预测比例 ≈ **0**，MI ≈ 0.6 | ✅ **文献现象在我们项目完整复现**；这是最高优先级的修复项 |
| 关系指标（MMD/DTW/PSD/PRD）信息量有限，应以下游 TSTR/TRTS + 临床评估为主 | 我们目前三层评估全部是关系/分布指标（无 TSTR/TRTS） | 需要补：train-on-synthetic-test-on-real 等下游指标 |
| 12 导联只有 8 个线性独立导联（III/aVR/aVL/aVF 是 I/II 的线性组合） | 我们直接生成并独立评估全部 12 导联 | 可改为"8 独立导联 + 线性重构 4 导联"，提升物理一致性、减少容量浪费 |
| 条件机制应支持无条件和引导采样（CFG/padding index） | 我们训练了 B_uncond 与 B_cond 两个独立模型 | 可合并为单模型 + CFG，省一次训练并获得 guidance 可调 |
| GAN 训练不稳定，需多 seed 报告方差 | 我们 D 路线 3 seed：MMD 0.042 / 0.0025 / 0.0057（方差大但已报告）✅ | 已符合文献建议；可在最终报告强调方差 |
| 心率/HRV 可作为可控条件（ECGEN 用 HR + SdNN） | 路线 C 只条件化"平均心率"，未条件化 RR 变异 | 可增加 RR 变异 / 目标 QT 等条件 → 更贴近"规律注入"本意 |
| 相位条件/生理约束可改善间期可控性（Biomedically informed） | 路线 C 的"规律注入"目前本质是 B_cond + HR 条件 | 需要独立的**规律监督信号**（见下方 P0-6），否则 C 与 B 的区别会被质疑 |

---

## 三、针对本项目的改进建议（按优先级）

### P0：投入小、收益大（1–3 天，可直接在现有代码上改）

1. **补齐下游指标 TSTR / TRTS**（最高优先级）
   - TSTR：只用生成数据训练分类器 → 在 PTB-XL 真测试集上评估；TRTS：真数据训练 → 在生成数据上评估；
   - 报告 G-mean / F1 / ROC-AUC（对齐 SSSD-ECG-nle 做法）；
   - 落地：新增 `experiments/eval_downstream.py`，把结果并入 `metrics.csv`。
   - 好处：直接回应"MMD 低≠有用"的质疑，是当前评估体系最明显的缺口。

2. **修复条件机制（SSSD-ECG-nle 式标签嵌入）**
   - `models/diffusion.py` 的 `CondEncoder`：每个标签 → 2 个 embedding（present / absent），再加一个"无病/中性"embedding + padding index；
   - 预期改善：NORM/中性样本生成、条件一致率（当前仅 **0.223**）；
   - 重训 c_full（其余消融可按需重跑）——这是本项目主方法，值得优先重训。

3. **提升分类器质量**（条件一致率与功能层都依赖它）
   - 当前 test macro-F1 **0.6838**；可加：更长训练 + label smoothing、类别加权/焦点损失、EMA、阈值校准（报告 ECE）；
   - 目标准确率提升后，功能层指标的解释力显著增强。

4. **补"条件—生成一致性"指标**
   - 目标 HR vs 生成信号测得 HR 的 MAE/相关系数（目前只比较了 RR-CV 分布）；
   - 目标标签 vs 分类器输出的 per-class 灵敏度（当前只有 macro 一致率）；
   - 归入 `eval_functional.py`。

5. **扩训练预算 / 模型容量（但要统一口径重跑）**
   - 方案：base 32→64（约 5.5M 参数）或直接上 SSSD-ECG 官方 S4 实现；epochs 100→300+；EMA 采样；
   - 建议先对 **c_full + B_cond** 做"加强版"，与现有基线对比后再决定是否全体重训；
   - 注意：实验设计要求同预算可比 → 要么全体升预算，要么在报告中明确"基线/加强版"两套预算。

6. **让路线 C 真正"规律注入"（否则 C 与 B 的差异站不住）**
   - 现状：C = B 骨干 + 标签/心率条件（`c_no_hr` 与 `b_cond` 指标完全一致即证据）；
   - 建议（择一或组合）：
     a. 增加**间期/规律监督损失**：辅助头预测 QT/PR/QRS 或 RR（由真实信号 delineate 得到），损失加权进扩散目标；
     b. **相位/间期条件**：以目标 QT/QRS/HR 作为显式条件（对齐 Biomedically informed / ECGEN 思路）；
     c. 采样后用**规律筛选**（best-of-N：按 QT/PR/QRS 与目标分布距离挑选），作为后处理基线。

### P1：一周内可做的增强

7. **8 独立导联生成 + 线性重构 12 导联**（对齐 SSSD-ECG / nle），提升导联一致性。
8. **单模型 + CFG**：训练时条件 dropout + padding index，采样用 guidance scale 网格（1.0/1.5/2.0/3.0），替代 B_uncond+B_cond 双模型。
9. **采样器升级**：DDIM 50 → DPM-Solver++（20 步内），对照质量/速度；用 EMA 权重。
10. **评估体系补强**：
    - ECG-FID（启用 `ecgen-eval` 的 ECG-FM / ECGFounder / HuBERT-ECG encoders）；
    - precision/recall/coverage（kNN 距离）；
    - t-SNE 可视化；临床 Turing test（我们已有盲评脚本，可正式跑）；
    - 跨库评估：MIT-BIH 之外加 CPSC2018 / Chapman-Shaoxing（若能获取）。
11. **路线 A 升级**：投影矩阵用真实 PTB-XL 数据最小二乘拟合（代替启发式），加 HRV/呼吸调制；可与扩散组成 hybrid（机理先验条件）。

### P2：研究性/加分项

12. **文本/报告条件生成**（DiffuSETS / Diff-TTE）：诊断报告 → ECG。
13. **个性化生成**（ECGTwin）：患者 embedding / 基础心拍条件。
14. **长程生成**（ECGEN 潜空间扩散）：10s → 30s/分钟级。
15. **隐私评估**：membership inference / 最近邻距离 / 复制检测，对齐"fidelity-utility-privacy"框架。
16. **生成数据增强实战**：用生成样本扩增少数类（HYP 仅 262 条测试样本），报告下游 F1 变化。

---

## 四、落地到本项目的执行清单（建议顺序）

| 步骤 | 改动文件/脚本 | 预计耗时 | 产出 |
|---|---|---|---|
| 1 | 新增 `experiments/eval_downstream.py`（TSTR/TRTS） | 0.5 天 | 新指标表 + 图 |
| 2 | 改 `models/diffusion.py` 条件编码器（nle 式）；重训 `c_full` | 1 天（训练 ~50min） | 新 c_full checkpoint/样本 |
| 3 | 改 `experiments/eval_functional.py`（HR 一致性、per-class 敏感度） | 0.5 天 | 功能层扩展指标 |
| 4 | 分类器加强训练 + 校准（`train_classifier.py`） | 0.5 天 | 新分类器（F1↑） |
| 5 | 训练预算/容量加强版（c_full + b_cond） | 1–2 天 | 加强版基线 |
| 6 | 规律注入（辅助间期损失或相位条件） | 2–3 天 | 真正的路线 C |
| 7 | 可选增强（8 导联重构 / CFG / DPM-Solver / ECG-FID / 隐私） | 3–5 天 | 论文级对比 |

> 建议把上面这些**整理**为后续实验清单；当前的交付结果（A/B/C/D + 三层评估 + 界面）作为**基线版本**保留，改进版单独编号（如 v2），不要覆盖基线，便于两端对比。

---

## 五、参考链接汇总

- SSSD-ECG: https://arxiv.org/abs/2301.08227 · https://github.com/AI4HealthUOL/SSSD-ECG
- SSSD-ECG-nle: https://arxiv.org/abs/2407.11108 · https://github.com/ispras/EcgLib
- ECGTwin: https://arxiv.org/abs/2508.02720
- DiffuSETS: https://arxiv.org/abs/2501.05932
- Diff-TTE: https://ieeexplore.ieee.org/document/11331862
- Biomedically Informed ECG Synconda 环境: https://xplorestaging.ieee.org/document/10821878
- ECGEN (latent diffusion, minute scale): https://www.sciencedirect.com/science/article/pii/S0169260725005541
- ECGGAN: https://arxiv.org/abs/1909.09150
- DeepFake ECG (MMD/DTW/PSD/PRD 评估族): https://www.nature.com/articles/s41598-021-01295-2
- ECG synconda 环境 metrics review (2025): https://doi.org/10.1016/j.compbiomed.2025.110879
- Deep generative models for physiological signals review (2025): https://doi.org/10.1016/j.artmed.2025.103127
- Synthetic ECG evaluation framework (fidelity/utility/privacy): https://uwspace.uwaterloo.ca/bitstreams/2c35ab52-959e-48f6-81a9-6e772d18fbbc/download
- Synthetic ECG for data augmentation: https://arxiv.org/abs/2411.18456
- PTB-XL: https://www.nature.com/articles/s41597-020-0495-6
- PTB-XL benchmarks: https://arxiv.org/abs/2004.13701
- ecgen-eval: https://github.com/vlbthambawita/ECGEN-Eval
