# 方法说明（Method）

> 面向作品集读者的方法细节。指标口径与复现命令见 README §4 与 `docs/EXPERIMENTS.md`。

---

## 1. 任务与数据

### 1.1 任务

在 PTB-XL 上生成 **10 秒 12 导联 ECG**（`(N,12,1000)` float32、100 Hz、逐导联 z-score），
并实现一个 **5 类多标签诊断分类器**（NORM/MI/STTC/CD/HYP）用于评估生成信号的功能与效用。
MIT-BIH 仅用于**跨域规律评估参考**，不参与训练/调参（实验设计要求，防泄漏）。

### 1.2 数据划分

| 数据集 | 用途 | 数量 |
|---|---|---|
| PTB-XL fold 1–8 | 训练 | 17,084 |
| PTB-XL fold 9 | 验证（选 checkpoint / 调阈值） | 2,146 |
| PTB-XL fold 10 | 测试（**只用于最终评估**） | 2,158 |
| MIT-BIH `'N'` 心搏 | 跨域参考分布 | 75,001（参考采样 1,000/域） |

### 1.3 预处理（写死在 `preprocess/`，可复现）

- 0.5–40 Hz Butterworth 带通（3 阶、零相位）+ 逐导联 z-score；
- 100 Hz 下 50 Hz 陷波在数学上不可定义（Nyquist）→ PTB-XL 分支跳过，MIT-BIH（360 Hz）正常陷波（偏差记录见 README §6）；
- 多标签：`scp_statements.csv` 中 `diagnostic==1` 的 code 聚合为 5 个超类，无诊断记录剔除（21,799 → 21,388）；
- 心率标签：Lead II Pan-Tompkins 式 R 峰检测（5–15 Hz 带通 → 微分 → 平方 → 80ms 滑窗 → 0.3s 不应期），
  与 NeuroKit2 交叉验证；第一版简单检测会误判 T 波（~108 bpm），已修复为 73.9 bpm，失败 262 条已记录；
- 间期目标（v2 规律注入用）：NeuroKit2 `ecg_delineate`（dwt）在整段 10s 信号上提取 QT/PR/QRS/RR-CV，
  中位 QT/PR/QRS ≈ 390/120/110 ms；失败样本在 `interval_targets_report.json` 中记录。

---

## 2. 四类生成路线

### 2.1 路线 A：机理模型（`models/mechanistic.py`）

McSharry 三耦合 ODE（ECGSYN 思路）在 500 Hz 内部积分后抽取 100 Hz，心率由输入 HR 控制（RR 序列带轻微变异性），
经 12 导联投影矩阵生成 12 导联信号。**无需训练**；生成 500 条件 + 1000 无条件。
局限：投影矩阵为几何启发式，波形细节与真实 ECG 有差距（形态层指标如实反映）。

### 2.2 路线 B：标准条件扩散（`models/diffusion.py`）

- DDPM ε-prediction，线性 beta 调度，1000 训练步；采样 DDIM 50 步；
- 骨干：1D U-Net（3 层下采样，residual block + timestep embedding + FiLM 条件 + 最低分辨率自注意力）；
- 条件模式：`none`（无条件）/ `label`（5 维多标签嵌入）/ `hr`（标准化心率标量）/ `both`；
- 条件 dropout p=0.15（训练时随机丢弃条件，使同一 checkpoint 也能无条件采样）；
- v1 配置：base 32（1.36–1.39M 参数），100 epochs，bs128，AdamW + cosine LR + AMP fp16 + 梯度裁剪。

### 2.3 路线 C：规律注入型扩散（v1 消融 5 配置 / v2 改进）

v1（消融配置）：

| 配置 | 条件 | 目的 |
|---|---|---|
| `c_full` | label + HR | 完整配置 |
| `c_no_label` | HR only | 无标签条件 |
| `c_no_hr` | label only | 无心率条件 |
| `c_label_noise` | label + HR + 10% 标签随机翻转 | 条件加噪鲁棒性 |
| `c_label_shuffle` | label + HR + 随机替换标签 | 条件替换（验证条件是否真被使用） |

v2（文献驱动改进，40 epochs / base 48 / EMA）：

- **nle 式标签嵌入**（SSSD-ECG-nle 2024）：每个标签 present/absent 两个 embedding + 可学习 null token，
  支持 CFG/条件 dropout（`models/diffusion.py: CondEncoderNLE`）；
- **间期辅助损失（规律注入）**：U-Net 瓶颈特征 → 预测 (QT, PR, QRS, RR-CV)，训练损失 +
  `λ × masked MSE`（λ=0.1）；目标来自 `preprocess/cache/interval_targets_*.npz`；
- **EMA 权重**（0.999，按 epoch 等效衰减）用于采样；
- 消融：`c_mask_v2`（mask 嵌入 + 辅助损失）、`c_nle_v2`（nle，无辅助损失）、`b_cond_v2`（nle 版路线 B）。

### 2.4 路线 D：GAN（`models/gan.py`）

Pulse2Pulse 风格：1D 生成器（latent z 输入）+ PatchGAN 判别器 + hinge loss + 谱归一化；
3 个固定 seed（42/43/44）各 100 epochs；报告均值±方差（GAN 方差是重要结论之一）。

### 2.5 生成协议

- 每条路线：500 条条件信号（条件从 PTB-XL 测试集真实条件采样）+ 1000 条无条件信号；
- 统一输出 `float32 (N,12,1000)`、100 Hz、逐导联 z-score、Lead II = 索引 1；
- 扩散：DDIM 50 步（v2 使用 EMA 权重）；GAN/机理：直接前向。

---

## 3. 评估体系

### 3.1 形态层（`experiments/eval_morphology.py`）

ecgen-eval 0.8.0：`mmd` / `dtw` / `psd` / `prd`（v1）+ `fd` / `swd` / `nn_dist` / `pr_dist` /
`heart_rate` / `spectral_entropy`（v2）；真实参考 = PTB-XL 测试集（100 Hz、12 导联）。

### 3.2 规律层（`experiments/eval_regularity.py`）

QT/PR/QRS/RR-CV 的分布距离：**Wasserstein（主指标）+ KS（统计量 + p 值）**；
域内参考 = PTB-XL 测试集；跨域参考 = MIT-BIH `'N'` 心搏；
**ΔWasserstein = 跨域 − 域内**（衡量域间稳健性）；每路线 delineate 失败数记录在 `n_failed_delineation`。

### 3.3 功能层（`experiments/eval_functional.py`）

- 条件一致率：给条件 c 生成信号，统计分类器预测为 c 的比例（每类 + 宏平均）；
- 无条件类别分布：1000 条无条件信号的 5 类预测比例 vs 真实测试集比例 → KL 散度；
- 无标签条件路线标记 `not_applicable` 并说明原因，不伪造数字。

### 3.4 v2 扩展（`eval_downstream.py` / `eval_consistency.py`）

- **下游效用**：TSTR（生成数据训练分类器→真实测试）/ CONTROL（同预算真实数据训练→真实测试）/
  TRTS（真实训练→生成数据测试）；
- **一致性诊断**：HR 一致性（目标 HR vs 生成信号测得 HR 的 MAE/bias/r/误差带命中率）、
  标签 TPR/TNR、导联一致性残差（III/aVR/aVL/aVF 对 I/II 的仿射残差）、
  k-NN coverage/precision、classifier-FID（分类器倒数第二层特征的 Fréchet 距离）。

### 3.5 统计与失败记录

- 固定 seed 42；GAN 3 seed 报告均值±方差；v2 与 v1 的对比注明预算差异；
- 所有失败（下载/预处理/训练/评估/delineate）写 `results/logs/` 与 `failures_*.csv`，不静默跳过；
- `metrics.csv` 完整性检查：缺失或无法解释的 NaN 会报错列出。

---

## 4. 诊断分类器

1D ResNet：conv stem（k=7, stride 2）+ 4 个残差阶段（base/2/4/8×base，各 2 个 basic block）+
全局平均池化 + 5 个 sigmoid logits；BCE 多标签损失。

| 版本 | 配置 | test macro-F1 |
|---|---|---|
| v1 | base 64，40 epochs，无 smoothing/EMA | 0.6838 |
| v2 | base 96，70 epochs，label smoothing 0.05 + EMA + 阈值校准 | 0.6868（fixed）/ 0.6970（tuned） |

分类器用于：功能层条件一致率、下游 TSTR/TRTS、一致性诊断（TPR/TNR、classifier-FID）。

---

## 5. 界面（`app/`）

Streamlit 三功能区：①上传诊断（分类器 + 间期测量）；②可控生成（路线/标签/HR → 生成 + 复核）；
③多路线对比（同条件并排 + FFT + 间期表）。界面与评估共用 `app/inference.py` 统一接口，
保证"演示结果 = 评估口径"；`tools/test_app.py`（AppTest）无头验收 PASS。
