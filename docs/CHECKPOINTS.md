# 模型权重说明（Checkpoints）

> 与数据集一样，**训练好的模型权重不随仓库分发**（约 186 MB、二进制、可复现训练即可重建）。
> 仓库包含：代码、训练/评估脚本、指标结果、图表与少量示例信号（`samples/`）。
> 因此：**看结果/图表不需要权重；跑 Streamlit 演示或重新生成信号需要权重或重新训练**。

---

## 1. 需要哪些权重

| 文件（`models/checkpoints/`） | 用途 | 训练命令要点 | 单次训练耗时（本机 8GB） |
|---|---|---|---|
| `classifier_best.pt` / `classifier_best_v2.pt` | 诊断分类器（v1 / v2） | `experiments/train_classifier.py [--tag v2 ...]` | ~12 / 25 分钟 |
| `b_uncond.pt` / `b_cond.pt` | 路线 B 扩散（无/有标签条件） | `train_diffusion.py --name b_uncond --cond-mode none` 等 | ~50 分钟/个 |
| `c_full.pt` 等 5 个（v1） | 路线 C 消融（5 配置） | `experiments/run_ablations.py` | ~50 分钟/个 |
| `c_full_v2.pt` / `c_nle_v2.pt` / `c_mask_v2.pt` / `b_cond_v2.pt` / `c_aux05_v2.pt` | v2/W2 改进模型 | `train_diffusion.py --name ... --base 48 --ema-decay 0.999 ...`（见 `tools/run_v2.ps1`、`tools/run_w2.ps1`） | ~20–40 分钟/个 |
| `d_seed42/43/44.pt` | 路线 D GAN（3 seed） | `experiments/run_route_d.py` | ~15–20 分钟/seed |

> 最快跑通界面的最小集合：`classifier_best_v2.pt`（或 v1）+ `c_full_v2.pt` + 可选 `b_cond.pt`。
> 完整复现全部结果：`tools/run_all.ps1`（含训练+生成+评估，数小时，可断点续跑）。

---

## 2. 训练权重（标准流程）

```powershell
# 0) 数据先就绪（不随仓库分发）：见 docs/DATA.md
python preprocess/check_data.py            # 应输出 READY

# 1) 最小可用集合（约 1 小时）
python experiments/train_classifier.py --tag v2 --base 96 --epochs 70 `
    --label-smoothing 0.05 --ema-decay 0.999 --threshold-tune
python experiments/train_diffusion.py --name c_full_v2 --cond-mode both --cond-style nle `
    --aux-intervals --aux-weight 0.1 --epochs 40 --base 48 --ema-decay 0.999

# 2) 可选：路线 A（无需训练）
python experiments/generate_a.py

# 3) 启动界面（需要至少一个分类器 + 一条扩散路线）
python -m streamlit run app/streamlit_app.py
```

完整复现（含所有路线、消融、W2 实验、评估、图表）：

```powershell
powershell -File tools/run_all.ps1          # 训练 + 生成 + 评估 + 汇总
powershell -File tools/run_eval_all.ps1     # 仅评估（已有权重时）
powershell -File tools/run_w2.ps1 -SkipTraining   # W2 三个实验
```

---

## 3. 可选：通过 GitHub Release 提供权重

若希望别人无需训练即可体验界面，可在 GitHub **Releases** 上传权重压缩包（不进入 git 历史）：

- 打包：`Compress-Archive -Path models/checkpoints/* -DestinationPath ecg-ckpt.zip`（约 100–200 MB）；
- 新建 Release（如 `v1.0-weights`），上传 `ecg-ckpt.zip`，在说明中写明解压到 `models/checkpoints/`；
- 在 README 的“快速开始”中放 Release 链接（本项目出于体积/可复现考虑默认不附带权重）。

> 注意：权重与数据一样受训练数据许可约束，仅用于研究/演示，不用于临床。

---

## 4. 常见问题

| 问题 | 处理 |
|---|---|
| 界面提示“没有任何可用路线” | 缺 `models/checkpoints/*.pt`；按 §2 训练或放好权重 |
| 只有分类器没有扩散模型 | 可先用“上传诊断”区块；生成/对比需要至少一条扩散路线 |
| 生成结果与仓库图表不一致 | 图表来自完整权重；重新训练存在随机性（同 seed 下应接近） |
| 想复现 `metrics.csv` | 需要全部权重 + 数据缓存，然后 `tools/run_eval_all.ps1`；也可直接查看仓库中的 `results/` |
