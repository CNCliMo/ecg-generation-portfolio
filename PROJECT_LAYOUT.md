# 项目目录结构（Directory Layout）

> 仓库根目录：`<REPO_ROOT>`。所有代码/文档/结果都在此目录内；
> **原始数据集、预处理缓存、模型权重、生成信号不随仓库分发**（见 `docs/DATA.md`、`docs/CHECKPOINTS.md`）。

---

## 1. 目录树（仓库内容）

```
ECG/
├── .gitignore                         # 排除环境/数据/缓存/权重/生成信号/临时文件
├── LICENSE                            # 代码 MIT；文档 CC BY 4.0
├── requirements.txt
├── README.md                          # 项目入口（环境/数据/训练/评估/界面/偏差记录）
├── PROJECT_LAYOUT.md                  # 本文件
├── configs/                           # 全局配置（seed/采样率/fold/预算/评估口径）
├── ecg_common/                        # 公共库：paths / config / data / intervals / logging
├── preprocess/                        # 下载、解压、缓存、参考参数、间期目标、数据检查
│   └── check_data.py                  # ★ 数据就绪检查（下载后运行）
├── models/                            # 机理 / 扩散 / GAN / 分类器代码（+ checkpoints/ 本地）
├── experiments/                       # 训练、生成、评估、汇总、画图
├── app/                               # Streamlit 界面 + inference
├── results/
│   ├── metrics*.csv                   # 三层评估 / 下游 / 一致性 / 对比指标
│   ├── summary.md / summary_v2.md     # 结果摘要
│   ├── w2_report.md                   # W2 实验报告
│   └── figures/                       # 结果图（入库，约 19 张 PNG + demo.gif）
├── samples/                           # 5 条示例信号（格式演示，约 0.65MB）
├── tools/                             # 统一入口 / 环境自检 / 完整性检查 / 一键脚本
└── docs/                              # 技术文档
    ├── METHOD.md                      # 方法与数据流程
    ├── EXPERIMENTS.md                 # 实验清单与协议
    ├── LIMITATIONS.md                 # 局限与负结果
    ├── DATA.md                        # 数据集下载指南（不随仓库分发）
    ├── CHECKPOINTS.md                 # 模型权重说明（不随仓库分发）
    ├── PAPERS_ECG_GENERATION.md       # 文献综述（20+ 篇）
    └── V2_DESIGN.md                   # v2 改进设计与结果
```

**以下内容仅存在本地、不进入仓库**：`.venv_iso/`（环境）、`data/`（原始数据）、
`preprocess/cache/`（预处理缓存）、`models/checkpoints/`（权重）、`results/generated*/`（生成信号）、
`results/logs/`（运行日志）、`.local_archive/`（开发历史归档）、`.tmp/`。

---

## 2. 运行方式速查

```powershell
cd <REPO_ROOT>

# 环境自检（应显示 .venv_iso、torch 2.5.1+cu121、CUDA True）
powershell -File tools\run.ps1 tools\env_check.py
powershell -File tools\run.ps1 tools\which_env.py

# 数据：下载 + 预处理（数据集不随仓库分发）
python preprocess/download_data.py
powershell -File tools\run.ps1 preprocess\extract_ptbxl.py
powershell -File tools\run.ps1 preprocess\make_ptbxl_cache.py
powershell -File tools\run.ps1 preprocess\make_mitbih_cache.py
powershell -File tools\run.ps1 preprocess\extract_reference_params.py
python preprocess/check_data.py            # 应输出 READY

# 训练 / 生成 / 评估
powershell -File tools\run.ps1 experiments\train_classifier.py
powershell -File tools\run.ps1 experiments\train_diffusion.py --name c_full --cond-mode both
powershell -File tools\run_eval_all.ps1   # 三层评估 + metrics.csv + 图表 + summary

# 界面
powershell -File tools\run_app.ps1        # http://127.0.0.1:8501
powershell -File tools\run.ps1 tools\test_app.py  # AppTest 无头验收

# 交付物完整性
powershell -File tools\run.ps1 tools\final_check.py  # → results/final_check.md
```

---

## 3. 注意

- 不要删除：`preprocess/cache/`（训练/评估输入）、`models/checkpoints/`、`results/generated/`（本地）。
- 失败样本记录：`results/logs/failures_*.csv`（本地，不入库）。
- `.venv_iso/` 为本机隔离环境；通用机器可按 README §4.1 重建。
- 实验清单/协议见 `docs/EXPERIMENTS.md`；改进方向见 `docs/PAPERS_ECG_GENERATION.md` §4。
