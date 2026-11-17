# 数据集获取指南（Data）

> **本仓库不包含数据集。** 出于体积（约 2.4 GB）与许可原因，PTB-XL、MIT-BIH 需使用者**自行下载一次**；
> 仓库只包含代码、文档、图表、少量示例信号（`samples/`，5 条派生样本）与指标结果。
> 本文档给出：下载地址、校验值、目录结构、自动/手动两种方式、以及下载后的预处理流程。

---

## 1. 需要哪两个数据集

| 数据集 | 用途 | 版本 | 大小 | 官方页面 |
|---|---|---|---|---|
| **PTB-XL**（12 导联 10 s ECG，21,799 条） | 训练 / 验证 / 测试（fold 1–8/9/10） | 1.0.3 | 压缩包 **1,743.4 MB** | [physionet.org/content/ptb-xl/1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) |
| **MIT-BIH Arrhythmia Database**（48 条长记录，360 Hz） | **仅**做跨域评估参考（不参与训练/调参） | 1.0.0 | **82.9 MB**（parquet 镜像）/ 约 100 MB（官方 WFDB zip） | [physionet.org/content/mitdb/1.0.0](https://physionet.org/content/mitdb/1.0.0/) |

引用请注明：Wagner et al., *PTB-XL, a large-scale publicly available electrocardiography dataset*, Scientific Data 2020；
Moody & Mark, *The impact of the MIT-BIH Arrhythmia Database*, IEEE EMB 2001。许可遵循 PhysioNet 页面标注（引用与再分发请遵守其条款）。

---

## 2. 自动下载（推荐）

```powershell
cd <repo root>
python preprocess/check_data.py            # 先看缺什么（缺数据时退出码 1）
python preprocess/download_data.py         # 下载两个数据集（约 1.9 GB）
python preprocess/check_data.py            # 再次确认
```

`preprocess/download_data.py` 默认走 **HuggingFace 镜像**（hf-mirror.com），支持断点续传/分块并行，
下载前会对比远端大小；已完整存在的文件自动跳过。参数：`--skip-ptbxl` / `--skip-mitbih`。

下载内容写入：

```
data/
├── ptb-xl/
│   ├── ptb-xl-data.zip          # 官方 1.0.3 归档（含 records100/、records500/、csv）
│   ├── ptbxl_database.csv
│   ├── scp_statements.csv
│   └── RECORDS
└── mit-bih/
    └── mitbih.parquet           # 官方 MIT-BIH 1.0.0 内容的公开 parquet 副本
```

---

## 3. 手动下载（镜像不可用 / 想要官方源）

### PTB-XL 1.0.3
1. 打开 <https://physionet.org/content/ptb-xl/1.0.3/>，下载官方归档
   （文件名以页面为准，约 1.7 GB）；
2. 放到 `data/ptb-xl/` 并**重命名为 `ptb-xl-data.zip`**；
3. 校验 SHA-256（可选，见 §5 校验值）。

### MIT-BIH 1.0.0
两条路任选：
- **官方 WFDB**：<https://physionet.org/content/mitdb/1.0.0/> 下载官方 zip，解压到 `data/mit-bih/`
  （使 `.hea/.dat/.atr` 直接在 `data/mit-bih/` 下，48 条记录）；
- **parquet 副本（与仓库脚本一致）**：HuggingFace 镜像
  `https://hf-mirror.com/datasets/epr-labs/mit-bih-arrhythmia-database/resolve/main/data/train-00000-of-00001.parquet`
  下载后保存为 `data/mit-bih/mitbih.parquet`。

> 说明：早期 HF 上的 `Meddies/ecg-mitdb` 原始 WFDB 仓库已变为私有（HTTP 403），因此本项目默认用
> `epr-labs/mit-bih-arrhythmia-database` 的 parquet 副本（内容即官方 mitdb 1.0.0：48 记录、MLII/V1 或 V5、
> 360 Hz、beat 注释 + WFDB 头信息）。

---

## 4. 下载后的预处理（一次性，约 15–30 分钟）

```powershell
python preprocess/extract_ptbxl.py          # 解压 records100 + 元数据（21,799 条）
python preprocess/make_ptbxl_cache.py       # 滤波/z-score/官方划分 → ptbxl_{train,val,test}.npz
python preprocess/recompute_hr.py           # Pan-Tompkins 心率标签（覆盖缓存中的 hr）
python preprocess/make_mitbih_cache.py      # MIT-BIH 正常心搏参考（'N' 标注，360 Hz 不重采样）
python preprocess/extract_reference_params.py  # QT/PR/QRS 参考分布（NeuroKit2 delineate）
python preprocess/extract_interval_targets.py  # 可选：v2 间期辅助损失目标
python preprocess/check_data.py             # 最终检查（应输出 READY）
```

产物位于 `preprocess/cache/`（约 1.1 GB）：`ptbxl_{train,val,test}.npz`、`mitbih.npz`、
`ref_params_*.csv`、`regularity_refs.npz` 等。它们**同样不随仓库分发**（可由上述脚本重建）。

> 磁盘预算：原始数据（zip + 解压）约 **2.4 GB**，缓存约 **1.1 GB**，模型 checkpoint/生成信号另计。
> 若想省空间：解压并生成缓存后可删除 `data/ptb-xl/records100/`（缓存已含所需信号）。

---

## 5. 校验值（SHA-256）

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `data/ptb-xl/ptb-xl-data.zip` | 1,743.4 MB | `b5a00ab0b5b2f2b19ab6de611eb4c38511f0c9827aa4ec8ae835a6f7ded5dfb9` |
| `data/mit-bih/mitbih.parquet` | 82.9 MB | `ef2c46f3de342b30ef56429b86d483621b4099314ab55ebd15537426bec59c79` |
| `data/ptb-xl/ptbxl_database.csv` | 6.3 MB | `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b` |
| `data/ptb-xl/scp_statements.csv` | 41 KB | `ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f` |
| `data/ptb-xl/RECORDS` | 1.1 MB | `56d37274b0b02339e9c30bff8c9a9f2a6fb3cb2bb7bcd5d7c55355451d53896a` |

校验命令（PowerShell 示例）：

```powershell
(Get-FileHash data/ptb-xl/ptb-xl-data.zip -Algorithm SHA256).Hash.ToLower()
(Get-FileHash data/mit-bih/mitbih.parquet -Algorithm SHA256).Hash.ToLower()
```

> 官方源与镜像的归档内容一致，但若官方 zip 重新打包，hash 可能不同——此时以“能正常解压 + `check_data.py` 通过”为准。

---

## 6. 常见问题

| 问题 | 处理 |
|---|---|
| 官方 PhysioNet 下载很慢（本项目实测 ~16 KB/s） | 用 `download_data.py` 的 HF 镜像；或换网络/离线下载后拷入 |
| hf-mirror 无法访问 | 换官方源手动下载（§3）；或使用其他公开镜像，但请核对 §5 校验值 |
| MIT-BIH 只有 WFDB 目录没有 parquet | `make_mitbih_cache.py` 同时支持 WFDB 目录，直接放 `data/mit-bih/*.hea` 即可 |
| `download_data.py` 报大小不一致/中断 | 重新运行即可（按远端大小续传/跳过）；仍失败则删除半成品 `.part` 后重试 |
| 磁盘不够 | 先只下 PTB-XL；MIT-BIH 仅用于跨域评估，可稍后补 |
| 不想跑训练，只想看结果/图表 | 不需要数据集：`results/` 已含指标与图表，`samples/` 含示例信号 |

---

## 7. 许可与再分发

- 数据集**不进仓库、不再分发**；`samples/` 中仅 5 条 PTB-XL 派生信号用于格式演示（注明来源，遵守 PTB-XL 许可）；
- 代码 MIT；文档 CC BY 4.0；
- 使用这些数据发表工作请引用对应数据集论文与 PhysioNet 页面。
