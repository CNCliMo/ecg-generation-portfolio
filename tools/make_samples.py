"""Extract a few small example signals into samples/ for quick inspection.

Source files are large generated/test sets; we keep only 4-5 examples per file so
the sample folder stays <1 MB and can be committed (with attribution, see samples/README.md).

Outputs:
  samples/ptbxl_test_examples.npz     signals/labels/ecg_id/hr  (5 samples, real)
  samples/gen_c_full_v2_examples.npz  signals/labels/hr         (5 generated, v2 main)
  samples/gen_A_examples.npz          signals/hr                (5 generated, mechanistic)
  samples/README.md                   provenance + loading instructions
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE = os.path.join(ROOT, "preprocess", "cache")
GEN = os.path.join(ROOT, "results", "generated")
OUT = os.path.join(ROOT, "samples")
os.makedirs(OUT, exist_ok=True)

N = 5
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ real ----
z = np.load(os.path.join(CACHE, "ptbxl_test.npz"), allow_pickle=False)
idx = rng.choice(len(z["signals"]), size=N, replace=False)
np.savez_compressed(
    os.path.join(OUT, "ptbxl_test_examples.npz"),
    signals=z["signals"][idx].astype(np.float32),
    labels=z["labels"][idx].astype(np.uint8),
    ecg_id=z["ecg_id"][idx] if "ecg_id" in z else np.arange(N),
    hr=z["hr"][idx].astype(np.float32) if "hr" in z else np.full(N, np.nan, np.float32),
)
print("saved ptbxl_test_examples.npz", z["signals"][idx].shape)

# ------------------------------------------------------------- generated ----
for name, src in (
    ("gen_c_full_v2_examples.npz", "c_full_v2_cond.npz"),
    ("gen_A_examples.npz", "A_cond.npz"),
):
    p = os.path.join(GEN, src)
    if not os.path.exists(p):
        print("skip (missing):", p)
        continue
    zz = np.load(p, allow_pickle=False)
    n = len(zz["signals"])
    sel = rng.choice(n, size=min(N, n), replace=False)
    out = {"signals": zz["signals"][sel].astype(np.float32)}
    if "labels" in zz:
        out["labels"] = zz["labels"][sel].astype(np.uint8)
    if "hr" in zz:
        out["hr"] = zz["hr"][sel].astype(np.float32)
    np.savez_compressed(os.path.join(OUT, name), **out)
    print("saved", name, out["signals"].shape)

readme = """# 示例信号（samples/）

本目录只放**少量示例信号**（每文件 5 条，float32 `(N,12,1000)`，100 Hz，逐导联 z-score，Lead II = 索引 1），
用于快速查看格式/画图，不参与训练与评估。

| 文件 | 内容 | 来源 |
|---|---|---|
| `ptbxl_test_examples.npz` | `signals, labels(5类多标签), ecg_id, hr` | PTB-XL fold 10 测试集随机 5 条（许可：PTB-XL，见 README §8；仅派生样本，不重新分发原始数据） |
| `gen_c_full_v2_examples.npz` | `signals, labels, hr` | 本项目 v2 主方法 `c_full_v2` 的条件生成（随机 5 条） |
| `gen_A_examples.npz` | `signals, labels, hr` | 本项目路线 A 机理模型的条件生成（随机 5 条） |

加载：

```python
import numpy as np
z = np.load("samples/ptbxl_test_examples.npz")
print(z["signals"].shape, z["labels"].shape)   # (5, 12, 1000)
```

完整数据/全部路线生成信号请按 README §4 下载与生成（不入库）。
"""
with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme)
print("saved samples/README.md")
print("DONE")
