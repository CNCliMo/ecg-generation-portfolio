# 示例信号（samples/）

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
