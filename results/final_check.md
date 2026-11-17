# 交付物完整性检查

| 状态 | 项目 | 详情 |
|---|---|---|
| ✅ | 缓存 ptbxl_train.npz |  |
| ✅ | 缓存 ptbxl_val.npz |  |
| ✅ | 缓存 ptbxl_test.npz |  |
| ✅ | 缓存 mitbih.npz |  |
| ✅ | 缓存 ref_params_ptbxl_test.csv |  |
| ✅ | 缓存 ref_params_mitbih.csv |  |
| ✅ | checkpoint classifier_best.pt（分类器） | 3.9MB, epoch=? |
| ✅ | checkpoint b_uncond.pt（路线B无条件） | 5.5MB, epoch=100 |
| ✅ | checkpoint b_cond.pt（路线B条件） | 5.7MB, epoch=100 |
| ✅ | checkpoint c_full.pt（路线C完整） | 5.7MB, epoch=100 |
| ✅ | checkpoint c_no_label.pt（C-去标签） | 5.7MB, epoch=100 |
| ✅ | checkpoint c_no_hr.pt（C-去心率） | 5.7MB, epoch=100 |
| ✅ | checkpoint c_label_noise.pt（C-条件加噪） | 5.7MB, epoch=100 |
| ✅ | checkpoint c_label_shuffle.pt（C-条件替换） | 5.7MB, epoch=100 |
| ✅ | checkpoint d_seed42.pt（路线D seed42） | 8.6MB, epoch=100 |
| ✅ | checkpoint d_seed43.pt（D seed43） | 8.6MB, epoch=100 |
| ✅ | checkpoint d_seed44.pt（D seed44） | 8.6MB, epoch=100 |
| ✅ | 生成信号 A | A_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 B_uncond | B_uncond_uncond.npz (1000, 12, 1000), finite=True |
| ✅ | 生成信号 B_cond | B_cond_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 C_full | C_full_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 C_no_label | C_no_label_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 C_no_hr | C_no_hr_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 C_label_noise | C_label_noise_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 C_label_shuffle | C_label_shuffle_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 D_seed42 | D_seed42_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 D_seed43 | D_seed43_cond.npz (500, 12, 1000), finite=True |
| ✅ | 生成信号 D_seed44 | D_seed44_cond.npz (500, 12, 1000), finite=True |
| ✅ | 指标 metrics.csv |  |
| ✅ | 指标 metrics_morphology.csv |  |
| ✅ | 指标 metrics_regularity.csv |  |
| ✅ | 指标 metrics_functional.csv |  |
| ✅ | 指标 class_distribution.csv |  |
| ✅ | 指标 summary.md |  |
| ✅ | 指标 classifier_test_report.md |  |
| ✅ | 指标 classifier_test_report.csv |  |
| ✅ | metrics.csv 路线覆盖 | 817 行, routes=['A', 'B_cond', 'B_uncond', 'REAL_PTBXL_TEST', 'b_cond_v2', 'c_aux05_v2', 'c_full', 'c_full_v2', 'c_label_noise', 'c_label_shuffle', 'c_mask_v2', 'c_nle_v2', 'c_no_hr', 'c_no_label', 'classifier', 'classifier_v2', 'd_seed42', 'd_seed43', 'd_seed44'] |
| ✅ | 图表 | 18 张 PNG |
| ✅ | 失败记录 CSV | 19 个 failures_*.csv |
| ✅ | 文档 README.md |  |
| ✅ | 文档 PROJECT_LAYOUT.md |  |
| ✅ | 文档 requirements.txt |  |
| ✅ | 文档 docs/DATA.md |  |
| ✅ | 文档 docs/CHECKPOINTS.md |  |
| ✅ | 文档 docs/METHOD.md |  |
| ✅ | 数据下载/检查脚本 |  |

**汇总：46 项通过，0 项待补/缺失。**
> 缺失项在明天补训练/生成/评估后重跑本脚本即可。
