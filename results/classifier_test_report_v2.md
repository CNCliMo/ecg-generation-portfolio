# Diagnostic classifier -- test report (fold 10) _v2

- best epoch: 13 (raw, val macro-F1 0.6903)
- params: 8.66 M (base=96), device: cuda, label_smoothing=0.05, ema_decay=0.999
- thresholds: fixed 0.5 / tuned [0.35, 0.25, 0.4, 0.4, 0.2]

## overall

| metric | value |
|---|---|
| fixed0.5_loss | 0.3228 |
| fixed0.5_macro_f1 | 0.6868 |
| fixed0.5_micro_f1 | 0.7437 |
| fixed0.5_exact_match | 0.5890 |
| fixed0.5_macro_auroc | 0.8975 |
| fixed0.5_macro_ap | 0.8129 |
| tuned_macro_f1 | 0.6970 |
| tuned_micro_f1 | 0.7388 |
| tuned_exact_match | 0.5306 |

## per class (fixed 0.5)

| class | precision | recall | F1 | AUROC | threshold | support |
|---|---|---|---|---|---|---|
| NORM | 0.8264 | 0.8650 | 0.8453 | 0.9321 | 0.35 | 963 |
| MI | 0.7291 | 0.6655 | 0.6958 | 0.9035 | 0.25 | 550 |
| STTC | 0.6902 | 0.8081 | 0.7445 | 0.9232 | 0.40 | 521 |
| CD | 0.8159 | 0.6613 | 0.7305 | 0.9111 | 0.40 | 496 |
| HYP | 0.5427 | 0.3397 | 0.4178 | 0.8175 | 0.20 | 262 |
