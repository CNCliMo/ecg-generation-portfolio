# Diagnostic classifier -- test report (fold 10)

- best epoch: 5 (val macro-F1 0.6859)
- params: 0.97 M, device: cuda
- threshold fixed at 0.5, BCEWithLogitsLoss, seed 42

## overall

| metric | value |
|---|---|
| loss | 0.3123 |
| macro_f1 | 0.6838 |
| micro_f1 | 0.7378 |
| exact_match | 0.5737 |
| macro_auroc | 0.8975 |
| macro_ap | 0.8190 |

## per class

| class | precision | recall | F1 | AUROC | support |
|---|---|---|---|---|---|
| NORM | 0.8424 | 0.8380 | 0.8402 | 0.9368 | 963 |
| MI | 0.6962 | 0.6709 | 0.6833 | 0.8914 | 550 |
| STTC | 0.7403 | 0.7716 | 0.7556 | 0.9295 | 521 |
| CD | 0.8453 | 0.6169 | 0.7133 | 0.9105 | 496 |
| HYP | 0.5345 | 0.3550 | 0.4266 | 0.8196 | 262 |
