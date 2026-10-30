# Head-to-head with published LLM data-science agents

Published figures: Data Interpreter (Hong et al., ACL Findings 2025, arXiv 2402.18679), Table 6, Normalized Performance Score. Accuracy for the three classification sets; 1/(1+RMSLE) for House Prices.

Ours: the complete framework run on an 80% split, scored once on the untouched 20% -- their protocol -- repeated over several random splits.

| Dataset | AutoGen | OpenInterpreter | TaskWeaver | XAgent | OpenDevin | Data Interpreter | **Ours (mean ± sd)** | Ours, per split |
|---|---|---|---|---|---|---|---|---|
| wine | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.96 | **0.994 ± 0.012** | 1.000, 1.000, 0.972, 1.000, 1.000 |
| breast_cancer | 0.97 | 0.96 | 0.96 | 0.94 | 0.96 | 0.99 | **0.977 ± 0.010** | 0.965, 0.991, 0.983, 0.974, 0.974 |
| titanic | 0.82 | 0.81 | 0.43 | 0.00 | 0.81 | 0.82 | **0.847 ± 0.029** | 0.799, 0.866, 0.866, 0.866, 0.838 |
| house_prices | 0.88 | 0.87 | 0.49 | 0.00 | 0.87 | 0.91 | **0.880 ± 0.007** | 0.880, 0.879, 0.872, 0.879, 0.891 |
| **Average of the four** | 0.917 | 0.910 | 0.720 | 0.485 | 0.900 | 0.920 | **0.925** | |

House Prices raw RMSLE (ours): 0.1363 ± 0.0091. Data Interpreter's NPS of 0.91 corresponds to RMSLE 0.0989.

## Cross-validation protocol: LAMBDA

LAMBDA (Sun et al., 2025, arXiv 2407.17535), Table 2: 5-fold cross-validated accuracy, best of the nine models it fitted.

| Dataset | LAMBDA (best of 9) | **Ours, selected model** | Ours, best candidate (their definition) |
|---|---|---|---|
| wine | 0.9889 | **0.9887 ± 0.0138** (Extra Trees) | 0.9887 (Extra Trees, of 10) |
| breast_cancer | 0.9807 | **0.9824 ± 0.0111** (Stacked Ensemble) | 0.9824 (Stacked Ensemble, of 10) |

## What is and is not controlled

- Same datasets, same metric, same 80/20 hold-out protocol.
- Not the same split rows: their split files are not reproduced, so several seeded splits are reported instead of one.
- Not the same language model: they used gpt-4-1106-preview. Our figures do not depend on the language model -- no agent here computes a number -- so this run used the offline backend.
- Runs with an error: 0
