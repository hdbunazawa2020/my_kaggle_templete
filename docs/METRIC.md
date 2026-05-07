# METRIC.md

> Define the official metric **exactly**. Misreading the metric is the most common Kaggle failure.

## Metric overview
- **Official metric**: {{e.g., macro ROC-AUC, RMSE, log-loss, MAP@k}}
- **Aggregation**: {{micro / macro / per-group}}
- **Special handling**: {{e.g., classes with zero positives are skipped, ties broken by ...}}

## Official scoring logic
```python
# Pseudocode of the official metric exactly as the host computes it.
# If a host-provided scoring file exists, link or paste it here.
```

## Plain-English interpretation
{{What does "+0.01 score" mean in practice? When does noise dominate?}}

## Local evaluation parity checklist
- [ ] local CV uses the **same** metric implementation
- [ ] local CV uses the **same** preprocessing
- [ ] local CV honors any class-skipping or per-group rules
- [ ] OOF concat metric is computed (= matches LB simulator more closely than per-fold mean)

## Known compression ratio (= CV-LB transfer)
| run | CV change | LB change | ratio |
|---|---|---|---|
| {{baseline}} | — | — | — |

> Update this table as runs accumulate. Knowing the compression ratio is critical for deciding what to submit.

## Threshold / calibration policy
- For ranking metrics (= ROC-AUC, MAP): no threshold needed.
- For decision metrics (= F1, accuracy): tune threshold on OOF, never on test.
- For probability metrics (= log-loss, Brier): apply per-class calibration only when CV improves consistently.
