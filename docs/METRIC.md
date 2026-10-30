# METRIC.md

> Define the official metric **exactly**. Misreading the metric is the most common Kaggle failure.

## Metric overview
- **Official metric**: {{e.g., macro ROC-AUC, RMSE, log-loss, MAP@k}}
- **Aggregation**: {{micro / macro / per-group}}
- **Special handling**: {{e.g., classes with zero positives are skipped, ties broken by ...}}

## ⭐ primary CV metric の宣言（初週に埋めて以後固定する）

- **primary**: {{fold-wise mean / concat(pooled)}} ← **これで全ての採否を決める**
- **secondary**: {{もう一方}}（参考値。採否の根拠にしない）
- **決めた理由**: {{公式 metric が macro 集計だから / fold 間の分布が揃っているから 等}}
- **決定日**: {{YYYY-MM-DD}}

> ⚠ **後から変更すると、それまでの実験が全部比較不能になる。** 変更する場合は
> 「いつ・なぜ変えたか」と「過去の数値をどう読み替えるか」を必ず併記する。

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
- [ ] primary の集計方法を **宣言済み**（fold-wise mean か concat か。`docs/CV.md` 参照）
- [ ] 両方の集計を計算し、**target 別に符号が食い違わないこと**を確認した
      （食い違ったら fold-scale artifact を疑う → `docs/VALIDATION_PITFALLS.md`）

## Known compression ratio (= CV-LB transfer)
| run | CV change | LB change | ratio |
|---|---|---|---|
| {{baseline}} | — | — | — |

> Update this table as runs accumulate. Knowing the compression ratio is critical for deciding what to submit.

## Threshold / calibration policy
- For ranking metrics (= ROC-AUC, MAP): no threshold needed.
- For decision metrics (= F1, accuracy): tune threshold on OOF, never on test.
- For probability metrics (= log-loss, Brier): apply per-class calibration only when CV improves consistently.
