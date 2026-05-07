# CV.md

## Validation strategy

- **Method**: {{GroupKFold / StratifiedGroupKFold / TimeSeriesSplit / custom}}
- **Grouping unit**: {{column or derived key — must match a real-world unit that test will respect}}
- **n_folds**: {{4 / 5 / 10}}
- **seed**: {{integer}}
- **Stratification target**: {{column}}

## Fold assignment policy
{{Where is the fold assignment stored? `train_df.csv["fold"]`? Derived deterministically per row? Store *one* canonical assignment so all scripts use the same folds.}}

## Why this method
{{Why this group / time choice? Which leakage risk does it close?}}

## Anti-leakage checks (= must hold)
- [ ] no source unit (= soundscape / patient / session) appears in >1 fold
- [ ] no temporal overlap if test is future
- [ ] no metadata leak (= site code, device id) across folds
- [ ] OOF concat aligns 1:1 with `train_df`

## Metric computation
- Per-fold metric is computed on val rows of that fold.
- Concat metric is computed once on stacked OOF (= closer to LB simulator).
- Report both. Concat is the trustworthy summary.

## Known CV-LB mismatch failure modes for this comp
- {{record observed ones here as you encounter them}}
