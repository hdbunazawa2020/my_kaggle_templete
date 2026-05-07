# /analyze_cv

Diagnose cross-validation quality.

## Read first
- `CLAUDE.md`
- `docs/METRIC.md`
- `docs/CV.md`
- `docs/EDA.md`
- `docs/EXPERIMENTS.md`

## Task
Analyze the current validation setup and determine whether CV is likely to be trustworthy.

## Checkpoints
- grouped vs random splitting (= grouping unit explicit?)
- same-source leakage (e.g., same soundscape / patient / session split across folds)
- adjacent-window leakage in time-series / audio
- site / domain leakage
- class coverage across folds (= rare class instability)
- mismatch between local inference path and submission inference path

## Output format
1. leakage risk (= high / med / low + evidence)
2. fold-quality issues
3. likely causes of CV-LB mismatch (with priority)
4. concrete fixes in priority order
