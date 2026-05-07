# /generate_submission

Create or review the submission pipeline.

## Read first
- `CLAUDE.md`
- `docs/SUBMISSION.md`
- `docs/METRIC.md`

## Task
Implement or review code that:
- loads the test data exactly as Kaggle delivers it
- runs inference under the same constraints as production
- writes `submission.csv` matching `sample_submission.csv` format

## Mandatory checks
- exact `row_id` format (= matches sample_submission row by row)
- all required class / target columns present and in the right order
- numeric values only, no NaN / inf
- runtime fits {{KAGGLE_RUNTIME_LIMIT}}
- hardware fits {{HARDWARE_CONSTRAINT}} (= no silent CUDA assumption if CPU-only)

## Output format
1. submission pipeline summary
2. files changed
3. validation checklist (= row_id ✓, columns ✓, dtype ✓, runtime ✓)
4. runtime warning if any
