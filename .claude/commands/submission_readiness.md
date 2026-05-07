# /submission_readiness

Check whether the current repo is ready to produce a valid submission.

## Read first
- `CLAUDE.md`
- `docs/SUBMISSION.md`
- `docs/AUTOMATION.md`

## Checks
- training pipeline → ckpt / artifacts exist for all needed folds
- inference notebook / script can read those artifacts on Kaggle (= path resolution works in both local and Kaggle env)
- `submission.csv` schema matches `sample_submission.csv`
- runtime budget estimate ≤ {{KAGGLE_RUNTIME_LIMIT}}
- model weights + helpers uploaded to Kaggle dataset (or attached)
- no hardcoded local paths leaked into the inference path

## Output format
1. readiness verdict (= ready / blocked / risky)
2. blocking issues (sorted by severity)
3. concrete fix for top blocker
4. runtime estimate
