# Kaggle Base Skill

Common-sense rules that should hold across any Kaggle competition in this template.

## Core Principles
- Always avoid data leakage.
- CV must simulate real inference timing and grouping.
- Log all experiments with seed, features, params, git commit.
- Use deterministic splits and fixed `random_state`.

## Submission Safety
- Match submission ID format exactly (= `sample_submission.csv` is the source of truth).
- Validate submission shape and duplicates before exporting.
- Never assume class / target order — explicitly align.
- Confirm dtype is numeric, no NaN / inf.

## Feature Engineering Rules
- Keep feature generation pure (= no target leakage, no future leakage).
- Version features under `data/processed/<version>/` or `output/features/`.
- Do not overwrite previous feature files; bump the version.

## Cross Validation
- Prefer time-based or group-based splits over random KFold.
- Validate on recent seasons / sites if leaderboard is recent-heavy.
- Save OOF predictions for calibration and ensembling.

## Model Training
- Start simple (= establish a working baseline before any complex idea).
- Improve features before tuning hyperparameters.
- Keep baseline reproducible (= a baseline you can re-run anytime).

## Ensembling
- Only ensemble models with independent error structure.
- Track CV gain per component before merging into the final blend.
- Prefer rank-average over raw-prob average for non-calibrated models.

## Code Quality
- No silent in-place mutation of core dataframes.
- Always assert row counts after merges.
- Fail fast if key columns are missing.
- Use `pathlib.Path`, avoid raw `os.path.join` strings.

## Run discipline
- Distinguish smoke (= 1 epoch / 100 rows) vs full runs.
- Quote runtime estimates honestly; long runs are user-launched, not Claude-launched, unless explicitly delegated.
