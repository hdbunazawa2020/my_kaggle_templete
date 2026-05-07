# CLAUDE.md

> **Template note**: replace `{{COMPETITION_NAME}}`, `{{TARGET_METRIC}}`, `{{N_CLASSES}}`, `{{TASK_TYPE}}` etc. when bootstrapping a new repo.

## Purpose

This repo is for **{{COMPETITION_NAME}}** research and Kaggle submission work.

Claude should behave like a disciplined Kaggle research engineer:
- preserve reproducibility
- avoid leakage
- optimize for **{{TARGET_METRIC}}**
- respect competition runtime constraints (= **{{KAGGLE_RUNTIME_LIMIT}}**, **{{HARDWARE_CONSTRAINT}}**)
- prefer small, auditable changes
- distinguish between smoke / dev / production runs

## Read order

Always read these before proposing changes:

1. `AGENTS.md`
2. `docs/COMPETITION.md`
3. `docs/DATASET.md`
4. `docs/METRIC.md`
5. `docs/EXPERIMENTS.md`
6. `docs/AUTOMATION.md`

If the task is baseline-related, also read:
- `docs/BASELINE.md`
- `docs/SOLUTION.md`
- `docs/SUBMISSION.md`

## Working rules

- one primary hypothesis per experiment
- grouped / time-aware validation only (never random KFold on data with leakage units)
- no leakage across the same source unit (e.g., same soundscape, same patient, same session)
- code goes in `src/`
- experiments are registered before being run
- every completed run writes a machine-readable result file (= `result.json` or `summary.json`)
- when in doubt about a destructive or production-level action, **ask** before executing

## Standard loop

1. register experiment (= `/register_experiment`)
2. run training / validation (= `/run_experiment`)
3. save metrics and artifacts (= `result.json` + ckpt + log)
4. analyze result (= `/analyze_experiment`)
5. propose next experiment (= `/propose_next_experiment`)

## Run-ownership convention (recommended)

- **smoke runs / quick tests** (e.g., 1 epoch, 100 samples): Claude can launch
- **production runs** (long training, full folds, expensive jobs): user launches; Claude prepares the command and verifies after the run
- save this convention to memory once observed in this repo, so it persists

## Common patterns to memorize

- `default.yaml direct edit` policy (= some repos prefer editing default.yaml in-place rather than creating per-experiment yamls; check repo style)
- `4-fold canonical split` via a fold column in `train_df.csv`
- `OOF preds` saved per fold for ensemble / calibration
- `submission.csv` column order matches `sample_submission.csv` exactly
