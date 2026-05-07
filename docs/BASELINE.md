# BASELINE.md

## Definition
A "baseline" here means the **smallest end-to-end pipeline** that produces a valid `submission.csv`.

It does **not** need to be competitive — it needs to be **trustworthy**.

## Required components
1. preprocessing → `data/processed/<ver>/`
2. fold assignment → `train_df.csv["fold"]` (canonical)
3. training → `experiments/<exp_id>/model/...`
4. OOF prediction → `experiments/<exp_id>/oof/...`
5. inference on test → produces `submission.csv`
6. CV reporting → `result.json`

## Baseline acceptance criteria
- [ ] runs end-to-end without manual fixes
- [ ] submission passes `submission_readiness` checks
- [ ] CV is computed honestly (= correct grouping, no leakage)
- [ ] runtime fits {{KAGGLE_RUNTIME_LIMIT}}
- [ ] reproducible from seed alone

## What the baseline must NOT do
- chase score before correctness
- introduce ensembling
- introduce TTA
- introduce post-processing
- mix multiple ideas

## Once baseline is valid
- record CV + LB
- record runtime breakdown
- this becomes the reference for every future experiment
