# SUBMISSION.md

## Submission format
- file: `submission.csv`
- columns: must match `sample_submission.csv` **exactly**
- row order: must match `sample_submission.csv` **exactly**
- dtype: numeric only, no NaN / inf

## Hard rules
- always reindex on `row_id` (or equivalent) against `sample_submission.csv` before saving
- never assume class / target order — derive it from `sample_submission`
- always run a final `assert df.isna().sum().sum() == 0`

## Runtime budget
- limit: **{{KAGGLE_RUNTIME_LIMIT}}**
- hardware: **{{HARDWARE_CONSTRAINT}}**
- internet: **{{ALLOWED / DISALLOWED}}**
- model size limit: **{{}}**

## Pipeline shape (= typical Kaggle code competition)
```
1. detect Kaggle vs local env
2. load test inputs as Kaggle delivers them
3. load model artifacts from attached datasets
4. run inference (= match training-time preprocessing)
5. apply post-processing (= prior, calibration, TTA, ensemble)
6. align to sample_submission (= reindex)
7. write submission.csv
```

## Notebook submission specifics
- inference notebook uploaded as a Kaggle Notebook
- all model files attached as Kaggle Datasets
- all dependencies wheel-installed inside the notebook (= no internet at sub time)
- env detection should fall back gracefully when run locally

## Final submission checklist (= run before every Sub)
- [ ] `submission.csv` row count == `sample_submission.csv` row count
- [ ] all required columns present, in right order
- [ ] all numeric, no NaN / inf, no negative where forbidden
- [ ] runtime estimate ≤ limit on the actual test set size
- [ ] model artifacts attached (= dataset, NOT just local)
- [ ] code path identical between dev and submission run
