---
name: submission-guard
description: Verify that the submission pipeline is correct, numeric, and runtime-aware before Kaggle submission.
---

# Submission Guard Skill

Use this skill when:
- generating `submission.csv`
- reviewing inference code
- preparing for Kaggle notebook submission

## Workflow

### Step 1: Read submission requirements
Read:
- `docs/SUBMISSION.md`
- `docs/METRIC.md`

Done when:
- expected `row_id` and column rules are clear

### Step 2: Verify inference path
Confirm:
- input data load matches Kaggle delivery format
- preprocessing matches training-time preprocessing
- model loading works on the target environment (= CPU / GPU as required)
- prediction output shape is correct

Done when:
- data flow is explicit end-to-end

### Step 3: Verify output correctness
Check:
- file name is `submission.csv`
- all required columns exist and in the right order (= match `sample_submission.csv`)
- row order is valid (= reindex to sample_submission)
- predictions are numeric
- no NaN / inf

Done when:
- output format is trustworthy

### Step 4: Verify runtime
Estimate:
- time per file / per batch
- total time on the full test set
- bottlenecks (= feature extraction vs model forward vs IO)

Done when:
- submission feasibility is stated clearly vs {{KAGGLE_RUNTIME_LIMIT}}

### Step 5: Report blockers
If any blocker exists, classify:
- formatting
- runtime
- logic
- leakage / mismatch

Done when:
- the user can fix the highest-priority blocker first

## Guardrails
- prefer `sample_submission.csv` as the output template
- do not assume hardware not available at submission
- do not ignore `row_id` edge cases (= duplicate handling, ordering)
