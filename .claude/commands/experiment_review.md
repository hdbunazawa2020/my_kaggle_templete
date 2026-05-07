# /experiment_review

Review an experiment plan or completed result.

## Read first
- `CLAUDE.md`
- `docs/EXPERIMENTS.md`
- `docs/METRIC.md`

## Task
Given an experiment plan or result:
- identify the main hypothesis
- determine whether too many variables changed
- estimate likely impact on {{TARGET_METRIC}}
- assess runtime impact
- recommend keep / reject / revisit

## Output format
1. hypothesis clarity (= 1 axis or many?)
2. experiment quality (= reproducible? confounds?)
3. interpretation of results (= signal vs noise vs leakage)
4. next-best experiment
