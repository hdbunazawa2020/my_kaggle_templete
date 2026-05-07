# /run_experiment

Run one registered experiment.

## Read first
- `CLAUDE.md`
- `docs/AUTOMATION.md`
- `docs/CV.md`
- `docs/METRIC.md`

## Task
Use the registered experiment config to run training / validation.

## Requirements
- do not silently change config
- write `result.json` (CV mean, per-fold, per-class if relevant, runtime)
- report runtime
- report CV
- state leakage risk
- if production-class run (long, expensive): prepare command + ask user to launch (= Claude does smoke runs only)

## Output format
1. command executed (= verbatim)
2. wall time
3. CV summary
4. result.json path
5. anomalies / warnings
