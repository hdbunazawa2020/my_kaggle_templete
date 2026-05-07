# /run_baseline

Build or improve the first valid baseline.

## Always do this first
1. Read `CLAUDE.md`
2. Read `docs/COMPETITION.md`
3. Read `docs/DATASET.md`
4. Read `docs/METRIC.md`
5. Read `docs/BASELINE.md`

## Task
Given the current repo state:
- identify the smallest missing components for a valid baseline
- propose the exact files to create or modify
- implement code in `src/`
- keep submission feasible under {{HARDWARE_CONSTRAINT}} / {{KAGGLE_RUNTIME_LIMIT}}

## Output format
1. current repo gap analysis
2. implementation plan
3. files created or modified
4. validation plan (= CV method + sanity check)
5. runtime concerns

## Guardrails
- do not introduce leakage across natural grouping units
- do not assume hardware not available at submission time
- do not recommend threshold tuning as the main optimization
