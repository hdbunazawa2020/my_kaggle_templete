# /propose_next_experiment

Choose the next experiment intelligently.

## Read first
- `CLAUDE.md`
- `docs/EXPERIMENTS.md`
- `docs/AUTOMATION.md`
- recent `result.json` files under `experiments/`

## Task
Review completed experiments and propose **exactly one** next experiment.

## Decision lens
- highest expected delta per unit of compute
- diversifies error structure (= helps ensemble)
- addresses known weakness (= rare class, domain gap, CV-LB mismatch)
- avoids stacking too many in-flight changes

## Output format
1. proposal (= 1 sentence hypothesis)
2. expected CV / LB delta
3. estimated wall time
4. files to create / change
5. why this beats the runner-up alternatives (= 2 lines)
