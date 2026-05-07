---
name: experiment-orchestrator
description: Coordinate the full one-hypothesis experiment loop in a Kaggle repo.
---

# Experiment Orchestrator

Use this skill when:
- starting a new experiment
- turning a hypothesis into a registered run
- choosing what to do after a run completes

## Workflow

### Step 1: Read repo context
Read:
- `CLAUDE.md`
- `docs/AUTOMATION.md`
- `docs/EXPERIMENTS.md`
- `docs/METRIC.md`

Done when:
- constraints, repo conventions, and current state are understood

### Step 2: Register a single hypothesis
Create a config that changes one main axis only. Use `/register_experiment`.

Done when:
- config and notes exist under `experiments/<exp_id>/`

### Step 3: Run
Use the standardized experiment runner. For long / production runs, prepare the command and ask the user to launch.

Done when:
- `result.json` exists

### Step 4: Analyze
Judge:
- CV improvement (mean + per-fold spread)
- runtime
- leakage risk
- likely LB transfer (= apply known compression ratio for this repo)

Done when:
- verdict is keep / reject / revisit

### Step 5: Propose one next experiment
Choose the highest-priority next move. Output exactly one.

Done when:
- exactly one next experiment is stated, with expected delta and wall time

## Anti-patterns
- multi-axis experiments
- skipping registration
- running production-class jobs without user OK
