---
name: kaggle-experiment
description: Run a disciplined Kaggle experiment workflow for any research repo using this template.
---

# Kaggle Experiment Skill

Use this skill when:
- implementing a baseline
- improving a model in a controlled way
- planning the next experiment
- reviewing whether an experiment is submission-feasible

## Workflow

### Step 1: Read project context
Read:
- `CLAUDE.md`
- `docs/COMPETITION.md`
- `docs/DATASET.md`
- `docs/METRIC.md`
- `docs/EXPERIMENTS.md`

Done when:
- task structure is clear
- metric implications are clear
- runtime / hardware constraints are clear

### Step 2: Identify one main hypothesis
Examples:
- add an auxiliary data source
- change fold grouping
- switch loss function
- replace a feature transform

Done when:
- exactly one primary hypothesis is stated

### Step 3: Define success criteria
State:
- target metric effect (= expected CV delta)
- expected rare-class effect
- expected runtime effect
- expected submission impact (= LB delta if known compression ratio applies)

Done when:
- experiment can be judged keep / reject / unclear by predefined rules

### Step 4: Implement minimally
Prefer:
- edits in `src/`
- small code deltas
- configurable parameters
- no unrelated refactors

Done when:
- experiment can run reproducibly

### Step 5: Validate
Check:
- CV procedure
- leakage risk
- runtime feasibility
- output correctness

Done when:
- validation plan is explicit

### Step 6: Log outcome
Write:
- what changed
- what improved
- what failed
- next recommended experiment

Done when:
- the experiment result can be compared against prior work

## Guardrails
- do not optimize multiple large axes at once
- do not ignore submission-side hardware / runtime constraints
- do not trust random splits on grouped data
- do not use thresholding as the main optimization in ranking / probabilistic metrics
