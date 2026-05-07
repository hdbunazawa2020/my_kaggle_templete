# EXPERIMENTS.md

## Purpose
This file defines how experiments are tracked.

The objective is to prevent the usual Kaggle failure modes:
- changing too many things at once
- losing reproducibility
- forgetting what actually improved CV
- not knowing which ideas helped rare classes
- getting trapped in CV-LB mismatch without evidence

---

## Core rules

### Rule 1: One primary hypothesis per experiment
Each experiment tests one main idea.

Good: add PCEN / change fold grouping / switch BCE→focal.
Bad: new model + new aug + new folds + new sampling all at once.

### Rule 2: Always preserve a clean baseline
Never overwrite the baseline logic without keeping a recoverable reference.

### Rule 3: Record failure, not only success
A strong workflow remembers what did **not** work.

### Rule 4: Production runs are user-launched
Smoke runs (= 1 epoch, 100 rows) Claude can launch.
Full / production runs require user OK; Claude prepares the command.

---

## Experiment id convention
- format: `<script_id>_<exp_name>` (e.g., `202_train_exp042`)
- always sequential, never reused
- registered before running

## Required artifacts per experiment
- `experiments/<exp_id>/notes.md` — hypothesis, expected delta, success criteria
- `experiments/<exp_id>/config.yaml` — exact config snapshot
- `experiments/<exp_id>/result.json` — CV mean, per-fold, runtime, status
- `experiments/<exp_id>/<exp_id>.log` — full run log
- model ckpts under `experiments/<exp_id>/model/` (or symlinked)
- OOF preds under `experiments/<exp_id>/oof/` (per fold)

## Decision rules
- **keep**: CV mean ↑ by > {{noise floor}} AND per-fold spread did not increase
- **reject**: CV ↓ or no signal vs noise
- **revisit**: mixed signal — needs ablation or seed-stability run

## CV-LB compression tracking
After each LB-tested run, append to `docs/METRIC.md` compression table.
This lets future experiments be judged against expected LB transfer instead of raw CV.
