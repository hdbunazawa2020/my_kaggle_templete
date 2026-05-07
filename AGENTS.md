# AGENTS.md

> Template note: replace `{{COMPETITION_NAME}}`, `{{TARGET_METRIC}}` when bootstrapping.

## Repo priorities
- {{COMPETITION_NAME}}
- {{TARGET_METRIC}}
- leakage-resistant CV
- runtime-feasible submission ({{HARDWARE_CONSTRAINT}}, {{KAGGLE_RUNTIME_LIMIT}})

## Experiment discipline
- one main hypothesis per experiment
- record config (all hyper-params + seed + git commit)
- record metrics (CV mean + per-fold + per-class where relevant)
- record runtime (training wall time + inference per file)
- record conclusion (keep / reject / revisit)

## Preferred slash commands
- `/register_experiment` — create exp_id + config + notes
- `/run_experiment` — run a registered experiment
- `/analyze_experiment` — judge one completed result
- `/propose_next_experiment` — choose the next move
- `/run_baseline` — first valid baseline
- `/analyze_cv` — diagnose validation quality
- `/experiment_review` — pre/post review
- `/generate_submission` — build submission pipeline
- `/submission_readiness` — pre-submit safety check

## Skills (auto-invoked when relevant)
- `kaggle-experiment` — full disciplined workflow
- `cv-diagnosis` — CV-LB mismatch triage
- `eda-base` — high-value EDA tied to modeling decisions
- `experiment-orchestrator` — one-hypothesis loop
- `result-analyzer` — turn outputs into a verdict
- `submission-guard` — pre-submit format/runtime check

## Anti-patterns to refuse
- changing >1 axis per experiment without splitting them
- random KFold on data with natural grouping units
- threshold tuning as the primary optimization in ROC-AUC competitions
- silently mutating shared dataframes
- skipping `result.json` because "it's just a quick run"
