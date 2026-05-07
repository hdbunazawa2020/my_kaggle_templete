# /analyze_experiment

Analyze one completed experiment.

## Read first
- `CLAUDE.md`
- `docs/EXPERIMENTS.md`
- `docs/METRIC.md`
- `docs/CV.md`

## Task
Read `experiments/<exp_id>/result.json` and `notes.md`, then decide:
- **keep** (CV improved meaningfully + runtime acceptable)
- **reject** (no improvement or regression)
- **revisit** (mixed signal, needs ablation or seed-stability check)

## Output format
1. delta vs baseline (= CV mean + per-fold spread)
2. runtime delta
3. leakage / overfit risk
4. CV → LB transfer estimate (= apply known compression ratio if any)
5. verdict (keep / reject / revisit)
6. one concrete next experiment if "keep"
