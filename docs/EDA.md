# EDA.md

## Purpose
EDA is not for plots — it is to make modeling decisions.

## Required EDA before baseline
1. **Target distribution** — class balance, label cardinality, rare classes
2. **Group structure** — what are the natural CV grouping units?
3. **Train↔test domain gap** — feature distribution shift, missingness diff, time gap
4. **Label noise / quality** — duplicate rows, contradictory labels, NaN target
5. **File / row-count sanity** — `len(test) == len(sample_submission)`?

## Each EDA notebook must end with
- a 1–3 line "decision" cell (= what this analysis changes about the modeling plan)
- otherwise it is cosmetic and should be removed

## Notebook conventions
- numbering: `0xx_eda_<topic>.ipynb`
- one topic per notebook
- save key tables to `data/processed/eda/` for reuse
