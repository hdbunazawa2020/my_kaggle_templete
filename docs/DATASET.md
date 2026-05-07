# DATASET.md

> Document the data layout and any non-obvious facts that bite later.

## Files provided
| file | rows | cols | description |
|---|---|---|---|
| `train.csv` | | | |
| `test.csv` | | | |
| `sample_submission.csv` | | | |
| `{{train_audio/}}` | | | |
| `{{train_soundscapes/}}` | | | |

## Prediction unit
{{e.g., one row per 5-second window per file, one row per patient, etc.}}

## Target column(s)
- `{{target}}` — {{type, range, NaN policy}}

## Natural grouping units (= candidate CV groups)
- {{e.g., soundscape filename, patient_id, session_id, site_id}}

## Time / domain shift
- {{train vs test temporal split? site mismatch? device mismatch?}}

## Class / target distribution highlights
- {{long tail? rare classes < N positives? expected per-class AUC variance?}}

## Auxiliary / external data
- {{pseudo-labels, pretrained embeddings, public datasets}}

## Known data quirks
- {{empty files, label noise, duplicates, mislabels, etc. — collect these as you find them}}

## Local layout convention
```
data/
├── raw/              # untouched competition files
├── processed/<ver>/  # versioned preprocessed artifacts
└── external/         # external datasets
```
