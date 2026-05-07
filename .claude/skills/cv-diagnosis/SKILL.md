---
name: cv-diagnosis
description: Diagnose validation leakage, fold weakness, and likely CV-LB mismatch in Kaggle-style competitions.
---

# CV Diagnosis Skill

Use this skill when:
- CV seems too good to be true
- LB underperforms CV
- fold variance is large
- rare classes / categories behave inconsistently

## Workflow

### Step 1: Inspect split unit
Determine whether splitting is by:
- row / sample
- group (= soundscape / patient / session / user / site)
- time (= date / season)
- nested combinations of the above

Done when:
- the actual grouping unit is explicit and matches the test-time prediction unit

### Step 2: Check leakage channels
Look for:
- same source unit across folds
- adjacent windows / time steps split across folds
- meta-feature leakage (= site identity, recording device, hospital code)
- target encoding fit on full train rather than per-fold

Done when:
- leakage risks are listed with severity

### Step 3: Check class / target coverage
Inspect:
- positives per fold
- missing classes per fold
- unstable rare classes
- target distribution drift between folds

Done when:
- fold-level coverage risks are known

### Step 4: Check metric alignment
Confirm:
- local score is computed with the same logic as the official metric
- submission path and validation path use the same preprocessing

Done when:
- scoring path parity is verified

### Step 5: Recommend fixes
Rank fixes by impact:
1. fold redesign (= correct grouping unit)
2. domain alignment (= train↔test distribution match)
3. inference parity (= same preprocessing)
4. class balancing
5. loss / model changes

Done when:
- next action is obvious

## Guardrails
- never trust random splits on grouped data
- do not propose threshold tuning as the primary fix in ROC-AUC / probabilistic metrics
- prioritize structural validation issues over model complexity
