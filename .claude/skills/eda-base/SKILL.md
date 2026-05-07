---
name: eda-base
description: Run high-value EDA tied to modeling decisions for Kaggle-style competitions. Replace with domain-specific EDA skill (audio / vision / tabular) once the modality is known.
---

# Base EDA Skill

Use this skill when:
- starting a new repo
- preparing the first baseline
- deciding folds or sampling strategy
- analyzing weak / rare classes
- diagnosing CV-LB mismatch

## Principle
Every EDA step must end in **a modeling decision**. Cosmetic plots are out of scope.

## Workflow

### Step 1: Read EDA intent
Read:
- `docs/EDA.md`
- `docs/DATASET.md`
- `docs/COMPETITION.md`

Done when:
- the goal of analysis is tied to a modeling decision

### Step 2: Analyze target distribution
Inspect:
- target counts (= class frequency, label cardinality)
- target counts in any auxiliary labeled set
- overlap between primary train and any pseudo / auxiliary labels

Done when:
- target imbalance and source coverage are visible

### Step 3: Analyze metadata structure
Inspect:
- categorical fields likely to drive grouping (= site, user, season, device)
- continuous fields with distribution drift train↔test
- missingness patterns

Done when:
- candidate grouping units for CV are identified
- likely covariate shift channels are known

### Step 4: Analyze test realism
Inspect:
- domain gap vs train (= noise, density, coverage, label quality)
- candidate leakage units
- distribution of inference-time features

Done when:
- the baseline training strategy is better informed

### Step 5: Recommend modeling decisions
Recommend:
- validation grouping (= which column to GroupKFold on)
- whether to start with a domain-restricted subset
- whether to filter low-quality data
- whether rare-class sampling is needed
- which augmentations match the test-time distribution

Done when:
- EDA leads to concrete modeling choices

## Guardrails
- avoid plots without modeling consequence
- always connect a finding to an actual decision
- never claim CV trustworthiness from a single fold
