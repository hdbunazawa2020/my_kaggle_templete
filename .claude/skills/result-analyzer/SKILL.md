---
name: result-analyzer
description: Convert experiment outputs into a decision and a next action.
---

# Result Analyzer

Use this skill when:
- a result file exists
- experiments need triage
- the user asks "what to do next?"

## Workflow

### Step 1: Read result
Read:
- `experiments/<exp_id>/result.json`
- `experiments/<exp_id>/notes.md`
- recent prior results for baseline comparison

Done when:
- score, runtime, and status are known

### Step 2: Compare to baseline
Compare:
- CV mean
- per-fold spread (= seed / fold stability)
- runtime
- implementation complexity

Done when:
- delta vs baseline is explicit

### Step 3: Estimate LB transfer
If a CV-LB compression ratio is known for this repo, apply it to estimate LB delta. Otherwise, mark "unknown — needs sub".

Done when:
- expected LB delta is stated or marked unknown

### Step 4: Classify
Verdict:
- **keep** (CV up + runtime OK + transfer plausible)
- **reject** (CV down or no signal)
- **revisit** (mixed signal; needs ablation)

Done when:
- one verdict is assigned

### Step 5: Propose one next action
Output exactly one next experiment OR one ablation OR one submission plan.
