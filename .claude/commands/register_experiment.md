# /register_experiment

Create a new experiment in a reproducible way.

## Read first
- `CLAUDE.md`
- `docs/EXPERIMENTS.md`
- `docs/AUTOMATION.md`

## Task
Given a hypothesis, create:
- experiment id (= sequential, e.g., `exp042` or `<script>_exp042`)
- `configs/experiments/<exp_id>.yaml` (or repo-specific config path)
- `experiments/<exp_id>/notes.md` with hypothesis + expected impact
- registration entry (= update progress log if the repo uses one)

## Requirements
- exactly one primary hypothesis
- clear experiment title
- explicit CV method (= which fold column / split)
- explicit runtime awareness (estimated train + inference time)
- explicit success criteria (= what CV delta would make this "keep")

## Output format
1. exp_id
2. files created
3. command to run it
4. expected wall time
