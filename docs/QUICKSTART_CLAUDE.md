# QUICKSTART_CLAUDE.md

## When you (Claude) start a fresh task in this repo

### 1. Read order (= every time)
1. `CLAUDE.md`
2. `AGENTS.md`
3. `docs/COMPETITION.md`
4. `docs/DATASET.md`
5. `docs/METRIC.md`
6. `docs/EXPERIMENTS.md`
7. `docs/AUTOMATION.md`

### 2. Recall memory
Check `~/.claude/projects/<this-repo>/memory/MEMORY.md`. It carries cross-conversation context (= run-ownership policy, default-yaml edit policy, repo-specific anti-patterns, current best LB, in-flight experiments).

### 3. Decide your role
- **smoke run / quick check** → you can run it
- **production run / long job** → prepare the command, ask the user to launch
- **destructive / shared-state / external-API action** → confirm before executing

### 4. Pick the right slash command
- new idea → `/register_experiment` → `/run_experiment` → `/analyze_experiment` → `/propose_next_experiment`
- before sub → `/submission_readiness` → `/generate_submission`
- CV-LB drift → `/analyze_cv`

### 5. End-of-turn discipline
- update relevant docs (= METRIC.md compression table, SOLUTION.md best-LB)
- update memory if a non-obvious lesson was learned
- mark TodoWrite items completed immediately

## When the user asks an exploratory question
- 2–3 sentences with a recommendation + main tradeoff
- present as redirectable, not decided
- do not implement until the user agrees

## When the user gives a terse instruction
- if context is unambiguous, act
- if context is ambiguous, ask a single tight clarifying question rather than guessing
