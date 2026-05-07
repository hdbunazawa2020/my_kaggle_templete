# REPO_STRUCTURE.md

## Directory layout
```
.
├── CLAUDE.md             # Claude root instructions
├── AGENTS.md             # repo discipline + slash commands
├── README.md             # human-facing setup / env vars
├── .claude/
│   ├── commands/         # /register_experiment, /run_experiment, ...
│   ├── skills/           # auto-invoked when relevant
│   ├── settings.json     # safe shared permissions
│   └── settings.local.json   # per-user, git-ignored
├── docs/                 # COMPETITION / METRIC / DATASET / CV / ...
├── configs/              # global Hydra config + experiments/<exp_id>/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── utils/
│   └── scripts/<id>_<name>/
├── data/
│   ├── raw/              # untouched competition files
│   ├── processed/<ver>/
│   └── external/
├── experiments/<exp_id>/ # config.yaml, notes.md, result.json, log, model/, oof/
├── notebooks/            # 0xx EDA / 1xx classical / 2xx-3xx NN / 9xx ensemble
├── public_notebooks/     # downloaded public NBs (raw + annotated)
└── outputs/              # ad-hoc analysis output
```

## Numbering convention
- 0xx: data preprocessing
- 1xx: classical ML (= GBDT etc.)
- 2xx: NN training
- 3xx: NN inference / submission
- 9xx: ensemble

Same numbering scheme is used inside `notebooks/` and `src/scripts/`.

## Where Claude writes vs reads
- writes: `src/`, `experiments/<exp_id>/`, `notebooks/<draft>/`, `docs/` (when instructed)
- reads: everything
- never modifies: `data/raw/`, `.git/`
