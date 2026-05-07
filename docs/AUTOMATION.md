# AUTOMATION.md

## Repo automation conventions

### Script entry points
- All training / inference scripts live under `src/scripts/<script_id>_<name>/`.
- All configs live under `src/scripts/conf/<script_id>_<name>/`.
- Hydra is used for config composition (= `defaults:` in `config.yaml`).

### Run shells
- `bash <script_id>.sh <exp_name>` — wraps the python entry + standard logging.
- Logs go to `experiments/<exp_id>/<exp_id>.log`.
- Long runs use `nohup ... &` + `disown`; PID captured in the log.

### Default-yaml direct edit policy (= optional, repo-specific)
Some repos prefer to edit `default.yaml` in place rather than creating a per-experiment yaml each time. Decide once and document here.
- decision: {{direct edit / per-exp yaml}}
- reason: {{}}

### Smoke vs production
- smoke: `epochs=1`, `n_files=100`, `BATCH=4` — Claude can launch
- production: full epochs / folds — user launches

### Result file convention
Every completed run must write `result.json` with at minimum:
```json
{
  "exp_id": "...",
  "cv_mean": 0.0,
  "cv_per_fold": [0.0, 0.0, 0.0, 0.0],
  "runtime_sec": 0,
  "git_commit": "...",
  "status": "ok | failed | partial"
}
```

### Background jobs
- prefer `run_in_background=True` for any job > 60 s
- always capture PID + log path so the user can monitor
- Claude does not poll; Claude is notified when a backgrounded job completes

### Kaggle dataset upload
- maintained by `src/scripts/upload_*_to_kaggle.sh`
- `kaggle datasets version -p ... --dir-mode zip`
- ⚠ `version` REPLACES the latest version; combine multiple files into one upload to avoid losing previously uploaded artifacts
