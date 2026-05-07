# CLAUDE_ENV_README.md

## Why this file
Claude reads this to understand what executes where (= local dev vs remote server vs Kaggle).

## Environment matrix

| location | what runs | path convention | notes |
|---|---|---|---|
| local dev | code edit, smoke tests | `${PROJECT_ROOT}` | == `/workspace/study/<PROJECT>` on this server |
| remote training | full train / inference | same as local | Claude prepares; user launches |
| Kaggle notebook | submission | `/kaggle/working/`, `/kaggle/input/` | code path must detect both envs |

## Path resolution helper (= recommended)
```python
import os
from pathlib import Path

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if IS_KAGGLE:
    PROJECT_ROOT = Path("/kaggle/working")
    DATA_ROOT = Path("/kaggle/input")
else:
    PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
    DATA_ROOT = PROJECT_ROOT / "data"
```

## What the inference notebook MUST handle
- detect Kaggle vs local
- locate competition data under `INPUT / "<comp-slug>"`, falling back through known candidates
- locate model artifacts under attached datasets, NOT under hardcoded local paths
- gracefully fall back when running locally for testing

## Cache locations (= server-specific)
On this server:
- code lives under `${WORKSPACE_ROOT}/study/<PROJECT>/`
- caches (HF / wandb / torch / uv) live under `${CACHE_ROOT}/`
- `.venv` lives under the project directory

See top-level `README.md` for the full env-var setup.
