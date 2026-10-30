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
│                         #  + 過去コンペの蓄積（下の「知識 docs」参照）
├── configs/              # global Hydra config + experiments/<exp_id>/
├── infra/
│   └── k8s/              # GPUクラスタ(k8s + Kueue)で回す場合の投入スクリプト一式
│                         #  ⚠ 単一サーバー運用なら丸ごと不要
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── utils/
│   └── scripts/<id>_<name>/
│       └── 300_submission/   # Code Competition 提出パイプライン（事故防止機構つき）
├── data/
│   ├── raw/              # untouched competition files
│   ├── processed/<ver>/
│   └── external/
├── experiments/<exp_id>/ # config.yaml, notes.md, result.json, log, model/, oof/
├── notebooks/            # 0xx EDA / 1xx classical / 2xx-3xx NN / 9xx ensemble
├── public_notebooks/     # downloaded public NBs (raw + annotated)
└── outputs/              # ad-hoc analysis output
```

## 知識 docs（過去コンペの蓄積 / 新コンペでも中身はそのまま効く）

| file | 中身 | いつ読む |
|---|---|---|
| `docs/LESSONS_FROM_PAST_COMPS.md` | ⭐ 横断教訓（BirdCLEF+ / ROGII / RSNA Knee の実証つき） | **最初に 1 回** |
| `docs/COMPETITION_START_CHECKLIST.md` | 開始〜1週目にやること | コンペ開始時 |
| `docs/VALIDATION_PITFALLS.md` | 検証設計の落とし穴（数値つき） | CV 設計時 / CV-LB 乖離時 |
| `docs/WEAK_LABEL_PLAYBOOK.md` | 弱ラベル・導出ラベルコンペの戦い方 | ラベルが導出物なら必読 |
| `docs/ENSEMBLE.md` | アンサンブル規律（固定重み > 重み探索） | 終盤の構成決め |

> これらは **コンペ固有の穴埋めが不要**（= clone してすぐ効く）。
> 一方 `COMPETITION.md` / `METRIC.md` / `DATASET.md` / `CV.md` は **埋めないと機能しない雛形**。

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
