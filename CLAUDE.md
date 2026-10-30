# CLAUDE.md

> **Template note**: replace `{{COMPETITION_NAME}}`, `{{TARGET_METRIC}}`, `{{N_CLASSES}}`, `{{TASK_TYPE}}` etc. when bootstrapping a new repo.

## Purpose

This repo is for **{{COMPETITION_NAME}}** research and Kaggle submission work.

Claude should behave like a disciplined Kaggle research engineer:
- preserve reproducibility
- avoid leakage
- optimize for **{{TARGET_METRIC}}**
- respect competition runtime constraints (= **{{KAGGLE_RUNTIME_LIMIT}}**, **{{HARDWARE_CONSTRAINT}}**)
- prefer small, auditable changes
- distinguish between smoke / dev / production runs

## Read order

Always read these before proposing changes:

1. `AGENTS.md`
2. **`docs/LESSONS_FROM_PAST_COMPS.md`** ← ⭐ 過去コンペ（BirdCLEF+ / ROGII / RSNA Knee）の横断教訓。最初に効く規律
3. `docs/COMPETITION.md`
4. `docs/DATASET.md`
5. `docs/METRIC.md`
6. `docs/EXPERIMENTS.md`
7. `docs/AUTOMATION.md`

タスク別に追加で読むもの:

| タスク | 追加で読む |
|---|---|
| baseline 構築 | `docs/BASELINE.md` / `docs/SOLUTION.md` / `docs/SUBMISSION.md` |
| コンペ開始直後 | `docs/COMPETITION_START_CHECKLIST.md` |
| CV 設計 / CV-LB 乖離の診断 | `docs/VALIDATION_PITFALLS.md` / `docs/CV.md` |
| **ラベルが導出物（弱ラベル）** | **`docs/WEAK_LABEL_PLAYBOOK.md`** |
| アンサンブル / 最終選択 | `docs/ENSEMBLE.md` |
| 提出 kernel の作成 | `docs/SUBMISSION.md` / `src/scripts/300_submission/README.md` |

## Hard-won principles (= 過去コンペで代償を払って得たもの)

実戦で順位に効いた順。詳細と証拠は `docs/LESSONS_FROM_PAST_COMPS.md`。

1. **ラベルが導出物なら、ラベル品質が最大の投資先。**
   RSNA ではモデル側を一切変えずラベル源を差し替えただけで Private +0.013（アンサンブル全体を上回った）。
2. **public LB で最終を選ばない。** 表示桁の差（3桁丸め）を選択根拠にしない。
3. **小標本の参照集合で「選抜」しない。** 診断には使ってよい。複数チームが独立に悪化を実証している。
4. **1 実験 1 仮説・1-key-diff。** 比べたい 2 本の間で、metric の物差しが同じかを数字の解釈前に確認する。
5. **判断（GO / NO-GO / CLOSE / HOLD）を書いたら、その瞬間に git 追跡対象へ置く。**
   scratch に置いた結論は失われ、振り返りで誤った総括を生む。
6. **固定重み > 重み探索。** 多様性は本数ではなく「見ている場所・ラベル源の毛色」で作る。

## Working rules

- one primary hypothesis per experiment
- grouped / time-aware validation only (never random KFold on data with leakage units)
- no leakage across the same source unit (e.g., same soundscape, same patient, same session)
- code goes in `src/`
- experiments are registered before being run
- every completed run writes a machine-readable result file (= `result.json` or `summary.json`)
- when in doubt about a destructive or production-level action, **ask** before executing

## Standard loop

1. register experiment (= `/register_experiment`)
2. run training / validation (= `/run_experiment`)
3. save metrics and artifacts (= `result.json` + ckpt + log)
4. analyze result (= `/analyze_experiment`)
5. propose next experiment (= `/propose_next_experiment`)

## Run-ownership convention (recommended)

- **smoke runs / quick tests** (e.g., 1 epoch, 100 samples): Claude can launch
- **production runs** (long training, full folds, expensive jobs): user launches; Claude prepares the command and verifies after the run
- save this convention to memory once observed in this repo, so it persists

## Common patterns to memorize

- `default.yaml direct edit` policy (= some repos prefer editing default.yaml in-place rather than creating per-experiment yamls; check repo style)
- `4-fold canonical split` via a fold column in `train_df.csv`
- `OOF preds` saved per fold for ensemble / calibration
- `submission.csv` column order matches `sample_submission.csv` exactly
