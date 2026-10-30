# AGENTS.md

> Template note: replace `{{COMPETITION_NAME}}`, `{{TARGET_METRIC}}` when bootstrapping.

## Repo priorities
- {{COMPETITION_NAME}}
- {{TARGET_METRIC}}
- leakage-resistant CV
- runtime-feasible submission ({{HARDWARE_CONSTRAINT}}, {{KAGGLE_RUNTIME_LIMIT}})

## Experiment discipline
- one main hypothesis per experiment
- record config (all hyper-params + seed + git commit)
- record metrics (CV mean + per-fold + per-class where relevant)
- record runtime (training wall time + inference per file)
- record conclusion (keep / reject / revisit)

## Preferred slash commands
- `/register_experiment` — create exp_id + config + notes
- `/run_experiment` — run a registered experiment
- `/analyze_experiment` — judge one completed result
- `/propose_next_experiment` — choose the next move
- `/run_baseline` — first valid baseline
- `/analyze_cv` — diagnose validation quality
- `/experiment_review` — pre/post review
- `/generate_submission` — build submission pipeline
- `/submission_readiness` — pre-submit safety check

## Skills (auto-invoked when relevant)
- `kaggle-experiment` — full disciplined workflow
- `cv-diagnosis` — CV-LB mismatch triage
- `eda-base` — high-value EDA tied to modeling decisions
- `experiment-orchestrator` — one-hypothesis loop
- `result-analyzer` — turn outputs into a verdict
- `submission-guard` — pre-submit format/runtime check

## Anti-patterns to refuse
- changing >1 axis per experiment without splitting them
- random KFold on data with natural grouping units
- threshold tuning as the primary optimization in ROC-AUC competitions
- silently mutating shared dataframes
- skipping `result.json` because "it's just a quick run"

### 実戦で代償を払った anti-pattern（RSNA Knee 2026）

- ❌ **小標本の専門家ラベルで重み / checkpoint を選ぶ** — 複数チームが独立に private 悪化を実証
- ❌ **OOF を fold 横断で連結してから metric を計算して primary にする** — fold-scale artifact で
  大悪化に見える偽陽性が出る（`docs/VALIDATION_PITFALLS.md`）
- ❌ **ラベル源が違う arm どうしの CV を、評価ラベルを揃えずに比較する** — 物差しが違うので差分が無意味
- ❌ **結論（CLOSE / HOLD 判定）を git 未追跡の scratch に置いたまま次へ進む** — 273 GB の
  未追跡ディレクトリに唯一の検証記録が眠り、危うく削除するところだった
- ❌ **HOLD の条件を書かずに保留する** — 「誰が・何を見たら・いつ判断するか」を書けないなら CLOSE にする
- ❌ **提出 kernel を作って生成コマンドを記録しない** — 後から 8 個の引数を復元する羽目になる
- ❌ **生成物（.ipynb）を直接編集する** — 正は `.py`。生成し直す
- ❌ **open-weight VLM の失敗をもって「その路線は無効」と判定する** — frontier model で試すまでは未検証

## 追加 docs（過去コンペの蓄積）
- `docs/LESSONS_FROM_PAST_COMPS.md` — 横断教訓（最初に読む）
- `docs/WEAK_LABEL_PLAYBOOK.md` — 弱ラベル / 導出ラベルコンペの戦い方
- `docs/VALIDATION_PITFALLS.md` — 検証設計の落とし穴（数値つき）
- `docs/ENSEMBLE.md` — アンサンブル規律
- `docs/COMPETITION_START_CHECKLIST.md` — 開始〜1週目にやること
