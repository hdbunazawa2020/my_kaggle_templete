# SUBMISSION.md

## Submission format
- file: `submission.csv`
- columns: must match `sample_submission.csv` **exactly**
- row order: must match `sample_submission.csv` **exactly**
- dtype: numeric only, no NaN / inf

## Hard rules
- always reindex on `row_id` (or equivalent) against `sample_submission.csv` before saving
- never assume class / target order — derive it from `sample_submission`
- always run a final `assert df.isna().sum().sum() == 0`

## Runtime budget
- limit: **{{KAGGLE_RUNTIME_LIMIT}}**
- hardware: **{{HARDWARE_CONSTRAINT}}**
- internet: **{{ALLOWED / DISALLOWED}}**
- model size limit: **{{}}**

## Pipeline shape (= typical Kaggle code competition)
```
1. detect Kaggle vs local env
2. load test inputs as Kaggle delivers them
3. load model artifacts from attached datasets
4. run inference (= match training-time preprocessing)
5. apply post-processing (= prior, calibration, TTA, ensemble)
6. align to sample_submission (= reindex)
7. write submission.csv
```

## Notebook submission specifics
- inference notebook uploaded as a Kaggle Notebook
- all model files attached as Kaggle Datasets
- all dependencies wheel-installed inside the notebook (= no internet at sub time)
- env detection should fall back gracefully when run locally

## Final submission checklist (= run before every Sub)
- [ ] `submission.csv` row count == `sample_submission.csv` row count
- [ ] all required columns present, in right order
- [ ] all numeric, no NaN / inf, no negative where forbidden
- [ ] runtime estimate ≤ limit on the actual test set size
- [ ] model artifacts attached (= dataset, NOT just local)
- [ ] code path identical between dev and submission run

---

## ⚠ Code Competition の事故と対策（RSNA Knee 2026 実戦）

実際に事故ってから入れた防御なので、**面倒でも外さないこと**。
実装は `src/scripts/300_submission/` にある。

### 1. 学習時の前処理仕様を kernel 側で取り違える 💣

**実例**: cache 仕様の引数を渡さずに kernel を生成し、学習と違う前処理
（枚数・範囲・厚み）で推論する kernel を提出 → **LB −0.005**。気づくまで時間を溶かした。

- ✅ **対策**: kernel 生成時に **学習済み arm の config から前処理仕様を自動導出**し、
  CLI 指定と食い違ったら**既定で hard fail**（逸脱は明示フラグでのみ許可）。
- ✅ **対策**: 未知の設定値は「黙って既定値を使う」のではなく**例外にする**。

### 2. 生成物を直接編集してしまう 💣

`.py` → `.ipynb` の生成物を直接編集し、**再生成を忘れて古いコードが Kaggle で 10 分走った**。

- ✅ **正は `.py`。`.ipynb` は生成物**。直したら必ず再生成する。

### 3. push してから落ちる（1 往復 10 分）💣

- ✅ **push 前にローカルで通す**。Kaggle と同じディレクトリ構造を手元に作って notebook を丸ごと実行する。
  ⚠ **symlink ではなく実体（ハードリンク）で作る** — `Path.glob` は symlink を辿らない。
- ✅ **トップレベルの前方参照チェック**を push 前に回す。
  `ast.parse` は通るのに実行時 `NameError` で落ちるパターンがある。

### 4. hidden test の実行時間 💣

**3 study の commit run が通っても、本番（数千 study）の timing は保証されない。**
時間ガードの設計ミスで「変更が無反映のまま提出された」失敗例が公開 writeup にもある。

- ✅ 1 サンプルあたりの処理時間を測り、**hidden test のサイズで外挿**してから提出する。
- ✅ 開発環境の実測値を Kaggle に外挿しない（ハードウェアが別物）。

### 5. 提出そのもの

- ⚠ Code Competition では **`kaggle competitions submit -f <file>` は使えない**。
  SDK の `CreateCodeSubmission`（kernel owner / slug / version）を使う。
- ⚠ **提出は枠を消費する**。`--yes` 相当の明示フラグが無い限り dry-run にする。
  **user の指示なしに実行しない。**
- 📝 **提出 kernel を作ったら、生成コマンドを必ず docs に記録する**（引数の復元作業が発生する）。
- 📊 **Public LB は桁が丸められていることがある**（RSNA は 3 桁）。
  同じ表示値の中に無視できない差が隠れている。表示上の同値で優劣を判断しない。
- 🕰 **締切後でも提出が採点されることがある**（RSNA は終了 7 日後の提出が採点された）。
  振り返りの追試に使える。ただし採点完了まで数時間かかることがある。
