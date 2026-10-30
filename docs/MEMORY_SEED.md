# MEMORY_SEED.md

> Seed entries to copy into `~/.claude/projects/<this-repo>/memory/` on first run, so Claude inherits cross-conversation conventions.
> Adapt to the user's profile and the new competition.

---

## Seed 1: user profile (= type=user)
File: `user_profile.md`
```
---
name: User profile
description: Background, role, language preference of the user
type: user
---

{{Toyota の Kaggle 研究者。 日本語で応答する}}
```

## Seed 2: run ownership (= type=feedback)
File: `feedback_production_run_owner.md`
```
---
name: Production run ownership
description: Smoke runs Claude can launch; production runs are user-launched
type: feedback
---

smoke runs (= 1 epoch, 100 samples) は Claude が起動して良い。
production runs (= 長時間 train、 全 fold inference 等) は user が起動する。

Why: 計算リソースは共有、 暴発を避けるため明示同意を入れる
How to apply: 「これは production」 と判断したら command を準備して user に投げる
```

## Seed 3: default.yaml direct edit policy (= type=feedback、 repo-specific)
File: `feedback_default_yaml_edit_workflow.md`
```
---
name: default.yaml direct edit workflow
description: Per-experiment yaml は作らず default.yaml を毎回書き換えて起動する運用
type: feedback
---

experiment 別の yaml ファイル (= exp042.yaml 等) は作らない。
default.yaml を直接編集して bash <script>.sh で起動。

Why: Hydra ++override の struct mode 問題で意図せず default に fallback したことがある
How to apply: 新 experiment の config を求められたら default.yaml の該当部分を edit する
```

## Seed 4: Kaggle runtime budget (= type=project)
File: `project_kaggle_runtime_limit.md`
```
---
name: Kaggle runtime limit
description: 推論時間制限の前提
type: project
---

このコンペは {{XX min}} 制限。
推論時間予算を立てる時の前提。

Why: 公開 NB が 9h 想定で書かれていたら危険
How to apply: 提出前に runtime estimate を必ず計算
```

## Seed 5: docs reference (= type=reference)
File: `reference_docs.md`
```
---
name: Authoritative docs in this repo
description: 必読 docs と古い doc の整理
type: reference
---

必読: docs/COMPETITION.md, METRIC.md, DATASET.md, EXPERIMENTS.md, AUTOMATION.md
それ以外の docs/ ファイルは時期固有 — 古くなりがちなので参照は注意
```

---

## 過去コンペで「書いておいてよかった」seed（RSNA Knee 2026 実績）

以下は実際に複数セッションにまたがって効いたもの。**新コンペでもほぼそのまま使える**。

## Seed 6: CV を信じる規律 (= type=feedback)
File: `feedback_trust_cv.md`
```
---
name: Trust CV, not small reference sets
description: 小標本の参照集合やpublic LBの表示差で選抜しない
type: feedback
---

primary は fold-wise に計算した CV。小標本の専門家ラベル集合（n が 2桁）は
**診断にのみ**使い、arm選抜・重み選抜には使わない。
public LB の表示桁内の差も選択根拠にしない。

Why: 複数の独立チームが「小標本にfitさせるとprivateで悪化する」と報告しており、
     我々のnested評価でも自由度を上げるほどouterが悪化することを確認した。
How to apply: 「どれを採用するか」を決める場面で、根拠が小標本かpublicの僅差なら止める。
```

## Seed 7: 実行環境の Python で検証する (= type=feedback)
File: `feedback_verify_with_project_interpreter.md`
```
---
name: Verify with the project interpreter
description: 構文・依存・notebookの検証は、実際に実行される環境のPythonで行う
type: feedback
---

「手元で import できた」を根拠にしない。**実際にそのコードが走る環境**の
interpreter / venv で検証する。

Why: 学習用venvと提出kernel環境、開発Podと実行Podでライブラリ構成が違う。
How to apply: 検証コマンドを書く時、どのvenvのpythonを使うかを毎回明示する。
```

## Seed 8: 検証範囲を超えた一般化をしない (= type=feedback)
File: `feedback_limit_generalization.md`
```
---
name: Limit generalization to tested conditions
description: 実測した条件の範囲を超えて断定しない
type: feedback
---

「この構成が最適」と書く前に、**何を何通り試した上での結論か**を併記する。
n=4 の比較で「一般に最適」とは書かない。

Why: 振り返り文書で過度な一般化をすると、次回に誤った前提を持ち込む。
How to apply: 結論には必ず「検証した範囲」の但し書きを付ける。
```

## Seed 9: 結論は追跡対象に置く (= type=feedback)
File: `feedback_conclusions_must_be_tracked.md`
```
---
name: Conclusions belong in tracked files
description: GO/NO-GO/CLOSE/HOLDの判断をscratchに置かない
type: feedback
---

判断を書いた瞬間に git 追跡対象へ移す。scratch / tmp は「消えてよいもの」専用。
HOLD にする時は「誰が・何を見たら・いつ判断するか」を同時に書く。

Why: RSNAで273GBの未追跡ディレクトリに唯一の検証記録（ある路線のCLOSE判定）が眠り、
     危うく削除するところだった。しかもその結論は振り返り文書に反映されておらず、
     「試していなかった」という誤った総括を書きかけた。
How to apply: 実験の結論を報告したら、それがどのtrackedファイルに載るかまで確認する。
```
