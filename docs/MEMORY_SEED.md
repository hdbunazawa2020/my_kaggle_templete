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
