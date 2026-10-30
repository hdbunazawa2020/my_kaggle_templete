# COMPETITION_START_CHECKLIST.md

> 🎯 **目的**: 新しいコンペを始めた最初の 1 週間でやることを、**実行可能なチェックリスト**にしたもの。
> この template を clone した直後に、上から順に潰していく。
>
> 💡 **なぜ初週が重要か**: 上位解法 23 件を読んだ結論として、**勝敗を分ける要素の多くは序盤の設計で決まる**。
> 特に「ラベルの作り方」「評価の測り方」「公開物との距離」は、終盤から変更しても間に合わない。
>
> 出典: RSNA Knee Abnormality Detection 2026 / BirdCLEF+ 2026 の実戦と、上位解法の横断分析。

---

## 📅 Day 0 — repo を立ち上げる

- [ ] この template を clone して `${PROJECT_NAME}` を決めた（`README.md` の初期セットアップ手順）
- [ ] 環境変数（`WORKSPACE_ROOT` / `CACHE_ROOT` 等）を設定し、**cache を永続ストレージに書かない**構成にした
- [ ] `.venv` をローカル高速ストレージ側に作った
- [ ] `CLAUDE.md` の `{{...}}` placeholder を**実値に置き換えた**
      （`COMPETITION_NAME` / `TARGET_METRIC` / `KAGGLE_RUNTIME_LIMIT` / `HARDWARE_CONSTRAINT` 等）
- [ ] `AGENTS.md` の placeholder も置き換えた
- [ ] `docs/MEMORY_SEED.md` を見て、Claude の memory に初期知識を seed した
- [ ] 実験記録の置き場（`experiments/<exp_id>/`）と命名規則を確認した

> ⚠ **git 管理外の作業ディレクトリに成果物を貯めない**。
> RSNA では、git 未追跡のスクラッチ領域に**重要な検証記録が丸ごと埋もれ、危うく削除するところだった**。
> 「後で整理する」で置いた場所は、高確率で失われる。

---

## 📅 Day 1〜2 — ルールと指標を確定する

### コンペ仕様

- [ ] `docs/COMPETITION.md` を埋めた（Code Competition か / runtime 制約 / hardware / internet 可否 / 提出枠）
- [ ] **外部データ・事前学習モデルの許可範囲**を確認し、`docs/COMPETITION.md` に書いた
- [ ] ホストの Discussion を読み、**ラベル定義の公式見解**を確認した
      （「report に書いてあるか」ではなく「基準を満たすか」で判定せよ、等の指示が出ることがある）
- [ ] ⏱ **hidden test の規模と実行時間制約**を確認した（Code Competition なら最重要）

### 指標

- [ ] `docs/METRIC.md` に**公式 metric の正確な定義**を書いた（擬似コードまで）
- [ ] 🔑 **primary metric を fold-wise（fold ごとに計算して平均）に固定**し、
      pooled/concat は secondary と明記した → `docs/VALIDATION_PITFALLS.md` §1
- [ ] fold 間のばらつきを必ず併記する運用にした
- [ ] Public LB の**表示桁数**を確認し、「表示同値の差を選択根拠にしない」と宣言した
- [ ] ランキング指標か較正指標か、**primary をどちらにするか**決めた

---

## 📅 Day 2〜3 — CV を設計する

- [ ] `docs/DATASET.md` を埋めた（prediction unit / grouping unit / 分布シフト / 既知の癖）
- [ ] `docs/CV.md` に **fold 戦略**を書いた（group / time / stratify の根拠つき）
- [ ] fold 割当を **1 箇所に正本化**した（全 script が同じ fold を使う）
- [ ] anti-leakage チェックを通した（同一ソース単位が複数 fold に出ていないか）
- [ ] 🔑 **小標本の高品質参照集合（専門家ラベル等）があるなら、
      「選抜には使わない・診断専用」と `docs/CV.md` に明記して宣言した**
      → `docs/VALIDATION_PITFALLS.md` §2
- [ ] 候補選択を伴う評価は **nested で測る**と決めた
- [ ] 🔑 **ラベル源を変える実験に備えて、評価ラベルを共通固定する仕組み**を用意した
      → `docs/VALIDATION_PITFALLS.md` §7

> 📌 site / scanner / device などの **メタデータが分布シフト源**になりうる場合は、
> この時点で「fold を切る軸」と「補正に使う軸」の両方の候補として控えておく。

---

## 📅 Day 3〜5 — baseline と提出経路を通す

- [ ] `docs/BASELINE.md` の定義に従い、**最小の end-to-end**（学習 → 推論 → `submission.csv`）を通した
- [ ] `submission.csv` の列順が `sample_submission.csv` と**完全一致**している
- [ ] ⏱ **提出経路を 1 回最後まで通した**（Code Competition なら kernel を push → 実行 → 提出）
      → `src/scripts/300_submission/` と `docs/SUBMISSION.md`
- [ ] hidden test での実行時間を見積もり、**余裕があることを確認**した
      （手元の少数サンプルでの commit 実行時間は、本番の実行時間を保証しない）
- [ ] 実験レジストリの運用を **1 本目から**始めた（`notes.md` / `config.yaml` / `result.json`）
- [ ] 採用ゲート（閾値・勝ち数・独立 seed 再現）を**事前登録する運用**を始めた

> ⚠ Code Competition では「**提出して初めて分かる失敗**」がある。
> 実行時間超過・環境差・出力形式の不一致は、**初週に 1 回通しておく**ことでしか潰せない。

---

## 📅 Day 5〜7 — 勝ち筋の仮説を立てる

### 公開物との距離を決める

- [ ] 公開 notebook / 公開ラベル / 公開 weight の有無を調べた
- [ ] Public LB に**同一スコアの塊**ができていないか確認した
      （= 公開 notebook をそのまま回した集団。コピーすると「その順位帯で頭打ち」になる）
- [ ] 🔑 公開ベースラインを使うなら、**「外付けブレンド」ではなく「自社ラベル / アーキを足す土台」**として
      扱う設計にした → `docs/ENSEMBLE.md` §5
- [ ] 取り込む部品は「**なぜ要るのかを実験で再導出**できたものだけ」にすると決めた

### ラベルの作り方を決める（弱ラベル / 擬似ラベル系なら最優先）

- [ ] ラベルが**与えられているか / 自分で作る必要があるか**を判定した
- [ ] 🔑 **自分で作る必要があるなら、ラベル生成の実現可能性を最優先で検証した**
      （API 費用・レート制限・prompt 設計・1 件あたり時間の当たり）
      → `docs/WEAK_LABEL_PLAYBOOK.md`
- [ ] 費用が発生する生成は、**少数サンプルで見積りを取ってから**本人承認を得る運用にした

> 💡 RSNA の最大の教訓: **学習レシピを完全固定してラベル源だけを差し替えた単体モデルが、
> 5 モデルのアンサンブルを Private で大きく上回った**。
> ラベルの質は、モデル側の工夫より桁違いに効きうる。**終盤から始めても間に合わない**。

### 入力設計

- [ ] 入力の crop / 正規化を、ピクセル単位ではなく **物理単位（mm・秒・Hz 等）**で設計した
- [ ] 前処理のメタデータ（plane / 機種 / 録音条件等）を、**ファイル名や series 名ではなく
      実際のヘッダ情報から**再構築した（名前は嘘をついていることがある）

### 分布シフト

- [ ] site / device / 機種などの**系統的バイアス**の有無を確認した
- [ ] （該当するなら）**学習時のみの補正**で吸収する設計を検討した
      ＝ 推論時には除去する前提の、群ごとのオフセット項

---

## 🏁 初週の終わりに宣言しておくこと（docs に書く）

以下は**後から決めると、既に見た数字に引きずられて決められなくなる**。初週に文章で宣言する。

| 宣言 | 書く場所 |
|---|---|
| primary metric は fold-wise / pooled のどちらか | `docs/METRIC.md` |
| 小標本の参照集合は選抜に使わない | `docs/CV.md` |
| 候補選択を含む評価は nested で測る | `docs/VALIDATION_PITFALLS.md` を参照する旨 |
| Public LB の表示同値の差で選ばない | `docs/METRIC.md` |
| 採用ゲートの閾値と勝ち数 | `docs/EXPERIMENTS.md` |
| 最終提出 2 枠の選び方（堅い構成 + 毛色の違う 1 つ） | `docs/SUBMISSION.md` |
| production run は誰が起動するか | `CLAUDE.md`（run-ownership） |

---

## ❌ 初週にやってはいけないこと

- ❌ **指標の定義を確認せずにモデルを回し始める**（最も多い失敗。測り方を間違えると全部やり直し）
- ❌ 公開 notebook をコピーして回し、それを基準線だと思い込む
- ❌ 小標本の参照集合でハイパラ / 重みを選ぶ
- ❌ 1 実験で複数の軸を同時に変える
- ❌ 「後で整理する」で git 管理外に成果物を置く
- ❌ 提出経路を終盤まで通さない（Code Competition では致命傷になりうる）

---

## 関連文書

- `docs/VALIDATION_PITFALLS.md` — 検証設計の落とし穴（最優先で読む）
- `docs/WEAK_LABEL_PLAYBOOK.md` — 弱ラベル / 擬似ラベルコンペの戦い方
- `docs/ENSEMBLE.md` — アンサンブルの規律
- `docs/LESSONS_FROM_PAST_COMPS.md` — 過去コンペ横断の規律
- `docs/EXPERIMENTS.md` — 実験の登録・判定ルール
- `docs/SUBMISSION.md` — 提出パイプラインと最終選択
