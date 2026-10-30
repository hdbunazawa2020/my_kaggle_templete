# POSTMORTEM_TEMPLATE.md — コンペ終了後の振り返りの型

> RSNA Knee 2026 で実際に使い、**機能することが確認できた**振り返りの構造。
> そのままコピーして `docs/COMPETITION_RETROSPECTIVE.md` と `docs/TOP_SOLUTIONS.md` を作る。
>
> 🎯 目的は「感想文を書くこと」ではなく、**次回に再利用できる形で知見を固定すること**。

---

## 🔑 最重要の原則

> **Private LB を見た後の後知恵と、締切前に実際に持っていた情報を必ず区別する。**

これを守らないと振り返りが結果論になり、次回に持ち越せるものが何も残らない。

---

# Part A — 自分たちの振り返り（`docs/COMPETITION_RETROSPECTIVE.md`）

## A1. 最終結果
- 順位（暫定/確定を明示）、Public / Private、順位変動
- ⚠ leaderboard は**終了後も再同期されることがある**（失格処理等）。取得日を書く。

## A2. 提出の Public–Private 監査
全提出を API で取得し、次を定量化する:
- Public と Private の相関・回帰式（= 圧縮率）
- **構成カテゴリ別**の落ち幅（例: 自前のみ vs 公開物を混ぜたもの）
- **時系列**の落ち幅の変化（終盤ほど過学習していないか）
- **同一 Public 表示値の中での Private のばらつき**（表示桁の丸めに隠れた差）

## A3. タイムライン — 主要な転換点だけ
全ログではなく「方針が変わった瞬間」だけ。各行に根拠文書へのリンク。

## A4. 効いたこと
**仮説 → 根拠 → CV/Public/Private → 教訓** の形式で書く。
「効いた」だけでなく **なぜ効いたと言えるのか** を必ず併記する。

## A5. 効かなかったこと
**なぜ期待したか → 何が反証したか** の形式。
⚠ **ここが最も価値が高い**。失敗は成功より再利用性が高いのに、放置されやすい。
- CLOSE した軸は、**検証した条件の範囲**も書く（「この条件では駄目だった」であって「無効」ではない）

## A6. 検証に関する教訓
独立したセクションとして立てる。最低 5 個。
→ 再利用可能なものは `docs/VALIDATION_PITFALLS.md` に昇格させる。

## A7. 意思決定の監査（後知恵を分離する）

| 締切前の判断 | その時点の evidence | Private の結果 | 判断は正当だったか |
|---|---|---|---|

- **良い判断 ≠ 幸運な結果 / 悪い結果 ≠ 悪い判断**
- 結果が良くても根拠が無かったものは「運」と明記する（再現できないので）

## A8. 事後検証（任意だが強く推奨）
上位解法から**仮説を立て、実際に実験で確かめる**。
- 自分のパイプラインの **1 箇所だけ**差し替え、他を完全固定する
- 締切後も提出が採点されることがある（= 一番安全な実験場）
- ⚠ ライセンス確認。再配布しない

---

# Part B — 上位解法のレビュー（`docs/TOP_SOLUTIONS.md` + `docs/top_solutions/`）

## 個別ファイルの書式（`docs/top_solutions/<rank>_<team>.md`）

```markdown
---
title:
author_team:
provisional_rank:
final_rank:
public_score:
private_score:
source_url:
source_type: writeup | discussion | code
reviewed_at:
---

## Result
## Labels / supervision
## Validation
## Input / preprocessing
## Model
## Training
## Ensemble
## Inference
## What did not work
## Claimed key gains
| 主張 | Evidence quality |   ← controlled ablation / CV観察 / Public LB観察 / Private結果 / 著者の推測
## Comparison with our solution
  - Same / Different / We tried and rejected / We did not try / Potentially important missed idea
## Source
```

> ⭐ **`Evidence quality` 列が肝**。「効いた」という主張の裏付けの強さを必ず分類する。
> 著者の推測と controlled ablation を同じ重みで扱うと、次回の投資判断を誤る。

## 横断ファイル（`docs/TOP_SOLUTIONS.md`）に書くもの

1. **比較表**（rank / team / score / labels / backbone / input / external / ensemble / key differentiator）
2. **Common patterns** — 複数チームが独立に収束した手法（= 最も信頼できる）
3. **Ideas unique to one or two solutions** — 当たれば大きいが再現性は不明
4. **我々も試したこと** / **我々が見落としたこと** / **正当な理由で却下したこと**
5. **Public-only の小技 vs CV でも裏付けのある改善**の区別
6. **Gap analysis** — 「何が自分と上位を分けたか」を候補ごとに evidence strong / moderate / weak で評価
7. **Actionable lessons** — Keep / Stop / Start を各 5 個程度 + **次回 1 週間以内にやること**

## 🔍 調査のルール

- Discussion の主張を事実として扱わない。**writeup > 著者の discussion > 第三者のコメント**
- 順位・スコア・手法は**一次情報（writeup 本文）で確認**する
- **Public だけの改善を Private の改善と同一視しない**
- 要約する。**丸ごとコピーしない**。URL は必ず残す
- 同じ解法の重複記事はマージする
- 本文に無い数値を**推測で補わない**（`unknown` と書く）

---

## 📌 やってみて分かった運用上の注意

- **writeup は終了後しばらく増え続ける**。1 回取得して終わりにせず、数日後に再取得する
  （RSNA では 1 週間後に 31 件増え、その中に **6 位・9 位**が含まれていた）
- **leaderboard の順位は再同期で変わる**。ファイル名を順位に固定すると後で崩れる
- **自分たちの結論と矛盾する上位解法**こそ丁寧に読む（= 自分の前提が間違っている可能性）
