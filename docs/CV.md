# CV.md

## Validation strategy

- **Method**: {{GroupKFold / StratifiedGroupKFold / TimeSeriesSplit / custom}}
- **Grouping unit**: {{column or derived key — must match a real-world unit that test will respect}}
- **n_folds**: {{4 / 5 / 10}}
- **seed**: {{integer}}
- **Stratification target**: {{column}}

## Fold assignment policy
{{Where is the fold assignment stored? `train_df.csv["fold"]`? Derived deterministically per row? Store *one* canonical assignment so all scripts use the same folds.}}

## Why this method
{{Why this group / time choice? Which leakage risk does it close?}}

## Anti-leakage checks (= must hold)
- [ ] no source unit (= soundscape / patient / session) appears in >1 fold
- [ ] no temporal overlap if test is future
- [ ] no metadata leak (= site code, device id) across folds
- [ ] OOF concat aligns 1:1 with `train_df`

## Metric computation — ⚠ primary をどちらにするかを初週に決めて固定する

2 つの集計方法があり、**どちらが正しいかはコンペ構造で決まる**。両方を出し、**primary を 1 つ宣言する**。

| 集計 | 計算 | 向いている構造 | failure mode |
|---|---|---|---|
| **concat（pooled）** | 全 fold の OOF を縦に積んでから 1 回だけ metric を計算 | test が**1 つのプール**として採点され、fold 間の難易度・陽性率が揃っている | fold 間でスコアのスケールが違うと **artifact** が出る |
| **fold-wise mean** | fold ごとに metric を計算してから平均 | **macro 系**（target / class ごとに平均する）metric、fold 間に分布差がある | fold あたりの n が小さいと分散が大きい |

### 実例（どちらも実戦）

- 🐦 **BirdCLEF+ 2026**: concat のほうが LB simulator に近かった。
- 🦵 **RSNA Knee 2026**: concat を primary にしていたら、ある target が
  **見かけ上 −0.08793 の大悪化**に見えた。実際は **fold ごとのスコアのスケール差による artifact** で、
  真の劣化ではなかった。→ **fold-wise macro を primary、concat を secondary** に変更した。

### 決め方

1. まず**両方計算する**。
2. **target / class 別に、2 つの集計で符号が食い違うものが無いか**を見る。
   食い違いが出たら **fold-scale artifact を疑う**（concat を primary にしてはいけない兆候）。
3. 公式 metric が **macro 集計**（class ごとに計算して平均）なら、**fold-wise を primary にするほうが安全**。
4. 決めたら `docs/METRIC.md` に**宣言として書く**。
   ⚠ **後から変えると、それまでの全実験が比較不能になる。**

詳細な落とし穴は `docs/VALIDATION_PITFALLS.md`。

## Known CV-LB mismatch failure modes for this comp
- {{record observed ones here as you encounter them}}
