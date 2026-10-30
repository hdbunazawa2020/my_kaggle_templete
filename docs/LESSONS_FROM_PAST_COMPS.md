# LESSONS_FROM_PAST_COMPS.md

> 過去コンペで **実証した** ドメイン非依存の規律 / プロセス学びだけを集めたもの。
> モデル技術（mel / Perch / tracker / 特定backbone 等）はドメイン固有なので持ち込まない。**「どう戦うか」の話**。
>
> 出典（すべて実戦ログ、または公開解法の読み込み）:
> - **BirdCLEF+ 2026** — 我々 private 0.943 / 139位 + 公開上位解法 1st〜24th の統合
> - **ROGII Wellbore Geology Prediction 2026** — 我々の実戦ログ
> - **RSNA Knee Abnormality Detection 2026** — 我々 private 0.945 + 上位解法 23件の読み込み + 終了後の追試
>
> ⚠ **各項の「実例」は過去コンペの具体値**。眺めるだけでは効かない。
> **自分のコンペの数字に置き換えて初めて意味を持つ**。

---

## 0. 使い方 — 新コンペで最初に決める 4 つ

1. **この問題は本質的に何を推定しているのか**を 1 文で書く（§11）。
   書けないうちにモデルを回さない。→ `docs/SOLUTION.md`
2. **primary metric をどう測るか**を固定する（§2 / `docs/VALIDATION_PITFALLS.md`）。
   fold ごとに計算して平均するのか、OOF を連結してから計算するのか。grouping unit は何か。
   ⚠ **後から変えると、それまでの実験が全部比較不能になる**。
3. **ラベルがどう作られたか**を確認する（→ `docs/WEAK_LABEL_PLAYBOOK.md`）。
   弱ラベル / 導出ラベルなら、**ラベル品質の改善がモデル改善を上回る**ことが多い。
4. **最終選択の規律**を先に宣言する（§1）。public で選ばない / 小標本の参照集合で選ばない。

> 📌 経験則: 順位を動かすのは **(a) 入力表現 (b) ラベル品質 (c) validation 設計 (d) 最終選択**。
> backbone 探索とアンサンブル本数は、この 4 つが固まった後の **仕上げ** でしかない。

---

## 1. 最終 submission の選択規律（最重要・ほぼ全上位で一致）

- **public LB だけで最終を選ぶな**。public を追った構成は private で溶ける。
  - BirdCLEF 実証: 最高 public 0.950 → private 0.943。地味な robust 版が private 0.945。
    **1st は「public-best + private-best」を 2 枠で hedge**。**11th / 5th / 我々は higher-private を選べず取り逃した**。
- **最終 2 枠は hedge**: `最も堅い CV の構成 1 つ + 毛色の違う 1 つ`。両方を攻めない。
- **実例: RSNA Knee 2026**: final は **primary CV が最良の構成** + **毛色の違う 1 つ**。public の小数点差で選ばない。
  （⚠ 当初は「OOF-concat が最良の構成」と書いていたが、**concat は fold-scale artifact を生むと判明し撤回**。下記 §2 と `docs/VALIDATION_PITFALLS.md` を参照）

## 1-b. 公開 Notebook との距離（★ ぶなぴーの実戦教訓 / 2026-09-05 追加）

> **公開 NB に引きずられると 95% のケースで失敗する。多くは LB に過学習しているため。**

- **RSNA Knee での実測**: Public LB の **0.9360 に 550 teams が同一スコアで固まっている**。
  = **公開 NB をそのまま回した集団**。銅ライン（~308 位 / 0.9360）と同義。
  → **コピーして回すと「銅で頭打ち」かつ「private でも全員同じ方向に動くので差がつかない」**。
- **正しい使い方 = #14**「**なぜその部品が要るのかを実験で再導出し、理由が分かった部品だけ取り込む**」。
- ⚠ **後処理・閾値・prior は特に危険**（public に fit している可能性が高い。#1 / #6）。

## 2. validation 設計が LB probing に勝つ（★ 1st place の核）

- **「local val は当てにならない」は作り方の問題**。
  - BirdCLEF 2nd は val 全滅で LB 頼み（相関 ~0.2）だったが、
    **1st は domain-tailored validation を設計し、LB 再調整より良い学習 param を val だけで決めた**。
  - 1st の val = 2 split の平均: ① **unseen-site split**（支配的 site を全部 train に寄せ、残りで汎化を測る）
    ② **coverage split**（各 class を最低 1 つ train に残しつつ最大限 val に回す）。
- **実例: RSNA Knee 2026（最重要）**:
  - 基本 = **MultilabelStratifiedKFold by `StudyInstanceUID`**（grouping unit は実測で確かめる。
    `patient_id` を想定していたが **`PatientID` は study と 1:1** だった）。
  - ⚠⚠ **【撤回された計画】** 当初は「**per-fold 平均ではなく OOF-concat をスコアにする**」と決めていた。
    → **実験で覆った**。concat は fold 間の予測スケール差で artifact を生み、ある target が
    見かけ上 **−0.08793** の大悪化に見えた（実際は劣化していなかった）。
    **最終的に fold-wise macro を primary、concat を secondary に変更**した。
    📌 **横展開**: どちらを primary にするかは**コンペ構造で決まる**。両方計算し、
    **target 別に符号が食い違わないか**を見てから決め、決めたら固定する（`docs/VALIDATION_PITFALLS.md` §1）。
  - さらに **hidden test を模した split を自作**: **leave-institution-out / leave-manufacturer-out / leave-period-out**。
    DICOM メタから施設・機種・撮影年が取れるなら必ず作る。**これが LB probing の代わりになり、提出枠を節約する**。
  - CV-LB の比（compression ratio）を `docs/METRIC.md` の表で追い、**乖離は最優先で分布 / leak を疑う**。

## 3. diversity ensemble + 賢い fusion（「同じ部品でも組み立てで +0.01」）

- BirdCLEF 上位（1st/2nd/3rd/5th/18th）は口を揃えて **diversity が最大の伸び**。
  **入力表現 × backbone × recipe を意図的にバラし、低相関ペアを高 weight**。
  **18th / 5th は我々と同じ OSS 部品で gold**（差は多様性 + fusion）。
  **ただし ensemble size ≠ quality**（18th: 4-model < 3-model）。
- **fusion は metric 依存**:
  - **AUC / 順位 metric** → rank fusion（per-class percentile 平均）が効く（calibration mismatch に強い）。
  - **log loss / RMSE など値が効く metric** → rank 平均は不適。**weighted average / stacking / calibration** を使う。
- **実例: RSNA Knee 2026（✅ metric = macro AUC 確定）**: **BirdCLEF と同じ rank fusion がそのまま有効**。
  per-target で percentile へ変換して平均すれば、モデル間の出力スケール差に強い。
  混ぜる軸: **CNN（convnext / effnet）× ViT 系 × 入力解像度 × series の選び方 × 2D slice vs 2.5D vs 3D 集約**。
  fusion 重みは **OOF で最適化**する。**安いモデルを多数**（5th: 推論を軽くして数で variance を下げる）
  — ⚠ ただし本コンペは **Efficiency Track があるため「多数」のコストが賞金に直結する**（`docs/METRIC.md`）。

## 4. 前年 / 公開の強資産を再利用（scratch より速く強い）

- 4th は **前年 2nd の pretrained backbone 流用**で大伸長、**from-scratch pretrain は逆に負け**と明言。
- **実例: RSNA Knee 2026**: RSNA は毎年コンペを開いている（RSNA 2022 頸椎骨折 / 2023 腹部外傷 / 2024 腰椎 …）。
  **過去 RSNA コンペの上位解法とその weights が最大の資産**。公開 starter・timm の医用 pretrain・
  類似解剖部位の解法を **土台に**する。車輪の再発明をしない。

## 5. 入力表現の調整は高レバレッジ

- BirdCLEF は **mel パラメータ調整だけで単体 0.92→0.94**。「入力表現」の伸び代を取り逃すな（我々の最大の取り逃し）。
- **実例: RSNA Knee 2026**: **windowing（center/width）、解像度、crop 範囲（膝関節の ROI 切り出し）、
  slice 選択・枚数、左右の正規化、PixelSpacing による物理サイズ揃え**。
  baseline 段階で本気で sweep する。**backbone を変えるより効くことが多い**。

## 6. 分布シフトを疑う（train ↔ test）

- BirdCLEF は **train↔test の site 分布が激ズレ**（LB probing で判明）。train 分布で fit した **blind prior は中立〜有害**。
  - ただし **3rd は shrinkage-to-mean（w=N/(N+K)）の prior を KEEP して 3 位**
    → **prior は「blind は危険、shrinkage / evidence 付きなら可」。必ず A/B**。
- **実例: RSNA Knee 2026**: **施設・機種・撮影年・年齢性別・陽性率** の train↔test 差を EDA で最初に確認。
  test に無い領域へ強い prior / 後処理を当てない。**分布差を val split に再現する**（#2）。

## 7. 半教師 / self-training（該当すれば最大の engine、ただし作法が要る）

- **BirdCLEF 優勝の主 engine は self-training**（supervised 0.935 → Noisy Student 2iter 0.950）。
  鍵 = **target ドメインの data を「学習に inject」+「val に使う」両方**。
  - 作法: **pseudo の寄与を本物より小さく cap / labeled と pseudo を別サンプルで同 batch /
    power transform / 2 iter で頭打ち**。素の pseudo を強く足すと逆効果。
- **★ 実例: RSNA Knee 2026 — データ構造が BirdCLEF とほぼ同型（2026-09-05、ぶなぴーの指摘で修正）**

  > ⚠ **これ以前の版で Perch v2 を Stage A に対応させていたのは誤り**だった。
  > Perch v2 は **foundation model の軸**であって、**弱ラベルの軸**ではない。2 つは別物。

  **正しい対応（データ構造の軸）**:

  | | BirdCLEF+ 2026 | RSNA Knee 2026 |
  |---|---|---|
  | **少量・明確な GT<br>（test と同形式）** | `train_soundscapes`<br>5 秒 × 234 種の 0/1、**〜100 件** | **58 labeled studies**<br>study × 12 の 0/1、専門医合意 |
  | **大量・弱いヒント<br>（形式が違う）** | `train_audio`<br>**40,000〜50,000 件**。`primary_label`（1 種）<br>+ `secondary_labels`（複数）を**録音全体**に付与 | **4,349 report-only studies**<br>9 言語の自由文 report |
  | **test** | 5 秒 × 234 種 | study × 12（約 1,300 studies） |
  | 少量 : 大量の比 | 約 **0.2%** | **1.3%** |

  → **「GT が明確なデータが少量、ヒント情報が大量」という構造が同じ**。
  **BirdCLEF はこの構造で 0.955–0.965 まで到達した実績がある** = この構造は攻略可能。

  **2 つのコンペの違い（どちらが難しいか）**:

  | 軸 | BirdCLEF | RSNA Knee | どちらが難しいか |
  |---|---|---|---|
  | 弱ラベルが**ラベル空間に載っているか** | ⭕ 種名そのもの（変換不要） | ❌ **9 言語の自由文**（Stage A の変換が要る） | **RSNA が難しい** |
  | **粒度のズレ** | ❌ 録音全体 → 5 秒（MIL 問題） | ⭕ study → study（**ズレなし**） | **BirdCLEF が難しい** |
  | 弱ラベルの**完全性** | ❌ secondary の記載漏れが多い（偽陰性） | ❌ report の記載漏れ + **厳格な閾値との不一致** | 同程度 |

  → **RSNA の固有の難所は「ラベル空間への変換」= Stage A に集中している。**
  逆に **BirdCLEF の最大の難所（粒度のズレ / MIL）は本コンペには無い** ので、
  そこに費やされた工夫（SED head、5 秒 window の切り方等）は**持ち込む必要がない**。

  **★ 流用すべきは「`train_soundscapes` をどう使ったか」と「CV をどう切ったか」**。
  上位解法の具体策は、**各コンペの repo 側**に `docs/TOP_SOLUTIONS.md` と `docs/top_solutions/` として残す
  （書式は `docs/POSTMORTEM_TEMPLATE.md`）。ドメイン固有なのでテンプレートには持ち込まない。

  ⚠ **pseudo の作法はそのまま使えない**: BirdCLEF の #7 作法は「真ラベル多 + pseudo 少」を
  前提にしたもの。本コンペは **58 : 4,349 と逆**で、**pseudo の重みを下げる cap は意味を持たない**
  （`docs/CV.md`）。**確実に流用できるのは「2 iter で頭打ち」「aug を強めに」**の 2 点。

## 8. 強表現の活用 / 蒸留

- BirdCLEF 上位ほぼ全員が **foundation model（Perch v2）を蒸留 or teacher 利用**。蒸留は **単体 gain の ~50%（5th）**。
  レシピ = **embedding を cosine/MSE で当てる head を前段 phase**。
- **★ 実例: RSNA Knee 2026（2026-09-05、ぶなぴーの指摘で格上げ）**:
  **BirdCLEF の Perch v2 に対応するのは「Stage A」ではなく「画像の大規模事前学習モデル」**。
  BirdCLEF 上位が**ほぼ全員 Perch v2 を使った**という事実は、
  **本コンペでも DINOv2 系 / 医用 foundation が同じ役割を果たす**ことを強く示唆する。

  候補: **DINOv2 / DINOv3**（自己教師・汎用視覚）、**RadImageNet**（医用画像 CNN pretrain）、
  **BiomedCLIP**、**MedSAM**、**timm の医用 pretrain**、**過去 RSNA コンペ上位の weights**（#4）。

  **調べるべきこと（GPU 不要・今すぐできる）**:
  - [ ] MRI / 筋骨格に効く公開 foundation の棚卸し
  - [ ] **ライセンスが「公開再配布」を許すか** ← 入賞時に weights 公開が義務（`docs/COMPETITION.md`）
  - [ ] Kaggle Dataset として持ち込めるサイズか（**internet 遮断下で動かす**）
  - [ ] 蒸留で使うか backbone として fine-tune するか（BirdCLEF 5th は**蒸留が単体 gain の ~50%**）

  ⚠ **「foundation を調べる」は Stage A と並行できる独立タスク**。GPU も Kaggle 情報も要らない。

## 9. データ品質（地力。上位共通）

- 7th = 手動 re-label、12th = **重複除去（leakage 防止）/ 破損データ復旧 / 追加データ回収**。
- **実例: RSNA Knee 2026**: **重複画像・重複 study の検出と除去（leakage 最重要）**、破損 DICOM、
  左右混在、外れ値の除去、**ラベルの作られ方（読影医何人・合意方式）から label noise を見積もる**。
  **noisy label は loss 側（label smoothing / asymmetric / focal）でも吸収できる**。

## 10. runtime / その他の落穂

- **runtime を最初から予算化**: BirdCLEF は 90分制約で境界 sub が timeout 失格。
  画像コンペは **DICOM decode が推論時間の支配項**になりがち → 最初に 1 study あたりの decode 時間を測る。
- **aug は強め / 変則でも generalize**（5th は物理的に無意味な filter で +0.01）。恐れず A/B する。
- **後処理は全モデル一律だと悪化**（5th）→ モデル毎に変える。**過剰 PP は private で逆効果**。
- **specialist 分割**: 難しい部分集合（稀な所見・小さい病変・特定機種）を別モデル・高 weight。
- **AI を accelerator に**: 1st「DS / idea は人間、software は agent（Claude Code）」。
  **コード生成・結果分析・discussion 要約に使い、idea / intuition は自分が出す**。

---

## 11. task framing を最初に固定する（★ ROGII の最大の学び）

ROGII では **「pointwise 回帰」だと思って始めた問題が、実は「系列アラインメント問題」だった**。
これに気づくまでの実験（幾何特徴 → drift 回帰、R²=0.07）は**全部無駄になった**。

- **教訓**: metric を下げにいく前に、**「この問題は本質的に何を推定しているのか」を 1 文で書く**。
  書けないうちにモデルを回さない。→ `docs/SOLUTION.md` の Task framing。
- **実例: RSNA Knee 2026**: 「これは image-level 分類か、study-level 集約か、局在検出か」を最初に決める。
  **予測単位（submission の行）と学習単位（画像）が違うなら、その集約こそが本丸**。

## 12. oracle 上界を測ってから設計する（★ ROGII で最も効いた診断）

ROGII では **「真値を知っていたら achievable な score」（oracle）** を測り、
`floor 15.91 → oracle 直線 5.96 → oracle 高次 2.29` と**伸びしろの地図**を作ったことで、
どの部分問題に投資すべきかが確定した。

- **やり方**: 各段の「もし X が完璧なら score はいくつか」を計算する。
  安い実験で **投資先の優先順位が決まる**。
- **実例: RSNA Knee 2026**:
  - 「陽性 study の位置が既知なら」「slice 選択が完璧なら」「calibration が完璧なら」の各 oracle を測る。
  - **床（全部 prior の定数予測）も必ず測る**。これを下回る手法は即棄却できる。

## 13. 「realistic 化」で崩れないかを必ず確認する

ROGII では oracle で 5.96 だった手法が、**実運用条件（prefix からの外挿）にすると 58.1 と床より大暴発**した。

- **教訓**: oracle と realistic の **ギャップが本当の難所**。oracle だけ見て採用しない。
- **実例: RSNA Knee 2026**: 「train で使えるが test で使えない情報」を使っていないか毎回確認する
  （= `docs/CV.md` の anti-leakage checklist）。

## 14. 公開アーキテクチャの「必然性」を自分で再導出する

ROGII では診断を重ねた結果、**公開上位 NB の構成（tracker + kNN + GBM + smoothing）が
なぜその形なのかを実データから再導出**できた。これで「どこを改善すべきか」が自分の言葉になった。

- **実例: RSNA Knee 2026**: 公開 NB をコピーして回すだけにしない。**「なぜこの部品が要るのか」を実験で確かめる**。
  理由が分かった部品だけが、改善の対象になる。

---

## 15. インフラは「思い込み」でなく実測で決める（★ RSNA Knee 2026-08-13 の実戦ログ）

このコンペの raw 展開で、**推測に基づく前提が 3 つとも実測で覆った**。
`LESSONS` に載せる価値があるのは、**どれも「測れば 5 分で分かる」ことを測らずに設計しかけた**点。

### 実際に起きたこと

**265 GB の zip の `unzip` が 24 時間終わらなかった。**

- 原因は **小ファイルの逐次 write**。819,640 ファイルを **1 スレッドで** NFS に書くと、
  ファイルごとに commit の round trip が発生してレイテンシ律速になる。
  **CPU 使用率 7.5%** がその証拠（= I/O 待ちで寝ている）。24h 平均で **write 2.9 MB/s**。
- **これは異常ではなく、この構成では正常**。殺して再実行しても同じだけかかる。

### そこから覆った前提

| 当初の前提 | 実測 | 教訓 |
|---|---|---|
| 「NFS のランダム読みが学習の I/O 律速になる」 | **RDMA NFS で cold 1.9 GB/s**。1 epoch 全舐め **5 分** | **read と write を混同していた**。遅いのは write |
| 「transfer syntax 混在。JPEG2000 decode が推論の支配項」 | 全 819,635 枚 census で **100% 無圧縮**。1.84 ms/枚 | **懸念が丸ごと空振り**。対策コードは無駄になっていた |
| 「host の RAM は 2 TB だから全部 page cache に載る」 | **cgroup `memory.max` は 128 GiB** | **`free` は host を見せる。コンテナの実力は cgroup** |

### 横展開できるルール

1. **「遅い」と感じたら、まず止める前に `/proc/<pid>/io` と CPU 使用率を見る。**
   CPU が低ければ I/O 律速で、**並列度を上げる以外に手は無い**。闇雲に kill しない。
   進捗率は `rchar ÷ 入力サイズ` で出せる。**残り時間が分かってから判断する**。
2. **read / write / decode を分けて測る。** 「I/O が遅い」は診断ではない。
   今回は「write が遅い・read は速い・decode はタダ」で、**打つ手が全部違った**。
3. **`free` でなく cgroup（`memory.max` / `cpu.max`）を見る。** コンテナ・k8s では host の値は嘘。
4. **展開・変換は必ず完全性を検証する。** 名前とバイトサイズを全件照合すれば、
   CRC を待たずに切り詰めを検出できる（今回は 819,640 件を照合し 欠損 0 を確認してから zip を削除）。
5. **「重い前処理が要る」という前提は、コンペ序盤に 5 分の実測で潰せることが多い。**
   `docs/DATASET.md` の「未確定リスト」は、**推測で埋めず、測って埋める**。

> ⚠ **ただし開発環境の実測値を Kaggle notebook に外挿しないこと。**
> ここは RDMA NFS / 16 core。Kaggle は別物。**#10 の「runtime を最初から予算化」は
> Kaggle 上で測り直して初めて意味を持つ**。

---

## 16. 定性表現（"small subset"）を、実数を測る前に設計の前提にしない（★ RSNA Knee 2026-09-05）

ホストの Data タブは "**Only a small subset** of training studies carry per-condition labels" と書いていた。
我々はこれを「一部」と読み、**「一部のラベルで val を作り、pseudo で train を増やす」設計**を
`docs/CV.md` に丸ごと書いた。

**実測したら 58 / 4,407 = 1.3% だった。** 書いた設計は **全部成立しなかった**:

| 書いていたルール | 58 件だと何が起きるか |
|---|---|
| 「val は必ずラベル付き subset から取る」 | 1 fold 11.6 study、最少ラベルの陽性 1.8 件 → **AUC が測定不能** |
| 「fold k の train に fold k の pseudo を入れない」 | train が 58 件しか無く、そもそも学習が成立しない |
| 「抽出器を fold 内 labeled だけで作る」 | 1 fold 46 件で抽出器を作る意味が無い |
| 「pseudo の寄与を本物より小さく cap」（#7） | 本物 58 : pseudo 4,349。**cap する対象が逆** |

- **教訓**: **"small" / "a few" / "sparse" のような定性語は、桁が分からない。**
  1.3% と 30% では設計が別物になる。**`wc -l` と `notna().sum()` で 1 分で分かることを、
  設計を書く前に測る**。これは #15（インフラは実測で決める）の**データ版**。
- **横展開ルール**: docs に「規模を表す形容詞」を書いたら、**そこに実数を測るタスクを必ず併記する**。
  実数が入るまで、その形容詞に依存した設計は **書いてもよいが「暫定」と明記する**。
- **副次的な学び**: 前提が壊れたと分かったら、**旧ルールを消さずに「撤回」と理由を残す**。
  消すと、後から「なぜこの設計にしたか」が追えなくなる（`docs/CV.md` はこの形にした）。

---

## 17. 「やったのに残らない」を防ぐ（★ RSNA Knee 2026 最大の運用教訓）

コンペ終了後の片付けで、**273 GB の git 未追跡ディレクトリ**の中に
「open-weight VLM に生画像を直接読ませる路線を 32 arm 規模で検証し、統計的に CLOSE と判定した記録」が
眠っているのを発見した。**危うく気づかずに削除するところだった。**

さらに悪いことに、**その結論は本編の振り返り文書に一度も反映されていなかった**。
その結果、終了後の振り返りで「我々は画像を読ませる路線を試していなかった」という
**誤った総括を書きかけた**（実際は「試して正しく閉じた。ただし frontier model では試していなかった」が正しい）。

### 横展開できるルール

1. 🔒 **価値のある結論を git 未追跡の場所に置かない。**
   scratch / tmp / 個人作業領域は「消えてよいもの」専用にする。
   **判断（GO / NO-GO / CLOSE / HOLD）を書いた瞬間に、追跡対象へ移す**。
2. 📝 **「閉じた」判断ほど本編に書き戻す。**
   失敗の記録は未来の自分が同じ穴を掘らないための最大の資産なのに、
   **成功しなかったがゆえに放置されやすい**。`docs/EXPERIMENTS.md` の「失敗も記録する」は、
   ここまで含めて初めて機能する。
3. ⏳ **HOLD は腐る。**
   「生徒モデルの結果を見てから判断する」と保留した有望な軸（効果量は有意だった）が、
   誰にも追跡されないまま終了した。**保留にするなら「誰が・何を見たら・いつ判断するか」を同時に書く**。
   書けないなら保留ではなく却下（CLOSE）にする。
4. 🔍 **巨大ディレクトリを消す前に、大きさではなく「種類」を見る。**
   今回は 273 GB のうち **99.9% が再ダウンロード可能なモデルキャッシュ**で、
   残り 0.1% に **二度と作れない実験記録** が入っていた。比率で判断すると全部消える。

---

## 18. ラベルを変えたら「物差し」も変わっている（★ RSNA Knee 2026-10-29）

ラベルソースを差し替えた arm と既存 arm の CV を比べたら **0.99 対 0.92** という劇的な差が出た。
だが改善ではなく、**評価に使う正解ラベル自体が arm ごとに違っていた**
（OOF に保存される `true_` 列が、その arm が学習に使ったラベルソースから作られていた）。

- ✅ **ルール**: ラベル生成を変える実験では、**本命と対照の評価ラベルを共通の 1 つに固定する**。
  仕組みとして `eval_source` のような「評価ラベルだけを差し替える指定」を最初から用意しておく。
- 🧭 **一般形**: 数字を解釈する前に *「この指標は、比較したい 2 本で同じ物差しになっているか?」* を問う。
  metric 定義・正解ラベル・対象サンプル・前処理のどれかが arm 依存だと、**差分は意味を失う**。
- ⚠ **中立な物差しが存在しないこともある**: RSNA では専門医の正解が全体の 1.3% にしかなく、
  残り 98.7% は「どのラベル源も推定値」だった。
  その場合に公正なのは **(a) 小標本の真の正解での診断値 (b) LB の実測** の 2 つだけ、と割り切る。

---

## 19. 終了後に「自分の仮説」を実験で検証する（★ 振り返りの型）

RSNA 終了後、上位 23 件の writeup から **「差を生んだのはラベル品質だ」という仮説**を立てた。
これを読み物で終わらせず、**7 位が公開したラベルに差し替え、他の条件を完全固定して再学習**した。

結果は **Private 0.958**。我々の最終提出 0.945（5 モデルアンサンブル + 公開ベースラインとのブレンド）を、
**ラベルだけ差し替えた単体モデルが上回った**。

- 💡 **効用**: 「ラベルが効いたらしい」が「**ラベルだけで +0.013**」になる。
  次回の投資配分を決める根拠としての強さが全く違う。
- 🔬 **やり方**: 上位の公開資産（ラベル / OOF / weights）があるなら、
  **自分のパイプラインの 1 箇所だけ差し替える**。他を完全固定するのが肝（差分の帰属が確定する）。
- ⚖ **注意**: ライセンス（CC-BY-NC-SA 等）を必ず確認し、再配布しない。社内の学習目的に限定する。
- ⏱ **締切後でも提出が採点されることがある**（RSNA は終了 7 日後の提出が採点された）。
  ただし採点完了まで数時間かかる場合がある。**終わったコンペは、一番安全な実験場**。

---

## 20. 後知恵と「その時の情報」を分けて振り返る

Private を見た後の振り返りは、放っておくと **結果論**になる。
RSNA の振り返りでは、判断ごとに次の 4 列の表を作った:

| 締切前の判断 | その時点で持っていた evidence | Private の結果 | その判断は正当だったか |
|---|---|---|---|

- 原則: **良い判断 ≠ 幸運な結果 / 悪い結果 ≠ 悪い判断**。
  evidence と整合していた判断は、結果が悪くても「正しい判断」として残す。
  逆に **結果が良くても根拠が無かったものは「運」と明記する**（再現できないので）。
- これをやると、次回に持ち越すべきなのは **判断のプロセス** であって
  個別のハイパラではない、ということがはっきりする。

---

## メタ: 先回りする 6 点

1. **target 分布のデータを評価専用に隔離するな** — 学習に inject + domain-aware val の両方で使う。
2. **validation を hidden test に似せて設計** — unseen-institution split。LB probing より信頼でき、提出枠も節約。
3. **diversity + 賢い fusion を最初から** — 同じ部品でも組み立てで +0.01。ただし **本数ではなく毛色**。
4. **強表現（pretrain / foundation）を早く調べて活用** — 自前天井に固執しない。
5. **最終選択は private / robust 重視 + hedge** — public で選ばない。
6. **ラベルが導出物なら、ラベル品質を最優先の投資先にする** — RSNA ではモデル側を一切変えず、
   ラベル源の差し替えだけで Private +0.013（§19 / `docs/WEAK_LABEL_PLAYBOOK.md`）。

---

## ★ frame（全実験に当てる羅針盤）

**「良いモデルを作る（最適化）」でなく「未知ドメインに適応 + 分散を管理する」frame で戦う。**
同じ CV +0.001 でも、off-domain なら無意味〜overfit。**metric を見る前に毎回3問**:

1. **Transfer**: 未知施設 / 未知機種（**leave-institution-out**）で効くか? — patient-CV の +0.001 に踊らない。
2. **Variance**: private の振れを下げるか? — diversity + 保守的 selection。
3. **Adapt**: test に近い信号を **測るだけでなく適応に使えているか?**

- アラーム: 「CV を +0.001 押すノブを回している」と気づいたら止まり、
  「**transfer か in-distribution fit か?**」を問う。
- 一行指針: **「スコアを上げる」でなく「未知患者・未知施設でスコアが上がり、かつ振れない構成を作る」**。
- 証拠: BirdCLEF で **上位は我々と同じ部品で 0.955–0.965**。差は部品でなく **frame**。
