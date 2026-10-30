# src/scripts/300_submission — Code Competition 提出パイプライン

学習済み weights を Kaggle Dataset にし、推論 notebook を生成して提出するまでの一式。
RSNA Knee（2026）で実戦投入したものを汎用化した。**ここに入っている検査は全て、
実際に事故ってから足されたもの**なので、面倒でも外さないこと。

---

## 0. 次回コンペで埋めるチェックリスト

- [ ] `submission_config.yaml` の `{{...}}` を全部埋める
      → `python src/scripts/300_submission/config.py` が ✅ になるまで
- [ ] `preproc.config_key` / `preproc.specs` を**学習側の実験 config に合わせる**
      （学習時の前処理バージョンと、それに対応する kernel 定数の対応表）
- [ ] `meta_fields` に **ckpt に入らない設定**を全部並べる（下の「無言で別モデル」参照）
- [ ] `banned_desc_words` に、このコンペで書いてはいけない語を足す
- [ ] `kernel/inference_template.py` の `{{...}}` を埋めて、実際に推論が通る形にする
- [ ] `.gitignore` が `*.json` を無視している場合、`kernel-metadata.json` を追跡したいなら `git add -f`

---

## 1. 流れ

```bash
# ① weights を Dataset の形に並べて upload
python src/scripts/300_submission/stage_arm.py --exp exp001 --upload

# ② kernel を生成（.py → .ipynb）
python src/scripts/300_submission/build_kernel.py \
    --arms '["exp001"]' \
    --datasets "<owner>/<src-dataset>,<owner>/<arm-dataset>"

# ③ push 前にローカルで通す（Kaggle は 1 往復 10 分）
python src/scripts/300_submission/run_kernel_local.py \
    --dataset-dir data/kaggle_upload/<slug>

# ④ push → Kaggle 上で commit run が通るのを確認
kaggle kernels push -p src/scripts/300_submission/kernel
kaggle kernels status <owner>/<kernel-slug>

# ⑤ 提出（★ 枠を消費する。--yes が無ければ dry-run）
python src/scripts/300_submission/submit.py --version 1 --desc "exp001 single" --yes
```

| ファイル | 役割 |
|---|---|
| `submission_config.yaml` | コンペ固有の設定を全部ここに集約 |
| `config.py` | 設定の読み込み＋**埋め忘れの hard fail** |
| `stage_arm.py` | weights → Kaggle Dataset の形に並べる / upload |
| `build_kernel.py` | `.py` → `.ipynb` 生成（**前処理仕様の自動導出＋検査**） |
| `check_kernel_order.py` | トップレベル前方参照の検出 |
| `run_kernel_local.py` | push 前のローカル疎通 |
| `submit.py` | Code Competition への提出（`CreateCodeSubmission`） |
| `kernel/inference_template.py` | 推論 kernel の雛形（**これが正本**） |

---

## 2. ⚠⚠ 守るべき 5 つ（全部、事故から生まれた）

### ① `.py` が正本、`.ipynb` は生成物
`.py` だけ直して `.ipynb` の再生成を忘れ、Kaggle で 10 分かけて**古いコードが走った**。
`.ipynb` を手で編集し始めると必ずこれをやる。

### ② 前処理仕様は実験 config から自動導出する（食い違ったら止める）
cache 仕様を指定せずに kernel を作り、24 枚 / span 0.15–0.85 で学習した arm を
**16 枚 / span 0–1 で推論する kernel** を提出した。LB −0.005。**エラーは一切出ない。**
`build_kernel.py` は学習側の config から導出し、CLI 指定と食い違えば既定で hard fail する
（意図的に逸脱するときだけ `--allow-preproc-override`）。

### ③ ckpt に入らない設定は `infer_meta.json` で運ぶ（= 無言で別モデル）
`persistent=False` の buffer や「層が増えるフラグ」は state_dict に現れない。
meta に無いと kernel は違う形のモデルを組み、しかも `load_state_dict` が
**missing / unexpected ゼロで通る**。出力だけが静かにズレる。
→ `submission_config.yaml` の `meta_fields` に全部並べる。`stage_arm.py` は読めない項目があれば止まる。

### ④ push 前に必ずローカルで通す
Kaggle は 1 往復 10 分。RSNA では 1 日に 6 回失敗して 1 時間以上溶かした。
`run_kernel_local.py` は `/kaggle/input` と同じ入れ子構造をローカルに作って notebook を丸ごと実行する。
⚠ **symlink ではなくハードリンク／実体で作る**こと。`Path.glob("**/...")` は symlink を辿らないので、
symlink だとパス解決が空振りして「ローカルでは通ったのに Kaggle で落ちる」になる。

### ⑤ 時間ガードは入れるが、発動したら必ずログに出す
実行時間超過は提出そのものを無効にする。一方で「足りないときに間引く」ガードは、
**本番で発動して直前の改善が丸ごと無反映になる**（412 位の解法がこれで取りこぼした）。
→ 発動したら大きくログに出し、提出後にログを見て判断できるようにする。
⚠⚠ **公開テスト数件で通った commit run は本番の実行時間を保証しない。** 1 件あたりの実測から外挿する。

---

## 3. ⚠ 提出 kernel を作ったら「生成コマンド」を必ず記録する

RSNA で kernel を作り直したとき、**生成コマンドが記録に残っておらず、
提出済みの `.ipynb` から 8 個の引数を逆算して復元する**羽目になった。
しかも復元を誤れば「入力の切り出し範囲が違う別物」を提出するところだった。

→ **提出した kernel は下の表に 1 行足す。** これは規律であって任意ではない。

### 提出記録

| 日付 | slot | 生成コマンド | Public | Private | 備考 |
|---|---|---|---|---|---|
| | | | | | |

---

## 4. 単一 CSV 提出のコンペの場合

Code Competition でなければこの一式は不要で、`kaggle competitions submit -f submission.csv` で足りる。
その場合も ③（学習と推論の設定が一致していること）と ⑤（列順と件数）の確認だけは残す価値がある。
