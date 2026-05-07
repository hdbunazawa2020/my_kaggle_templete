# my_kaggle_templete

Kaggle 研究の出発点 template。 環境設定 + Hydra + Claude Code 連携 (= `.claude/`, `docs/`) を bundle。

新コンペを開始する時はこの repo を clone → `${PROJECT_NAME}` を変えるだけで Claude が disciplined Kaggle engineer として動ける構成。

---

## ⚠️ 重要: 実行環境の原則 (= 必読)

本 template は **「コードは永続ストレージ、 cache はローカル高速ストレージ」** という二層構造を前提とする。
パスは環境変数で抽象化しており、 サーバー / クラウドが変わっても `~/.bashrc` の export を差し替えるだけで通用する。

### 環境変数 (= プロジェクトルート + cache root)

| 変数 | 役割 | このサーバーでの値 (= 例) |
|---|---|---|
| `${WORKSPACE_ROOT}` | コード/データを置くルート (= 永続) | `/workspace` |
| `${CACHE_ROOT}` | HF/wandb/torch/uv cache の root (= ローカル高速) | `/workspace/cache` |
| `${PROJECT_NAME}` | このコンペの名前 | 例: `BirdCLEF2026` |
| `${USER}` | ユーザー名 | 例: `hidebu` |

> **旧版との対応**: 旧 README は `/mnt/nfs` (= 永続) と `/mnt/nva` (= local) を前提にしていたが、 現環境ではこの 2 つが `/workspace` に統合されている (`/mnt/nva` は存在しない)。 環境ごとに `WORKSPACE_ROOT` / `CACHE_ROOT` を切り替えれば旧 path 派にも新環境にも適合する。

### 基本原則
| 種類 | 保存場所 |
|------|----------|
| コード | `${WORKSPACE_ROOT}/study/${PROJECT_NAME}/` |
| データセット (raw) | `${WORKSPACE_ROOT}/study/${PROJECT_NAME}/data/raw/` |
| 仮想環境 (.venv) | `${WORKSPACE_ROOT}/study/${PROJECT_NAME}/.venv/` (= ローカル高速 SSD 内) |
| wandb ログ | `${CACHE_ROOT}/wandb` |
| HuggingFace cache | `${CACHE_ROOT}/huggingface` |
| torch cache | `${CACHE_ROOT}/torch` |
| uv cache | `${CACHE_ROOT}/uv` |
| 一時ファイル | `${CACHE_ROOT}/tmp` |

👉 **永続ストレージに巨大 cache を書かないことが最重要** (= NFS / EFS 等を遅くする原因)

---

## 🚀 初期セットアップ (= 必須手順)

### 0. 環境変数の宣言 (= 一度だけ、 `~/.bashrc` に)
```bash
# このサーバーの実体に合わせて編集する。 別環境では path を差し替えるだけで OK
echo 'export WORKSPACE_ROOT=/workspace' >> ~/.bashrc
echo 'export CACHE_ROOT=/workspace/cache' >> ~/.bashrc

echo 'export WANDB_DIR=${CACHE_ROOT}/wandb' >> ~/.bashrc
echo 'export HF_HOME=${CACHE_ROOT}/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=${CACHE_ROOT}/huggingface' >> ~/.bashrc
echo 'export TORCH_HOME=${CACHE_ROOT}/torch' >> ~/.bashrc
echo 'export UV_CACHE_DIR=${CACHE_ROOT}/uv' >> ~/.bashrc
echo 'export TMPDIR=${CACHE_ROOT}/tmp' >> ~/.bashrc
source ~/.bashrc
```

### 1. cache ディレクトリ作成
```bash
mkdir -p ${CACHE_ROOT}/{wandb,huggingface,torch,uv,tmp}
```

### 2. プロジェクト clone + 仮想環境作成
```bash
export PROJECT_NAME=<このコンペ名>     # 例: BirdCLEF2026

cd ${WORKSPACE_ROOT}/study
git clone <this-template> ${PROJECT_NAME}     # template から派生
cd ${PROJECT_NAME}
uv venv .venv
source .venv/bin/activate
```

### 3. 依存関係インストール
```bash
uv sync --active
```

### 4. VSCode 設定
Python Interpreter を以下に設定:
```
${WORKSPACE_ROOT}/study/${PROJECT_NAME}/.venv/bin/python
```

### 5. Claude Code を有効化
```bash
# .claude/settings.json は安全な共通 permissions のみ
# 個人 / プロジェクト固有の追加 allow は .claude/settings.local.json (= git-ignored) に書く
```

### ❌ やってはいけないこと
* 永続ストレージに `.venv` を作る (= 学習が遅くなる原因)
* wandb ログを永続ストレージに書く
* HuggingFace cache を永続ストレージに置く
* checkpoint を無制限に保存する (= Kaggle 提出に必要なものだけ残す)

→ 共有ストレージが詰まり、 サーバ全体がフリーズする原因

---

## 🤖 Claude Code 連携

### 何が入っているか
| ファイル | 役割 |
|---|---|
| `CLAUDE.md` | Claude の振る舞いルート定義 (= reproducibility, one hypothesis, standard loop) |
| `AGENTS.md` | 規律 + slash command の一覧 |
| `.claude/commands/*.md` | 9 個の slash command (`/register_experiment` 等) |
| `.claude/skills/*/SKILL.md` | 6 個の auto-invoked skill (= cv-diagnosis, kaggle-experiment, ...) |
| `.claude/settings.json` | 共通 safe permissions |
| `docs/*.md` | 12+ 個の Claude 必読 doc 雛形 (COMPETITION/METRIC/CV/EXPERIMENTS/AUTOMATION 等) |

### コンペ開始時にやること (= Claude が実力発揮する条件)
1. `docs/COMPETITION.md` を埋める (= competition の overview / runtime 制約 / hardware)
2. `docs/METRIC.md` を埋める (= 公式 metric の正確な定義)
3. `docs/DATASET.md` を埋める (= file 構成 + grouping unit)
4. `docs/CV.md` を埋める (= fold 戦略)
5. `CLAUDE.md` の `{{...}}` placeholder (= COMPETITION_NAME / TARGET_METRIC / KAGGLE_RUNTIME_LIMIT 等) を実値に置き換える
6. `docs/MEMORY_SEED.md` を見て `~/.claude/projects/<this-repo>/memory/` に initial memory を seed

### Claude が起動時に読むもの
順序は `CLAUDE.md` の Read order セクションで指定。 `docs/QUICKSTART_CLAUDE.md` も参照。

---

## 📂 ディレクトリ構成

```
.
├── CLAUDE.md / AGENTS.md / README.md
├── .claude/
│   ├── commands/       (9 個)
│   ├── skills/         (6 個 + kaggle_base.md)
│   └── settings.json
├── docs/               (14 個雛形)
├── configs/            (Hydra global config)
├── src/
│   ├── datasets/ models/ training/ inference/ utils/
│   └── scripts/<id>_<name>/
├── data/
│   ├── raw/            (= 競技 file、 触らない)
│   ├── processed/<ver>/
│   └── external/
├── experiments/<exp_id>/   (config.yaml, notes.md, result.json, log, model/, oof/)
├── notebooks/          (= 0xx EDA / 1xx classical / 2xx-3xx NN / 9xx ensemble)
├── public_notebooks/   (= 公開 NB の raw + annotated)
└── outputs/
```

詳細は `docs/REPO_STRUCTURE.md`。

### 連番ルール (notebook / script 共通)
- **0xx**: データ前処理
- **1xx**: classical ML (= GBDT 等)
- **2xx**: NN 学習
- **3xx**: NN 推論 / submission
- **9xx**: ensemble

---

## 🏃 実行方法

### コマンドライン
ワーキングディレクトリは `src/scripts/`。 各 script は引数を取らず、 Hydra config で動作:
```bash
cd src/scripts
python 000_data_preprocess/000_data_preprocess.py
```

### Hydra 設定
- main file: `src/scripts/conf/config.yaml`
- `defaults:` に各 script 用 config を記載
- 実行時 override: overrideは基本的に使わない。defaultsを書き換えて実行すること。
- 詳細: [hydra docs](https://hydra.cc/docs/intro/)

### 新規 script 作成
```bash
python scripts/generate_template.py --name <new_script_name>
```

---

## 📝 開発ルール

### Docstring
[Google style](https://google.github.io/styleguide/pyguide.html) に準拠。

### typehint
Python 3.9 以降でも動くように、 `typing.List` / `Dict` ではなく `from __future__ import annotations` 後に組込み `list` / `dict` を使用。

### Path
できる限り `pathlib.Path` を使用 (= OS による `/` `\\` 差を吸収)。

### フォーマット / lint
[Ruff](https://docs.astral.sh/ruff/) で統一 (= formatter + linter 兼用、 Rust 製で高速)。
```bash
uv add --dev ruff
ruff format <ファイル名 or ディレクトリ名>     # black 互換 formatter
ruff check  <ファイル名 or ディレクトリ名>     # lint
ruff check --fix <ファイル名 or ディレクトリ名> # lint + auto-fix
```
設定は `pyproject.toml` の `[tool.ruff]` セクションで管理する。

VSCode は [Ruff 拡張](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) を入れる (= save-on-format で自動整形)。

---

## 📚 参考
- 各種有用 link: [useful_kaggle_links.md](useful_kaggle_links.md)
- 過去 score: [score_progress.md](score_progress.md)
