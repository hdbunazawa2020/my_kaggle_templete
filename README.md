# my_kaggle_templete
- This is My_kaggle_templete (keep updating)

---
# ⚠️ 重要: 実行環境の原則（必読）
本テンプレートは、**NFS環境での安定動作**を前提とする。

## 基本原則
| 種類 | 保存場所 |
|------|----------|
| コード | NFS (/mnt/nfs) |
| データセット | NFS (/mnt/nfs) |
| 仮想環境 (.venv) | ローカル (/mnt/nva) |
| wandbログ | ローカル (/mnt/nva) |
| HuggingFace cache | ローカル (/mnt/nva) |
| uv cache | ローカル (/mnt/nva) |
| 一時ファイル | ローカル (/mnt/nva) |

👉 **NFSに書き込みをしないことが最重要**
---

# 🚀 初期セットアップ（必須手順）
## ① ローカルディレクトリ作成
```bash
mkdir -p /mnt/nva/home/${USER}/<PROJECT_NAME>/.venv
mkdir -p /mnt/nva/home/${USER}/cache/wandb
mkdir -p /mnt/nva/home/${USER}/cache/huggingface
mkdir -p /mnt/nva/home/${USER}/cache/uv
mkdir -p /mnt/nva/home/${USER}/tmp
mkdir -p /mnt/nva/home/${USER}/outputs
```
② 仮想環境作成（ローカル）
```bash
cd /mnt/nfs/home/${USER}/study/<PROJECT_NAME>
uv venv /mnt/nva/home/${USER}/<PROJECT_NAME>/.venv
source /mnt/nva/home/${USER}/<PROJECT_NAME>/.venv/bin/activate
```
③ 依存関係インストール
```
uv sync --active
```
④ 環境変数設定（重要）
```bash
export WANDB_DIR=/mnt/nva/home/${USER}/cache/wandb
export HF_HOME=/mnt/nva/home/${USER}/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/nva/home/${USER}/cache/huggingface
export TORCH_HOME=/mnt/nva/home/${USER}/cache/torch
export UV_CACHE_DIR=/mnt/nva/home/${USER}/cache/uv
export TMPDIR=/mnt/nva/home/${USER}/tmp
```
```bash
echo 'export WANDB_DIR=/mnt/nva/home/${USER}/cache/wandb' >> ~/.bashrc
echo 'export HF_HOME=/mnt/nva/home/${USER}/cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/mnt/nva/home/${USER}/cache/huggingface' >> ~/.bashrc
echo 'export TORCH_HOME=/mnt/nva/home/${USER}/cache/torch' >> ~/.bashrc
echo 'export UV_CACHE_DIR=/mnt/nva/home/${USER}/cache/uv' >> ~/.bashrc
echo 'export TMPDIR=/mnt/nva/home/${USER}/tmp' >> ~/.bashrc
```
⑤ VSCode設定（重要）
Python Interpreter を以下に設定：
```bash
/mnt/nva/home/${USER}/<PROJECT_NAME>/.venv/bin/python
```
❌ やってはいけないこと
* .venv を /mnt/nfs に作る
* wandbログをNFSに書く
* HuggingFace cacheをNFSに置く
* checkpointを無制限に保存する

👉 これをやると：
* 学習が遅くなる
* NFSが詰まる
* サーバがフリーズする

# ディレクトリ構成
---
### 1. [input](/input)
- 入出力データを格納するディレクトリ。


### 2. [notebook](/notebook)
- 各種python notebookを格納するディレクトリ。
Notebookでは、各種の検討をしたり、可視化を伴う後処理をしたい場合などに活用する。

- 連番の振り方
    - 0xx: データの前処理
    - 1xx: シンプルな機械学習の学習、推論(決定木など)
    - 2xx: Neural Network系モデルの学習、推論
    - 3xx: Neural Network系モデルの学習、推論
    - 9xx: アンサンブル

### ３. [script](/script)
- 各種python scriptを格納するディレクトリ.

- 連番の振り方
    - 0xx: データの前処理
    - 1xx: シンプルな機械学習の学習、推論(決定木など)
    - 2xx: Neural Network系モデルの学習、推論
    - 3xx: Neural Network系モデルの学習、推論
    - 9xx: アンサンブル

[conf](/script/conf)
: 各スクリプトの設定ファイルを格納する

### 4. [output](/output)
- 各検討のoutputを保存する。
- output/`連番`_`scriptの名前`_`EXPの連番`/ という形でディレクトリ分けされるようにする。

# 実行方法
### コマンドラインでの実行
- 基本的に、ワーキングディレクトリはscriptディレクトリとする.
各スクリプトは基本的に入力を受け取らないので, 以下のように実行できる:
```py
python 000_data_preprocess/000_data_preprocess.py
```
## 設定ファイルについて
[hydra](https://hydra.cc/)を用いる。

メインファイルは[config.yaml](/script/conf/config.yaml)である。
この中の`defaults`に各スクリプトの設定ファイルを記載する。
例えば, 000_data_preprocessの設定は, `conf/000_data_preprocess/000_data_preprocess_default.yaml`に記載されており, `config.yaml`には
```yaml
defaults:
    - 000_data_preprocess: 000_data_preprocess_default
```
として設定されている。`000_data_preprocess`ディレクトリ下に別のyamlファイルを作成し, ここに設定することも可能である。
また, 実行時に以下のように指定することもできる。
```sh
python 000_data_preprocess.py 000_data_preprocess=000_data_preprocess_default
```
詳細は[hydraのドキュメント](https://hydra.cc/docs/intro/)に記載されている。

## 新規スクリプト作成
基本的には, `generate_template.py`を実行すれば必要なファイルがすべて生成される。

`generate_template.py` の引数は以下の通り:
`--name`: スクリプト名 (拡張子を含まない)

## 開発

### Docstringの書き方

[google style](https://google.github.io/styleguide/pyguide.html)に準拠する。

### typehintの書き方

Python3.9以降でも動くように、typingのListやDictを使わない。
つまり、 `from __future__ import annotation` をしてlistやdictを使用する。

### パスの書き方

できる限りpathlibを使用する。
OSが違う場合, 区切りが `/` や `\\` のように違うものが使われる可能性ある。

### フォーマット

基本的に[black](https://black.readthedocs.io/en/stable/)でフォーマットを統一する。
以下のコマンドでblackをインストールできる。
```sh
pip install black
```
以下のコマンドで単一のファイルを修正できる。
```sh
black <スクリプト名>
```
以下のコマンドでディレクトリ内のすべてのファイルを修正できる。
```sh
black <ディレクトリ名>
```
VSCodeを使用している場合は, 拡張機能で[Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)を追加できる。
