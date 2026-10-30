# %% [markdown]
# # 提出 kernel（テンプレート）
#
# ⚠ **このファイルが正本。`.ipynb` は `build_kernel.py` の生成物なので直接編集しない。**
#   直したら必ず `python src/scripts/300_submission/build_kernel.py --arms '[...]'` を走らせる。
#
# `# %%` でセルを区切る。`# %% [markdown]` は markdown セルになる。

# %%
# =============================================================================
# 生成時に差し替わる定数（build_kernel.py が `KEY = ...` を丸ごと置換する）
# =============================================================================
# ⚠ ここの名前は submission_config.yaml の preproc.specs のキーと一致させること。
#   値は学習時の実験 config から **自動導出**され、食い違えば build_kernel が止まる。
ARMS = ["{{EXP_NAME}}"]
N_SLICES = 0
CACHE_SPAN = (0.0, 1.0)
CACHE_SLAB_MM = 0.0

# 推論の振る舞い（必要に応じて --set で上書きする）
ENSEMBLE = "prob"      # "prob"（確率平均）/ "rank"（順位平均）
TTA_MODE = "off"
BUDGET_H = 8.0         # 時間ガード（下の「⚠ 時間ガード」を読むこと）
DEBUG_N = 0            # >0 なら先頭 N 件だけ推論（ローカル疎通用）

# %%
# =============================================================================
# import とパス解決
# =============================================================================
# ⚠ Kaggle の /kaggle/input は {competitions, datasets} の入れ子。
#   **絶対パスを決め打ちせず再帰探索する**（dataset 名が変わっても動くように）。
# ⚠ `from __future__ import annotations` は使えない（ファイル先頭でないと SyntaxError になり、
#   セル結合後は必ず定数定義より後ろに来るため）。Kaggle の Python は新しいので不要。
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T_START = time.time()
IN_DIR = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find_parent_of(rel: str) -> Path:
    """`/kaggle/input` 以下から `rel` を含む親ディレクトリを探す。

    Args:
        rel: 例 `"exp001/fold0.pt"`。

    Returns:
        見つかった親ディレクトリ。

    Raises:
        FileNotFoundError: 見つからないとき（attach 忘れが原因のことが多い）。
    """
    hits = sorted(IN_DIR.glob(f"**/{rel}"))
    if not hits:
        raise FileNotFoundError(f"{rel} が /kaggle/input に見つからない（dataset の attach 忘れ？）")
    return hits[0].parent.parent if "/" in rel else hits[0].parent


# ⚠ 自作コードを Dataset で配る場合、同名の pip パッケージに負けることがある。
#   sys.path の **先頭**に入れ、`src/` には `__init__.py` を置いておく。
SRC = find_parent_of("{{SRC_MARKER_FILE}}")  # 例: "src/datasets/__init__.py"
sys.path.insert(0, str(SRC))

DATA = next(iter((IN_DIR / "competitions").iterdir()))  # コンペ配布物のディレクトリ
print(f"SRC : {SRC}")
print(f"DATA: {DATA}")
print(f"ARMS: {ARMS}")

# %%
# =============================================================================
# 入力の準備（{{ここをコンペに合わせて書く}}）
# =============================================================================
sample = pd.read_csv(DATA / "sample_submission.csv")
test_ids = sample["{{ID_COLUMN}}"].tolist()
if DEBUG_N:
    test_ids = test_ids[:DEBUG_N]
print(f"test: {len(test_ids)} 件")

# %%
# =============================================================================
# ⚠ 時間ガード
# =============================================================================
# Kaggle の実行時間上限を超えると **提出そのものが無効**になる。一方で、
# 「時間が足りないときに処理を間引く」ガードを入れると、**本番で間引きが発動して
# 直前の改善が丸ごと無反映になる**事故が起きる（412 位の解法がこれで取りこぼした）。
#
# → ガードは入れるが、**発動したら必ずログに大きく出す**。提出後にログを見て、
#   発動していたら「その提出は意図した構成ではない」と判断できるようにする。
#
# ⚠⚠ **公開テストの 3 件で通った commit run は、本番の実行時間を保証しない。**
#   本番の hidden test は件数が桁違いなので、1 件あたりの実測から外挿して見積もること。
def time_left_h() -> float:
    """残り時間（時間単位）。"""
    return BUDGET_H - (time.time() - T_START) / 3600


# %%
# =============================================================================
# 推論
# =============================================================================
preds = []
for arm in ARMS:
    meta = json.loads((find_parent_of(f"{arm}/infer_meta.json") / arm / "infer_meta.json").read_text())
    print(f"  {arm}: {meta}")
    # ⚠ meta に無い設定でモデルを組むと、load_state_dict が通っても別モデルになる。
    #    stage_arm.py の meta_fields に必要な項目が全部入っているか確認すること。
    # model = build_model(meta)
    # p = predict(model, ...)
    # preds.append(p)
    if time_left_h() < 0.5:
        print("⚠⚠ 時間ガード発動: 残り arm をスキップした（この提出は意図した構成ではない）")
        break

# {{ここで preds を統合する}}
# if ENSEMBLE == "rank":
#     merged = np.mean([pd.DataFrame(p).rank(pct=True).to_numpy() for p in preds], axis=0)
# else:
#     merged = np.mean(preds, axis=0)

# %%
# =============================================================================
# submission.csv を書く
# =============================================================================
# ⚠ 列順は sample_submission.csv と **完全に一致**させる（順序違いは静かに 0 点になる）。
sub = sample.copy()
# sub[TARGET_COLS] = merged
assert list(sub.columns) == list(pd.read_csv(DATA / "sample_submission.csv").columns)
sub.to_csv("submission.csv", index=False)
print(f"✅ submission.csv ({len(sub)} 行) / 総所要 {(time.time() - T_START) / 60:.1f} 分")
print(sub.describe())
