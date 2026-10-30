#!/usr/bin/env bash
# =============================================================================
# submit_folds.sh — 1 実験の N fold を **N GPU へ並列に**投げる
# =============================================================================
#   ./infra/k8s/submit_folds.sh <exp> [folds]
#
# 例:
#   ./infra/k8s/submit_folds.sh exp042            # fold 0,1,2,3,4 を 5 job で
#   ./infra/k8s/submit_folds.sh exp042 0          # fold0 だけ（サーベイ用）
#   ./infra/k8s/submit_folds.sh exp042 0,3        # fold0+3 だけ
#
# ## なぜ速くなるか
# fold は互いに独立なので、1 job が 5 fold を直列に回すのが唯一のボトルネックだった。
# 並列化しても **精度は一切落ちない**（同じ OOF をそのまま再現する）。
#
# ## fold0 だけのサーベイについて（RSNA 実測値。コンペごとに測り直すこと）
#   1 fold のノイズ σ = 0.0014（5-fold OOF は 0.0006）
#   fold0 と OOF の相関 r = 0.995 / fold0+fold3 の平均で r = 0.998
#   ⚠ **σ の 2 倍未満の差は 1 fold では見えない**。篩にかける用途に限る。
#
# ⚠ 全 job が終わったら **必ず**統合する（各 job は result.json を書かない）:
#   .venv/bin/python src/scripts/210_collect_results/merge_folds.py --exp <exp>
# =============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
ENVFILE="${PAI_ENV_FILE:-$DIR/cluster.env}"
# shellcheck disable=SC1090
set -a; . "$ENVFILE"; set +a

EXP="${1:?使い方: ./infra/k8s/submit_folds.sh <exp> [folds]}"
FOLDS="${2:-0,1,2,3,4}"
PY="${PAI_PY:?cluster.env に PAI_PY が必要}"
TRAIN="${PAI_TRAIN_SCRIPT:-src/scripts/200_train_model/200_train_model.py}"

# -----------------------------------------------------------------------------
# ⚠⚠ **投げる前に config を検証する。**
#   RSNA では `train.batch_size` を二重に書いて YAML を壊し、その後に起動した job が
#   軒並み `found duplicate key` で落ちた。25 job 投げてから気づくと損失が大きい。
#   コンペ側に preflight_conf.py 相当があるなら、ここで必ず通す。
# -----------------------------------------------------------------------------
PREFLIGHT="$ROOT/src/scripts/200_train_model/preflight_conf.py"
if [ -f "$PREFLIGHT" ]; then
  "$ROOT/.venv/bin/python" "$PREFLIGHT" --arm "$EXP" \
    || { echo "❌ preflight に失敗したので投げません"; exit 1; }
  echo
else
  echo "⚠ preflight_conf.py が無いので config 検証をスキップします（作ることを強く推奨）"
fi

IFS=',' read -ra KS <<< "$FOLDS"
echo "★ $EXP を fold [${KS[*]}] = ${#KS[@]} job に分けて投げます"
for k in "${KS[@]}"; do
  "$DIR/submit.sh" run 1 "ARMS=$EXP FOLDS=$k $PY $TRAIN" | grep -E "created" | sed "s/^/  fold$k: /"
done
echo
echo "⚠ 全 job 完了後に必ず:"
echo "   .venv/bin/python src/scripts/210_collect_results/merge_folds.py --exp $EXP"
