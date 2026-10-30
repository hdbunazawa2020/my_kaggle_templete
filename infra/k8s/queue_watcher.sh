#!/usr/bin/env bash
# =============================================================================
# queue_watcher.sh — GPU 枠が空き次第、queue の job を **1 つずつ** 投入する常駐スクリプト
# =============================================================================
# ## 何をするものか
# 「投げたい学習コマンドを 1 行 1 コマンドで書いたテキストファイル」を上から順に消化する。
# 枠が空くまで待ち、1 本投げ、**Kueue に ADMITTED されたことを確認してから**次へ進む。
# 夜間に 20 本の学習を流しっぱなしにしたいときに使う。
#
# ## なぜ必要か
# - 一度に大量投入すると Kueue が再キュー上限を超えて workload を **deactivate（永久停止）**する。
#   RSNA では 40 job を一度に投げて 14 job が 12 時間死んだまま放置された。
# - 「投げる → ADMITTED を確認 → 次」の順序を守るだけでこれを避けられる。
#
# ## 使い方（必ずクラスタ側の Pod 内 tmux で動かす）
#   tmux new -d -s gpuq "CAP=12 RESERVE=2 bash infra/k8s/queue_watcher.sh"
#   ⚠ 手元の端末で動かさない。端末が落ちると監視が止まる（手元環境は予告なく止まりうる）。
#
#   queue      : $QUEUE_DIR/queue.txt       1 行 1 コマンド（# と空行は無視）
#   優先 queue : $QUEUE_DIR/queue_prio.txt  先に消化される（短時間の評価 job 用）
#   done       : $QUEUE_DIR/queue.done      投入に成功した行
#   log        : $QUEUE_DIR/queue.log       timestamp / job / workload / admitted / retry / delete
#   停止       : touch $QUEUE_DIR/QUEUE_STOP
#
# ## 二重投入ガード（3 段）
#   ① done ファイルに記録済みの行は投げない
#   ② 完了済みの成果物が既にあるなら投げない（DONE_ARTIFACT を設定した場合）
#   ③ 同じ exp+fold の job が既に生きているなら投げない
# =============================================================================
set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
ENVFILE="${PAI_ENV_FILE:-$DIR/cluster.env}"
# shellcheck disable=SC1090
set -a; . "$ENVFILE"; set +a

QUEUE_DIR="${QUEUE_DIR:-$ROOT/.queue}"
Q="$QUEUE_DIR/queue.txt"
QP="$QUEUE_DIR/queue_prio.txt"
D="$QUEUE_DIR/queue.done"
L="$QUEUE_DIR/queue.log"

CAP="${CAP:-8}"               # 同時 Running の上限（Kueue の実運用上限に合わせる）
RESERVE="${RESERVE:-2}"       # 後処理 job 用に空けておく枠数（training はこのぶん少ない上限になる）
MAXRETRY="${MAXRETRY:-8}"
POLL="${POLL:-120}"           # 枠待ちの再確認間隔（秒）

# ⚠ **コンペごとに埋める**: 既に完走している実験を二重に投げないためのガード。
#   コマンドから取り出した EXP / FOLD を埋め込んだパスが存在したら投げない。
#   空にするとこのガードは無効（① と ③ だけになる）。
DONE_ARTIFACT="${DONE_ARTIFACT:-experiments/{EXP}/oof/fold{FOLD}.csv}"

mkdir -p "$QUEUE_DIR"; touch "$D" "$Q"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
say "=== queue watcher 開始 (CAP=$CAP / RESERVE=$RESERVE / MAXRETRY=$MAXRETRY) ==="

# コマンドから実験名と fold を取り出す（`ARMS=xxx FOLDS=0` / `--exp xxx --folds 0` の両形式）
parse() {
  EXP=$(echo "$1" | grep -oE '\-\-exp[= ][A-Za-z0-9_]+' | sed 's/.*[= ]//')
  [ -z "${EXP:-}" ] && EXP=$(echo "$1" | grep -oE 'ARMS=[A-Za-z0-9_,]+' | cut -d= -f2)
  FOLD=$(echo "$1" | grep -oE '\-\-folds?[= ][0-9]+' | sed 's/.*[= ]//')
  [ -z "${FOLD:-}" ] && FOLD=$(echo "$1" | grep -oE 'FOLDS=[0-9]+' | cut -d= -f2)
}

declare -A RETRY
while :; do
  [ -f "$QUEUE_DIR/QUEUE_STOP" ] && { say "STOP ファイル検出。終了"; exit 0; }

  # ---- 次に投げる 1 行を決める（優先 queue が先）--------------------------------
  NEXT=""; FROM_PRIO=0
  for QQ in "$QP" "$Q"; do
    [ -f "$QQ" ] || continue
    [ "$QQ" = "$QP" ] && ISP=1 || ISP=0
    while IFS= read -r line; do
      case "$line" in ''|'#'*) continue;; esac
      grep -Fxq "$line" "$D" && continue                       # ガード①: 投入済み
      parse "$line"
      if [ -n "$DONE_ARTIFACT" ] && [ -n "${EXP:-}" ] && [ -n "${FOLD:-}" ]; then
        ART="${DONE_ARTIFACT//\{EXP\}/$EXP}"; ART="${ART//\{FOLD\}/$FOLD}"
        if [ -f "$ROOT/$ART" ]; then                           # ガード②: 成果物が既にある
          say "skip（成果物あり: $ART）: $EXP fold$FOLD"; echo "$line" >> "$D"; continue
        fi
      fi
      if [ -n "${EXP:-}" ] && [ -n "${FOLD:-}" ]; then          # ガード③: 同じ job が生存中
        ALIVE=$(kubectl get jobs -n "$PAI_NAMESPACE" -o json 2>/dev/null \
          | grep -oE '"PAI_CMD_B64"[^}]*' | grep -c "$EXP" || true)
        if [ "${ALIVE:-0}" -gt 0 ]; then
          say "skip（同一 exp の job が生存中）: $EXP fold$FOLD"; sleep "$POLL"; continue
        fi
      fi
      NEXT="$line"; FROM_PRIO=$ISP; break
    done < "$QQ"
    [ -n "$NEXT" ] && break
  done
  [ -z "$NEXT" ] && { say "queue は空（処理済み）。${POLL}s 後に再確認"; sleep "$POLL"; continue; }

  # ---- 枠が空くまで待つ ----------------------------------------------------------
  RUN=$(kubectl get jobs -n "$PAI_NAMESPACE" --no-headers 2>/dev/null \
        | grep -c "${PAI_JOB_PREFIX}-run.*Running" || true)
  # ⚠ 後処理（評価）job を training より優先するため、training には RESERVE 枚ぶん低い上限を使う。
  #   これが無いと、夜間に待機 training が枠を埋め尽くし、完走済み arm の評価が
  #   次の training 完了まで（十数時間）始まらない、という詰まり方をする。
  if [ "$FROM_PRIO" = "1" ]; then EFFCAP=$CAP; else EFFCAP=$((CAP - RESERVE)); fi
  if [ "${RUN:-99}" -ge "$EFFCAP" ]; then
    say "枠待ち: Running=$RUN >= $EFFCAP"; sleep "$POLL"; continue
  fi

  # ---- 1 本だけ投げる --------------------------------------------------------------
  cd "$ROOT" || exit 1
  OUT=$(PAI_MAX_GPU="${PAI_MAX_GPU:-$CAP}" "$DIR/submit.sh" run 1 "$NEXT" 2>&1)
  JOB=$(echo "$OUT" | grep -oE "${PAI_JOB_PREFIX}-run-[a-z0-9]+" | head -1)
  if [ -z "$JOB" ]; then
    say "submit 失敗（枠 or preflight）: $(echo "$OUT" | tail -1 | cut -c1-120)"; sleep "$POLL"; continue
  fi
  # ⚠ retry カウンタの key は **コマンド全体の hash**。先頭 N 文字だと fold 間で衝突する
  K=$(printf '%s' "$NEXT" | md5sum | cut -c1-16)
  RETRY[$K]=$(( ${RETRY[$K]:-0} + 1 ))
  say "submit  job=$JOB  retry=${RETRY[$K]}  cmd=$(echo "$NEXT" | cut -c1-90)"

  # ---- ADMITTED を確認してから次へ進む -----------------------------------------------
  OK=0
  for _ in $(seq 1 30); do          # 最大 5 分待つ
    A=$(kubectl get workloads -n "$PAI_NAMESPACE" --no-headers 2>/dev/null | grep "$JOB" | awk '{print $4}')
    [ "$A" = "True" ] && { OK=1; break; }
    sleep 10
  done
  if [ "$OK" = 1 ]; then
    say "        ADMITTED job=$JOB"
    echo "$NEXT" >> "$D"
  else
    # ⚠ 待たせ続けると Kueue が再キュー上限を超えて deactivate する。先に消す。
    say "        ⚠ TIMEOUT(5min) 未 ADMITTED → delete $JOB（deactivation 回避）"
    kubectl delete job -n "$PAI_NAMESPACE" "$JOB" >/dev/null 2>&1
    if [ "${RETRY[$K]}" -ge "$MAXRETRY" ]; then
      say "        ⛔ retry 上限 $MAXRETRY に到達。この行は保留して queue を進める"
      { echo "# HELD(retry>=$MAXRETRY) $NEXT"; echo "$NEXT"; } >> "$D"
    fi
    sleep 180
  fi
done
