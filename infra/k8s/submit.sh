#!/usr/bin/env bash
# =============================================================================
# submit.sh — GPU Job を k8s クラスタへ投入する
# =============================================================================
# 編集 Pod / ノートPC には GPU が無い前提。学習は必ずこれで投げる。
#
# 使い方:
#   ./infra/k8s/submit.sh run <ngpu> "<command>"    # 任意コマンド
#   ./infra/k8s/submit.sh <preset>                  # 下の PRESETS に定義したもの
#
# 例:
#   ./infra/k8s/submit.sh run 1 "ARMS=exp001 .venv-k8s/bin/python src/scripts/200_train_model/200_train_model.py"
#   ./infra/k8s/submit.sh preflight
#
# コマンドは repo root から、`.env` を source した状態で走る。
#
# ⚠ arm を並列で回したいときは 1 arm = 1 job として複数投げる。
#   ただし一度に大量投入しないこと（infra/k8s/README.md の Kueue の節を読む）。
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$DIR/job.template.yaml"
ENVFILE="${PAI_ENV_FILE:-$DIR/cluster.env}"

[ -f "$ENVFILE" ] || {
  echo "ERROR: $ENVFILE がありません。cluster.env.example をコピーして埋めてください" >&2
  exit 1
}
# shellcheck disable=SC1090
set -a; . "$ENVFILE"; set +a
[ -f "$TEMPLATE" ] || { echo "ERROR: template がありません: $TEMPLATE" >&2; exit 1; }

PY="${PAI_PY:?cluster.env に PAI_PY が必要}"
TRAIN="${PAI_TRAIN_SCRIPT:-src/scripts/200_train_model/200_train_model.py}"

# -----------------------------------------------------------------------------
# preset: 名前 -> "NGPU|COMMAND"（コマンドは repo root から走る）
# ⚠ **コンペごとにここを埋める**。よく使う組み合わせだけで十分。
# -----------------------------------------------------------------------------
declare -A PRESETS=(
  # 長時間 job を焼く前に「GPU が見えるか / venv が壊れていないか / データが揃っているか」を
  # 1 分で確かめる。image を変えたら必ず先にこれを通すこと。
  [preflight]="1|$PY src/scripts/210_collect_results/preflight.py"
  [collect]="1|$PY src/scripts/210_collect_results/210_collect_results.py"
  # [train-all]="1|$PY $TRAIN"
)

usage() {
  echo "Usage:"
  echo "  $0 run <ngpu> \"<command>\""
  echo "  $0 <preset>"
  echo
  echo "Presets:"
  for k in $(echo "${!PRESETS[@]}" | tr ' ' '\n' | sort); do
    printf "  %-16s ngpu=%s  cmd=%s\n" "$k" "${PRESETS[$k]%%|*}" "${PRESETS[$k]#*|}"
  done
  exit 1
}

[ $# -ge 1 ] || usage
KEY="$1"
if [ "$KEY" = "run" ]; then
  [ $# -eq 3 ] || { echo "ERROR: 'run' には <ngpu> と \"<command>\" が要ります" >&2; usage; }
  NGPU="$2"; CMD="$3"; JOBNAME="run"
elif [ -n "${PRESETS[$KEY]:-}" ]; then
  NGPU="${PRESETS[$KEY]%%|*}"; CMD="${PRESETS[$KEY]#*|}"; JOBNAME="$KEY"
else
  echo "ERROR: 未知の preset '$KEY'" >&2; usage
fi

# k8s のオブジェクト名は英小文字・数字・'-' のみ
JOBNAME="$(echo "$JOBNAME" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
[ -n "$JOBNAME" ] || JOBNAME="job"
[[ "$NGPU" =~ ^[0-9]+$ ]] || { echo "ERROR: <ngpu> は整数。受け取った値: '$NGPU'" >&2; exit 1; }

# -----------------------------------------------------------------------------
# ⚠⚠ 共有 GPU の占領防止
#   我々の未完了 job（Running + Suspended）が要求する GPU 合計に今回の NGPU を足して
#   PAI_MAX_GPU を超えるなら投げない。
#   上限を変えるときは `PAI_MAX_GPU=24 ./infra/k8s/submit.sh ...` のように
#   **空き状況を確認したその回だけ** env で上書きする。
# -----------------------------------------------------------------------------
PAI_MAX_GPU="${PAI_MAX_GPU:-8}"
if [ "$NGPU" -gt 0 ]; then
  # ⚠ Kueue に deactivated された job（spec.active=false の workload）は二度と動かないので数えない
  DEAD=$(kubectl get workloads -n "$PAI_NAMESPACE" -o json 2>/dev/null | python3 -c '
import json,sys
try: items=json.load(sys.stdin)["items"]
except Exception: items=[]
for w in items:
    if w.get("spec",{}).get("active") is False:
        for o in w["metadata"].get("ownerReferences",[]): print(o["name"])' 2>/dev/null | tr '\n' ' ')
  IN_USE=$(kubectl get jobs -n "$PAI_NAMESPACE" -l "app=$PAI_APP_LABEL" -o json 2>/dev/null \
    | DEAD="$DEAD" python3 -c '
import json,sys,os
dead=set(os.environ.get("DEAD","").split()); tot=0
try: items=json.load(sys.stdin)["items"]
except Exception: items=[]
for j in items:
    st=j.get("status",{})
    if st.get("succeeded") or st.get("failed"): continue   # 終わった job は数えない
    if j["metadata"]["name"] in dead: continue              # deactivated は数えない
    tot+=int(j["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"].get("nvidia.com/gpu",0))
print(tot)' 2>/dev/null || echo 0)
  if [ $((IN_USE + NGPU)) -gt "$PAI_MAX_GPU" ]; then
    echo "⛔ GPU 上限: 使用中 ${IN_USE} + 今回 ${NGPU} > PAI_MAX_GPU=${PAI_MAX_GPU}。共有クラスタなので待つ。" >&2
    echo "   超えたいときは空き状況を確認してから PAI_MAX_GPU=<新上限> を付けて投げる。" >&2
    "$DIR/gpu_status.sh" >&2 || true
    exit 2
  fi
  echo "  GPU: 使用中 ${IN_USE} + 今回 ${NGPU} ≤ ${PAI_MAX_GPU}"
fi

# コマンドは base64 で渡す（パイプ・クォート・'&' があっても YAML/sed が壊れない）
CMD_B64="$(printf '%s' "$CMD" | base64 -w0)"
VENV_DIR="$(dirname "$(dirname "$PY")")"
CERT_FILE="${PAI_CERT_FILE:-${PAI_PROJECT_DIR}/${VENV_DIR}/lib/python${PAI_PY_VER:-3.12}/site-packages/certifi/cacert.pem}"

echo "Submitting: JOB=$JOBNAME  NGPU=$NGPU"
echo "  CMD:   $CMD"
echo "  IMAGE: $PAI_IMAGE"

RENDERED="$(sed \
    -e "s|%%JOBNAME%%|${JOBNAME}|g" \
    -e "s|%%JOB_PREFIX%%|${PAI_JOB_PREFIX}|g" \
    -e "s|%%APP_LABEL%%|${PAI_APP_LABEL}|g" \
    -e "s|%%NAMESPACE%%|${PAI_NAMESPACE}|g" \
    -e "s|%%QUEUE_NAME%%|${PAI_QUEUE_NAME}|g" \
    -e "s|%%NGPU%%|${NGPU}|g" \
    -e "s|%%IMAGE%%|${PAI_IMAGE}|g" \
    -e "s|%%CMD_B64%%|${CMD_B64}|g" \
    -e "s|%%PROJECT_DIR%%|${PAI_PROJECT_DIR}|g" \
    -e "s|%%DATA_PVC%%|${PAI_DATA_PVC}|g" \
    -e "s|%%DATA_MOUNT%%|${PAI_DATA_MOUNT}|g" \
    -e "s|%%HOME_PVC%%|${PAI_HOME_PVC}|g" \
    -e "s|%%HOME_MOUNT%%|${PAI_HOME_MOUNT}|g" \
    -e "s|%%HOME%%|${PAI_HOME}|g" \
    -e "s|%%POD_USER%%|${PAI_USER}|g" \
    -e "s|%%UID%%|${PAI_UID}|g" \
    -e "s|%%GID%%|${PAI_GID}|g" \
    -e "s|%%FSGROUP%%|${PAI_FSGROUP}|g" \
    -e "s|%%PY%%|${PY}|g" \
    -e "s|%%CPU%%|${PAI_CPU}|g" \
    -e "s|%%MEM%%|${PAI_MEM}|g" \
    -e "s|%%SHM%%|${PAI_SHM}|g" \
    -e "s|%%GPU_TAINT_KEY%%|${PAI_GPU_TOLERATION_KEY}|g" \
    -e "s|%%CERT_FILE%%|${CERT_FILE}|g" \
    -e "s|%%WANDB_PROJECT%%|${WANDB_PROJECT:-}|g" \
    -e "s|%%WANDB_ENTITY%%|${WANDB_ENTITY:-}|g" \
    "$TEMPLATE")"

# Kueue / GPU taint を使わないクラスタでは該当行を落とす
RENDERED="$(printf '%s' "$RENDERED" | QN="${PAI_QUEUE_NAME:-}" TK="${PAI_GPU_TOLERATION_KEY:-}" python3 -c '
import os, sys
qn, tk = os.environ.get("QN", ""), os.environ.get("TK", "")
out, skip = [], 0
for ln in sys.stdin.read().splitlines():
    if skip > 0:                       # tolerations ブロックの残り行を捨てる
        skip -= 1
        continue
    if not qn and "kueue.x-k8s.io/queue-name" in ln:
        continue
    if not tk and ln.strip() == "tolerations:":
        skip = 3                        # - key / operator / effect の 3 行
        continue
    out.append(ln)
print("\n".join(out))')"

echo "$RENDERED" | kubectl create -f -

echo
echo "Watch: kubectl get pods -n $PAI_NAMESPACE -l job=$JOBNAME -w"
echo "Logs:  kubectl logs -n $PAI_NAMESPACE -f -l job=$JOBNAME --tail=50"
echo "Stop:  kubectl delete jobs -n $PAI_NAMESPACE -l job=$JOBNAME"
echo
echo "⚠ 投入から 30 分たったら Suspended が残っていないか必ず確認する（infra/k8s/README.md）:"
echo "   kubectl get workloads -n $PAI_NAMESPACE --no-headers | awk '\$3==\"False\"{print \$1}'"
