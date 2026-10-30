#!/usr/bin/env bash
# =============================================================================
# gpu_status.sh — 共有クラスタの GPU 状況を「読める範囲で」1 行にまとめる
# =============================================================================
#   ./infra/k8s/gpu_status.sh
#   → 全体 N 枚（M ノード）/ 我々 使用 x 枚・待機 y 枚 / 空き: 推定 ...
#
# ⚠ 一般ユーザの RBAC では `pods -A` / `clusterqueue` / metrics.k8s.io は **読めない**ことが多い。
#   よって「他チームが何枚使っているか」は直接は分からない。空きは次の 2 つから推定する:
#     ① 我々の Pending pod に付く scheduler の FailedScheduling イベント
#        「0/30 nodes are available: Y Insufficient nvidia.com/gpu」→ 空きノードは (全体 − Y) 台
#     ② 直近で我々の pod が即 Running になったか（= その時点で空きがあった）
#
# 上限（PAI_MAX_GPU）を超えて使いたいときは、この出力を添えて確認を取る。
# =============================================================================
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ENVFILE="${PAI_ENV_FILE:-$DIR/cluster.env}"
# shellcheck disable=SC1090
[ -f "$ENVFILE" ] && { set -a; . "$ENVFILE"; set +a; }
NS="${PAI_NAMESPACE:?cluster.env に PAI_NAMESPACE が必要}"

# 全体（ノード数と GPU 総数）
read -r NN TOTAL < <(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null \
                     | awk '$1>0{n++; s+=$1} END{print n+0, s+0}')

# 我々（namespace の pod 要求）
read -r USE PEND < <(kubectl get pods -n "$NS" -o json 2>/dev/null | python3 -c '
import json,sys
run=pend=0
try: items=json.load(sys.stdin)["items"]
except Exception: items=[]
for p in items:
    g=sum(int(c.get("resources",{}).get("requests",{}).get("nvidia.com/gpu",0)) for c in p["spec"]["containers"])
    if p["status"]["phase"]=="Running": run+=g
    elif p["status"]["phase"]=="Pending": pend+=g
print(run, pend)' 2>/dev/null || echo "? ?")

# 空きの推定: 直近 30 分の FailedScheduling から「GPU 不足ノード数 Y」を拾う
Y=$(kubectl get events -n "$NS" --field-selector reason=FailedScheduling -o json 2>/dev/null | python3 -c '
import json,sys,re,datetime as dt
now=dt.datetime.now(dt.timezone.utc); best=None
try: items=json.load(sys.stdin)["items"]
except Exception: items=[]
for e in items:
    t=e.get("lastTimestamp") or e.get("eventTime") or ""
    try: ts=dt.datetime.fromisoformat(t.replace("Z","+00:00"))
    except Exception: continue
    if (now-ts).total_seconds()>1800: continue
    m=re.search(r"(\d+) Insufficient nvidia.com/gpu", e.get("message",""))
    if m and (best is None or ts>best[0]): best=(ts,int(m.group(1)))
print(best[1] if best else "")' 2>/dev/null)

if [ -n "$Y" ]; then
  EST="推定: GPU に空きがあるノード $((NN - Y))/${NN} 台（直近 30 分の scheduler イベント・枚数までは読めない）"
elif [ "${PEND:-0}" = "0" ] && [ "${USE:-0}" != "?" ]; then
  EST="推定: 我々の pod は全部即 Running（直近 30 分に GPU 待ちなし）= 少なくとも我々の要求分は空いていた"
else
  EST="不明（scheduler イベント無し・枚数を読む権限も無し）"
fi

echo "全体 ${TOTAL} 枚（${NN} ノード）/ 我々（${NS}）使用 ${USE} 枚・GPU 待ち ${PEND} 枚 / 空き: ${EST}"
echo "⚠ 他チームの使用量は一般ユーザ権限では読めない。上限を超えたいときは、この行を添えて確認を取る"
