# infra/k8s — GPU クラスタで学習を回す

Kubernetes + Kueue のクラスタに学習 Job を投げるための一式。
**単一サーバで回すなら、このディレクトリは丸ごと無視してよい**（テンプレート本体は単一サーバ前提のまま）。

ここに書いてあるのは RSNA Knee（2026-08〜10）で実際に踏んだ穴とその対処。
**半日〜12 時間を溶かした事故が複数含まれている**ので、クラスタを使う前に一度通して読むこと。

---

## 0. 次回コンペでやること（チェックリスト）

- [ ] `cp cluster.env.example cluster.env` して全ての `{{...}}` を埋める
- [ ] `cluster.env` を `.gitignore` に追加する（クラスタ名・PVC 名は社内情報のことが多い）
- [ ] repo root の `.env` に API キー類を置く（Job が起動時に source する）
- [ ] `submit.sh` の `PRESETS` によく使うコマンドを 2〜3 個書く
- [ ] `queue_watcher.sh` の `DONE_ARTIFACT` をこのコンペの成果物パスに直す
- [ ] `./infra/k8s/submit.sh preflight` が通ることを確認してから長時間 job を投げる

---

## 1. 構成

| ファイル | 役割 |
|---|---|
| `cluster.env.example` | クラスタ固有の値を全部ここに集約（コピーして `cluster.env` を作る） |
| `job.template.yaml` | Job マニフェストの雛形。`%%TOKEN%%` は `submit.sh` が埋める |
| `submit.sh` | Job 投入。**GPU 使用上限のガード付き** |
| `submit_folds.sh` | 1 実験の N fold を N job に分けて並列投入 |
| `gpu_status.sh` | 読める範囲でクラスタの GPU 空き状況を要約 |
| `queue_watcher.sh` | 枠が空き次第 1 本ずつ投げる常駐スクリプト（夜間用） |

### 設計の前提
- **image にコードを焼かない。** repo も data も venv も共有 PVC の上にあり、Job は同じ PVC を
  マウントする。コードを直したら次の job からすぐ反映される（ビルド待ちゼロ）。
- **秘密情報は image にも YAML にも入れない。** Job は起動時に repo root の `.env` を source する。

---

## 2. ⚠⚠ Kueue: `Suspended` のまま動かないとき

**GPU の取り合いとは限らない。** Kueue が**再キュー回数の上限超過で workload を無効化**していることがある。
こうなった workload は **二度と動かない**（放っておいても復活しない）。

```bash
# 症状: kubectl get jobs → Suspended が大量、Running が 0。だが新しく投げた job は動く
kubectl get workloads -n "$PAI_NAMESPACE" --no-headers | awk '$3=="False"'   # admit されていない
kubectl describe workload -n "$PAI_NAMESPACE" <name> | grep -A3 Conditions
#   Message: The workload is deactivated due to exceeding the maximum number of re-queuing retries
#   Reason:  DeactivatedDueToRequeuingLimitExceeded
```

- ⚠ `kubectl patch workload ... {"spec":{"active":true}}` は **一般ユーザ権限では Forbidden**。復活させられない。
- **対処**: 死んだ job を消して、**足りない fold だけ**投げ直す。
- **予防**:
  - 一度に大量投入しない（`queue_watcher.sh` が 1 本ずつ投げるのはこのため）
  - **投入から 30 分たったら必ず確認する**:
    ```bash
    kubectl get workloads -n "$PAI_NAMESPACE" --no-headers | awk '$3=="False"{print $1}'   # 空でなければ死んでいる
    ```

> 実例: 03:20 に 40 job を一度に投げ、**14 job が 12 時間死んだまま**放置された。
> 別の日には評価 job 1 本が **4 時間気づかれず死んでいた**（Running が 0 なのを誰も見ていなかった）。

### 📌 Suspended が 30 分を超えたら deactivation を疑う
これを合言葉にしておくと早く気づける。`Suspended` 自体は正常な待ち状態なので、**時間**で判断する。

---

## 3. ⚠ Head-of-line blocking

Kueue は原則 **キューの先頭から順に** admit する。
先頭に「8 GPU 要求」のような大きい job が居座ると、**後ろの 1 GPU job は空きがあっても流れない**。

- 大きい job と小さい job を同じキューに混ぜない。混ぜるなら大きい方を後に投げる。
- `queue_watcher.sh` は「短時間の評価 job 用に `RESERVE` 枚だけ枠を空けておく」仕組みを持っている。
  これが無いと、夜間に待機 training が枠を埋め尽くし、**完走済み arm の評価が十数時間始まらない**。

---

## 4. ⚠ コンテナレジストリ（Harbor 等）の脆弱性ブロック

- **Critical CVE を含む image は `pull` が 412 Precondition Failed で拒否される。**
  "Prevent vulnerable images from running" が有効なレジストリでは、ある日の scan を境に
  **それまで動いていた image が全部 pull 不能**になりうる（設定変更には admin 権限が要る）。
- **同名タグへの push も 412 になる。** → **image を更新するときはタグを必ず上げる**（`v9` → `v10`）。
- 回避策: ベースイメージ自体が弱点になることもある（`FROM` 行の pull すら 412）。
  torch の cu wheel は CUDA ランタイムを同梱するので、**薄い public image + PVC 上の venv** にすれば
  レジストリの状態に依存しなくなる。

### ⚠ 巨大 image は Kueue の期限に負ける
cold node で 6.5GB の image を pull すると 25 分かかり、**PodsReady の 30 分期限**に引っかかって
job が deactivated になった（一晩で 18 job が死んだ）。薄い image ほど安全。

---

## 5. ⚠ 2 つの venv を使い分ける

| venv | python の実体 | 用途 |
|---|---|---|
| `.venv` | **image 内**の python | 開発 Pod でのみ使う |
| `.venv-k8s` | **PVC 上**の python | **k8s Job で使う**。image に依存しない |

image 内の python を指した venv を Job で使うと、image を変えた瞬間に動かなくなる。

---

## 6. GPU 使用上限の運用

共有クラスタでは「いま何枚まで使ってよいか」を決めておく（`cluster.env` の `PAI_MAX_GPU`）。
`submit.sh` は**自分の未完了 job の要求 GPU 合計**を数えて、上限を超える投入を止める。

超えたいときは `gpu_status.sh` の出力を添えて確認を取り、**その回だけ** env で上書きする:
```bash
PAI_MAX_GPU=24 ./infra/k8s/submit.sh run 1 "..."
```

> 上限ガードが無かった頃、気づかないうちに同時 30 GPU を占有していたことがある。

---

## 7. 📌 長時間処理は必ずクラスタ側に置く

**手元の端末（devcontainer / ノート PC / クラウド IDE）は予告なく止まる。**
アイドルタイムアウト・Spot インスタンスの中断・ネットワーク切断、いずれも日常的に起きる。

→ **長時間処理と監視ループはクラスタ側に置く**:
- 学習そのもの: k8s Job（`submit.sh`）
- 定期実行: CronJob
- 夜間のキュー消化・監視: **dev Pod 内の tmux**（`queue_watcher.sh`）

手元が落ちても困らない設計にしておけば、「朝起きたら 8 時間ぶんの GPU が無駄になっていた」を避けられる。

---

## 8. Job 名の付け方

社内ダッシュボードの「全 namespace のジョブ一覧」に他チームと並んで表示される。
**コンペ名や個人名をそのまま出さない中立な接頭辞**にしておくと角が立たない（`cluster.env` の `PAI_JOB_PREFIX`）。

⚠ 運用コマンドのセレクタは `job=<name>` **ラベル**なので、接頭辞を変えても手順は変わらない:
```bash
kubectl get pods -n "$PAI_NAMESPACE" -l job=run -w
kubectl logs  -n "$PAI_NAMESPACE" -f -l job=run --tail=50
kubectl delete jobs -n "$PAI_NAMESPACE" -l job=run
```

---

## 9. その他の実測メモ

| 事象 | 対処 |
|---|---|
| public image は既定 root。共有ストレージに root 所有ファイルが生まれる | `securityContext` で uid/gid を開発 Pod に合わせる |
| `HOME` が `/root` になり PVC 上の uv/wandb 設定を見失う | `HOME` を env で明示 |
| `getpass.getuser()` が `KeyError: getpwuid()` で落ちる（torch._inductor が呼ぶ） | `USER` / `LOGNAME` を env で与える |
| 薄い image に CA 証明書が無く HTTPS が張れない | venv 同梱の certifi を `SSL_CERT_FILE` 等に指定 |
| NFS 上の `TMPDIR` だと DataLoader worker 終了時に大量エラー | `TMPDIR=/dev/shm` |
| torch/triton のコンパイルキャッシュを NFS に置くと遅い・ロック事故 | `/dev/shm` に逃がす |
