"""stage_arm — 実験の weights を **提出用 Kaggle Dataset の形**に並べ、upload する。

    # 並べるだけ（upload しない）
    python src/scripts/300_submission/stage_arm.py --exp exp001
    # upload まで行う
    python src/scripts/300_submission/stage_arm.py --exp exp001 --upload

## なぜ必要か
kernel は `/kaggle/input` を再帰探索して `<arm>/fold0.pt` と `<arm>/infer_meta.json` を探す。
この `infer_meta.json` を**手で書いていた**ため生成手順が repo に残らず、
「どの設定で学習した重みなのか」が後から分からなくなった。必ずこのスクリプトで生成する。

## ⚠ 2 つの実戦上の注意

1. **`weights/<arm>/` と 1 段深くする。**
   `kaggle datasets create --dir-mode zip` は `-p` 直下の各ディレクトリを `<dir>.zip` に固めるが、
   Kaggle 側で展開されると **その dir 名が消えてフラットに置かれる**。
   RSNA では `<slug>/fold0.pt` になってしまい、kernel が探す `<arm>/fold0.pt` が見つからず提出が失敗した。
   `weights/<arm>/` と入れ子にすれば zip の中に `<arm>/` が残る。

2. **ckpt に入らない設定は必ず `infer_meta.json` に運ぶ。**
   `persistent=False` の buffer や、層の有無を決めるフラグは state_dict に現れない。
   meta に無いと kernel は**違う形のモデルを組み**、しかも `load_state_dict` が
   missing/unexpected ゼロで通ってしまう（= 無言で別モデルを推論する）。
   運ぶ項目は `submission_config.yaml` の `meta_fields` に列挙する。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "data/kaggle_upload"


def _dig(d: dict, dotted: str) -> Any:
    """`"model.backbone"` のようなドット区切りで dict を辿る。"""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="カンマ区切りで複数可（1 dataset にまとめる）")
    ap.add_argument("--slug", default=None, help="既定 <dataset_slug_prefix>-<先頭 exp>")
    ap.add_argument("--upload", action="store_true", help="⚠ Kaggle へ upload する")
    ap.add_argument("--n-ckpt", type=int, default=None, help="期待する ckpt 数（fold 数）。合わなければ止める")
    a = ap.parse_args()

    cfg = load()
    exps = [x.strip() for x in a.exp.split(",") if x.strip()]
    slug = a.slug or f"{cfg['dataset_slug_prefix']}-{exps[0]}"
    root = STAGE / slug
    total = 0.0

    for e in exps:
        ed = ROOT / "experiments" / e
        ck = sorted((ed / "model").glob("fold*.pt")) or sorted((ed / "model").glob("full.pt"))
        if not ck:
            raise SystemExit(f"[!] {e}: model/fold*.pt も model/full.pt も見つかりません（{ed/'model'}）")
        if a.n_ckpt is not None and len(ck) != a.n_ckpt:
            raise SystemExit(f"[!] {e}: ckpt が {len(ck)} 個（--n-ckpt={a.n_ckpt} と不一致）: {[c.name for c in ck]}")

        cfg_path = ed / "config.yaml"
        if not cfg_path.exists():
            raise SystemExit(f"[!] {e}: config.yaml がありません（学習設定を推測してはいけない）")
        ec = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        res_path = ed / "result.json"
        res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}

        meta: dict[str, Any] = {
            "exp": e,
            "n_ckpt": len(ck),
            "cv": res.get("cv") or res.get("cv_mean"),
            cfg["preproc"]["config_key"]: ec.get(cfg["preproc"]["config_key"]),
        }
        missing = []
        for f in cfg.get("meta_fields") or []:
            v = _dig(ec, f)
            if v is None:
                missing.append(f)
            meta[f.split(".")[-1]] = v
        if missing:
            # ⚠ 止める。meta の取りこぼしは「無言で別モデル」に直結する。
            raise SystemExit(
                f"[!] {e}: meta_fields {missing} が config.yaml から読めません。\n"
                f"    submission_config.yaml の meta_fields を実際の config 構造に合わせてください。"
            )

        # ⚠ 1 段深くする（上の注意 1）
        d = root / "weights" / e
        d.mkdir(parents=True, exist_ok=True)
        for c in ck:
            if not (d / c.name).exists():
                shutil.copy2(c, d / c.name)
        (d / "infer_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        sz = sum(f.stat().st_size for f in d.iterdir()) / 1e9
        total += sz
        print(f"  ✅ {e:26s} {sz:.2f} GB  ckpt={len(ck)}  meta={ {k: meta[k] for k in list(meta)[:4]} }")

    (root / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": slug[:50], "id": f"{cfg['kaggle_owner']}/{slug}", "licenses": [{"name": "other"}]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✅ {root}\n   {len(exps)} arm / {total:.2f} GB")

    if not a.upload:
        print("\n（upload するには --upload）")
        return

    cmd = ["kaggle", "datasets", "create", "-p", str(root), "--dir-mode", "zip"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out.strip()[:400])
    if "already exists" in out or "409" in out:  # ⚠ 既にあるなら version を上げる
        r = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(root), "--dir-mode", "zip", "-m", f"{a.exp} weights"],
            capture_output=True,
            text=True,
        )
        print((r.stdout + r.stderr).strip()[:400])
    print(f"\n→ kernel-metadata.json の dataset_sources に \"{cfg['kaggle_owner']}/{slug}\" を足す")
    print("   （build_kernel.py --datasets で指定するか、submission_config.yaml に追記する）")


if __name__ == "__main__":
    main()
