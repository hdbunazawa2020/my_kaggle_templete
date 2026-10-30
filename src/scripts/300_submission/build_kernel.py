"""build_kernel — `# %%` 形式の `.py` から Kaggle 提出用 `.ipynb` を生成する。

⚠⚠ **`.py` が正本で `.ipynb` は生成物**。`.ipynb` を直接編集しない。
    （RSNA では `.py` だけ直して `.ipynb` の再生成を忘れ、Kaggle で 10 分かけて
      **古いコードが走った**。生成物を手で触ると必ずこれをやる。）

## このスクリプトが持つ 3 つの事故防止機構

1. **前処理仕様の自動導出 + hard fail**
   学習時の前処理バージョンを実験 config から読み、kernel 定数に注入する。
   CLI で明示した値と食い違ったら **既定で止める**（`--allow-preproc-override` で明示的にのみ逸脱可）。
   → 「黙って別の前処理で推論する kernel」を提出する事故を防ぐ。

2. **トップレベル前方参照チェック**（`check_kernel_order.py`）
   `ast.parse` は通るのに commit run が `NameError` で落ちる事故を push 前に捕まえる。

3. **生成内容の目視出力**
   どの arm / どの定数で作ったかを必ず print する。push 前に目で確認する。

## 使い方

    python src/scripts/300_submission/build_kernel.py --arms '["exp001"]'
    python src/scripts/300_submission/build_kernel.py --arms '["exp001","exp002"]' \\
        --slot b --set TTA_MODE='"off"' --set K_EVAL=22

`--set KEY=VALUE` は kernel ソースのトップレベル定数 `KEY = ...` を丸ごと置き換える。
（文字列を入れたいときは `--set NAME='"value"'` のようにクォートごと渡す）

⚠ **提出 kernel を作ったら生成コマンドを必ず記録する**（README.md 参照）。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def _derive_preproc(cfg: dict, arms: list[str]) -> tuple[str, dict]:
    """arm の config.yaml から前処理バージョンを読み、kernel 定数の dict を返す。

    Args:
        cfg: submission_config.yaml の内容。
        arms: 1 つの kernel に載せる実験名のリスト。

    Returns:
        `(バージョン名, {定数名: 値})`。

    Raises:
        SystemExit: config が無い / arm 間でバージョンが食い違う / 未対応バージョン。
    """
    import yaml

    key = cfg["preproc"]["config_key"]
    vers: dict[str, str] = {}
    for e in arms:
        p = ROOT / "experiments" / e / "config.yaml"
        if not p.exists():
            raise SystemExit(f"[!] {e} の config.yaml が無い（前処理仕様を導出できない）: {p}")
        c = yaml.safe_load(p.read_text(encoding="utf-8"))
        v = c.get(key)
        if v is None:
            raise SystemExit(f"[!] {e} の config.yaml に '{key}' が無い")
        vers[e] = v

    uniq = set(vers.values())
    if len(uniq) != 1:
        raise SystemExit(f"[!] 1 kernel 内の arm で {key} が食い違っている: {vers}")
    ver = uniq.pop()

    unsupported = cfg["preproc"].get("unsupported") or {}
    if ver in unsupported:
        raise SystemExit(f"[!] {key}={ver} は kernel で再現できない: {unsupported[ver]}")

    specs = cfg["preproc"]["specs"]
    if ver not in specs:
        raise SystemExit(
            f"[!] 未知の {key}={ver}。submission_config.yaml の preproc.specs に\n"
            f"    この前処理バージョンの kernel 定数を追記してから作り直すこと。\n"
            f"    （黙って既定値で作ると学習と推論の前処理がズレる）"
        )
    print(f"  ★ 前処理仕様を自動導出: {key}={ver} → {specs[ver]}")
    return ver, dict(specs[ver])


def _replace_const(src: str, name: str, value: str) -> str:
    """トップレベル定数 `name = ...` を value に置き換える。

    ⚠ リストや dict など複数行にまたがる定数も 1 文としてまとめて置換する
      （1 行だけ替えると続きの行が残って壊れる）。
    """
    pat = re.compile(rf"^{re.escape(name)} = .*?(?=^\S|\Z)", re.S | re.M)
    if not pat.search(src):
        raise SystemExit(f"[!] kernel ソースにトップレベル定数 `{name} = ...` が見つかりません")
    return pat.sub(f"{name} = {value}   # build_kernel.py が生成時に設定\n", src, count=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help='例: \'["exp001","exp002"]\'')
    ap.add_argument("--slot", default="", help="'' なら本体、'b' なら 2 本目の kernel（並列実行用）")
    ap.add_argument("--datasets", default=None, help="dataset_sources をカンマ区切りで明示指定")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="kernel のトップレベル定数を上書き（複数可）",
    )
    ap.add_argument(
        "--allow-preproc-override",
        action="store_true",
        help="⚠ 自動導出した前処理仕様と違う値を意図的に使う（既定は hard fail）",
    )
    a = ap.parse_args()

    cfg = load()
    src_path = ROOT / cfg["kernel"]["source"]
    if not src_path.exists():
        raise SystemExit(f"[!] kernel ソースがありません: {src_path}")

    try:
        arms = list(ast.literal_eval(a.arms))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[!] --arms が Python リテラルとして読めません: {exc}") from exc
    if not arms:
        raise SystemExit("[!] --arms が空です")

    # --- ① 前処理仕様の自動導出と突き合わせ -------------------------------------------
    _, derived = _derive_preproc(cfg, arms)
    overrides: dict[str, str] = {}
    for kv in a.set:
        if "=" not in kv:
            raise SystemExit(f"[!] --set は KEY=VALUE 形式: {kv}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v

    conflicts = [
        f"--set {k}={overrides[k]} vs 導出 {derived[k]}"
        for k in derived
        if k in overrides and str(overrides[k]).strip() != str(derived[k]).strip()
    ]
    if conflicts:
        if not a.allow_preproc_override:
            raise SystemExit(
                "[!] CLI の指定が前処理仕様の導出値と食い違う: "
                + " / ".join(conflicts)
                + "\n    学習と推論の前処理がズレるので既定では止める。"
                "\n    意図的なら --allow-preproc-override を付けること。"
            )
        print("  ⚠⚠ --allow-preproc-override: 導出値と食い違うまま作る → " + " / ".join(conflicts))

    consts: dict[str, str] = {k: str(v) for k, v in derived.items()}
    consts.update(overrides)  # CLI が最後に勝つ（上で衝突は検査済み）
    consts["ARMS"] = json.dumps(arms)

    # --- ② ソースに定数を流し込む --------------------------------------------------------
    src = src_path.read_text(encoding="utf-8")
    for k, v in consts.items():
        src = _replace_const(src, k, v)

    ast.parse(src)  # 文法エラーは push 前に落とす

    # --- ③ トップレベル前方参照チェック（ast.parse では捕まらない）-----------------------
    from check_kernel_order import forward_refs

    hits = forward_refs(src, str(src_path))
    if hits:
        lines = src.splitlines()
        print(f"  ⛔ **後で定義される名前を読んでいる箇所が {len(hits)} 件** ⇒ .ipynb を書かない")
        for ln, nm in hits:
            print(f"     L{ln}: `{nm}`   {lines[ln - 1].strip()[:100]}")
        raise SystemExit("[!] このまま push すると commit run が NameError で落ちる")

    # --- ④ .ipynb と kernel-metadata.json を書く --------------------------------------
    slot = f"-{a.slot}" if a.slot else ""
    out_dir = ROOT / cfg["kernel"]["out_root"] / f"kernel{slot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / f"{cfg['kernel_slug']}{slot}.ipynb"

    nb = nbf.v4.new_notebook()
    for blk in src.split("# %%"):
        if not blk.strip():
            continue
        if blk.lstrip().startswith("[markdown]"):
            body = blk.lstrip()[len("[markdown]") :]
            md = "\n".join(
                ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                for ln in body.strip("\n").splitlines()
            )
            nb.cells.append(nbf.v4.new_markdown_cell(md.strip()))
        else:
            nb.cells.append(nbf.v4.new_code_cell(blk.strip("\n")))
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, out_nb)

    sources = (
        [x.strip() for x in a.datasets.split(",") if x.strip()]
        if a.datasets
        else list(cfg["kernel"]["base_dataset_sources"])
    )
    meta = {
        "id": f"{cfg['kaggle_owner']}/{cfg['kernel_slug']}{slot}",
        "title": f"{cfg['kernel_title']}{slot.upper()}",
        "code_file": out_nb.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(cfg["kernel"].get("is_private", True)),
        "enable_gpu": bool(cfg["kernel"].get("enable_gpu", True)),
        "enable_internet": bool(cfg["kernel"].get("enable_internet", False)),
        "dataset_sources": sources,
        "competition_sources": [cfg["competition"]],
        "kernel_sources": [],
        "machine_shape": cfg["kernel"].get("machine", "None"),
    }
    # ⚠ .gitignore が *.json を一括無視している場合、この生成物は git から見えない。
    #   追跡したいなら `git add -f` すること。
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # --- ⑤ 何を作ったかを必ず目視させる -------------------------------------------------
    print(f"  ✅ {out_nb}  ({len(nb.cells)} cells)")
    print(f"     ARMS = {arms}")
    for k in sorted(consts):
        if k != "ARMS":
            print(f"     {k} = {consts[k]}")
    print(f"     dataset_sources = {sources}")
    print("     → 次: python src/scripts/300_submission/run_kernel_local.py で疎通確認")
    print("     ⚠ 生成コマンドを README.md の記録欄に必ず残すこと")


if __name__ == "__main__":
    main()
