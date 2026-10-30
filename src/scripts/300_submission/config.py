"""submission_config.yaml の読み込みと検証。

⚠ **埋め忘れた `{{...}}` が残っていたら hard fail する**。
提出は枠を消費する不可逆操作なので、「設定が半端なまま走る」余地を残さない。

実行:
    python src/scripts/300_submission/config.py      # 設定の中身と検証結果を表示
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "submission_config.yaml"
PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")


def _walk(node: Any, path: str = "") -> list[tuple[str, str]]:
    """未置換のプレースホルダを (yaml パス, 値) で列挙する。"""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            sub = f"{path}.{k}" if path else str(k)
            if isinstance(k, str) and PLACEHOLDER.search(k):
                found.append((sub, f"(キー名) {k}"))   # キー側の埋め忘れも拾う
            found += _walk(v, sub)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += _walk(v, f"{path}[{i}]")
    elif isinstance(node, str) and PLACEHOLDER.search(node):
        found.append((path, node))
    return found


def load(strict: bool = True) -> dict:
    """submission_config.yaml を読む。

    Args:
        strict: True なら未置換プレースホルダが 1 つでもあれば SystemExit。

    Returns:
        設定の dict。
    """
    if not CONFIG_PATH.exists():
        raise SystemExit(f"[!] {CONFIG_PATH} がありません")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    if strict:
        todo = _walk(cfg)
        if todo:
            print(f"[!] submission_config.yaml に未置換の項目が {len(todo)} 件あります:")
            for p, v in todo:
                print(f"    {p}: {v}")
            raise SystemExit(
                "    → コンペ固有の値で埋めてください。"
                "埋めないまま提出すると枠を無駄にします。"
            )

    spec = cfg.get("preproc", {}).get("specs") or {}
    if strict and not spec:
        raise SystemExit("[!] preproc.specs が空です。学習時の前処理バージョンを 1 つ以上定義してください")
    return cfg


def check_desc(desc: str, cfg: dict | None = None) -> None:
    """提出説明文に禁止語が含まれていないか検査する。

    Args:
        desc: 提出説明文。
        cfg: 設定 dict（省略時は読み込む）。
    """
    cfg = cfg or load()
    low = desc.lower()
    hit = [w for w in (cfg.get("banned_desc_words") or []) if w.lower() in low]
    if hit:
        raise SystemExit(
            f"[!] 提出説明に禁止語 {hit} が含まれます。\n"
            f"    説明: {desc!r}\n"
            f"    desc は自分の arm 名と blend 比だけにする（submission_config.yaml 参照）。"
        )


def main() -> None:
    cfg = load(strict=False)
    todo = _walk(cfg)
    print(f"■ {CONFIG_PATH.relative_to(ROOT)}")
    print(f"  competition : {cfg.get('competition')}")
    print(f"  kernel      : {cfg.get('kaggle_owner')}/{cfg.get('kernel_slug')}")
    print(f"  preproc key : {cfg.get('preproc', {}).get('config_key')}")
    print(f"  specs       : {list((cfg.get('preproc', {}).get('specs') or {}).keys())}")
    if todo:
        print(f"\n  ⛔ 未置換 {len(todo)} 件 — 提出前に埋めること")
        for p, v in todo:
            print(f"     {p}: {v}")
        sys.exit(1)
    print("\n  ✅ プレースホルダの埋め残しなし")


if __name__ == "__main__":
    main()
