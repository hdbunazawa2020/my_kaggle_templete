"""check_kernel_order — 提出 kernel の **トップレベル前方参照**を静的に検出する。

■ なぜ必要か（RSNA での実例）
  時間ガードの位置を入れ替えたら、**後ろで定義される名前**をトップレベルで読む形になり、
  Kaggle の commit run が `NameError: name 'xxx' is not defined` で ERROR になった。
  - `ast.parse` は通ってしまう（構文は正しい）
  - ローカル疎通（run_kernel_local.py）は GPU 無しだと数十分かかる
  ⇒ **この 1 点だけを秒で見る**検査を build_kernel.py から自動で通す。

■ 何を見るか / 見ないか
  - トップレベル文を実行順に走査し、Name の load が **その時点で束縛済みか**を確認する。
  - ⚠ `def` / `class` / `lambda` の **本体は見ない**（呼ばれるのは後なので後方の名前を読んでよい）。
    ただし decorator / default 引数 / 基底クラスは定義時に評価されるので見る。
  - ⚠ 内包表記・ジェネレータ式は自分のスコープで target を束縛する ⇒ 外に漏らさない。
  - `if` / `for` / `while` / `try` / `with` の中はトップレベル実行なので見る。
    分岐内の束縛は「束縛されうる」として楽観的に加える（**偽陽性を出さない**ため）。
  ⇒ 見逃しはありうるが、偽陽性は出さない。目的は「移動・並べ替えで壊れたときに気づく」こと。

実行:
    python src/scripts/300_submission/check_kernel_order.py [file.py]
"""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

BUILTIN = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
COMP = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
DEFER = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _targets(node: ast.AST | None) -> set[str]:
    """代入先として現れる名前を集める。"""
    return {
        n.id
        for n in (ast.walk(node) if node is not None else [])
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }


def _binds(node: ast.AST) -> set[str]:
    """`node` を実行したあと **モジュール名前空間に残る**名前を返す。"""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    if isinstance(node, COMP + DEFER):
        return set()  # 内包表記 / lambda / def の中の束縛は外に出ない
    out: set[str] = set()
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        out.add(node.id)
    if isinstance(node, ast.ExceptHandler) and node.name:
        out.add(node.name)
    for sub in ast.iter_child_nodes(node):
        out |= _binds(sub)
    return out


def forward_refs(text: str, filename: str = "<kernel>") -> list[tuple[int, str]]:
    """トップレベルで **後に定義される名前を読んでいる箇所**を返す。

    Args:
        text: kernel の `.py` ソース。
        filename: エラー表示用のファイル名。

    Returns:
        `(行番号, 名前)` のリスト。空なら問題なし。
    """
    known: set[str] = set(BUILTIN)
    bad: list[tuple[int, str]] = []

    def loads(node: ast.AST, local: frozenset[str] = frozenset()):
        if isinstance(node, DEFER):
            for sub in (
                list(getattr(node, "decorator_list", []))
                + [d for d in node.args.defaults if d]
                + [d for d in getattr(node.args, "kw_defaults", []) if d]
                + ([node.returns] if getattr(node, "returns", None) else [])
            ):
                yield from loads(sub, local)
            return
        if isinstance(node, ast.ClassDef):
            for sub in node.decorator_list + list(node.bases) + list(node.keywords):
                yield from loads(sub, local)
            return
        if isinstance(node, COMP):
            inner = local | {n for g in node.generators for n in _targets(g.target)}
            for g in node.generators:
                yield from loads(g.iter, inner)
                for c in g.ifs:
                    yield from loads(c, inner)
            for f in ("elt", "key", "value"):
                if getattr(node, f, None) is not None:
                    yield from loads(getattr(node, f), inner)
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in local and node.id not in known:
                yield (node.lineno, node.id)
            return
        for sub in ast.iter_child_nodes(node):
            yield from loads(sub, local)

    def scan(body: list[ast.stmt]) -> None:
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bad.extend(loads(st))
                known.add(st.name)
                continue
            if isinstance(
                st, (ast.If, ast.For, ast.While, ast.With, ast.AsyncFor, ast.AsyncWith, ast.Try)
            ):
                for f in ("test", "iter"):
                    if getattr(st, f, None) is not None:
                        bad.extend(loads(getattr(st, f)))
                for it in getattr(st, "items", []) or []:
                    bad.extend(loads(it.context_expr))
                    known.update(_targets(it.optional_vars))
                known.update(_targets(getattr(st, "target", None)))
                for f in ("body", "orelse", "finalbody"):
                    scan(getattr(st, f, []) or [])
                for h in getattr(st, "handlers", []) or []:
                    if h.type:
                        bad.extend(loads(h.type))
                    if h.name:
                        known.add(h.name)
                    scan(h.body)
                continue
            bad.extend(loads(st))
            known.update(_binds(st))

    scan(ast.parse(text, filename=filename).body)
    seen: set[tuple[int, str]] = set()
    return [x for x in bad if x not in seen and not seen.add(x)]


def main() -> None:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    else:
        from config import load

        p = Path(__file__).resolve().parents[3] / load()["kernel"]["source"]
    text = p.read_text(encoding="utf-8")
    hits = forward_refs(text, str(p))
    print(f"■ {p.name} のトップレベル前方参照検査")
    if hits:
        src = text.splitlines()
        print(f"⛔ **その行より後で定義される名前を読んでいる箇所が {len(hits)} 件**")
        for ln, nm in hits:
            print(f"   L{ln}: `{nm}`   {src[ln - 1].strip()[:100]}")
        print("\n⚠ push すると commit run が NameError で落ちる。直してから push する。")
        sys.exit(1)
    print("  ✅ 前方参照なし ⇒ トップレベルの実行順で NameError にはならない")


if __name__ == "__main__":
    main()
