"""run_kernel_local — 提出 notebook を **Kaggle に push する前にローカルで通す**。

## なぜ必要か
Kaggle は 1 往復（push → 実行 → ログ取得）に **約 10 分**かかる。
RSNA ではこれを怠って 1 日に 6 回失敗し、1 時間以上を溶かした:

| 失敗 | 原因 |
|---|---|
| `ModuleNotFoundError` | `src/` に `__init__.py` が無く、同名の pip パッケージに負けた |
| `Dataset が attach されていない` | `/kaggle/input` が `{competitions,datasets}` の入れ子構造だった |
| `NameError` | 書き換えで定義が落ちた（→ check_kernel_order.py で検出するようにした） |
| `CUDA error: no kernel image` | 割当 GPU が古く、新しい torch が非対応だった → machine_shape を明示 |
| `weights が揃っていない` | `.py` を直して `.ipynb` の再生成を忘れた／パス解決が 1 階層ずれた |

→ **Kaggle と同じディレクトリ構造をローカルに作って notebook を丸ごと実行する。**
   再現できないのは GPU 種別だけで、それ以外はここで捕まえられる。

## 使い方
    python src/scripts/300_submission/build_kernel.py --arms '["exp001"]'
    python src/scripts/300_submission/run_kernel_local.py --dataset-dir data/kaggle_upload/<slug>

⚠ `--dataset-dir` には「Kaggle Dataset にした中身」のローカル側を渡す。
  複数 dataset を attach する kernel なら、それらを並べた親ディレクトリを渡せばよい
  （kernel は `/kaggle/input` を再帰探索する前提で書く）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def _copy(src: Path, dst: Path) -> None:
    """まずハードリンク（一瞬・容量ゼロ）、駄目なら実コピー。

    ⚠ 別ファイルシステム間ではハードリンクを張れない（Invalid cross-device link）。
      そのとき `cp -al` は **ディレクトリの殻だけ**を残して失敗するので、
      殻を消してからコピーし直す（入れ子になって中身が見つからなくなる事故の防止）。
    """
    if subprocess.run(["cp", "-al", str(src), str(dst)], capture_output=True).returncode == 0:
        return
    subprocess.run(["rm", "-rf", str(dst)], check=True)
    subprocess.run(["cp", "-r", str(src), str(dst)], check=True)


def build_fake_input(work: Path, ds_dir: Path, cfg: dict, comp_files: list[str]) -> Path:
    """`/kaggle/input/{competitions,datasets}/...` と同じ構造を作る。

    ⚠ **symlink ではなく実体（ハードリンク）にすること。**
      Python の `Path.glob("**/...")` は **symlink を辿らない**ので、
      symlink で作ると notebook のパス解決が空振りして再現にならない。
    """
    inp = work / "fakeinput"
    if inp.exists():
        return inp
    (inp / "datasets" / cfg["kaggle_owner"]).mkdir(parents=True)
    (inp / "competitions").mkdir(parents=True)
    _copy(ds_dir, inp / "datasets" / cfg["kaggle_owner"] / ds_dir.name)
    comp = inp / "competitions" / cfg["competition"]
    comp.mkdir()
    for f in comp_files:
        src = ROOT / "data/raw" / f
        if not src.exists():
            print(f"  ⚠ {src} が無いのでスキップ（kernel が必要とするなら失敗します）")
            continue
        _copy(src, comp / Path(f).name)
    return inp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Kaggle Dataset にした中身のローカル側")
    ap.add_argument("--nb", default=None, help="実行する .ipynb（既定は本体 kernel）")
    ap.add_argument("--work", default="/tmp/kernel_local")
    ap.add_argument(
        "--comp-files",
        default="sample_submission.csv,test.csv",
        help="data/raw から fakeinput へコピーするコンペ配布物（カンマ区切り）",
    )
    a = ap.parse_args()

    cfg = load()
    nb_path = (
        Path(a.nb).resolve()
        if a.nb
        else ROOT / cfg["kernel"]["out_root"] / "kernel" / f"{cfg['kernel_slug']}.ipynb"
    )
    if not nb_path.exists():
        raise SystemExit(f"[!] {nb_path} がありません。先に build_kernel.py を実行してください")

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    inp = build_fake_input(
        work,
        Path(a.dataset_dir).resolve(),
        cfg,
        [x.strip() for x in a.comp_files.split(",") if x.strip()],
    )
    (work / "fakework").mkdir(exist_ok=True)

    nb = nbf.read(nb_path, as_version=4)
    code = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
    sub = cfg["submission_file"]
    code = (
        code.replace('Path("/kaggle/input")', f'Path("{inp}")')
        .replace('Path("/kaggle/working")', f'Path("{work}/fakework")')
        .replace(f'to_csv("{sub}", index=False)', f'to_csv("{work}/fakework/{sub}", index=False)')
    )
    script = work / "nb_local.py"
    script.write_text(code, encoding="utf-8")

    print(f"  input = {inp}")
    r = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT, env={**os.environ, "PYTHONPATH": ""}
    )
    if r.returncode:
        print("\n  [!] ローカルで落ちた。**この状態で push しない**（Kaggle は 1 往復 10 分）")
        sys.exit(1)
    out = work / "fakework" / sub
    print(f"\n  ✅ ローカル疎通 OK（{out}）→ push してよい:")
    print(f"     kaggle kernels push -p {cfg['kernel']['out_root']}/kernel")


if __name__ == "__main__":
    main()
