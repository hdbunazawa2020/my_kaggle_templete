"""submit — Code Competition へ notebook version を提出する。

⚠⚠ **提出枠を消費する不可逆操作**。`--yes` を付けない限り dry-run。

## 背景
Code Competition では `kaggle competitions submit -f <file>` は **使えない**
（提出するのは CSV ではなく「notebook の version」なので）。
`kagglesdk` の `CreateCodeSubmission`（kernel_owner / kernel_slug / kernel_version）で提出する。

## 使い方

    # 残り枠と提出履歴を見る（提出しない）
    python src/scripts/300_submission/submit.py --list

    # 実際に提出する
    python src/scripts/300_submission/submit.py --version 3 --desc "exp001+exp002 rank 0.5:0.5" --yes

⚠ `--desc` には **自分の arm 名と blend 比だけ**を書く。
  禁止語は `submission_config.yaml` の `banned_desc_words` で管理する。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import check_desc, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def _limits(client, competition: str):
    from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionLimitsRequest

    r = ApiGetSubmissionLimitsRequest()
    r.competition_name = competition
    return client.competitions.competition_api_client.get_submission_limits(r)


def main() -> None:
    cfg = load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, help="提出する notebook の version 番号")
    ap.add_argument("--slug", default=None, help=f"kernel slug（既定 {cfg['kernel_slug']}）")
    ap.add_argument("--file-name", default=cfg["submission_file"], help="notebook が出力するファイル名")
    ap.add_argument("--desc", default="", help="提出の説明（★ 自分の arm 名のみ）")
    ap.add_argument("--list", action="store_true", help="残り枠と提出履歴だけ見る")
    ap.add_argument("--yes", action="store_true", help="⚠ 実際に提出する（無いと dry-run）")
    a = ap.parse_args()

    check_desc(a.desc, cfg)
    slug = a.slug or cfg["kernel_slug"]
    owner, comp = cfg["kaggle_owner"], cfg["competition"]

    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
        except ImportError:
            pass
    if not os.getenv("KAGGLE_API_TOKEN") and not (Path.home() / ".kaggle/kaggle.json").exists():
        raise SystemExit("[!] Kaggle の認証情報がありません（KAGGLE_API_TOKEN か ~/.kaggle/kaggle.json）")

    from kagglesdk import KaggleClient
    from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest

    with KaggleClient() as k:
        lim = _limits(k, comp)
        print(f"  残り枠 {lim.num_allowed_now} / 今日 {lim.num_today} 件提出済 / 通算 {lim.num_total}")
        if a.list or not a.version:
            return
        if lim.num_allowed_now < 1:
            raise SystemExit("[!] 今日の提出枠が残っていません")

        print(f"  提出しようとしているもの: {owner}/{slug} version {a.version}")
        print(f"    説明: {a.desc}")
        if not a.yes:
            print("\n  [dry-run] --yes を付けると実際に提出する（★ 枠を 1 消費する）")
            return

        r = ApiCreateCodeSubmissionRequest()
        r.competition_name = comp
        r.kernel_owner, r.kernel_slug, r.kernel_version = owner, slug, a.version
        r.file_name = a.file_name  # ⚠ 必須（省略すると 400 Bad Request）
        r.submission_description = a.desc or f"version {a.version}"
        res = k.competitions.competition_api_client.create_code_submission(r)
        print(f"  ✅ 提出した: {getattr(res, 'message', res)}")
        print(f"     残り枠 {_limits(k, comp).num_allowed_now}")


if __name__ == "__main__":
    main()
