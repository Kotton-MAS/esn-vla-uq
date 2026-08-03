"""argparse によるコマンドラインインタフェース本体。

サブコマンドの引数定義と実処理はそれぞれの層 (`diagnostics.commands` /
`data.commands`) に委譲し、ここではパーサの組み立てとロギング初期化のみを行う。
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from esn_vla_uq import __version__
from esn_vla_uq.data.commands import add_gen_sample_data_arguments, run_gen_sample_data
from esn_vla_uq.diagnostics.commands import add_diagnose_arguments, run_diagnose

logger = logging.getLogger(__name__)

DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _build_common_parser() -> argparse.ArgumentParser:
    """全サブコマンドが継承する共通オプションのパーサを返す。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="乱数シード (既定: %(default)s)",
    )
    common.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="出力先ディレクトリ (既定: %(default)s)",
    )
    common.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=DEFAULT_LOG_LEVEL,
        help="ログレベル (既定: %(default)s)",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """トップレベルのパーサを構築する。"""
    parser = argparse.ArgumentParser(
        prog="esn-vla-uq",
        description=(
            "Echo State Network による VLA ポリシーの不確実性定量化ツール。"
            "既定の入力は同梱の合成データ (source: synthetic) であり、"
            "実 LIBERO 評価の結果ではない。"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    common = _build_common_parser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_diagnose_arguments(
        subparsers.add_parser(
            "diagnose",
            parents=[common],
            help="リザバー診断 (スペクトル半径 / ESP / メモリ容量) を実行する",
        )
    )
    add_gen_sample_data_arguments(
        subparsers.add_parser(
            "gen-sample-data",
            parents=[common],
            help="合成ロールアウトのサンプルデータを生成する",
        )
    )
    return parser


def _configure_logging(level_name: str) -> None:
    """CLI エントリでのみ呼ぶロギング初期化。

    タイムスタンプは CLAUDE.md のログ出力ルールに従い UTC で記録する。
    """
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.getLevelNamesMapping()[level_name],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def _run_diagnose(args: argparse.Namespace) -> int:
    """`diagnose` サブコマンドを実行する (実体は `diagnostics.commands`)。"""
    return run_diagnose(args)


def _run_gen_sample_data(args: argparse.Namespace) -> int:
    """`gen-sample-data` サブコマンドを実行する (実体は `data.commands`)。"""
    return run_gen_sample_data(args)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI エントリポイント。終了コードを返す。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(str(args.log_level))

    if args.command == "diagnose":
        return _run_diagnose(args)
    if args.command == "gen-sample-data":
        return _run_gen_sample_data(args)
    parser.error(f"unknown command: {args.command}")
