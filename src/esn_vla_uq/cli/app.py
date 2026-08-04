"""argparse によるコマンドラインインタフェース本体。

サブコマンドの引数定義と実処理はそれぞれの層 (`diagnostics.commands` /
`data.commands`) に委譲し、ここではパーサの組み立て・ロギング初期化・
トップレベルの例外処理のみを行う。

未捕捉の例外をそのまま送出するとトレースバックが stderr に出る。トレースバックは
ソースの絶対パス (`/home/<ユーザー名>/...`) を含み、公開後は issue への貼り付けで
環境情報が漏れる (S3、CWE-209)。`main` は例外を捕まえ、INFO 以下では例外の型と
メッセージだけを出し、トレースバックは DEBUG でのみ出す。
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from esn_vla_uq import __version__
from esn_vla_uq.calibration.commands import add_calibrate_arguments, run_calibrate
from esn_vla_uq.data.commands import add_gen_sample_data_arguments, run_gen_sample_data
from esn_vla_uq.demo.commands import add_demo_arguments, run_demo
from esn_vla_uq.diagnostics.commands import add_diagnose_arguments, run_diagnose

logger = logging.getLogger(__name__)

DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

EXIT_OK: Final[int] = 0
"""正常終了。"""

EXIT_ERROR: Final[int] = 1
"""想定内の失敗 (不正な入力・出力先の衝突など)。"""

EXIT_INTERRUPTED: Final[int] = 130
"""Ctrl-C による中断 (128 + SIGINT)。"""


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


class _Subcommand(NamedTuple):
    """サブコマンド 1 つ分の定義。

    A6 への対応。以前はサブコマンドを 1 つ足すのに、import・パーサ登録・
    ハンドラのラッパ・`_dispatch` の分岐という 4 箇所を同時に編集する必要が
    あった。定義をこのテーブル 1 箇所にまとめ、`set_defaults(handler=...)` で
    振り分ける (argparse の標準機能)。
    """

    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
    handler: Callable[[argparse.Namespace], int]


SUBCOMMANDS: Final[tuple[_Subcommand, ...]] = (
    _Subcommand(
        name="diagnose",
        help="リザバー診断 (スペクトル半径 / ESP / メモリ容量) を実行する",
        add_arguments=add_diagnose_arguments,
        handler=run_diagnose,
    ),
    _Subcommand(
        name="gen-sample-data",
        help="合成ロールアウトのサンプルデータを生成する",
        add_arguments=add_gen_sample_data_arguments,
        handler=run_gen_sample_data,
    ),
    _Subcommand(
        name="calibrate",
        help="conformal 予測区間の較正評価 (被覆率 / ECE / 失敗検知) を実行する",
        add_arguments=add_calibrate_arguments,
        handler=run_calibrate,
    ),
    _Subcommand(
        name="demo",
        help="不確実性バーのデモアニメーション (GIF) を生成する",
        add_arguments=add_demo_arguments,
        handler=run_demo,
    ),
)
"""サブコマンドの一覧。追加するときはここに 1 要素足すだけでよい。"""


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
    for subcommand in SUBCOMMANDS:
        subparser = subparsers.add_parser(
            subcommand.name, parents=[common], help=subcommand.help
        )
        subcommand.add_arguments(subparser)
        subparser.set_defaults(handler=subcommand.handler)
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


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """`set_defaults(handler=...)` で登録したハンドラを呼ぶ。"""
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
    return handler(args)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI エントリポイント。終了コードを返す。

    未捕捉の例外をトレースバックごと stderr に出さない (S3、CWE-209)。
    トレースバックはソースの絶対パスを含み、公開後は issue への貼り付けで
    環境情報が漏れる。利用者に見せるのは例外の型とメッセージまでとし、
    完全なトレースバックは `--log-level DEBUG` を明示したときだけ出す。

    Args:
        argv: コマンドライン引数 (省略時は `sys.argv[1:]`)。

    Returns:
        終了コード。正常終了 0、想定内の失敗 1、Ctrl-C による中断 130。
        引数解析の失敗は argparse が `SystemExit(2)` を送出する
        (使い方の誤りであり、実行時エラーとは区別する)。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(str(args.log_level))

    try:
        return _dispatch(parser, args)
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        return EXIT_INTERRUPTED
    except Exception as error:
        # 例外の型名とメッセージまでに留める。メッセージ自体はパッケージ内で
        # 送出したものが大半で、パスは `display_path` 済み。
        logger.error("%s: %s", type(error).__name__, error)
        logger.debug("unhandled exception", exc_info=True)
        return EXIT_ERROR
