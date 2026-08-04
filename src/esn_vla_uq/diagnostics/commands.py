"""`diagnose` サブコマンドのハンドラ。

CLI 本体 (`esn_vla_uq.cli.app`) はここで定義した関数を呼ぶだけにして、診断の
ロジックを `diagnostics` レイヤ側に閉じ込める (`data.commands` と同じ構成)。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from esn_vla_uq.cli import options
from esn_vla_uq.diagnostics.report import write_report
from esn_vla_uq.diagnostics.runner import (
    DEFAULT_DIAGNOSTICS_N_INPUTS,
    run_diagnostics,
)
from esn_vla_uq.esn.config import (
    DEFAULT_LEAK_RATE,
    DEFAULT_N_RESERVOIR,
    DEFAULT_SPECTRAL_RADIUS,
    ESNConfig,
)
from esn_vla_uq.logging_paths import display_path

logger = logging.getLogger(__name__)


def add_diagnose_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """`diagnose` 固有の引数を登録する。"""
    parser.add_argument(
        "--n-reservoir",
        type=int,
        default=DEFAULT_N_RESERVOIR,
        help="リザバーのニューロン数 N (既定: %(default)s)",
    )
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=DEFAULT_SPECTRAL_RADIUS,
        help="再帰行列の目標スペクトル半径 rho (既定: %(default)s)",
    )
    parser.add_argument(
        "--leak-rate",
        type=float,
        default=DEFAULT_LEAK_RATE,
        help="リーク率 a (既定: %(default)s)",
    )
    parser.add_argument(
        "--n-inputs",
        type=int,
        default=DEFAULT_DIAGNOSTICS_N_INPUTS,
        help=(
            "spectral/esp を計算するリザバーの入力次元 D_u (既定: %(default)s)。"
            "メモリ容量診断は D_u=1 を要求するため、これが 1 以外のときは"
            "メモリ容量だけ別の D_u=1 リザバーで測定する"
        ),
    )
    parser.add_argument(
        "--skip-memory-capacity",
        action="store_true",
        help=(
            "メモリ容量診断を省略する "
            "(遅延ごとの read-out 学習を伴い計算コストが高いため)"
        ),
    )
    return parser


@dataclass(frozen=True)
class DiagnoseOptions:
    """`diagnose` の型付き設定。

    `argparse.Namespace` の無型アクセスを `from_namespace` の 1 箇所に閉じ込め、
    ハンドラ本体は型の付いた値だけを扱う (A7)。
    """

    seed: int
    output_dir: Path
    n_reservoir: int
    spectral_radius: float
    leak_rate: float
    n_inputs: int
    skip_memory_capacity: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> DiagnoseOptions:
        """`Namespace` から型を検証しつつ取り出す。"""
        return cls(
            seed=options.get_int(args, "seed"),
            output_dir=options.get_path(args, "output_dir"),
            n_reservoir=options.get_int(args, "n_reservoir"),
            spectral_radius=options.get_float(args, "spectral_radius"),
            leak_rate=options.get_float(args, "leak_rate"),
            n_inputs=options.get_int(args, "n_inputs"),
            skip_memory_capacity=options.get_bool(args, "skip_memory_capacity"),
        )


def run_diagnose(args: argparse.Namespace) -> int:
    """`diagnose` を実行する (`Namespace` の解釈のみを担う)。"""
    return execute_diagnose(DiagnoseOptions.from_namespace(args))


def execute_diagnose(opts: DiagnoseOptions) -> int:
    """リザバー診断を実行し JSON レポートを書き出す。

    Args:
        args: `add_diagnose_arguments` と共通オプションを含む名前空間。

    Returns:
        終了コード (成功時 0)。
    """
    seed = opts.seed
    config = ESNConfig(
        n_reservoir=opts.n_reservoir,
        spectral_radius=opts.spectral_radius,
        leak_rate=opts.leak_rate,
        seed=seed,
    )
    report = run_diagnostics(
        config,
        n_inputs=opts.n_inputs,
        seed=seed,
        skip_memory_capacity=opts.skip_memory_capacity,
    )
    report.log_summary()
    path = write_report(report, opts.output_dir)
    logger.info(
        "diagnose done: n_reservoir=%d seed=%d esp_verdict=%s report=%s",
        config.n_reservoir,
        seed,
        report.esp.verdict,
        display_path(path),
    )
    return 0
