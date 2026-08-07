"""`calibrate` サブコマンドのハンドラ。

CLI 本体 (`esn_vla_uq.cli.app`) はここで定義した関数を呼ぶだけにして、較正評価の
ロジックを `calibration` レイヤ側に閉じ込める。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from esn_vla_uq.calibration.metrics import ReliabilityCurve
from esn_vla_uq.calibration.plot import (
    DIAGRAM_TITLE_TEMPLATE,
    write_reliability_diagram,
)
from esn_vla_uq.calibration.report import (
    DIAGRAM_FILENAME,
    REPORT_SUBDIR,
    write_report,
)
from esn_vla_uq.calibration.runner import DEFAULT_N_SPLITS, run_calibration
from esn_vla_uq.cli import options
from esn_vla_uq.cli.inputs import load_rollouts
from esn_vla_uq.esn.config import (
    DEFAULT_LEAK_RATE,
    DEFAULT_N_RESERVOIR,
    DEFAULT_READOUT_FEATURES,
    DEFAULT_SPECTRAL_RADIUS,
    SUPPORTED_READOUT_FEATURES,
    ESNConfig,
    ReadoutFeatures,
)
from esn_vla_uq.logging_paths import display_path
from esn_vla_uq.uncertainty.conformal import DEFAULT_ALPHA, DEFAULT_WASHOUT
from esn_vla_uq.uncertainty.nonconformity import (
    DEFAULT_SCORE_KIND,
    SUPPORTED_SCORE_KINDS,
    ScoreKind,
)
from esn_vla_uq.uncertainty.split import (
    DEFAULT_SPLIT_STRATEGY,
    SUPPORTED_SPLIT_STRATEGIES,
    SplitStrategy,
)

logger = logging.getLogger(__name__)


def add_calibrate_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """`calibrate` 固有の引数を登録する。"""
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "評価する .npz、または収集した openpi ログのディレクトリ "
            "(既定: 同梱の合成サンプルデータ)"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="有意水準。名目被覆率は 1 - alpha (既定: %(default)s)",
    )
    parser.add_argument(
        "--score-kind",
        choices=SUPPORTED_SCORE_KINDS,
        default=DEFAULT_SCORE_KIND,
        help=(
            "非適合度スコア。absolute は区間幅が定数になりステップを区別しない "
            "(既定: %(default)s)"
        ),
    )
    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLIT_STRATEGIES,
        default=DEFAULT_SPLIT_STRATEGY,
        help=(
            "較正データの分割方針。across_task は交換可能性が崩れるため"
            "被覆率保証が弱い (既定: %(default)s)"
        ),
    )
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
        help="再帰行列のスペクトル半径 (既定: %(default)s)",
    )
    parser.add_argument(
        "--leak-rate",
        type=float,
        default=DEFAULT_LEAK_RATE,
        help="リーク率 (既定: %(default)s)",
    )
    parser.add_argument(
        "--readout",
        choices=SUPPORTED_READOUT_FEATURES,
        default=DEFAULT_READOUT_FEATURES,
        help=(
            "read-out の設計行列。input_reservoir は [1, u, x]、reservoir_only は "
            "[1, x]、input_only は [1, u] (リザバー無しの baseline) "
            "(既定: %(default)s)"
        ),
    )
    parser.add_argument(
        "--washout",
        type=int,
        default=DEFAULT_WASHOUT,
        help=(
            "エピソードごとに先頭から捨てる標本数。既定の 0 は初期過渡も"
            "「予測しづらい区間」として評価に含める。ESNConfig.washout とは別物で、"
            "較正経路に効くのはこちら (既定: %(default)s)"
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=(
            "評価する分割の数。被覆率は単一分割では代表値にならないため "
            "複数分割の平均で報告する (既定: %(default)s)"
        ),
    )
    parser.add_argument(
        "--diagram",
        action="store_true",
        help=(
            "reliability diagram を PNG で書き出す "
            "(matplotlib が必要: uv sync --extra viz)"
        ),
    )
    return parser


@dataclass(frozen=True)
class CalibrateOptions:
    """`calibrate` の型付き設定 (A7)。"""

    seed: int
    output_dir: Path
    input: Path | None
    alpha: float
    score_kind: ScoreKind
    split: SplitStrategy
    n_reservoir: int
    spectral_radius: float
    leak_rate: float
    readout: ReadoutFeatures
    washout: int
    n_splits: int
    diagram: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> CalibrateOptions:
        """`Namespace` から型を検証しつつ取り出す。"""
        return cls(
            seed=options.get_int(args, "seed"),
            output_dir=options.get_path(args, "output_dir"),
            input=options.get_optional_path(args, "input"),
            alpha=options.get_float(args, "alpha"),
            score_kind=options.get_choice(args, "score_kind", SUPPORTED_SCORE_KINDS),
            split=options.get_choice(args, "split", SUPPORTED_SPLIT_STRATEGIES),
            n_reservoir=options.get_int(args, "n_reservoir"),
            spectral_radius=options.get_float(args, "spectral_radius"),
            leak_rate=options.get_float(args, "leak_rate"),
            readout=options.get_choice(args, "readout", SUPPORTED_READOUT_FEATURES),
            washout=options.get_int(args, "washout"),
            n_splits=options.get_int(args, "n_splits"),
            diagram=options.get_bool(args, "diagram"),
        )


def run_calibrate(args: argparse.Namespace) -> int:
    """`calibrate` を実行する (`Namespace` の解釈のみを担う)。"""
    return execute_calibrate(CalibrateOptions.from_namespace(args))


def execute_calibrate(opts: CalibrateOptions) -> int:
    """較正評価を実行してレポートを書き出す。

    Args:
        args: `add_calibrate_arguments` と共通オプションを含む名前空間。

    Returns:
        終了コード (成功時 0)。
    """
    seed = opts.seed
    input_passthrough, use_reservoir = ESNConfig.readout_flags(opts.readout)
    config = ESNConfig(
        n_reservoir=opts.n_reservoir,
        spectral_radius=opts.spectral_radius,
        leak_rate=opts.leak_rate,
        input_passthrough=input_passthrough,
        use_reservoir=use_reservoir,
        seed=seed,
    )
    dataset = load_rollouts(opts.input)

    report = run_calibration(
        dataset,
        config,
        alpha=opts.alpha,
        score_kind=opts.score_kind,
        split_strategy=opts.split,
        split_seed=seed,
        n_splits=opts.n_splits,
        washout=opts.washout,
    )
    report.log_summary()
    path = write_report(report, opts.output_dir)

    if opts.diagram:
        _write_diagram(report.reliability, opts.output_dir, report.data_source)

    # 失敗エピソードを含まない出所では AUROC が定義できず `None` になる
    # (`DetectionSummary.unavailable_reason`)。`%.4f` に None を渡すと logging が
    # 整形に失敗し、**この 1 行だけが消えてスタックトレースが stderr に出る**。
    # 数値が出ないこと自体は異常ではないので、`demo` と同じく "n/a" と書く。
    auroc = report.detection.mean_auroc
    logger.info(
        "calibrate done: readout=%s score_kind=%s split=%s coverage=%.4f±%.4f "
        "width=%.4f auroc=%s report=%s",
        config.readout_features,
        report.conformal["score_kind"],
        report.split["strategy"],
        report.coverage.mean,
        report.coverage.std,
        report.coverage.mean_interval_width,
        "n/a" if auroc is None else f"{auroc:.4f}",
        display_path(path),
    )
    return 0


def _write_diagram(curve: ReliabilityCurve, output_dir: Path, data_source: str) -> None:
    """reliability diagram を書き出す。matplotlib が無ければ警告に留める。

    図が出せないことで較正評価そのものを失敗にはしない。数値はすでにレポート
    JSON に書かれている。
    """
    path = output_dir / REPORT_SUBDIR / DIAGRAM_FILENAME
    try:
        write_reliability_diagram(
            curve, path, title=DIAGRAM_TITLE_TEMPLATE.format(data_source=data_source)
        )
    except ImportError as error:
        logger.warning("reliability diagram を書き出せませんでした: %s", error)
        return
    logger.info("saved reliability diagram: path=%s", display_path(path))
