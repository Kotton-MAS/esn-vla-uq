"""`calibrate` サブコマンドのハンドラ。

CLI 本体 (`esn_vla_uq.cli.app`) はここで定義した関数を呼ぶだけにして、較正評価の
ロジックを `calibration` レイヤ側に閉じ込める。
"""

from __future__ import annotations

import argparse
import logging
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
from esn_vla_uq.data.io import load_bundled_sample, load_dataset
from esn_vla_uq.esn.config import (
    DEFAULT_LEAK_RATE,
    DEFAULT_N_RESERVOIR,
    DEFAULT_SPECTRAL_RADIUS,
    ESNConfig,
)
from esn_vla_uq.logging_paths import display_path
from esn_vla_uq.uncertainty.conformal import DEFAULT_ALPHA
from esn_vla_uq.uncertainty.nonconformity import (
    DEFAULT_SCORE_KIND,
    SUPPORTED_SCORE_KINDS,
)
from esn_vla_uq.uncertainty.split import (
    DEFAULT_SPLIT_STRATEGY,
    SUPPORTED_SPLIT_STRATEGIES,
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
        help="評価する .npz のパス (既定: 同梱の合成サンプルデータ)",
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


def run_calibrate(args: argparse.Namespace) -> int:
    """較正評価を実行してレポートを書き出す。

    Args:
        args: `add_calibrate_arguments` と共通オプションを含む名前空間。

    Returns:
        終了コード (成功時 0)。
    """
    seed = int(args.seed)
    config = ESNConfig(
        n_reservoir=int(args.n_reservoir),
        spectral_radius=float(args.spectral_radius),
        leak_rate=float(args.leak_rate),
        seed=seed,
    )
    input_path: Path | None = args.input
    dataset = load_bundled_sample() if input_path is None else load_dataset(input_path)

    report = run_calibration(
        dataset,
        config,
        alpha=float(args.alpha),
        score_kind=args.score_kind,
        split_strategy=args.split,
        split_seed=seed,
        n_splits=int(args.n_splits),
    )
    report.log_summary()
    path = write_report(report, Path(args.output_dir))

    if args.diagram:
        _write_diagram(report.reliability, Path(args.output_dir), report.data_source)

    logger.info(
        "calibrate done: score_kind=%s split=%s coverage=%.4f±%.4f "
        "auroc=%.4f report=%s",
        report.conformal["score_kind"],
        report.split["strategy"],
        report.coverage.mean,
        report.coverage.std,
        report.detection.mean_auroc,
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
