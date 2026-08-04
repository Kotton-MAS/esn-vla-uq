"""`demo` サブコマンドのハンドラ。"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from esn_vla_uq.cli import options
from esn_vla_uq.cli.inputs import load_rollouts
from esn_vla_uq.demo.animate import (
    DEFAULT_FPS,
    DEFAULT_MAX_FRAMES,
    write_demo_animation,
)
from esn_vla_uq.demo.frames import build_demo_frames
from esn_vla_uq.esn.config import DEFAULT_N_RESERVOIR, ESNConfig
from esn_vla_uq.logging_paths import display_path
from esn_vla_uq.uncertainty.conformal import DEFAULT_ALPHA
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

REPORT_SUBDIR: Final[str] = "demo"
"""`--output-dir` 配下の書き出し先サブディレクトリ。"""

DEFAULT_FILENAME: Final[str] = "uncertainty_demo.gif"
"""既定の出力ファイル名。"""


def add_demo_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """`demo` 固有の引数を登録する。"""
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
        "--output",
        type=Path,
        default=None,
        help=(
            "出力する GIF のパス "
            f"(既定: <output-dir>/{REPORT_SUBDIR}/{DEFAULT_FILENAME})"
        ),
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help=(
            "描画するエピソード。既定はテスト集合の失敗エピソードのうち "
            "失敗開始以降の不確実性の上がり方が最大のもの"
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
            "非適合度スコア。absolute は区間幅が定数になりバーが跳ねない "
            "(既定: %(default)s)"
        ),
    )
    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLIT_STRATEGIES,
        default=DEFAULT_SPLIT_STRATEGY,
        help=(
            "較正データの分割方針。1 タスク 1 エピソードのデータでは "
            "within_task が 3 分割できないため across_task が要る "
            "(既定: %(default)s)"
        ),
    )
    parser.add_argument(
        "--n-reservoir",
        type=int,
        default=DEFAULT_N_RESERVOIR,
        help="リザバーのニューロン数 N (既定: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="GIF のフレームレート (既定: %(default)s)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="GIF に含める最大フレーム数 (既定: %(default)s)",
    )
    return parser


@dataclass(frozen=True)
class DemoOptions:
    """`demo` の型付き設定 (A7)。"""

    seed: int
    output_dir: Path
    input: Path | None
    output: Path | None
    episode_id: str | None
    alpha: float
    score_kind: ScoreKind
    split: SplitStrategy
    n_reservoir: int
    fps: int
    max_frames: int

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> DemoOptions:
        """`Namespace` から型を検証しつつ取り出す。"""
        return cls(
            seed=options.get_int(args, "seed"),
            output_dir=options.get_path(args, "output_dir"),
            input=options.get_optional_path(args, "input"),
            output=options.get_optional_path(args, "output"),
            episode_id=options.get_optional_str(args, "episode_id"),
            alpha=options.get_float(args, "alpha"),
            score_kind=options.get_choice(args, "score_kind", SUPPORTED_SCORE_KINDS),
            split=options.get_choice(args, "split", SUPPORTED_SPLIT_STRATEGIES),
            n_reservoir=options.get_int(args, "n_reservoir"),
            fps=options.get_int(args, "fps"),
            max_frames=options.get_int(args, "max_frames"),
        )


def run_demo(args: argparse.Namespace) -> int:
    """`demo` を実行する (`Namespace` の解釈のみを担う)。"""
    return execute_demo(DemoOptions.from_namespace(args))


def execute_demo(opts: DemoOptions) -> int:
    """デモアニメーションを生成する。

    Args:
        args: `add_demo_arguments` と共通オプションを含む名前空間。

    Returns:
        終了コード (成功時 0)。
    """
    seed = opts.seed
    config = ESNConfig(n_reservoir=opts.n_reservoir, seed=seed)
    dataset = load_rollouts(opts.input)

    frames = build_demo_frames(
        dataset,
        config,
        alpha=opts.alpha,
        score_kind=opts.score_kind,
        split_strategy=opts.split,
        split_seed=seed,
        episode_id=opts.episode_id,
    )
    output = opts.output
    path = (
        output
        if output is not None
        else opts.output_dir / REPORT_SUBDIR / DEFAULT_FILENAME
    )
    write_demo_animation(frames, path, fps=opts.fps, max_frames=opts.max_frames)

    ratio = frames.uncertainty_ratio_after_onset()
    lag = frames.detection_lag_steps()
    logger.info(
        "demo done: episode_id=%s task=%s success=%s n_steps=%d "
        "uncertainty_ratio_after_onset=%s detection_lag_steps=%s output=%s",
        frames.episode_id,
        frames.task_name,
        frames.success,
        frames.n_steps,
        "n/a" if ratio is None else f"{ratio:.2f}x",
        "n/a" if lag is None else str(lag),
        display_path(path),
    )
    if lag is not None:
        logger.warning(
            "不確実性の立ち上がりは失敗開始の %d ステップ後です。"
            "予兆ではなく反応であり、遅れはチャンク周期 (推論間隔) で決まります",
            lag,
        )
    return 0
