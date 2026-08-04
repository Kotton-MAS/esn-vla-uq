"""`gen-sample-data` サブコマンドのハンドラ。

CLI 本体 (`esn_vla_uq.cli.app`) はここで定義した関数を呼ぶだけにして、データ生成の
ロジックを `data` レイヤ側に閉じ込める。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from esn_vla_uq.data.io import save_dataset
from esn_vla_uq.data.sources import SyntheticRolloutSource
from esn_vla_uq.data.synthetic import DEFAULT_N_EPISODES
from esn_vla_uq.logging_paths import display_path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_NAME = "synthetic_rollouts.npz"
"""`--output` 省略時に `--output-dir` 配下へ書き出すファイル名。"""


def add_gen_sample_data_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """`gen-sample-data` 固有の引数を登録する。"""
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_N_EPISODES,
        help="生成するエピソード数 (既定: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "出力する .npz のパス "
            f"(既定: <output-dir>/{DEFAULT_OUTPUT_NAME}。サイドカー JSON も同名で生成)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "既存の .npz とサイドカー JSON を上書きする "
            "(既定では、どちらかが存在するとエラーで中断する)"
        ),
    )
    return parser


def run_gen_sample_data(args: argparse.Namespace) -> int:
    """合成サンプルデータを生成して保存する。

    Args:
        args: `add_gen_sample_data_arguments` と共通オプションを含む名前空間。

    Returns:
        終了コード (成功時 0)。
    """
    seed = int(args.seed)
    n_episodes = int(args.n_episodes)
    output: Path | None = args.output
    archive_path = (
        output if output is not None else Path(args.output_dir) / DEFAULT_OUTPUT_NAME
    )

    dataset = SyntheticRolloutSource(seed=seed, n_episodes=n_episodes).load()
    saved_path = save_dataset(dataset, archive_path, overwrite=bool(args.force))
    n_success = sum(1 for episode in dataset.episodes if episode.success)
    logger.info(
        "gen-sample-data done: source=%s seed=%d n_episodes=%d n_success=%d "
        "total_steps=%d archive=%s",
        dataset.source,
        dataset.seed,
        dataset.n_episodes,
        n_success,
        dataset.total_steps,
        display_path(saved_path),
    )
    return 0
