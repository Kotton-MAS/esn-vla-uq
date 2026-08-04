"""`--input` の解決 (同梱サンプル / `.npz` / openpi ログディレクトリ)。

サブコマンドが受け取るのは「データセットの在り処」であって出所ではない。
どの供給元で読むかはパスの形から決める。

- 省略 → 同梱の合成サンプルデータ
- ファイル (`.npz`) → `load_dataset` (本リポジトリの保存形式)
- ディレクトリ → `OpenpiLogSource` (収集した openpi ログ)

ファイルとディレクトリは取り違えようがないので、`--source` のような追加の
フラグは設けない。どちらとして読んだかは INFO ログに出す。

**この dispatch を `data` 層に置かない。** 具象供給元を選ぶのは利用者入力の解釈で
あり CLI の仕事である。`data/io.py` や `data/invariants.py` は具象を知らないまま
にしておく (A1 / S7)。
"""

from __future__ import annotations

import logging
from pathlib import Path

from esn_vla_uq.data.io import load_bundled_sample, load_dataset
from esn_vla_uq.data.schema import RolloutDataset
from esn_vla_uq.data.sources.openpi import MANIFEST_NAME, OpenpiLogSource
from esn_vla_uq.logging_paths import display_path

logger = logging.getLogger(__name__)


def load_rollouts(path: Path | None) -> RolloutDataset:
    """`--input` の値からロールアウトデータセットを読む。

    Args:
        path: `--input` の値。`None` なら同梱の合成サンプルデータ。

    Returns:
        検証済みの `RolloutDataset`。

    Raises:
        FileNotFoundError: パスが存在しない、またはディレクトリに
            `manifest.json` が無い場合。
        ValueError: データがスキーマ検証に通らない場合。
    """
    if path is None:
        dataset = load_bundled_sample()
        logger.info("input: 同梱の合成サンプルデータ (source=%s)", dataset.source)
        return dataset

    if not path.exists():
        raise FileNotFoundError(
            f"--input: 見つかりません ({display_path(path)})。"
            "本リポジトリの .npz か、収集した openpi ログのディレクトリを"
            "指定してください"
        )

    if path.is_dir():
        if not (path / MANIFEST_NAME).exists():
            raise FileNotFoundError(
                f"--input: ディレクトリに {MANIFEST_NAME} がありません "
                f"({display_path(path)})。openpi ログは "
                "scripts/collect_openpi_rollouts.py が書き出した形式である"
                "必要があります"
            )
        dataset = OpenpiLogSource(path).load()
        logger.info(
            "input: openpi ログ (source=%s n_episodes=%d chunk_horizon=%d path=%s)",
            dataset.source,
            dataset.n_episodes,
            dataset.chunk_horizon,
            display_path(path),
        )
        return dataset

    dataset = load_dataset(path)
    logger.info(
        "input: npz (source=%s n_episodes=%d path=%s)",
        dataset.source,
        dataset.n_episodes,
        display_path(path),
    )
    return dataset
