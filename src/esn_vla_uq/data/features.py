"""`RolloutDataset` から ESN 入力配列を取り出す変換。

`RolloutDataset` は永続化と検証のための形 (エピソードのリスト) であり、ESN が
必要とする形 (`[T, D_u]` の float64 配列とエピソード境界) とは違う。この変換は
Sprint 1 まで呼び出し側の責務だったが、Sprint 2 では uncertainty 層・較正評価・
ノートブックの 3 者が同じ変換を必要とする。各自が実装すると、以下 2 つの判断が
実装ごとにばらつく (A8)。

**1. エピソード境界でリザバー状態をリセットするか**

する。エピソードは互いに独立した試行であり、直前のエピソード末尾の状態を次の
エピソードへ持ち越すと、実際には観測していない過去に依存した特徴量になる。
連結済み配列を `Reservoir.run` へそのまま渡すとまさにそれが起きるため、本
モジュールは**エピソードごとに切り出した区間の列**を第一級の表現として返し
(`DatasetInputs.segments`)、`esn.reservoir.run_episodes` がそれを区間ごとに
初期状態から駆動する。これは「どちらでも動くが片方が誤り」という種類の選択で
あり、呼び出し側ごとに決めさせない (`docs/design.md` 3.9 節)。

**2. NaN の扱い**

`state` / `action` は `Episode.validate()` の `check_all_finite` により有限性が
保証されているため、この 2 つから作る入力に NaN は入らない。NaN を含みうるのは
`action_chunk` だけで、非推論ステップは全要素 NaN と定義されている
(`data/schema.py`)。したがって**チャンク由来の特徴量は `is_inference_step` が
真のステップでのみ定義される**。本モジュールが返す入力には `action_chunk` を
含めず、代わりに `DatasetInputs.is_inference_step` を渡して、チャンク由来の量を
作る側が「どのステップで定義されているか」を必ず参照できるようにする。
NaN を 0 などで埋めてしまうと「予測が無かった」と「予測が 0 だった」が
区別できなくなるため、この層では埋めない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.data.schema import RolloutDataset

FeatureSet = Literal["state", "action", "state_action"]
"""ESN 入力に使うフィールドの組み合わせ。

- ``"state"``: `state` のみ (`D_u = state_dim`)
- ``"action"``: `action` のみ (`D_u = action_dim`)
- ``"state_action"``: `state` と `action` を軸 1 で連結
  (`D_u = state_dim + action_dim`)

`action_chunk` は含めない (モジュール docstring の「NaN の扱い」を参照)。
"""

SUPPORTED_FEATURE_SETS: Final[tuple[str, ...]] = get_args(FeatureSet)
"""`FeatureSet` が許可する値の実行時タプル。"""

DEFAULT_FEATURE_SET: Final[FeatureSet] = "state"
"""既定の特徴量。観測できる量だけからなり、行動を入力に含めない。"""


@dataclass(frozen=True)
class DatasetInputs:
    """ESN へ渡す形に整えたデータセット。

    Attributes:
        values: 全エピソードを連結した入力 `float64[T_total, D_u]`。
        episode_starts: 連結表現における各エピソードの開始インデックス。
        episode_lengths: 各エピソードのステップ数。
        is_inference_step: 連結表現での `bool[T_total]`。チャンク由来の量が
            定義されるステップを示す (モジュール docstring 参照)。
        feature: どのフィールドから作ったか。
    """

    values: NDArray[np.float64]
    episode_starts: NDArray[np.int64]
    episode_lengths: NDArray[np.int64]
    is_inference_step: NDArray[np.bool_]
    feature: FeatureSet

    @property
    def n_inputs(self) -> int:
        """入力次元 `D_u`。`Reservoir(config, n_inputs)` にそのまま渡せる。"""
        return int(self.values.shape[1])

    @property
    def n_episodes(self) -> int:
        """エピソード数。"""
        return int(self.episode_lengths.shape[0])

    @property
    def total_steps(self) -> int:
        """連結後の総ステップ数 `T_total`。"""
        return int(self.values.shape[0])

    @property
    def segments(self) -> list[NDArray[np.float64]]:
        """エピソードごとに切り出した入力 `[T_i, D_u]` の列。

        `esn.reservoir.run_episodes` にそのまま渡すと、エピソード境界で
        リザバー状態がリセットされる。連結済みの `values` を直接
        `Reservoir.run` へ渡してはならない (モジュール docstring 参照)。
        """
        return [
            self.values[start : start + length]
            for start, length in zip(
                self.episode_starts.tolist(), self.episode_lengths.tolist(), strict=True
            )
        ]


def dataset_inputs(
    dataset: RolloutDataset, *, feature: FeatureSet = DEFAULT_FEATURE_SET
) -> DatasetInputs:
    """`RolloutDataset` を ESN 入力へ変換する。

    Args:
        dataset: 変換対象。`validate()` 済みであることを前提とする
            (`load_dataset` / `generate_dataset` の戻り値は検証済み)。
        feature: 使うフィールドの組み合わせ (`FeatureSet`)。

    Returns:
        連結した入力とエピソード境界を持つ `DatasetInputs`。dtype は float64
        (`Reservoir.run` が float64 を要求するため、ここで一度だけ変換する)。

    Raises:
        ValueError: `feature` が未知の場合。
    """
    if feature not in SUPPORTED_FEATURE_SETS:
        raise ValueError(
            f"feature: 未知の特徴量です (actual={feature!r}, "
            f"supported={list(SUPPORTED_FEATURE_SETS)})"
        )

    per_episode = [
        _episode_values(episode.state, episode.action, feature)
        for episode in dataset.episodes
    ]
    values = np.concatenate(per_episode, axis=0)
    is_inference_step = np.concatenate(
        [episode.is_inference_step for episode in dataset.episodes], axis=0
    )
    return DatasetInputs(
        values=values,
        episode_starts=dataset.episode_starts,
        episode_lengths=dataset.episode_lengths,
        is_inference_step=is_inference_step,
        feature=feature,
    )


def _episode_values(
    state: NDArray[np.float32], action: NDArray[np.float32], feature: FeatureSet
) -> NDArray[np.float64]:
    """1 エピソード分の入力 `[T_i, D_u]` を float64 で作る。"""
    if feature == "state":
        return state.astype(np.float64)
    if feature == "action":
        return action.astype(np.float64)
    return np.concatenate((state, action), axis=1).astype(np.float64)
