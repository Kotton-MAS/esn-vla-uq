"""1 ステップ先 action 予測タスクの構築。

`RolloutDataset` を「入力 `u[t] = [state[t], action[t]]` から目標
`y[t] = action[t+1]` を予測する」教師ありタスクへ変換する
(`docs/plans/sprint2_v0.1.md`)。

この定義を選んだ理由は、観測できる情報から次の行動がどれだけ予測できるかを
測るためである。ポリシーの挙動が不安定になる (= 失敗に向かう) 区間では次の行動が
予測しづらくなる、という仮説を検証できる形にする。

**エピソード境界を跨がない。** 目標は同一エピソード内の次ステップに限る。
各エピソードの最終ステップは目標が存在しないため落とし、`T_i - 1` 標本になる。
リザバー状態も区間ごとに初期化する (`esn.reservoir.run_episodes`、
`docs/design.md` 3.9 節)。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.data.features import FeatureSet, dataset_inputs
from esn_vla_uq.data.schema import Episode, RolloutDataset

DEFAULT_PREDICTION_FEATURE: FeatureSet = "state_action_chunk"
"""予測タスクの既定入力。

要件書が定める入力は「action chunk 系列と固有受容感覚」。 だけでは
合成データが失敗区間に注入するチャンク分散の増大が入力に現れず、不確実性が
失敗を検知できない (実測 AUROC 0.50)。
"""

MIN_EPISODE_STEPS: int = 2
"""1 標本を作るのに必要な最小ステップ数 (入力 1 つと目標 1 つ)。"""


@dataclass(frozen=True)
class EpisodeSamples:
    """1 エピソード分の予測タスク標本。

    Attributes:
        episode_id: 元エピソードの識別子。
        task_name: タスク名。較正データ分割 (`uncertainty/split.py`) が使う。
        success: エピソードの成否。
        failure_onset: 失敗が始まったステップ (元エピソードのステップ番号)。
            成功エピソードと、`failure_onset` の概念を持たない出所では `None`。
        inputs: `float64[T_i - 1, D_u]`。`u[t] = [state[t], action[t]]`。
        targets: `float64[T_i - 1, D_y]`。`y[t] = action[t+1]`。
        target_steps: `int64[T_i - 1]`。各標本の目標が元エピソードの
            どのステップかを表す (`t+1`)。失敗開始位置との突き合わせに使う。
    """

    episode_id: str
    task_name: str
    success: bool
    failure_onset: int | None
    inputs: NDArray[np.float64]
    targets: NDArray[np.float64]
    target_steps: NDArray[np.int64]
    difficulty_column: int | None = None

    @property
    def n_samples(self) -> int:
        """標本数 `T_i - 1`。"""
        return int(self.inputs.shape[0])

    @property
    def n_inputs(self) -> int:
        """入力次元 `D_u`。"""
        return int(self.inputs.shape[1])

    @property
    def n_targets(self) -> int:
        """目標次元 `D_y`。"""
        return int(self.targets.shape[1])

    def after_failure_onset(self) -> NDArray[np.bool_]:
        """各標本が失敗開始以降かを示す `bool[T_i - 1]`。

        `failure_onset` が `None` (成功エピソード、または概念を持たない出所) の
        ときは全て False。失敗検知の評価ラベルとして使う。
        """
        if self.failure_onset is None:
            return np.zeros(self.n_samples, dtype=np.bool_)
        flags: NDArray[np.bool_] = self.target_steps >= self.failure_onset
        return flags


def build_samples(
    dataset: RolloutDataset, *, feature: FeatureSet = DEFAULT_PREDICTION_FEATURE
) -> list[EpisodeSamples]:
    """データセットを 1 ステップ先 action 予測の標本列へ変換する。

    Args:
        dataset: 変換対象。`validate()` 済みであることを前提とする。

    Returns:
        エピソードごとの `EpisodeSamples`。順序は `dataset.episodes` と同じ。
        ステップ数が `MIN_EPISODE_STEPS` 未満のエピソードは標本を作れないため
        除外する。

    Raises:
        ValueError: 標本を 1 つも作れない場合。
    """
    inputs = dataset_inputs(dataset, feature=feature)
    samples = [
        _episode_samples(episode_inputs, episode, inputs.difficulty_column)
        for episode_inputs, episode in zip(
            inputs.segments, dataset.episodes, strict=True
        )
        if episode.n_steps >= MIN_EPISODE_STEPS
    ]
    if not samples:
        raise ValueError(
            "1 ステップ先予測の標本を作れるエピソードがありません "
            f"(最小ステップ数={MIN_EPISODE_STEPS})"
        )
    return samples


def _episode_samples(
    episode_inputs: NDArray[np.float64],
    episode: Episode,
    difficulty_column: int | None,
) -> EpisodeSamples:
    """1 エピソード分の入力・目標・ステップ番号を切り出す。"""
    # 目標は次ステップの action。最終ステップには目標が無いので入力側を 1 つ削る。
    return EpisodeSamples(
        episode_id=episode.episode_id,
        task_name=episode.task_name,
        success=episode.success,
        failure_onset=episode.failure_onset,
        inputs=episode_inputs[:-1],
        targets=episode.action[1:].astype(np.float64),
        target_steps=np.arange(1, episode.n_steps, dtype=np.int64),
        difficulty_column=difficulty_column,
    )


def stack_targets(samples: Sequence[EpisodeSamples]) -> NDArray[np.float64]:
    """標本列の目標を連結して `[N, D_y]` にする。"""
    if not samples:
        raise ValueError("samples: 1 件以上必要です")
    return np.concatenate([sample.targets for sample in samples], axis=0)


DetectionLabel = Literal["failure_onset", "episode_success"]
"""失敗検知の評価に使うラベルの種類。

- ``"failure_onset"``: 失敗開始以降のステップを陽性とする。**細かいが、
  `failure_onset` を持つ出所でしか作れない**(合成データのみ)。
- ``"episode_success"``: 失敗エピソードの全ステップを陽性とする。粗いが
  どの出所でも作れる。openpi の評価ループには失敗開始時刻の概念が無い
  (`data/sources/openpi.py`) ため、実ログではこちらになる。
"""


def stack_failure_labels(samples: Sequence[EpisodeSamples]) -> NDArray[np.bool_]:
    """標本列の「失敗開始以降か」ラベルを連結して `[N]` にする。"""
    if not samples:
        raise ValueError("samples: 1 件以上必要です")
    return np.concatenate([sample.after_failure_onset() for sample in samples], axis=0)


def stack_episode_failure_labels(
    samples: Sequence[EpisodeSamples],
) -> NDArray[np.bool_]:
    """失敗エピソードの全ステップを陽性としたラベルを `[N]` で返す。"""
    if not samples:
        raise ValueError("samples: 1 件以上必要です")
    return np.concatenate(
        [
            np.full(sample.n_samples, not sample.success, dtype=np.bool_)
            for sample in samples
        ],
        axis=0,
    )


def detection_labels(
    samples: Sequence[EpisodeSamples],
) -> tuple[NDArray[np.bool_], DetectionLabel]:
    """使える中で最も細かいラベルと、その種類を返す。

    `failure_onset` を持つ標本が 1 つでもあればそちらを使う。持たない出所
    (openpi) では全て False になってしまい、AUROC が計算できないため
    エピソード単位の成否へ落とす。**どちらを使ったかはレポートに残す**
    (粒度が違う数値を並べて比較できてしまうため)。
    """
    if any(sample.failure_onset is not None for sample in samples):
        return stack_failure_labels(samples), "failure_onset"
    return stack_episode_failure_labels(samples), "episode_success"


def input_segments(samples: Sequence[EpisodeSamples]) -> list[NDArray[np.float64]]:
    """`esn.reservoir.run_episodes` に渡す入力区間の列を返す。"""
    return [sample.inputs for sample in samples]
