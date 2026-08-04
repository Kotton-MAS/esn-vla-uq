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

from esn_vla_uq.data.schema import Episode, RolloutDataset

FeatureSet = Literal["state", "action", "state_action", "state_action_chunk"]
"""ESN 入力に使うフィールドの組み合わせ。

- ``"state"``: `state` のみ (`D_u = state_dim`)
- ``"action"``: `action` のみ (`D_u = action_dim`)
- ``"state_action"``: `state` と `action` を軸 1 で連結
  (`D_u = state_dim + action_dim`)
- ``"state_action_chunk"``: 上記に**チャンク由来の要約量** 2 本を足す
  (`D_u = state_dim + action_dim + 2`)

`action_chunk` の生の値 (`[H, D_a]`) は入力に含めない。非推論ステップでは全要素
NaN であり、そのまま入れると NaN が状態に伝播する。代わりに推論ステップで
要約量を計算し、次の推論まで前方補完する (`CHUNK_FEATURE_NAMES`)。これは
実運用でも観測できる量である (ポリシーが出したチャンクをそのまま見るだけで、
正解を必要としない)。
"""

CHUNK_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log_chunk_dispersion",
    "steps_since_inference",
)
"""チャンク由来の要約量。

- ``log_chunk_dispersion``: チャンクをホライズン方向に 2 階差分した二乗平均の
  対数。滑らかなトレンド成分が落ち、チャンク内のばらつき (flow matching の
  サンプリング分散に相当) が残る。数桁にわたる量なので対数を取る。
- ``steps_since_inference``: 直近の推論ステップからの経過ステップ数。前方補完
  した要約量がどれだけ古いかを表す。

要件書は入力を「action chunk 系列と固有受容感覚」と定めている。固有受容感覚
(`state`) と実行された行動 (`action`) だけでは、合成データが失敗区間に注入する
**チャンク分散の増大**が入力に現れない。実測では、この 2 本を足すことで失敗検知
AUROC が 0.50 (信号なし) から改善する。
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
    def difficulty_column(self) -> int | None:
        """`DIFFICULTY_FEATURE` が入力の何列目か。含まれないなら `None`。"""
        if self.feature != "state_action_chunk":
            return None
        offset = CHUNK_FEATURE_NAMES.index(DIFFICULTY_FEATURE)
        return self.n_inputs - len(CHUNK_FEATURE_NAMES) + offset

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

    per_episode = [_episode_values(episode, feature) for episode in dataset.episodes]
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


DIFFICULTY_FEATURE: Final[str] = "log_chunk_dispersion"
"""区間幅の変調に使う観測量の名前 (`CHUNK_FEATURE_NAMES` のいずれか)。

split conformal の被覆率保証は「入力だけから決まる」任意の難易度関数に対して
成り立つ。残差の大きさを当てにいく必要はなく、**観測できて失敗と結びつく量**を
選んでよい (`uncertainty/nonconformity.py`)。
"""

CHUNK_DISPERSION_EPSILON: Final[float] = 1e-12
"""``log(dispersion + eps)`` の eps。分散がちょうど 0 のチャンクで -inf にしない。"""

_MIN_HORIZON_FOR_DISPERSION: Final[int] = 3
"""2 階差分に必要な最小ホライズン。"""


def chunk_features(episode: Episode) -> NDArray[np.float64]:
    """チャンク由来の要約量 `[T_i, 2]` を作る。

    推論ステップで要約量を計算し、次の推論ステップまで前方補完する。実機では
    ポリシーが出したチャンクをそのまま観測できるため、正解を必要としない。

    Args:
        episode: 対象エピソード。`validate()` 済みであることを前提とする。

    Returns:
        `[T_i, 2]` の float64 配列 (`CHUNK_FEATURE_NAMES` の順)。

    Raises:
        ValueError: 推論ステップが 1 つも無い場合 (`Episode.validate()` が
            禁じているので通常は起きない)。
    """
    inference_steps = np.nonzero(episode.is_inference_step)[0]
    if inference_steps.size == 0:
        raise ValueError(
            "is_inference_step: 推論ステップがありません "
            f"(episode_id={episode.episode_id!r})"
        )

    chunks = episode.action_chunk[episode.is_inference_step].astype(np.float64)
    if chunks.shape[1] >= _MIN_HORIZON_FOR_DISPERSION:
        roughness = np.diff(chunks, n=2, axis=1)
        dispersion = np.mean(roughness**2, axis=(1, 2))
    else:
        # ホライズンが短く 2 階差分を取れない場合は分散で代替する。
        dispersion = np.var(chunks, axis=(1, 2))

    # 各ステップを「直近の推論ステップ」へ対応づける (前方補完)。
    step_index = np.arange(episode.n_steps)
    source = np.searchsorted(inference_steps, step_index, side="right") - 1
    log_dispersion = np.log(dispersion + CHUNK_DISPERSION_EPSILON)[source]
    steps_since = (step_index - inference_steps[source]).astype(np.float64)
    return np.stack((log_dispersion, steps_since), axis=1)


def _episode_values(episode: Episode, feature: FeatureSet) -> NDArray[np.float64]:
    """1 エピソード分の入力 `[T_i, D_u]` を float64 で作る。"""
    if feature == "state":
        return episode.state.astype(np.float64)
    if feature == "action":
        return episode.action.astype(np.float64)
    blocks = [episode.state.astype(np.float64), episode.action.astype(np.float64)]
    if feature == "state_action_chunk":
        blocks.append(chunk_features(episode))
    return np.concatenate(blocks, axis=1)
