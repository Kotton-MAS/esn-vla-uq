"""ロールアウトデータの供給元を抽象化する Protocol。

Sprint 1 では合成データ (`SyntheticRolloutSource`) のみを提供する。Sprint 2 で追加する
`OpenpiLogSource` は同じ `RolloutSource` Protocol を満たすだけでよく、openpi を
ランタイム依存に加える必要はない (疎結合設計)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from esn_vla_uq.data.schema import RolloutDataset
from esn_vla_uq.data.synthetic import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MIN_STEPS,
    DEFAULT_N_EPISODES,
    DEFAULT_SUCCESS_RATE,
    generate_dataset,
)


@runtime_checkable
class RolloutSource(Protocol):
    """ロールアウトデータセットの供給元。"""

    def load(self) -> RolloutDataset:
        """検証済みの `RolloutDataset` を返す。"""
        ...


@dataclass(frozen=True)
class SyntheticRolloutSource:
    """合成ロールアウトを生成する供給元。

    Attributes:
        seed: 乱数シード。同じ設定なら常に同一のデータセットを返す。
        n_episodes: エピソード数。
        success_rate: 成功エピソードの割合。
        min_steps: エピソード長の下限。
        max_steps: エピソード長の上限。
    """

    seed: int
    n_episodes: int = DEFAULT_N_EPISODES
    success_rate: float = DEFAULT_SUCCESS_RATE
    min_steps: int = DEFAULT_MIN_STEPS
    max_steps: int = DEFAULT_MAX_STEPS

    def load(self) -> RolloutDataset:
        """合成データセットを生成して返す (`source == "synthetic"`)。"""
        return generate_dataset(
            seed=self.seed,
            n_episodes=self.n_episodes,
            success_rate=self.success_rate,
            min_steps=self.min_steps,
            max_steps=self.max_steps,
        )
