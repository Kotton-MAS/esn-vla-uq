"""合成ロールアウトを供給する具象 `RolloutSource`。

`data/synthetic.py` (生成モデルそのもの) の薄いアダプタであり、生成ロジックは
持たない。Protocol の定義は `data/sources/base.py` にあり、本モジュールは
そちらを import しない (実装が Protocol を満たすかは構造的部分型として
`runtime_checkable` なテストで確認する)。
"""

from __future__ import annotations

from dataclasses import dataclass

from esn_vla_uq.data.schema import RolloutDataset
from esn_vla_uq.data.synthetic import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MIN_STEPS,
    DEFAULT_N_EPISODES,
    DEFAULT_SUCCESS_RATE,
    generate_dataset,
)


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
