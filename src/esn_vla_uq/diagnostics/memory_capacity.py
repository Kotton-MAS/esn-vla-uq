"""線形メモリ容量 (Jaeger 2001) の測定。

`docs/design.md` の 4.3 節に従う。i.i.d. スカラー入力 ``u[t] ~ Uniform(-0.8, 0.8)``
でリザバーを駆動し、遅延 ``k = 1..K`` ごとに教師信号 ``y_k[t] = u[t - k]`` への
線形 read-out を学習して ``MC_k = corr(y_hat_k, u[t - k])^2`` を求める。
``total_mc = sum_k MC_k`` の理論上界は ``N`` (リザバーのニューロン数)。

**正則化強度への感度**: メモリ容量は ``ridge_alpha`` に敏感である。正則化を強めると
``W_out`` のノルムが縮んで高遅延成分の相関が過小評価され、``total_mc`` が系統的に
小さく出る。そのため本モジュールの既定 ``ridge_alpha`` は診断専用の微小値
``1e-8`` とし、`ESNConfig.ridge_alpha` (既定 ``1e-6``) とは独立に扱う。測定値を
比較する際は必ず ``ridge_alpha`` を揃えること。

**負の ``MC_k`` をクリップしない**: ``corr(...)^2`` は理論上 ``[0, 1]`` だが、生値を
そのまま返して丸めない (数値不安定性を隠さないため。`docs/design.md` 4.3 節)。
相関が定義できない (予測または目標の分散が 0) ケースのみ ``0.0`` を返す。

**実装上の同値変形**: 遅延ごとの read-out は設計行列 ``X = [1, x]`` を共有し目標
列だけが異なる。リッジ回帰の閉形式解は出力列ごとに独立に解けるため、``K`` 本の
read-out を個別に解く代わりに多出力の目標行列 ``Y = [y_1, ..., y_K]`` として 1 回で
解いても各列の解は同一である (同じ Gram 行列・同じ罰則行列)。計算量を
``O(K T p^2)`` から ``O(T p^2 + K T p)`` に落とすためこの形で実装する。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.esn.readout import RidgeReadout
from esn_vla_uq.esn.reservoir import Reservoir

logger = logging.getLogger(__name__)

DEFAULT_MC_SEED: Final[int] = 0
"""入力系列を生成する既定シード。

同モジュールの他の既定値が `Final` 定数になっているのに合わせる (C2)。
"""

MEMORY_CAPACITY_INPUT_DIM: Final[int] = 1
"""メモリ容量診断の既定の入力次元 (``D_u = 1``)。

**リザバーの入力次元がこれである必要は無い。** 定義が要求するのは「駆動信号が
スカラーの i.i.d. であること」であって、リザバーの `W_in` が何列あるかではない。
`D_u > 1` のリザバーに対しては ``input_channel`` の 1 本だけにスカラーを流し、
残りを 0 にして測る (`linear_memory_capacity`)。

**なぜこれが要るのか。** `Reservoir` は `W_in -> b -> W` の順に同一 RNG から
引くため、同じ `seed` でも `n_inputs` が違えば **`W` まで別の行列**になる
(`esn/reservoir.py`)。`D_u=1` のリザバーで測った MC は、較正で実際に使う
`D_u=17` のリザバーの記憶ではない。診断値と較正性能を突き合わせるには
**同じリザバーで測る**必要がある (`docs/next-research-directions.md` ②)。
"""

DEFAULT_MC_INPUT_CHANNEL: Final[int] = 0
"""スカラー駆動信号を流す入力チャンネルの既定値。"""

DEFAULT_MC_N_TRAIN: Final[int] = 3000
"""read-out 学習に使うステップ数。"""

DEFAULT_MC_N_TEST: Final[int] = 1000
"""``MC_k`` の評価に使うステップ数。"""

DEFAULT_MC_WASHOUT: Final[int] = 200
"""先頭の過渡区間として捨てるステップ数 (`ESNConfig.washout` とは独立)。"""

DEFAULT_MC_RIDGE_ALPHA: Final[float] = 1e-8
"""診断専用の微小リッジ係数 (`ESNConfig.ridge_alpha` とは独立)。"""

MAX_DELAY_CAP: Final[int] = 200
"""既定の最大遅延 ``K = min(2 * N, MAX_DELAY_CAP)`` の上限。"""

MEMORY_HORIZON_THRESHOLD: Final[float] = 0.1
"""``memory_horizon`` の判定閾値 (``MC_k < 0.1`` となる最小の k)。"""

INPUT_LOW: Final[float] = -0.8
INPUT_HIGH: Final[float] = 0.8


@dataclass(frozen=True)
class MemoryCapacityResult:
    """線形メモリ容量の測定結果。

    Attributes:
        total_mc: ``sum_{k=1}^{K} MC_k``。理論上界は ``N``。
        per_delay: ``MC_1, ..., MC_K`` (長さ K)。生値であり負値もクリップしない。
        memory_horizon: ``MC_k < 0.1`` となる最小の k (存在しなければ K)。
        mc_per_neuron: ``total_mc / N``。
    """

    total_mc: float
    per_delay: list[float]
    memory_horizon: int
    mc_per_neuron: float

    @property
    def n_delays(self) -> int:
        """評価した最大遅延 K。"""
        return len(self.per_delay)

    def to_dict(self) -> dict[str, object]:
        """JSON シリアライズ可能な辞書へ変換する (診断レポート用)。

        フィールドは `dataclasses.asdict` で列挙し、フィールドを足したときに
        診断レポート JSON から黙って欠落しないようにする (A2)。`n_delays` は
        フィールドではなく `per_delay` から導出する property なので `asdict`
        には現れず、JSON の読み手が `per_delay` を数えずに済むよう明示的に
        加える。
        """
        return {**asdict(self), "n_delays": self.n_delays}


def default_max_delay(n_reservoir: int) -> int:
    """既定の最大遅延 ``K = min(2 * N, MAX_DELAY_CAP)``。"""
    return min(2 * n_reservoir, MAX_DELAY_CAP)


def _squared_correlation(
    predicted: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    """Pearson 相関係数の二乗を返す。分散 0 の場合は 0.0 (相関が未定義)。"""
    centered_predicted = predicted - predicted.mean()
    centered_target = target - target.mean()
    denominator = float(
        np.sqrt(
            float(centered_predicted @ centered_predicted)
            * float(centered_target @ centered_target)
        )
    )
    if denominator == 0.0:
        return 0.0
    correlation = float(centered_predicted @ centered_target) / denominator
    return correlation * correlation


def _delayed_targets(
    inputs: NDArray[np.float64], max_delay: int, channel: int
) -> NDArray[np.float64]:
    """``targets[t, k - 1] = u[t - k]`` の行列 `[T, K]` を作る。

    ``t < k`` の区間には過去が無いため 0 を置く。学習・評価はいずれも先頭
    ``washout >= K`` ステップを除いた区間で行うため、この区間は参照されない。

    `inputs` は `[T, D_u]` で、駆動信号が入っているのは ``channel`` の列だけ
    (他は 0)。目標はその列の遅延である。
    """
    n_steps = inputs.shape[0]
    flat = inputs[:, channel]
    targets = np.zeros((n_steps, max_delay), dtype=np.float64)
    for delay in range(1, max_delay + 1):
        targets[delay:, delay - 1] = flat[:-delay]
    return targets


def _validate_parameters(
    *,
    n_train: int,
    n_test: int,
    washout: int,
    max_delay: int,
    ridge_alpha: float,
) -> None:
    """メモリ容量診断のパラメータを検証する。"""
    if n_train < 1:
        raise ValueError(f"n_train は 1 以上である必要があります (実値: {n_train})")
    if n_test < 2:
        raise ValueError(f"n_test は 2 以上である必要があります (実値: {n_test})")
    if washout < 0:
        raise ValueError(f"washout は 0 以上である必要があります (実値: {washout})")
    if max_delay < 1:
        raise ValueError(f"max_delay は 1 以上である必要があります (実値: {max_delay})")
    if ridge_alpha < 0.0:
        raise ValueError(
            f"ridge_alpha は 0 以上である必要があります (実値: {ridge_alpha})"
        )
    if washout < max_delay:
        raise ValueError(
            "washout は max_delay 以上である必要があります "
            f"(washout: {washout}, max_delay: {max_delay})"
        )


def linear_memory_capacity(
    reservoir: Reservoir,
    *,
    n_train: int = DEFAULT_MC_N_TRAIN,
    n_test: int = DEFAULT_MC_N_TEST,
    washout: int = DEFAULT_MC_WASHOUT,
    max_delay: int | None = None,
    ridge_alpha: float = DEFAULT_MC_RIDGE_ALPHA,
    seed: int = DEFAULT_MC_SEED,
    input_channel: int = DEFAULT_MC_INPUT_CHANNEL,
) -> MemoryCapacityResult:
    """線形メモリ容量を測定する。

    駆動信号はスカラーである (定義の要求)。リザバーの入力次元が 1 より大きい
    ときは ``input_channel`` の列だけにその信号を流し、残りの列は 0 にする。
    **測っているのはそのリザバーの `W` が持つ記憶であり、`input_channel` 以外の
    列の `W_in` は結果に関与しない。** 較正では 17 列すべてに信号が入るので、
    実運用の動作点そのものではない点に注意する
    (`docs/next-research-directions.md` ②)。

    Args:
        reservoir: 診断対象のリザバー。入力次元に制限は無い。
        n_train: read-out 学習に使うステップ数。
        n_test: ``MC_k`` の評価に使うステップ数。
        washout: 先頭で捨てる過渡区間の長さ (``max_delay`` 以上)。
        max_delay: 最大遅延 K。省略時は ``min(2 * N, 200)``。
        ridge_alpha: 診断専用の微小リッジ係数 (既定 ``1e-8``)。
        seed: 入力系列を生成する `np.random.default_rng` の種。
        input_channel: スカラー信号を流す入力チャンネル。

    Returns:
        `MemoryCapacityResult`。

    Raises:
        ValueError: `input_channel` がリザバーの入力次元の範囲外、または
            パラメータが範囲外の場合。
    """
    if not 0 <= input_channel < reservoir.n_inputs:
        raise ValueError(
            "input_channel はリザバーの入力次元の範囲内である必要があります "
            f"(実値: {input_channel}, n_inputs: {reservoir.n_inputs})"
        )
    delays = (
        default_max_delay(reservoir.n_reservoir) if max_delay is None else max_delay
    )
    _validate_parameters(
        n_train=n_train,
        n_test=n_test,
        washout=washout,
        max_delay=delays,
        ridge_alpha=ridge_alpha,
    )

    rng = np.random.default_rng(seed)
    n_steps = washout + n_train + n_test
    # 駆動信号はスカラー。`input_channel` 以外の列は 0 に保つ (定義どおり
    # 1 本の i.i.d. 信号だけでリザバーを駆動する)。
    inputs = np.zeros((n_steps, reservoir.n_inputs), dtype=np.float64)
    inputs[:, input_channel] = rng.uniform(INPUT_LOW, INPUT_HIGH, size=n_steps)
    states = reservoir.run(inputs)
    targets = _delayed_targets(inputs, delays, input_channel)

    train = slice(washout, washout + n_train)
    test = slice(washout + n_train, n_steps)

    # 設計行列は [1, x] (入力パススルー無し)。生入力 u[t] を特徴に含めると
    # 「過去の記憶」ではない自由度が 1 本増え、理論上界 total_mc <= N が崩れる。
    readout = RidgeReadout(ridge_alpha, input_passthrough=False).fit(
        states[train], inputs[train], targets[train]
    )
    predictions = readout.predict(states[test], inputs[test])

    test_targets = targets[test]
    per_delay = [
        _squared_correlation(predictions[:, index], test_targets[:, index])
        for index in range(delays)
    ]
    total_mc = float(sum(per_delay))
    logger.debug(
        "memory capacity measured: n_reservoir=%d K=%d total_mc=%.4f alpha=%g",
        reservoir.n_reservoir,
        delays,
        total_mc,
        ridge_alpha,
    )
    return MemoryCapacityResult(
        total_mc=total_mc,
        per_delay=per_delay,
        memory_horizon=_memory_horizon(per_delay),
        mc_per_neuron=total_mc / float(reservoir.n_reservoir),
    )


def _memory_horizon(per_delay: list[float]) -> int:
    """``MC_k < 0.1`` となる最小の k。存在しなければ K を返す。"""
    for index, value in enumerate(per_delay, start=1):
        if value < MEMORY_HORIZON_THRESHOLD:
            return index
    return len(per_delay)
