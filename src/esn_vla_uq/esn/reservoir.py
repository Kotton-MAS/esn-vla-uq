"""リザバー行列の生成と状態の時間発展。

状態更新式 (`docs/design.md` の「ESN の数学仕様」節と一致させる)::

    x[t] = (1 - a) * x[t-1] + a * tanh(W_in @ u[t] + W @ x[t-1] + b)

初期状態 `x[-1]` は既定で零ベクトル。washout の破棄は呼び出し側の責務とし、
ヘルパ `discard_washout` を提供する。

疎行列は scipy を導入せず「密行列 + マスク」で表現し、スペクトル半径は
`np.linalg.eigvals` の密計算で求める (N の実用上限は `docs/design.md` を参照)。
"""

from __future__ import annotations

import logging
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.esn.config import ESNConfig

logger = logging.getLogger(__name__)


class Activation(Protocol):
    """リザバーの活性化関数の契約 (差し替え口)。

    Sprint 1 では `tanh_activation` 固定で運用する。
    """

    def __call__(self, x: NDArray[np.float64], /) -> NDArray[np.float64]:
        """要素ごとに活性化を適用して同 shape の配列を返す。"""
        ...


def tanh_activation(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """既定の活性化関数 (tanh)。"""
    return cast("NDArray[np.float64]", np.tanh(x))


def _max_abs_eigenvalue(matrix: NDArray[np.float64]) -> float:
    """正方行列の固有値の絶対値の最大 (= スペクトル半径) を返す。

    公開 API としてのスペクトル半径は `diagnostics` 側で提供する。ここでは
    `W` のスケーリングに必要な内部ヘルパとしてのみ用いる。
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"正方行列が必要です (実 shape: {matrix.shape})")
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def discard_washout(states: NDArray[np.float64], washout: int) -> NDArray[np.float64]:
    """状態行列 `[T, N]` の先頭 `washout` ステップを破棄する。

    `washout` が系列長以上の場合は残る行が無くなるため `ValueError`。
    """
    if states.ndim != 2:
        raise ValueError(
            f"states は 2 次元 [T, N] が必要です (実 shape: {states.shape})"
        )
    if washout < 0:
        raise ValueError(f"washout は 0 以上である必要があります (実値: {washout})")
    n_steps = states.shape[0]
    if washout >= n_steps:
        raise ValueError(
            f"washout が系列長以上です (washout: {washout}, 系列長: {n_steps})"
        )
    return states[washout:]


def _as_input_matrix(inputs: NDArray[np.float64], n_inputs: int) -> NDArray[np.float64]:
    """入力を `[T, n_inputs]` の float64 配列として検証する。"""
    array = np.asarray(inputs, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"inputs は 2 次元 [T, n_inputs] が必要です (実 shape: {array.shape})"
        )
    if array.shape[1] != n_inputs:
        raise ValueError(
            "inputs の入力次元がリザバーと一致しません "
            f"(期待: {n_inputs}, 実値: {array.shape[1]})"
        )
    if array.shape[0] < 1:
        raise ValueError("inputs の系列長は 1 以上である必要があります (実値: 0)")
    return array


class Reservoir:
    """入力行列 `W_in` / 再帰行列 `W` / バイアス `b` を保持するリザバー。

    行列は `np.random.default_rng(config.seed)` のみから生成する
    (グローバルな `np.random` は使わない)。生成順は `W_in` -> `b` -> `W` で固定し、
    同一 seed・同一 `n_inputs` なら常に同じ行列になる。
    """

    def __init__(
        self,
        config: ESNConfig,
        n_inputs: int,
        activation: Activation = tanh_activation,
    ) -> None:
        if n_inputs < 1:
            raise ValueError(
                f"n_inputs は 1 以上である必要があります (実値: {n_inputs})"
            )
        self.config = config
        self.n_inputs = n_inputs
        self.activation = activation

        rng = np.random.default_rng(config.seed)
        self.W_in: NDArray[np.float64] = _make_input_matrix(rng, config, n_inputs)
        self.b: NDArray[np.float64] = _make_bias(rng, config)
        self.W: NDArray[np.float64] = _make_recurrent_matrix(rng, config)
        logger.debug(
            "reservoir built: n_reservoir=%d n_inputs=%d seed=%d",
            config.n_reservoir,
            n_inputs,
            config.seed,
        )

    @property
    def n_reservoir(self) -> int:
        """リザバーのニューロン数 N。"""
        return self.config.n_reservoir

    def initial_state(self) -> NDArray[np.float64]:
        """既定の初期状態 (零ベクトル) を返す。"""
        return np.zeros(self.n_reservoir, dtype=np.float64)

    def run(
        self,
        inputs: NDArray[np.float64],
        initial_state: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """入力系列 `[T, n_inputs]` を駆動し状態行列 `[T, N]` を返す。

        washout の破棄は行わない (`discard_washout` を呼び出し側で使う)。
        """
        u = _as_input_matrix(inputs, self.n_inputs)
        x = self._as_state_vector(initial_state)
        leak_rate = self.config.leak_rate

        n_steps = u.shape[0]
        states = np.empty((n_steps, self.n_reservoir), dtype=np.float64)
        for t in range(n_steps):
            pre_activation = self.W_in @ u[t] + self.W @ x + self.b
            x = (1.0 - leak_rate) * x + leak_rate * self.activation(pre_activation)
            states[t] = x
        return states

    def _as_state_vector(
        self, initial_state: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        """初期状態を検証して float64 のコピーとして返す。"""
        if initial_state is None:
            return self.initial_state()
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (self.n_reservoir,):
            raise ValueError(
                "initial_state の shape が不正です "
                f"(期待: {(self.n_reservoir,)}, 実値: {state.shape})"
            )
        return state.copy()


def _make_input_matrix(
    rng: np.random.Generator, config: ESNConfig, n_inputs: int
) -> NDArray[np.float64]:
    """`W_in` = Uniform(-1, 1) * input_scaling、shape `[N, n_inputs]`。"""
    raw = rng.uniform(-1.0, 1.0, size=(config.n_reservoir, n_inputs))
    return raw * config.input_scaling


def _make_bias(rng: np.random.Generator, config: ESNConfig) -> NDArray[np.float64]:
    """`b` = Uniform(-1, 1) * bias_scaling、shape `[N]`。"""
    raw = rng.uniform(-1.0, 1.0, size=config.n_reservoir)
    return raw * config.bias_scaling


def _make_recurrent_matrix(
    rng: np.random.Generator, config: ESNConfig
) -> NDArray[np.float64]:
    """`W` = (density マスク * Uniform(-1, 1)) を目標スペクトル半径にスケールする。"""
    size = (config.n_reservoir, config.n_reservoir)
    mask = rng.random(size) < config.density
    raw = rng.uniform(-1.0, 1.0, size=size)
    matrix = np.where(mask, raw, 0.0)

    radius = _max_abs_eigenvalue(matrix)
    if radius == 0.0:
        raise ValueError(
            "生成された W のスペクトル半径が 0 のためスケールできません "
            f"(n_reservoir: {config.n_reservoir}, density: {config.density}, "
            f"seed: {config.seed})"
        )
    return matrix * (config.spectral_radius / radius)
