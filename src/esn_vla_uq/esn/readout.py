"""リッジ回帰による線形 read-out。

閉形式解 ``W_out = (X^T X + lambda * P)^{-1} X^T Y`` を `np.linalg.solve` で解く
(`np.linalg.inv` は使わない)。`P` はバイアス列に対応する対角成分だけ 0 とした
単位行列であり、これによりバイアス項は正則化の対象外となる。

設計行列 `X` は `input_passthrough` と `use_states` の組で決まる。

| `input_passthrough` | `use_states` | 設計行列    |
| ------------------- | ------------ | ----------- |
| True (既定)         | True (既定)  | `[1, u, x]` |
| False               | True         | `[1, x]`    |
| True                | False        | `[1, u]`    |

``use_states=False`` はリザバー無しの baseline (`ESNConfig.use_reservoir`) を
作るためにある。両方を False にすると設計行列がバイアス列だけになるので拒否する。
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

BIAS_COLUMN_INDEX = 0


class RidgeReadout:
    """リッジ回帰 read-out (バイアス項は非正則化)。"""

    def __init__(
        self,
        alpha: float,
        *,
        input_passthrough: bool = True,
        use_states: bool = True,
    ) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha は 0 以上である必要があります (実値: {alpha})")
        if not input_passthrough and not use_states:
            raise ValueError(
                "input_passthrough と use_states を同時に False にはできません "
                "(設計行列がバイアス列だけになります)"
            )
        self.alpha = float(alpha)
        self.input_passthrough = input_passthrough
        self.use_states = use_states
        self._w_out: NDArray[np.float64] | None = None

    @property
    def is_fitted(self) -> bool:
        """`fit` 済みかどうか。"""
        return self._w_out is not None

    @property
    def w_out(self) -> NDArray[np.float64]:
        """学習済み係数行列 `W_out` (`[n_features, n_outputs]`)。"""
        if self._w_out is None:
            raise RuntimeError(
                "RidgeReadout は未 fit です (fit を呼んでから w_out を参照してください)"
            )
        return self._w_out

    def n_features(self, n_inputs: int, n_reservoir: int) -> int:
        """設計行列の列数を返す。"""
        n = 1
        if self.input_passthrough:
            n += n_inputs
        if self.use_states:
            n += n_reservoir
        return n

    def design_matrix(
        self, states: NDArray[np.float64], inputs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """設計行列を組み立てる。

        `input_passthrough` と `use_states` の組で `[1, u, x]` / `[1, x]` /
        `[1, u]` のいずれかになる (モジュール docstring の表)。先頭列は
        バイアス項で、リッジ罰則の対象外にする (`fit` 参照)。

        Args:
            states: リザバー状態 `[T, N]`。``use_states=False`` のときは列数 0 の
                `[T, 0]` を渡してよい (リザバーを駆動しない baseline のため)。
                系列長の検証には使うので形は必要である。
            inputs: 入力系列 `[T, D_u]`。

        Returns:
            設計行列 `[T, P]`。`P` は `n_features` が返す値。

        Raises:
            ValueError: いずれかが 2 次元でない、または系列長が食い違う場合。
        """
        x = _as_2d(states, "states")
        u = _as_2d(inputs, "inputs")
        if x.shape[0] != u.shape[0]:
            raise ValueError(
                "states と inputs の系列長が一致しません "
                f"(states: {x.shape[0]}, inputs: {u.shape[0]})"
            )
        ones = np.ones((x.shape[0], 1), dtype=np.float64)
        blocks = [ones]
        if self.input_passthrough:
            blocks.append(u)
        if self.use_states:
            blocks.append(x)
        return np.concatenate(blocks, axis=1)

    def fit(
        self,
        states: NDArray[np.float64],
        inputs: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> RidgeReadout:
        """閉形式のリッジ解 ``W_out = (X^T X + alpha P)^{-1} X^T Y`` を求める。

        逆行列は作らず `np.linalg.solve` で解く。バイアス列は罰則の対象外に
        する (定数項を縮めると予測が系統的に原点へ寄るため)。

        washout の除去は呼び出し側の責務である (`ESN.fit` が行う)。

        Args:
            states: リザバー状態 `[T, N]`。
            inputs: 入力系列 `[T, D_u]`。`input_passthrough` が偽でも系列長の
                検証に使うため必須。
            targets: 目標 `[T, n_outputs]`。

        Returns:
            自分自身 (メソッドチェーン用)。

        Raises:
            ValueError: 設計行列と `targets` の系列長が食い違う場合。
            numpy.linalg.LinAlgError: Gram 行列が特異な場合 (`alpha` を上げる)。
        """
        design = self.design_matrix(states, inputs)
        y = _as_2d(targets, "targets")
        if design.shape[0] != y.shape[0]:
            raise ValueError(
                "設計行列と targets の系列長が一致しません "
                f"(設計行列: {design.shape[0]}, targets: {y.shape[0]})"
            )

        n_features = design.shape[1]
        penalty = np.eye(n_features, dtype=np.float64)
        # バイアス列 (定数 1) は正則化対象外にする。
        penalty[BIAS_COLUMN_INDEX, BIAS_COLUMN_INDEX] = 0.0

        gram = design.T @ design + self.alpha * penalty
        rhs = design.T @ y
        self._w_out = np.linalg.solve(gram, rhs)
        logger.debug(
            "readout fitted: n_features=%d n_outputs=%d alpha=%g",
            n_features,
            y.shape[1],
            self.alpha,
        )
        return self

    def predict(
        self, states: NDArray[np.float64], inputs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """学習済み `W_out` で予測する。

        Args:
            states: リザバー状態 `[T, N]`。
            inputs: 入力系列 `[T, D_u]`。

        Returns:
            予測 `[T, n_outputs]`。

        Raises:
            RuntimeError: `fit` を呼ぶ前に呼んだ場合。
            ValueError: 設計行列の列数が学習時と食い違う場合。
        """
        w_out = self.w_out
        design = self.design_matrix(states, inputs)
        if design.shape[1] != w_out.shape[0]:
            raise ValueError(
                "設計行列の列数が学習時と一致しません "
                f"(期待: {w_out.shape[0]}, 実値: {design.shape[1]})"
            )
        return design @ w_out


def _as_2d(array: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """2 次元の float64 配列として検証する。"""
    validated = np.asarray(array, dtype=np.float64)
    if validated.ndim != 2:
        raise ValueError(
            f"{name} は 2 次元配列が必要です (実 shape: {validated.shape})"
        )
    return validated
