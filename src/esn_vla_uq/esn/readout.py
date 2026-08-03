"""リッジ回帰による線形 read-out。

閉形式解 ``W_out = (X^T X + lambda * P)^{-1} X^T Y`` を `np.linalg.solve` で解く
(`np.linalg.inv` は使わない)。`P` はバイアス列に対応する対角成分だけ 0 とした
単位行列であり、これによりバイアス項は正則化の対象外となる。

設計行列 `X` は `input_passthrough=True` (既定) のとき `[1, u, x]`、
`False` のとき `[1, x]`。
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

BIAS_COLUMN_INDEX = 0


class RidgeReadout:
    """リッジ回帰 read-out (バイアス項は非正則化)。"""

    def __init__(self, alpha: float, *, input_passthrough: bool = True) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha は 0 以上である必要があります (実値: {alpha})")
        self.alpha = float(alpha)
        self.input_passthrough = input_passthrough
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
        if self.input_passthrough:
            return 1 + n_inputs + n_reservoir
        return 1 + n_reservoir

    def design_matrix(
        self, states: NDArray[np.float64], inputs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """設計行列 `[1, u, x]` (または `[1, x]`) を組み立てる。"""
        x = _as_2d(states, "states")
        u = _as_2d(inputs, "inputs")
        if x.shape[0] != u.shape[0]:
            raise ValueError(
                "states と inputs の系列長が一致しません "
                f"(states: {x.shape[0]}, inputs: {u.shape[0]})"
            )
        ones = np.ones((x.shape[0], 1), dtype=np.float64)
        blocks = [ones, u, x] if self.input_passthrough else [ones, x]
        return np.concatenate(blocks, axis=1)

    def fit(
        self,
        states: NDArray[np.float64],
        inputs: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> RidgeReadout:
        """閉形式のリッジ解を求めて `self` を返す。"""
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
        """学習済み `W_out` で `[T, n_outputs]` を予測する。"""
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
