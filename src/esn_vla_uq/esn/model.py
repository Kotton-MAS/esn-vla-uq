"""ESN 本体 (リザバー + リッジ read-out) の学習・予測 API。

教師強制 (teacher forcing) は行わない。`fit` は入力系列でリザバーを駆動し、
washout を除いた状態と入力からリッジ read-out を 1 回の閉形式で解く。
`predict` は入力長と同じ長さの予測を返す (先頭 washout ステップは過渡応答を
含むため、評価時は呼び出し側で `ESNConfig.washout` 分を捨てる)。
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.esn.config import ESNConfig
from esn_vla_uq.esn.readout import RidgeReadout
from esn_vla_uq.esn.reservoir import Activation, Reservoir, tanh_activation

logger = logging.getLogger(__name__)


class ESN:
    """Echo State Network。

    リザバーは最初に入力が与えられた時点で `ESNConfig.seed` から構築される。
    構築は入力次元 `n_inputs` にのみ依存するため、同一 seed・同一入力次元なら
    `fit` / `transform` の呼び出し順に依らず同じ行列になる。
    """

    def __init__(
        self, config: ESNConfig, activation: Activation = tanh_activation
    ) -> None:
        self.config = config
        self._activation = activation
        self._reservoir: Reservoir | None = None
        self._readout = RidgeReadout(
            config.ridge_alpha, input_passthrough=config.input_passthrough
        )
        self._targets_were_1d = False

    @property
    def is_fitted(self) -> bool:
        """`fit` 済みかどうか。"""
        return self._readout.is_fitted

    @property
    def reservoir(self) -> Reservoir:
        """構築済みリザバー。未構築なら `RuntimeError`。"""
        if self._reservoir is None:
            raise RuntimeError(
                "リザバーが未構築です (fit または transform を先に呼んでください)"
            )
        return self._reservoir

    @property
    def readout(self) -> RidgeReadout:
        """リッジ read-out。"""
        return self._readout

    def transform(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """入力系列 `[T, n_inputs]` からリザバー状態行列 `[T, N]` を得る。"""
        u = np.asarray(inputs, dtype=np.float64)
        if u.ndim != 2:
            raise ValueError(
                f"inputs は 2 次元 [T, n_inputs] が必要です (実 shape: {u.shape})"
            )
        reservoir = self._ensure_reservoir(int(u.shape[1]))
        return reservoir.run(u)

    def fit(self, inputs: NDArray[np.float64], targets: NDArray[np.float64]) -> ESN:
        """入力・目標系列から read-out を学習して `self` を返す。"""
        u = np.asarray(inputs, dtype=np.float64)
        y, self._targets_were_1d = _as_2d_targets(targets)
        if u.ndim != 2:
            raise ValueError(
                f"inputs は 2 次元 [T, n_inputs] が必要です (実 shape: {u.shape})"
            )
        if u.shape[0] != y.shape[0]:
            raise ValueError(
                "inputs と targets の系列長が一致しません "
                f"(inputs: {u.shape[0]}, targets: {y.shape[0]})"
            )

        washout = self.config.washout
        if washout >= u.shape[0]:
            raise ValueError(
                f"washout が系列長以上です (washout: {washout}, 系列長: {u.shape[0]})"
            )

        states = self.transform(u)
        self._readout.fit(states[washout:], u[washout:], y[washout:])
        logger.debug(
            "esn fitted: n_reservoir=%d n_inputs=%d n_train=%d",
            self.config.n_reservoir,
            u.shape[1],
            u.shape[0] - washout,
        )
        return self

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """学習済み read-out で `[T, n_outputs]` を予測する。

        `fit` に 1 次元の `targets` を渡していた場合は `[T]` を返す。
        未 fit の場合は `RuntimeError`。
        """
        if not self._readout.is_fitted:
            raise RuntimeError(
                "ESN は未 fit です (predict の前に fit を呼んでください)"
            )
        u = np.asarray(inputs, dtype=np.float64)
        states = self.transform(u)
        predictions = self._readout.predict(states, u)
        if self._targets_were_1d:
            return predictions[:, 0]
        return predictions

    def _ensure_reservoir(self, n_inputs: int) -> Reservoir:
        """必要ならリザバーを構築し、入力次元の整合を検証する。"""
        if self._reservoir is None:
            self._reservoir = Reservoir(
                self.config, n_inputs, activation=self._activation
            )
        elif self._reservoir.n_inputs != n_inputs:
            raise ValueError(
                "入力次元が構築済みリザバーと一致しません "
                f"(期待: {self._reservoir.n_inputs}, 実値: {n_inputs})"
            )
        return self._reservoir


def _as_2d_targets(targets: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
    """目標系列を `[T, n_outputs]` に整形し、元が 1 次元だったかを返す。"""
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        return y.reshape(-1, 1), True
    if y.ndim == 2:
        return y, False
    raise ValueError(
        "targets は 1 次元 [T] か 2 次元 [T, n_outputs] が必要です "
        f"(実 shape: {y.shape})"
    )
