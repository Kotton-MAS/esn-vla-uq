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

    `ESNConfig.use_reservoir=False` のときも `transform` はリザバーを駆動する
    (状態を返すのが `transform` の契約であるため)。read-out が状態を使わなく
    なるだけで、計算は省かれない。リザバー無し baseline を**安く**回したい場合は
    `uncertainty/conformal.py` の経路を使う (そちらは駆動自体を飛ばす)。
    """

    def __init__(
        self, config: ESNConfig, activation: Activation = tanh_activation
    ) -> None:
        self.config = config
        self._activation = activation
        self._reservoir: Reservoir | None = None
        self._readout = RidgeReadout(
            config.ridge_alpha,
            input_passthrough=config.input_passthrough,
            use_states=config.use_reservoir,
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
        """入力系列からリザバー状態行列を得る (read-out を経由しない)。

        washout は適用しない。診断モジュールや `uncertainty` 層が生の状態を
        必要とするための入口である。

        **エピソード境界を跨ぐ系列をそのまま渡してはならない。** 状態は系列の
        先頭から連続に発展するため、独立した試行を連結して渡すと前の試行の
        末尾状態が次へ持ち越される。複数エピソードを扱う場合は
        `esn.reservoir.run_episodes` を使う (`docs/design.md` 3.9 節)。

        Args:
            inputs: 入力系列 `[T, n_inputs]`。初回呼び出し時の第 2 軸が
                リザバーの入力次元を決め、以降はそれと一致する必要がある。

        Returns:
            状態行列 `[T, N]` (`N = config.n_reservoir`)。各行が時刻 t の
            `x[t]` で、初期状態 `x[-1]`(零ベクトル) は含まない。

        Raises:
            ValueError: `inputs` が 2 次元でない、または入力次元が初回と
                食い違う場合。
        """
        u = np.asarray(inputs, dtype=np.float64)
        if u.ndim != 2:
            raise ValueError(
                f"inputs は 2 次元 [T, n_inputs] が必要です (実 shape: {u.shape})"
            )
        reservoir = self._ensure_reservoir(int(u.shape[1]))
        return reservoir.run(u)

    def fit(self, inputs: NDArray[np.float64], targets: NDArray[np.float64]) -> ESN:
        """入力・目標系列からリッジ read-out を学習する。

        リザバーは学習しない。閉形式のリッジ解で線形 read-out だけを解く
        (勾配法もアンサンブルも使わない)。教師強制 (teacher forcing) は行わない。

        先頭 `config.washout` ステップは学習から除く。リザバーの初期過渡が
        read-out の解を歪めるためである。`washout=0` を指定すれば除かない。

        Args:
            inputs: 入力系列 `[T, n_inputs]`。
            targets: 目標系列 `[T, n_outputs]`。1 次元 `[T]` も受け付け、その
                場合は `predict` も `[T]` を返す。

        Returns:
            自分自身 (メソッドチェーン用)。

        Raises:
            ValueError: `inputs` が 2 次元でない、`inputs` と `targets` の
                系列長が違う、または `washout` が系列長以上の場合。
        """
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
        """学習済み read-out で予測する。

        `transform` と同じくエピソード境界を跨ぐ系列をそのまま渡してはならない
        (`docs/design.md` 3.9 節)。

        Args:
            inputs: 入力系列 `[T, n_inputs]`。入力次元は `fit` 時と一致する
                必要がある。

        Returns:
            予測 `[T, n_outputs]`。`fit` に 1 次元の `targets` を渡していた
            場合は `[T]`。

        Raises:
            RuntimeError: `fit` を呼ぶ前に呼んだ場合。
            ValueError: `inputs` が 2 次元でない、または入力次元が `fit` 時と
                食い違う場合。
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
