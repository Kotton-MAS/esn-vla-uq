"""Split conformal prediction による予測区間。

手順 (`docs/plans/sprint2_v0.1.md`):

1. **fit 集合**でリザバーを駆動し、read-out (`y[t] = action[t+1]` の予測) と
   スケール推定 read-out (`normalized` のとき) を学習する。
2. **較正集合**で非適合度スコアを計算し、その ``ceil((n+1)(1-alpha))/n`` 分位点
   ``q`` を取る。
3. **テスト集合**の予測区間を ``[y_hat - q * sigma(x), y_hat + q * sigma(x)]``
   とする。

分位点に ``ceil((n+1)(1-alpha))/n`` を使うのが有限標本での被覆率保証
``P(y in C(x)) >= 1 - alpha`` を与える鍵で、単純な経験分位点ではこの保証は
出ない。標本数 ``n`` が小さいと ``ceil((n+1)(1-alpha)) > n`` になりうる
(例: ``n=5, alpha=0.1``)。この場合は有限標本では要求された水準を保証できないため、
区間を無限大にせず **明示的なエラー**にする (黙って `inf` を返すと、レポート上は
被覆率 100% の「良い」結果に見えてしまう)。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.esn.config import ESNConfig
from esn_vla_uq.esn.readout import RidgeReadout
from esn_vla_uq.esn.reservoir import Reservoir, run_episodes
from esn_vla_uq.uncertainty.nonconformity import (
    DEFAULT_SCORE_KIND,
    ScoreKind,
    ScoreModel,
    fit_score_model,
)
from esn_vla_uq.uncertainty.targets import EpisodeSamples, input_segments, stack_targets

logger = logging.getLogger(__name__)

DEFAULT_ALPHA: Final[float] = 0.1
"""既定の有意水準 (名目被覆率 90%)。"""

DEFAULT_WASHOUT: Final[int] = 0
"""区間ごとの washout。

エピソードは 150〜250 ステップと短く、`ESNConfig` の既定 washout (100) を各区間に
適用すると標本の半分以上が消える。予測タスクでは初期過渡も「予測しづらい区間」
として意味を持つため、既定では捨てない。捨てたい場合は明示的に指定する。
"""

DEFAULT_INPUT_LAGS: Final[int] = 0
"""read-out の設計行列に足す入力のラグ数 (遅延埋め込み)。

``k > 0`` にすると設計行列に ``u[t-1], ..., u[t-k]`` が加わる。**リザバー無しで
これだけを足したものが「ただの遅延線」baseline** であり、リザバーが効かないときに
その理由を切り分けるための対照になる (`docs/design.md` 16 節)。

- 遅延を足しても幅が縮まない → **タスクが記憶を要求していない**。どんなリザバーでも
  原理的に効かない。
- 縮む → タスクは記憶を要求しており、**ESN という記憶の実装のほうに改善余地がある**。

リザバーはこのラグを見ない (生の ``u`` で駆動する)。ラグは read-out の設計行列
だけに入る。
"""


def lag_segments(
    segments: Sequence[NDArray[np.float64]], n_lags: int
) -> list[NDArray[np.float64]]:
    """区間ごとに ``[u[t], u[t-1], ..., u[t-k]]`` を横に並べた配列を返す。

    **エピソード境界を跨がない。** 区間の先頭では過去が存在しないので、その区間の
    先頭値で埋める (端点保持)。

    行を落とす選択もあるが、採らない。エピソード先頭は**最も予測しやすい区間**で
    あり (`docs/design.md` 12.3 節)、そこを落とすと残った行の非適合度が上がって
    幅が系統的に広がる。ラグを足した条件だけ行が減ると、その効果とラグの効果を
    分離できなくなる。**全条件で同じ行を保つほうが比較として正しい。**

    Args:
        segments: 区間ごとの入力 `[T_i, D_u]`。
        n_lags: 足すラグの数 ``k``。0 ならそのまま返す。

    Returns:
        区間ごとの `[T_i, D_u * (k + 1)]`。

    Raises:
        ValueError: `n_lags` が負の場合。
    """
    if n_lags < 0:
        raise ValueError(f"n_lags: 0 以上が必要です (actual={n_lags})")
    if n_lags == 0:
        return list(segments)
    lagged: list[NDArray[np.float64]] = []
    for segment in segments:
        blocks = [segment]
        for lag in range(1, n_lags + 1):
            shifted = np.empty_like(segment)
            shifted[:lag] = segment[0]
            shifted[lag:] = segment[:-lag]
            blocks.append(shifted)
        lagged.append(np.concatenate(blocks, axis=1))
    return lagged


@dataclass(frozen=True)
class PredictionIntervals:
    """テスト標本に対する予測区間。

    Attributes:
        predicted: 点予測 `[N, D_y]`。
        lower: 区間下端 `[N, D_y]`。
        upper: 区間上端 `[N, D_y]`。
        half_width: 区間の半幅 `[N, D_y]`。``q * sigma(x)``。
        uncertainty: ステップ単位の不確実性スコア `[N]`。半幅の次元方向 max。
            **正解を知らずに計算できる**ことが要件 (実運用では正解が無い)。
    """

    predicted: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    half_width: NDArray[np.float64]
    uncertainty: NDArray[np.float64]

    def covers(self, targets: NDArray[np.float64]) -> NDArray[np.bool_]:
        """各標本で**全次元が同時に**区間内かを `[N]` で返す。"""
        inside = (targets >= self.lower) & (targets <= self.upper)
        result: NDArray[np.bool_] = np.all(inside, axis=1)
        return result


def conformal_quantile_index(n_calibration: int, alpha: float) -> int:
    """有限標本の被覆率保証を与える順序統計量の番号 (1 始まり) を返す。

    ``ceil((n + 1) * (1 - alpha))`` を返す。

    Raises:
        ValueError: `alpha` が範囲外、または標本数が水準に対して不足する場合。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha: 0 < alpha < 1 である必要があります (actual={alpha})")
    if n_calibration < 1:
        raise ValueError(f"較正標本が空です (n_calibration={n_calibration})")
    index = math.ceil((n_calibration + 1) * (1.0 - alpha))
    if index > n_calibration:
        minimum = math.ceil(1.0 / alpha) - 1
        raise ValueError(
            "較正標本が少なすぎて要求水準の被覆率を有限標本で保証できません "
            f"(n_calibration={n_calibration}, alpha={alpha}, "
            f"必要な最小標本数={minimum})"
        )
    return index


class SplitConformalPredictor:
    """ESN の 1 ステップ先 action 予測に split conformal を掛ける。

    `fit` -> `calibrate` -> `predict_intervals` の順に呼ぶ。
    """

    def __init__(
        self,
        config: ESNConfig,
        *,
        alpha: float = DEFAULT_ALPHA,
        score_kind: ScoreKind = DEFAULT_SCORE_KIND,
        washout: int = DEFAULT_WASHOUT,
        input_lags: int = DEFAULT_INPUT_LAGS,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"alpha: 0 < alpha < 1 である必要があります (actual={alpha})"
            )
        if washout < 0:
            raise ValueError(f"washout: 0 以上が必要です (actual={washout})")
        if input_lags < 0:
            raise ValueError(f"input_lags: 0 以上が必要です (actual={input_lags})")
        self.config = config
        self.alpha = alpha
        self.score_kind = score_kind
        self.washout = washout
        self.input_lags = input_lags
        self._reservoir: Reservoir | None = None
        self._readout: RidgeReadout | None = None
        self._score_model: ScoreModel | None = None
        self._quantile: float | None = None
        self._n_calibration: int | None = None

    @property
    def quantile(self) -> float:
        """較正で決まった分位点 ``q``。"""
        if self._quantile is None:
            raise RuntimeError("calibrate() を先に呼んでください")
        return self._quantile

    @property
    def n_calibration(self) -> int:
        """較正に使った標本数。"""
        if self._n_calibration is None:
            raise RuntimeError("calibrate() を先に呼んでください")
        return self._n_calibration

    def fit(self, samples: Sequence[EpisodeSamples]) -> SplitConformalPredictor:
        """fit 集合で read-out とスケール推定を学習する。

        Args:
            samples: fit 集合。

        Returns:
            自分自身。
        """
        states, inputs, targets = self._design(samples)
        readout = RidgeReadout(
            alpha=self.config.ridge_alpha,
            input_passthrough=self.config.input_passthrough,
            use_states=self.config.use_reservoir,
        )
        readout.fit(states, inputs, targets)
        residuals = targets - readout.predict(states, inputs)
        self._readout = readout
        self._score_model = fit_score_model(
            self.score_kind,
            residuals,
            states,
            inputs,
            difficulty_column=samples[0].difficulty_column,
        )
        logger.debug(
            "conformal fit: n_samples=%d score_kind=%s",
            states.shape[0],
            self.score_kind,
        )
        return self

    def calibrate(self, samples: Sequence[EpisodeSamples]) -> SplitConformalPredictor:
        """較正集合で分位点 ``q`` を決める。

        Args:
            samples: 較正集合。fit 集合と重なってはならない (重なると残差が
                楽観的になり区間が過小になる)。

        Returns:
            自分自身。

        Raises:
            ValueError: 較正標本が水準に対して不足する場合。
        """
        states, inputs, targets = self._design(samples)
        scores = self._scores(states, inputs, targets)
        index = conformal_quantile_index(scores.shape[0], self.alpha)
        self._quantile = float(np.sort(scores)[index - 1])
        self._n_calibration = int(scores.shape[0])
        logger.debug(
            "conformal calibrated: n=%d alpha=%g quantile=%g",
            self._n_calibration,
            self.alpha,
            self._quantile,
        )
        return self

    def predict_intervals(
        self, samples: Sequence[EpisodeSamples]
    ) -> PredictionIntervals:
        """テスト集合の予測区間を返す。"""
        quantile = self.quantile
        states, inputs, _targets = self._design(samples)
        predicted = self._require_readout().predict(states, inputs)
        half_width = quantile * self._require_score_model().scale(states, inputs)
        uncertainty: NDArray[np.float64] = np.max(half_width, axis=1)
        return PredictionIntervals(
            predicted=predicted,
            lower=predicted - half_width,
            upper=predicted + half_width,
            half_width=half_width,
            uncertainty=uncertainty,
        )

    def nonconformity_scores(
        self, samples: Sequence[EpisodeSamples]
    ) -> NDArray[np.float64]:
        """標本の非適合度スコア `[N]` を返す。

        較正評価に使う。任意の水準 ``alpha`` の被覆率は「テストのスコアが較正
        スコアの分位点以下である割合」に等しいため、reliability curve は
        較正スコアとテストスコアだけから引ける (`calibration/metrics.py`)。
        """
        states, inputs, targets = self._design(samples)
        return self._scores(states, inputs, targets)

    def stacked_targets(self, samples: Sequence[EpisodeSamples]) -> NDArray[np.float64]:
        """`predict_intervals` の結果と行が対応する目標 `[N, D_y]` を返す。"""
        _states, _inputs, targets = self._design(samples)
        return targets

    def to_dict(self) -> dict[str, object]:
        """レポート用の辞書。"""
        return {
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "score_kind": self.score_kind,
            "washout": self.washout,
            "input_lags": self.input_lags,
            "quantile": self.quantile,
            "n_calibration": self.n_calibration,
        }

    def _require_readout(self) -> RidgeReadout:
        if self._readout is None:
            raise RuntimeError("fit() を先に呼んでください")
        return self._readout

    def _require_score_model(self) -> ScoreModel:
        if self._score_model is None:
            raise RuntimeError("fit() を先に呼んでください")
        return self._score_model

    def _scores(
        self,
        states: NDArray[np.float64],
        inputs: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """非適合度スコア `[N]` を計算する。"""
        residuals = targets - self._require_readout().predict(states, inputs)
        return self._require_score_model().score(residuals, states, inputs)

    def _design(
        self, samples: Sequence[EpisodeSamples]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """リザバー状態・入力・目標を、washout 適用後に連結して返す。

        リザバーはエピソード境界で初期化する (`run_episodes`)。

        `ESNConfig.use_reservoir=False` (リザバー無し baseline) のときは
        **リザバーを構築も駆動もしない。** 状態の代わりに列数 0 の `[T, 0]` を
        返す。read-out もスコアモデルも状態を使わないので、`O(T N^2)` の駆動が
        丸ごと不要になる。形だけ合わせておけば下流は分岐しなくて済む。
        """
        if not samples:
            raise ValueError("samples: 1 件以上必要です")
        segments = input_segments(samples)
        if self.config.use_reservoir:
            reservoir = self._ensure_reservoir(segments[0].shape[1])
            states = run_episodes(reservoir, segments)
        else:
            states = np.zeros((sum(s.shape[0] for s in segments), 0), dtype=np.float64)
        # ラグはリザバーを駆動したあとで足す。リザバーは生の u だけを見る。
        inputs = np.concatenate(lag_segments(segments, self.input_lags), axis=0)
        targets = stack_targets(samples)
        if self.washout == 0:
            return states, inputs, targets
        keep = self.retained_mask(samples)
        return states[keep], inputs[keep], targets[keep]

    def retained_mask(self, samples: Sequence[EpisodeSamples]) -> NDArray[np.bool_]:
        """washout 適用後に残る標本を示す `[N]` を返す。

        **区間ごとに先頭 `washout` 標本を落とす。** `washout=0` なら全て True。

        呼び出し側が標本と行を対応させたい配列 (失敗検知のラベルなど) を持って
        いるときは、それにこのマスクを掛ける必要がある。`predict_intervals` の
        戻り値は washout 後の行数であり、`targets.detection_labels` が返す
        ラベルは washout 前の行数である。**両者を突き合わせる前に揃えないと、
        長さが違えば例外、たまたま同じなら黙って別のステップと突き合わせる。**
        """
        masks: list[NDArray[np.bool_]] = []
        for sample in samples:
            mask = np.ones(sample.n_samples, dtype=np.bool_)
            if self.washout >= sample.n_samples:
                raise ValueError(
                    "washout が標本数以上のエピソードがあります "
                    f"(episode_id={sample.episode_id!r}, "
                    f"washout={self.washout}, n_samples={sample.n_samples})"
                )
            mask[: self.washout] = False
            masks.append(mask)
        return np.concatenate(masks, axis=0)

    def _ensure_reservoir(self, n_inputs: int) -> Reservoir:
        """入力次元に合ったリザバーを 1 度だけ構築する。"""
        if self._reservoir is None:
            self._reservoir = Reservoir(self.config, n_inputs)
        elif self._reservoir.n_inputs != n_inputs:
            raise ValueError(
                "入力次元が学習時と一致しません "
                f"(期待: {self._reservoir.n_inputs}, 実値: {n_inputs})"
            )
        return self._reservoir
