"""非適合度スコア (nonconformity score)。

`docs/design.md` 8 節の未解決論点 3 (残差の正規化方法) への回答。2 種類を実装する。

- ``"absolute"``: ``s = max_j |r_j|``
- ``"normalized"``: ``s = max_j |r_j| / (sigma_j(x) + beta)``

`sigma(x)` は「この入力における残差の大きさ」の推定値で、リザバー状態から
``log(|r| + eps)`` を予測する第 2 の ridge read-out で求める
(Papadopoulos らの normalized nonconformity)。

**2 つを実装するのは比較のためではなく、`absolute` では要件を満たせないことを
示すため。** `absolute` の区間幅は入力に依存しない定数になる。被覆率は名目値どおりに
なるが、全ステップで同じ幅なので「どのステップが危ないか」を一切区別しない。
要件書が求めるデモ GIF (失敗直前に不確実性バーが跳ねる) や失敗検知は
`normalized` でしか成立しない。この対比は
`tests/test_conformal.py::test_absolute_score_cannot_discriminate` が数値で固定する。

多次元目標の扱い: `action` は 7 次元ある。次元ごとに独立した区間を出すと被覆率が
次元ごとの周辺被覆になり「区間に入った」の意味が曖昧になるため、次元方向の
max でスカラー化し、**全次元が同時に区間内に入る確率**として被覆率を定義する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.esn.readout import RidgeReadout

ScoreKind = Literal["absolute", "normalized"]
"""非適合度スコアの種類。"""

SUPPORTED_SCORE_KINDS: Final[tuple[str, ...]] = get_args(ScoreKind)
"""`ScoreKind` が許可する値の実行時タプル。"""

DEFAULT_SCORE_KIND: Final[ScoreKind] = "normalized"
"""既定のスコア。`absolute` は入力に依存しない定数幅になるため既定にしない。"""

DEFAULT_SCALE_FLOOR: Final[float] = 1e-3
"""``sigma(x)`` に加える下駄 ``beta``。

推定スケールが 0 に近づくとスコアが発散し、少数の標本が分位点を支配する。
下駄を置くことで、スケールが極端に小さい領域でも区間が潰れないようにする。
"""

DEFAULT_SCALE_ALPHA: Final[float] = 1e-3
"""スケール推定用 read-out のリッジ正則化強度。"""

SCALE_CLIP_QUANTILE: Final[float] = 0.01
"""``log sigma`` の丸め込み範囲を決める分位点。

学習時に観測した ``log|r|`` の 1%〜99% 分位点を上下限にする。両端 1% を落とすのは、
残差がちょうど 0 に近い標本が下限を極端に小さくし、丸め込みが効かなくなるため。
"""

_LOG_EPSILON: Final[float] = 1e-12
"""``log(|r| + eps)`` の eps。残差がちょうど 0 の標本で -inf にしないため。"""


@dataclass(frozen=True)
class ScoreModel:
    """非適合度スコアの計算方法 (学習済み)。

    Attributes:
        kind: スコアの種類。
        scale_readout: `normalized` のときの ``sigma(x)`` 推定用 read-out。
            `absolute` のときは `None`。
        scale_floor: ``sigma(x)`` に加える下駄。
        log_scale_bounds: 予測した ``log sigma`` を丸め込む範囲 (下限, 上限)。
            `absolute` のときは `None`。
    """

    kind: ScoreKind
    scale_readout: RidgeReadout | None
    scale_floor: float
    log_scale_bounds: tuple[float, float] | None = None

    def scale(
        self, states: NDArray[np.float64], inputs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """各標本・各次元のスケール ``sigma_j(x)`` を `[N, D_y]` で返す。

        `absolute` では全要素 1.0 (= スケーリングしない)。

        `normalized` では予測した ``log sigma`` を学習時に観測した範囲へ
        丸め込んでから ``exp`` する。丸め込みが無いと、学習データの外側へ
        わずかに外挿しただけで ``exp`` が発散し、区間幅が行動の実スケール
        (0.01 程度) の 1000 倍以上になる (実測: 平均幅 44)。区間が広いだけなら
        被覆率は上がるが、**幅が意味を失い不確実性スコアとして使えなくなる**。
        """
        if self.scale_readout is None:
            return np.ones((states.shape[0], 1), dtype=np.float64)
        log_scale = self.scale_readout.predict(states, inputs)
        if self.log_scale_bounds is not None:
            lower, upper = self.log_scale_bounds
            log_scale = np.clip(log_scale, lower, upper)
        estimated: NDArray[np.float64] = np.exp(log_scale)
        return estimated + self.scale_floor

    def score(
        self,
        residuals: NDArray[np.float64],
        states: NDArray[np.float64],
        inputs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """非適合度スコア `[N]` を返す (次元方向の max)。"""
        scaled = np.abs(residuals) / self.scale(states, inputs)
        result: NDArray[np.float64] = np.max(scaled, axis=1)
        return result


def fit_score_model(
    kind: ScoreKind,
    residuals: NDArray[np.float64],
    states: NDArray[np.float64],
    inputs: NDArray[np.float64],
    *,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
    scale_alpha: float = DEFAULT_SCALE_ALPHA,
) -> ScoreModel:
    """スコアモデルを学習する。

    `normalized` のときだけ学習が要る。``log(|r_j| + eps)`` を目標とする ridge
    read-out を、**fit 集合の残差**に対して学習する。較正集合の残差を使うと
    較正集合が二重に使われ、conformal の交換可能性が壊れる。

    Args:
        kind: スコアの種類。
        residuals: fit 集合の残差 `[N, D_y]`。
        states: fit 集合のリザバー状態 `[N, N_res]`。
        inputs: fit 集合の入力 `[N, D_u]`。
        scale_floor: ``sigma(x)`` に加える下駄。
        scale_alpha: スケール推定 read-out の正則化強度。

    Returns:
        学習済みの `ScoreModel`。

    Raises:
        ValueError: `kind` が未知の場合。
    """
    if kind not in SUPPORTED_SCORE_KINDS:
        raise ValueError(
            f"kind: 未知のスコアです (actual={kind!r}, "
            f"supported={list(SUPPORTED_SCORE_KINDS)})"
        )
    if kind == "absolute":
        return ScoreModel(kind=kind, scale_readout=None, scale_floor=scale_floor)

    log_magnitude = np.log(np.abs(residuals) + _LOG_EPSILON)
    readout = RidgeReadout(alpha=scale_alpha, input_passthrough=True)
    readout.fit(states, inputs, log_magnitude)
    # 学習時に**実際に観測した** log|r| の範囲を丸め込みの上下限にする。
    # 予測値の範囲ではなく観測値の範囲を使うのは、read-out が学習集合上で
    # すでに外挿している場合にその外挿まで許してしまわないため。
    bounds = (
        float(np.quantile(log_magnitude, SCALE_CLIP_QUANTILE)),
        float(np.quantile(log_magnitude, 1.0 - SCALE_CLIP_QUANTILE)),
    )
    return ScoreModel(
        kind=kind,
        scale_readout=readout,
        scale_floor=scale_floor,
        log_scale_bounds=bounds,
    )
