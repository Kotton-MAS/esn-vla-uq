"""非適合度スコア (nonconformity score)。

`docs/design.md` 8 節の未解決論点 3 (残差の正規化方法) への回答。2 種類を実装する。

- ``"absolute"``: ``s = max_j |r_j| / c_j``
- ``"normalized"``: ``s = max_j |r_j| / (c_j * g(x))``

``c_j`` は**次元ごとの定数スケール** (残差の中央値)、``g(x)`` は**入力ごとの
スカラー難易度**。

**次元スケールを先に割る理由**: 合成データの残差中央値は 6 DoF デルタが
0.005〜0.007 に対しグリッパが 0.026 と約 5 倍ある。次元をそのまま比べると
``max_j`` がグリッパ次元に支配され、他の次元の情報が捨てられる。

## 難易度 g(x) を「観測量」から取る理由

``g(x)`` は入力に含まれる観測量 (`data/features.py` の `DIFFICULTY_FEATURE`、
チャンク分散の対数) を、fit 集合での中央値が 1 になるよう中心化して使う。
**残差の大きさを推定するモデルは使わない。**

split conformal の被覆率保証は「較正データを見ずに、入力だけから決まる」任意の
``sigma(x)`` に対して成り立つ。``sigma`` が残差の良い推定である必要は無い。
推定が下手なら区間幅が無駄に広くなるだけで、被覆率は保たれる。この自由度を
使い、**観測できて失敗と結びつく量**を選ぶ。

当初はリザバー状態から ``log|r|`` を予測する第 2 の ridge read-out で ``g(x)``
を推定していた (Papadopoulos らの normalized nonconformity)。実装して測った
結果、この方針は本データでは機能しなかった。記録として残す。

| 試した内容 | 失敗検知 AUROC |
| --- | --- |
| 次元ごとに ``log|r_j|`` を予測 | 0.612 ± 0.099 (平均幅が実スケールの 4000 倍) |
| 次元を揃えて ``max_j`` を予測 | 0.44 ± 0.20 |
| 同上を ``mean_j`` に変更 | 0.46 ± 0.20 |
| 目標をエピソード内で平滑化 | 0.28 ± 0.10 (悪化) |
| 観測量 (チャンク分散) を使う (現行) | **0.869 ± 0.075** |

学習型が 0.5 を下回る (= 反相関する)のは、read-out の**学習集合内**残差で
難易度を学習していたため。てこ比の高い点は in-sample 残差がほぼ 0 に潰れる一方、
out-of-sample では最も誤差が大きい。平滑化すると反相関が強まった (0.37 → 0.28)
ことからも、雑音ではなく系統的な反転だと分かる。

さらに本データでは、**真の残差を使っても** 失敗検知 AUROC は 0.68 ± 0.11 が
上限だった。一方チャンク分散は 0.87 ± 0.075。残差の大きさは失敗の在り処では
ないため、推定器をいくら改良しても届かない。

## 代償

観測量ベースの ``g(x)`` は残差の推定ではないため、被覆率がやや名目を下回る
(0.864 対 0.900、ECE 0.042)。`absolute` は被覆率が正確 (0.903、ECE 0.002) だが
区間幅が定数で失敗を区別しない (AUROC は定義上 0.5)。この対比は
`tests/test_calibration.py` が数値で固定する。

多次元目標の扱い: スコアを次元方向の max でスカラー化し、**全次元が同時に
区間内に入る確率**として被覆率を定義する (同時被覆)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
from numpy.typing import NDArray

ScoreKind = Literal["absolute", "normalized"]
"""非適合度スコアの種類。"""

SUPPORTED_SCORE_KINDS: Final[tuple[ScoreKind, ...]] = get_args(ScoreKind)
"""`ScoreKind` が許可する値の実行時タプル。"""

DEFAULT_SCORE_KIND: Final[ScoreKind] = "normalized"
"""既定のスコア。`absolute` は入力に依存しない定数幅になるため既定にしない。"""

DIFFICULTY_CLIP_QUANTILE: Final[float] = 0.02
"""中心化した対数難易度を丸め込む分位点 (fit 集合での分布)。

観測量が外れ値を取ったときに区間幅が発散しないようにする。
"""

_DIMENSION_SCALE_FLOOR: Final[float] = 1e-9
"""``c_j`` の下限。ある次元の残差が恒等的に 0 でもゼロ除算しないため。"""


@dataclass(frozen=True)
class ScoreModel:
    """非適合度スコアの計算方法 (学習済み)。

    Attributes:
        kind: スコアの種類。
        dimension_scale: 次元ごとの定数スケール ``c_j`` (`[D_y]`)。
        difficulty_readout: `normalized` のときの ``g(x)`` 推定用 read-out。
            `absolute` のときは `None`。
        log_difficulty_bounds: 予測した ``log g`` を丸め込む範囲 (下限, 上限)。
            `absolute` のときは `None`。
    """

    kind: ScoreKind
    dimension_scale: NDArray[np.float64]
    difficulty_column: int | None
    log_difficulty_center: float = 0.0
    log_difficulty_bounds: tuple[float, float] | None = None

    def difficulty(
        self, states: NDArray[np.float64], inputs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """入力ごとのスカラー難易度 ``g(x)`` を `[N, 1]` で返す。

        `absolute` では全要素 1.0 (= 入力に依存しない)。`normalized` では入力の
        `difficulty_column` を対数難易度として読み、fit 集合の中央値が 1 になる
        よう中心化してから ``exp`` する。中心化により ``q`` が `absolute` の
        分位点と同程度の大きさに収まる。
        """
        if self.difficulty_column is None:
            return np.ones((states.shape[0], 1), dtype=np.float64)
        log_difficulty = inputs[:, self.difficulty_column : self.difficulty_column + 1]
        centered = log_difficulty - self.log_difficulty_center
        if self.log_difficulty_bounds is not None:
            lower, upper = self.log_difficulty_bounds
            centered = np.clip(centered, lower, upper)
        result: NDArray[np.float64] = np.exp(centered)
        return result

    def scale(
        self, states: NDArray[np.float64], inputs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """各標本・各次元のスケール ``c_j * g(x)`` を `[N, D_y]` で返す。"""
        return self.difficulty(states, inputs) * self.dimension_scale

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


def dimension_scale(residuals: NDArray[np.float64]) -> NDArray[np.float64]:
    """次元ごとの定数スケール ``c_j`` を残差の中央値から求める。

    平均ではなく中央値を使う。残差は裾が重く、平均は少数の大きな残差に
    引きずられる。
    """
    scale: NDArray[np.float64] = np.median(np.abs(residuals), axis=0)
    return np.maximum(scale, _DIMENSION_SCALE_FLOOR)


def fit_score_model(
    kind: ScoreKind,
    residuals: NDArray[np.float64],
    states: NDArray[np.float64],
    inputs: NDArray[np.float64],
    *,
    difficulty_column: int | None = None,
    clip_quantile: float = DIFFICULTY_CLIP_QUANTILE,
) -> ScoreModel:
    """スコアモデルを学習する。

    次元スケール ``c_j`` は両方のスコアで求める (次元間を揃えるのは `absolute`
    でも有効なため)。`normalized` はさらに ``difficulty_column`` の観測量を
    難易度として使い、fit 集合での中央値が 1 になるよう中心化する。

    学習には **fit 集合**のみを使う。較正集合を使うと較正集合が二重に使われ、
    conformal の交換可能性が壊れる。

    Args:
        kind: スコアの種類。
        residuals: fit 集合の残差 `[N, D_y]`。
        states: fit 集合のリザバー状態 `[N, N_res]` (現在は未使用。難易度を
            リザバー状態から学習する実装に戻す場合の拡張点)。
        inputs: fit 集合の入力 `[N, D_u]`。
        difficulty_column: 難易度として読む入力の列。`None` なら `normalized`
            を指定しても定数幅になる。
        clip_quantile: 中心化した対数難易度を丸め込む分位点。

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
    scale = dimension_scale(residuals)
    if kind == "absolute" or difficulty_column is None:
        return ScoreModel(kind=kind, dimension_scale=scale, difficulty_column=None)

    log_difficulty = inputs[:, difficulty_column]
    center = float(np.median(log_difficulty))
    centered = log_difficulty - center
    bounds = (
        float(np.quantile(centered, clip_quantile)),
        float(np.quantile(centered, 1.0 - clip_quantile)),
    )
    return ScoreModel(
        kind=kind,
        dimension_scale=scale,
        difficulty_column=difficulty_column,
        log_difficulty_center=center,
        log_difficulty_bounds=bounds,
    )
