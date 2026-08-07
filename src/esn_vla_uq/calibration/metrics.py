"""較正評価の指標 (被覆率 / reliability curve / ECE / 失敗検知 AUROC)。

依存は numpy のみ。作図 (`calibration/plot.py`) とは分離してあり、matplotlib が
無い環境でも全ての数値が計算できる。

**ECE の定義について**: 分類の ECE は「予測確率の区間ごとに、平均予測確率と実際の
正解率の差」を取る。回帰の予測区間にはこの形の確率が無いため、直接は転用できない。
本モジュールでは名目被覆率 ``1 - alpha`` を横軸、経験被覆率を縦軸にした
reliability curve を引き、**両者の差の絶対値の平均**を ECE と呼ぶ。分類の ECE とは
別物なので、レポート JSON にも定義を書き出す (`ECE_DEFINITION`)。

**有効標本数の注意**: 被覆率はステップ単位で数えるが、同一エピソード内のステップは
強く相関している。したがって被覆率の分散を決めるのは**エピソード数**であって
ステップ数ではない。同梱の合成データ (40 エピソード、較正 8 エピソード) では、
名目 90% に対する実測被覆率が分割の乱数種によって 0.63〜1.00 まで振れる
(平均は 0.896 で名目どおり)。単一の分割の被覆率をもって「較正がずれている」と
判断してはならない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

ECE_DEFINITION: Final[str] = (
    "mean(|nominal_coverage - empirical_coverage|) over the evaluated nominal levels. "
    "This is the regression/interval analogue of the classification ECE, not the "
    "classification ECE itself."
)
"""レポートに書き出す ECE の定義。読み手が分類の ECE と混同しないようにする。"""

DEFAULT_NOMINAL_LEVELS: Final[tuple[float, ...]] = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
)
"""reliability curve を評価する名目被覆率。"""


@dataclass(frozen=True)
class ReliabilityCurve:
    """名目被覆率と経験被覆率の対応。

    Attributes:
        nominal: 実際に評価できた名目被覆率 `[L]`。
        empirical: 対応する経験被覆率 `[L]`。
        unsupported: 較正標本が足りず評価できなかった名目被覆率。
            **黙って落とさず必ず記録する。** 高い水準ほど多くの較正標本を要する
            ため、落ちるのは決まって曲線の右端であり、ECE が実勢より小さく出る。
    """

    nominal: tuple[float, ...]
    empirical: tuple[float, ...]
    unsupported: tuple[float, ...] = ()

    def expected_calibration_error(self) -> float:
        """名目と経験の差の絶対値の平均 (モジュール docstring の定義)。"""
        gaps = np.abs(np.asarray(self.nominal) - np.asarray(self.empirical))
        return float(np.mean(gaps))

    def max_calibration_error(self) -> float:
        """名目と経験の差の絶対値の最大。"""
        gaps = np.abs(np.asarray(self.nominal) - np.asarray(self.empirical))
        return float(np.max(gaps))

    def to_dict(self) -> dict[str, object]:
        """レポート用の辞書。"""
        return {
            "nominal": list(self.nominal),
            "empirical": list(self.empirical),
            "unsupported_levels": list(self.unsupported),
            "ece": self.expected_calibration_error(),
            "max_calibration_error": self.max_calibration_error(),
            "ece_definition": ECE_DEFINITION,
        }


def conformal_coverage(
    calibration_scores: NDArray[np.float64],
    test_scores: NDArray[np.float64],
    alpha: float,
) -> float:
    """水準 ``alpha`` における経験被覆率を返す。

    conformal では「目標が区間に入る」ことと「テストのスコアが較正スコアの
    分位点以下である」ことが同値なので、スコアだけから被覆率が引ける。

    Args:
        calibration_scores: 較正集合の非適合度スコア `[n]`。
        test_scores: テスト集合の非適合度スコア `[m]`。
        alpha: 有意水準。名目被覆率は ``1 - alpha``。

    Returns:
        経験被覆率。

    Raises:
        ValueError: 標本が空、または `alpha` が水準に対して標本不足の場合。
    """
    # 循環 import を避けるためここで import する (conformal -> calibration の
    # 依存は無く、calibration -> uncertainty の一方向のみ)。
    from esn_vla_uq.uncertainty.conformal import conformal_quantile_index

    if test_scores.shape[0] == 0:
        raise ValueError("test_scores: 1 件以上必要です")
    index = conformal_quantile_index(calibration_scores.shape[0], alpha)
    quantile = float(np.sort(calibration_scores)[index - 1])
    return float(np.mean(test_scores <= quantile))


def reliability_curve(
    calibration_scores: NDArray[np.float64],
    test_scores: NDArray[np.float64],
    nominal_levels: tuple[float, ...] = DEFAULT_NOMINAL_LEVELS,
) -> ReliabilityCurve:
    """名目被覆率ごとの経験被覆率を求める。

    較正標本が少なく有限標本で保証できない水準 (高い名目被覆率) は評価から外し、
    **`unsupported` に記録する**。当初はここで `ValueError` にしていたが、その
    設計だと標本の少ないデータセット (収集直後の openpi ログなど) で較正評価
    そのものが止まってしまい、計算できる被覆率や ECE まで得られなくなる。

    落とした事実を残すのは、消えるのが決まって曲線の右端 (高い水準) であり、
    黙って落とすと ECE が実勢より小さく出るためである。

    Raises:
        ValueError: どの水準も評価できない場合 (較正標本が極端に少ない)。
    """
    nominal: list[float] = []
    empirical: list[float] = []
    unsupported: list[float] = []
    for level in nominal_levels:
        try:
            coverage = conformal_coverage(calibration_scores, test_scores, 1.0 - level)
        except ValueError:
            unsupported.append(level)
            continue
        nominal.append(level)
        empirical.append(coverage)
    if not nominal:
        raise ValueError(
            "reliability curve: どの名目水準も評価できませんでした "
            f"(n_calibration={calibration_scores.shape[0]}, "
            f"levels={list(nominal_levels)})"
        )
    return ReliabilityCurve(
        nominal=tuple(nominal),
        empirical=tuple(empirical),
        unsupported=tuple(unsupported),
    )


def rank_data(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """同順位を平均順位で扱う順位付け (`scipy.stats.rankdata` 相当)。

    同順位の扱いは AUROC の正しさに直結する。`absolute` スコアは全ステップで
    区間幅が同一 (= 全て同順位) になるため、素朴に `argsort` の順番をそのまま
    順位にすると、**元の並び順**に依存した意味のない AUROC が出る
    (実測: 本来 0.5 のところ 0.67 になった)。
    """
    sorter = np.argsort(values, kind="stable")
    inverse = np.empty_like(sorter)
    inverse[sorter] = np.arange(values.shape[0])
    sorted_values = values[sorter]
    is_new = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    dense = is_new.cumsum()[inverse]
    boundaries = np.concatenate((np.nonzero(is_new)[0], [values.shape[0]]))
    ranks: NDArray[np.float64] = 0.5 * (
        boundaries[dense] + boundaries[dense - 1] + 1
    ).astype(np.float64)
    return ranks


def detection_auroc(scores: NDArray[np.float64], positive: NDArray[np.bool_]) -> float:
    """スコアが陽性クラスをどれだけ順位付けできるかを AUROC で返す。

    Mann-Whitney U 統計量から求める (scipy を導入しない)。

    Args:
        scores: ステップ単位の不確実性スコア `[N]`。
        positive: 陽性ラベル `[N]` (失敗開始以降か)。

    Returns:
        AUROC。全て同順位のスコアでは厳密に 0.5 になる。

    Raises:
        ValueError: どちらかのクラスが空の場合。
    """
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            f"AUROC には両クラスの標本が必要です (陽性={n_positive}, 陰性={n_negative})"
        )
    ranks = rank_data(scores)
    u_statistic = float(ranks[positive].sum()) - n_positive * (n_positive + 1) / 2.0
    return u_statistic / (n_positive * n_negative)


DEFAULT_AUROC_Z: Final[float] = 1.96
"""95% 区間に対応する正規分布の分位点。"""


def auroc_confidence_interval(
    auroc: float,
    *,
    n_positive: int,
    n_negative: int,
    z: float = DEFAULT_AUROC_Z,
) -> tuple[float, float]:
    """AUROC の信頼区間を Hanley-McNeil の標準誤差から求める。

    ``SE = sqrt((A(1-A) + (n1-1)(Q1-A^2) + (n2-1)(Q2-A^2)) / (n1 n2))``
    ただし ``Q1 = A/(2-A)``、``Q2 = 2A^2/(1+A)``。

    **``n_positive`` / ``n_negative`` には有効標本数を渡すこと。** ステップ単位で
    AUROC を計算していても、同一エピソード内のステップは強く相関しているため、
    分散を決めるのは**エピソード数**である (`docs/design.md` 9.3 節、10.9 節)。
    ステップ数を渡すと区間が桁違いに狭く出て、実体のない信号を有意だと読む。
    どちらを渡すかが判定の分かれ目になるので、引数は必須キーワードにしてある。

    点推定だけでは「検知できない」と言えない。失敗が数十本しかなければ真の値が
    0.65 でも 0.48 は普通に出る。**区間を出して初めて実用水準を排除できる**
    (`docs/design.md` 10.16 節)。

    Args:
        auroc: 点推定 (`detection_auroc` の戻り値)。
        n_positive: 陽性の有効標本数 (通常は失敗エピソード数)。
        n_negative: 陰性の有効標本数 (通常は成功エピソード数)。
        z: 正規近似の分位点。既定は 95% 区間。

    Returns:
        ``(下限, 上限)``。``[0, 1]` に丸める。

    Raises:
        ValueError: `auroc` が `[0, 1]` の外、標本数が 1 未満、または `z` が負。
    """
    if not 0.0 <= auroc <= 1.0:
        raise ValueError(f"auroc: [0, 1] の範囲が必要です (actual={auroc})")
    if n_positive < 1 or n_negative < 1:
        raise ValueError(
            "有効標本数は両クラスとも 1 以上が必要です "
            f"(n_positive={n_positive}, n_negative={n_negative})"
        )
    if z < 0.0:
        raise ValueError(f"z: 0 以上が必要です (actual={z})")

    q1 = auroc / (2.0 - auroc)
    q2 = 2.0 * auroc * auroc / (1.0 + auroc)
    variance = (
        auroc * (1.0 - auroc)
        + (n_positive - 1) * (q1 - auroc * auroc)
        + (n_negative - 1) * (q2 - auroc * auroc)
    ) / (n_positive * n_negative)
    margin = z * math.sqrt(max(variance, 0.0))
    return (max(0.0, auroc - margin), min(1.0, auroc + margin))
