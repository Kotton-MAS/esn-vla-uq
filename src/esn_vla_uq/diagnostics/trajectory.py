"""リザバー状態軌道の時間的な統計量。

## なぜ要るのか

実 LIBERO の失敗はすべてタイムアウトであり (`docs/design.md` 10.15 節)、その実体は
**ポリシーが同じ動作を繰り返して止まること**である。瞬時量 (チャンク分散・行動の
大きさ・行動の変化) はどれも失敗と無相関だった (10.12〜10.14 節)。**「詰まり」は
時間方向の構造であって、1 ステップの値には現れない**という見立てから、リザバー
状態の軌道そのものを特徴にできないかを測るために置いた。

ここが扱うのは以下の 3 つで、いずれも**過去だけを見る** (窓は `[t-window+1, t]`)。
実運用では未来を知らずに計算できる必要があるため。

- `state_autocorrelation`: 軌道の自己相似性。同じ動作の反復で上がる。
- `state_participation_ratio`: 軌道が張る実効次元。1 点に潰れると下がる。
- `state_novelty`: 直近の状態集合からの隔たり。反復すると下がる。

## 結論: これでも検知できない

実測で決着した (`docs/design.md` 14 節)。3 特徴とも実用水準 (0.6) を区間で排除する。
**この結論をもって失敗検知を閉じている。**

軌道の記述としては正しく動くので、リザバーが実データ上でどう振る舞っているかを
見る用途には使える。検知に使えるという主張だけが否定された。

## 先頭は NaN で返す

窓が埋まらない先頭区間は `NaN` を返す。0 などで埋めると「窓が足りない」と
「値が 0 だった」が区別できなくなる (`data/features.py` と同じ方針)。呼び出し側は
`np.isfinite` で明示的に落とすこと。
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

DEFAULT_WINDOW: Final[int] = 20
"""既定の窓幅 (ステップ)。

LIBERO の制御は 20 Hz なので 1 秒に相当する。エピソードは 150〜520 ステップあり、
窓が長すぎると落下開始のような局所的な変化を平滑化してしまう。
"""

DEFAULT_LAG: Final[int] = 1
"""自己相関の既定ラグ。"""

_NORM_FLOOR: Final[float] = 1e-12
"""ゼロ除算を避けるノルムの下限。"""


def _validate(states: NDArray[np.float64], window: int, extra: int = 0) -> int:
    """入力を検証して系列長を返す。"""
    if states.ndim != 2:
        raise ValueError(
            f"states は 2 次元 [T, N] が必要です (実 shape: {states.shape})"
        )
    if window < 2:
        raise ValueError(f"window は 2 以上が必要です (実値: {window})")
    if states.shape[1] < 1:
        raise ValueError("states の列数は 1 以上が必要です")
    return int(states.shape[0])


def state_autocorrelation(
    states: NDArray[np.float64],
    *,
    window: int = DEFAULT_WINDOW,
    lag: int = DEFAULT_LAG,
) -> NDArray[np.float64]:
    """窓内の状態ベクトルとその ``lag`` ステップ前との余弦類似度の平均。

    同じ動作を繰り返しているとき、状態は前の時刻と似た向きを取り続けるため
    上がる。値域は ``[-1, 1]``。

    Args:
        states: リザバー状態 `[T, N]`。
        window: 平均を取る窓幅。
        lag: 比較するラグ。

    Returns:
        `[T]`。窓が埋まらない先頭 ``window + lag - 1`` ステップは `NaN`。

    Raises:
        ValueError: 形状・窓幅・ラグが不正な場合。
    """
    n_steps = _validate(states, window)
    if lag < 1:
        raise ValueError(f"lag は 1 以上が必要です (実値: {lag})")

    result = np.full(n_steps, np.nan, dtype=np.float64)
    if n_steps <= lag:
        return result

    current = states[lag:]
    previous = states[:-lag]
    numerator = np.einsum("ij,ij->i", current, previous)
    norms = np.linalg.norm(current, axis=1) * np.linalg.norm(previous, axis=1)
    similarity = numerator / np.maximum(norms, _NORM_FLOOR)

    # similarity[i] は時刻 i + lag の値。窓 [t-window+1, t] の平均を取る。
    cumulative = np.concatenate([[0.0], np.cumsum(similarity)])
    for index in range(window - 1, similarity.shape[0]):
        total = cumulative[index + 1] - cumulative[index + 1 - window]
        result[index + lag] = total / window
    return result


def state_participation_ratio(
    states: NDArray[np.float64], *, window: int = DEFAULT_WINDOW
) -> NDArray[np.float64]:
    """窓内の状態が張る実効次元 (participation ratio)。

    ``PR = (sum_i lambda_i)^2 / sum_i lambda_i^2``。窓内の共分散の固有値
    ``lambda_i`` から定義され、1 方向に潰れていれば 1、等方なら窓幅に近づく。

    固有値分解はしない。共分散 ``C = X^T X / (w-1)`` の非零固有値は Gram 行列
    ``G = X X^T / (w-1)`` (窓幅かける窓幅) と一致するので、``tr(G)`` と
    ``tr(G^2) = ||G||_F^2`` から直接求める。``N=200`` の共分散を毎ステップ
    作ると `O(T N^2)` になるが、この形なら `O(T w^2 N)` で済む。

    Args:
        states: リザバー状態 `[T, N]`。
        window: 窓幅。

    Returns:
        `[T]`。窓が埋まらない先頭 ``window - 1`` ステップは `NaN`。

    Raises:
        ValueError: 形状・窓幅が不正な場合。
    """
    n_steps = _validate(states, window)
    result = np.full(n_steps, np.nan, dtype=np.float64)
    for end in range(window - 1, n_steps):
        block = states[end - window + 1 : end + 1]
        centered = block - block.mean(axis=0, keepdims=True)
        gram = centered @ centered.T
        trace = float(np.trace(gram))
        frobenius = float(np.sum(gram * gram))
        if frobenius <= _NORM_FLOOR:
            # 窓内で状態が全く動いていない (共分散が 0)。実効次元は定義できない
            # ため、潰れの極限である 1.0 を返す。
            result[end] = 1.0
            continue
        result[end] = trace * trace / frobenius
    return result


def state_novelty(
    states: NDArray[np.float64], *, window: int = DEFAULT_WINDOW
) -> NDArray[np.float64]:
    """直近 ``window`` ステップの状態集合からの隔たり。

    ``min_{s in [t-window, t-1]} ||x[t] - x[s]||`` を状態のノルムで割ったもの。
    同じ状態へ戻り続けているとき (= 動作の反復) に下がる。

    Args:
        states: リザバー状態 `[T, N]`。
        window: 参照する過去の長さ。

    Returns:
        `[T]`。先頭 ``window`` ステップは `NaN`。

    Raises:
        ValueError: 形状・窓幅が不正な場合。
    """
    n_steps = _validate(states, window)
    result = np.full(n_steps, np.nan, dtype=np.float64)
    for step in range(window, n_steps):
        past = states[step - window : step]
        distances = np.linalg.norm(past - states[step], axis=1)
        scale = max(float(np.linalg.norm(states[step])), _NORM_FLOOR)
        result[step] = float(distances.min()) / scale
    return result
