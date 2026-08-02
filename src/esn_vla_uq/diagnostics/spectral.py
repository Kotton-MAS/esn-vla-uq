"""スペクトル半径と最大特異値。

定義は `docs/design.md` の「診断指標の定義」節 (4.1) に従う。

- ``spectral_radius(W)``: ``max(|eigvals(W)|)``
- ``effective_spectral_radius(W, a)``: リーク統合を含む実効更新行列
  ``A = (1 - a) I + a W`` のスペクトル半径。``a = 1`` のとき ``A = W`` に退化する。
- ``largest_singular_value(A)``: ``A`` の最大特異値 (ESP の十分条件判定に使う)

疎行列ライブラリは導入せず、密行列に対する `np.linalg.eigvals` / `np.linalg.svd` を
用いる (N の実用上限は `docs/design.md` の 3.8 節を参照)。numpy の戻り値は
`float(...)` で明示変換する。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _as_matrix(matrix: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """2 次元の float64 配列として検証する。"""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"{name} は 2 次元配列である必要があります (実 shape: {array.shape})"
        )
    return array


def _as_square_matrix(matrix: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """正方かつ 2 次元の float64 配列として検証する。"""
    array = _as_matrix(matrix, name)
    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{name} は正方行列である必要があります (実 shape: {array.shape})"
        )
    return array


def spectral_radius(matrix: NDArray[np.float64]) -> float:
    """正方行列のスペクトル半径 ``max(|eigvals|)`` を返す。"""
    square = _as_square_matrix(matrix, "matrix")
    return float(np.max(np.abs(np.linalg.eigvals(square))))


def effective_update_matrix(
    recurrent: NDArray[np.float64], leak_rate: float
) -> NDArray[np.float64]:
    """実効更新行列 ``A = (1 - a) I + a W`` を組み立てる。

    Args:
        recurrent: 再帰行列 ``W`` (`[N, N]`)。
        leak_rate: リーク率 ``a`` (``0 < a <= 1``)。

    Returns:
        実効更新行列 ``A`` (`[N, N]`)。

    Raises:
        ValueError: ``W`` が正方でない、または ``a`` が範囲外の場合。
    """
    w = _as_square_matrix(recurrent, "recurrent")
    if not 0.0 < leak_rate <= 1.0:
        raise ValueError(
            "leak_rate は 0 < leak_rate <= 1 の範囲である必要があります "
            f"(実値: {leak_rate})"
        )
    identity = np.eye(w.shape[0], dtype=np.float64)
    return (1.0 - leak_rate) * identity + leak_rate * w


def effective_spectral_radius(
    recurrent: NDArray[np.float64], leak_rate: float
) -> float:
    """実効更新行列 ``A = (1 - a) I + a W`` のスペクトル半径を返す。"""
    return spectral_radius(effective_update_matrix(recurrent, leak_rate))


def largest_singular_value(matrix: NDArray[np.float64]) -> float:
    """行列の最大特異値 ``sigma_max`` を返す。

    任意の正方行列で ``rho(A) <= sigma_max(A)`` が成り立つため、ESP の十分条件
    (``sigma_max(A) < 1``) は必要条件 (``rho(A) < 1``) より強い。
    """
    array = _as_matrix(matrix, "matrix")
    return float(np.max(np.linalg.svd(array, compute_uv=False)))
