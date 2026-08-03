"""行列の形状検証とスペクトル量。

パッケージ内の**最下層**。依存は numpy のみで、`esn` / `diagnostics` / `data` の
どのモジュールも import しない (逆にこれらから import される)。

以前は同じ計算が 2 箇所に存在していた。`esn/reservoir.py` は `W` を目標スペクトル
半径へスケールするために `_max_abs_eigenvalue` を持ち、`diagnostics/spectral.py`
は診断値として `spectral_radius` を公開していた。両者は独立した実装だったため、
片方だけを N の大きい領域向けに反復法へ差し替えると、「`W` は目標 rho に
スケール済みである」ことを実測で検証しているはずの診断が、別のアルゴリズムの出力同士を
比べる無意味な検査に変わってしまう。この 2 者が**同じ関数を呼ぶ**ことがその検証
の前提なので、ここへ寄せる (A5)。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def as_matrix(matrix: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """2 次元の float64 配列として検証する。

    Args:
        matrix: 検証対象。
        name: エラーメッセージに含める引数名。

    Returns:
        float64 に変換した 2 次元配列。

    Raises:
        ValueError: 2 次元でない場合。
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"{name} は 2 次元配列である必要があります (実 shape: {array.shape})"
        )
    return array


def as_square_matrix(matrix: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """正方かつ 2 次元の float64 配列として検証する。

    Args:
        matrix: 検証対象。
        name: エラーメッセージに含める引数名。

    Returns:
        float64 に変換した正方行列。

    Raises:
        ValueError: 2 次元でない、または正方でない場合。
    """
    array = as_matrix(matrix, name)
    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{name} は正方行列である必要があります (実 shape: {array.shape})"
        )
    return array


def spectral_radius(matrix: NDArray[np.float64], name: str = "matrix") -> float:
    """正方行列のスペクトル半径 ``max(|eigvals|)`` を返す。

    Args:
        matrix: 正方行列。
        name: エラーメッセージに含める引数名。

    Returns:
        スペクトル半径。

    Raises:
        ValueError: `matrix` が正方でない場合。
    """
    square = as_square_matrix(matrix, name)
    return float(np.max(np.abs(np.linalg.eigvals(square))))


def largest_singular_value(matrix: NDArray[np.float64], name: str = "matrix") -> float:
    """行列の最大特異値 ``sigma_max`` を返す。

    任意の正方行列で ``rho(A) <= sigma_max(A)`` が成り立つため、ESP の十分条件
    (``sigma_max(A) < 1``) は必要条件 (``rho(A) < 1``) より強い。

    Args:
        matrix: 2 次元配列 (正方でなくてよい)。
        name: エラーメッセージに含める引数名。

    Returns:
        最大特異値。

    Raises:
        ValueError: `matrix` が 2 次元でない場合。
    """
    array = as_matrix(matrix, name)
    return float(np.max(np.linalg.svd(array, compute_uv=False)))
