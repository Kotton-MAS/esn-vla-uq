"""スペクトル半径と最大特異値。

定義は `docs/design.md` の「診断指標の定義」節 (4.1) に従う。

- ``spectral_radius(W)``: ``max(|eigvals(W)|)``
- ``effective_spectral_radius(W, a)``: リーク統合を含む実効更新行列
  ``A = (1 - a) I + a W`` のスペクトル半径。``a = 1`` のとき ``A = W`` に退化する。
- ``largest_singular_value(A)``: ``A`` の最大特異値 (ESP の十分条件判定に使う)

`spectral_radius` / `largest_singular_value` の実体は最下層の
`esn_vla_uq.linalg` にあり、ここでは診断 API として再エクスポートする。
`esn/reservoir.py` が `W` をスケールする際に使うのも同じ関数であり、
「`W` は目標 rho にスケール済み」を実測で検証するという診断の意味は、両者が
同一実装を共有していることに依存する (A5、`esn_vla_uq/linalg.py` の docstring)。

疎行列ライブラリは導入せず、密行列に対する `np.linalg.eigvals` / `np.linalg.svd` を
用いる (N の実用上限は `docs/design.md` の 3.8 節を参照)。numpy の戻り値は
`float(...)` で明示変換する。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.linalg import (
    as_square_matrix,
    largest_singular_value,
    spectral_radius,
)

__all__ = [
    "effective_spectral_radius",
    "effective_update_matrix",
    "largest_singular_value",
    "spectral_radius",
]


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
    w = as_square_matrix(recurrent, "recurrent")
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
