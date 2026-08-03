"""`esn_vla_uq.linalg` のテストと、実装が一本化されていることの検証 (A5)。"""

from __future__ import annotations

import numpy as np
import pytest

from esn_vla_uq import linalg
from esn_vla_uq.diagnostics import spectral as diagnostics_spectral
from esn_vla_uq.esn import Reservoir
from esn_vla_uq.esn import reservoir as esn_reservoir
from esn_vla_uq.esn.config import ESNConfig

N_RESERVOIR = 40


def test_spectral_radius_of_diagonal_matrix() -> None:
    matrix = np.diag([0.2, -0.7, 0.5])
    assert linalg.spectral_radius(matrix) == pytest.approx(0.7)


def test_spectral_radius_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="正方行列"):
        linalg.spectral_radius(np.zeros((2, 3)))


def test_spectral_radius_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2 次元"):
        linalg.spectral_radius(np.zeros(3))


def test_spectral_radius_error_message_uses_given_name() -> None:
    """`name` 引数がエラーメッセージに現れる (呼び出し側ごとの文脈を残すため)。"""
    with pytest.raises(ValueError, match="W は正方行列"):
        linalg.spectral_radius(np.zeros((2, 3)), "W")


def test_largest_singular_value_of_diagonal_matrix() -> None:
    matrix = np.diag([0.2, -0.7, 0.5])
    assert linalg.largest_singular_value(matrix) == pytest.approx(0.7)


def test_largest_singular_value_accepts_non_square() -> None:
    """最大特異値は正方行列でなくても定義される。"""
    assert linalg.largest_singular_value(np.eye(2, 3)) == pytest.approx(1.0)


def test_largest_singular_value_bounds_spectral_radius() -> None:
    """任意の正方行列で ``rho(A) <= sigma_max(A)``。"""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(12, 12))
    radius = linalg.spectral_radius(matrix)
    assert radius <= linalg.largest_singular_value(matrix) + 1e-9


def test_reservoir_and_diagnostics_share_one_implementation() -> None:
    """`W` のスケーリングと診断値が**同一の関数**を呼ぶこと (A5)。

    この 2 つが別実装に分かれると、片方だけを反復法などへ差し替えた瞬間に
    `test_measured_spectral_radius_matches_config` が「別々のアルゴリズムの
    出力を比べる」テストに変質し、検証としての意味を失う。同一性そのものを
    固定する。
    """
    # `mypy --strict` の `no_implicit_reexport` により、import しただけの名前への
    # 属性アクセスは禁じられる。同一性そのものが検証対象なので `__dict__` で取る。
    assert esn_reservoir.__dict__["spectral_radius"] is linalg.spectral_radius
    assert diagnostics_spectral.__dict__["spectral_radius"] is linalg.spectral_radius
    assert (
        diagnostics_spectral.__dict__["largest_singular_value"]
        is linalg.largest_singular_value
    )


def test_measured_spectral_radius_matches_config() -> None:
    """設定した rho が実際に達成されていることを実測で確認する。"""
    config = ESNConfig(n_reservoir=N_RESERVOIR, spectral_radius=0.75, seed=3)
    reservoir = Reservoir(config, n_inputs=2)
    assert linalg.spectral_radius(reservoir.W) == pytest.approx(0.75)
