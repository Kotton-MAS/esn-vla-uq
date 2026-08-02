"""`esn_vla_uq.diagnostics.spectral` のテスト (Sprint 1 T4)。"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.diagnostics import (
    effective_spectral_radius,
    effective_update_matrix,
    largest_singular_value,
    spectral_radius,
)
from esn_vla_uq.esn import ESNConfig, Reservoir

RTOL = 1e-12


@pytest.fixture
def diagonal() -> NDArray[np.float64]:
    """既知のスペクトル半径 0.7 を持つ対角行列。"""
    return np.diag([0.3, -0.7])


@pytest.fixture
def reservoir() -> Reservoir:
    return Reservoir(ESNConfig(n_reservoir=60, spectral_radius=0.9, seed=0), 1)


def test_spectral_radius_matches_known_value(diagonal: NDArray[np.float64]) -> None:
    # 受け入れ基準: diag(0.3, -0.7) -> 0.7。
    assert spectral_radius(diagonal) == pytest.approx(0.7, rel=RTOL)


def test_spectral_radius_of_reservoir_matches_configuration(
    reservoir: Reservoir,
) -> None:
    measured = spectral_radius(reservoir.W)
    assert measured == pytest.approx(reservoir.config.spectral_radius, rel=1e-8)


def test_effective_update_matrix_degenerates_at_unit_leak_rate(
    diagonal: NDArray[np.float64],
) -> None:
    assert np.array_equal(effective_update_matrix(diagonal, 1.0), diagonal)


def test_effective_spectral_radius_with_leak_rate(
    diagonal: NDArray[np.float64],
) -> None:
    # A = 0.5 I + 0.5 diag(0.3, -0.7) = diag(0.65, 0.15) -> rho = 0.65
    assert effective_spectral_radius(diagonal, 0.5) == pytest.approx(0.65, rel=RTOL)


def test_effective_spectral_radius_equals_spectral_radius_at_unit_leak_rate(
    reservoir: Reservoir,
) -> None:
    assert effective_spectral_radius(reservoir.W, 1.0) == pytest.approx(
        spectral_radius(reservoir.W), rel=RTOL
    )


def test_largest_singular_value_of_normal_matrix(
    diagonal: NDArray[np.float64],
) -> None:
    assert largest_singular_value(diagonal) == pytest.approx(0.7, rel=RTOL)


def test_largest_singular_value_exceeds_spectral_radius_for_nilpotent_matrix() -> None:
    # 非正規行列では sigma_max >> rho になりうる (ESP 十分条件が保守的である理由)。
    nilpotent = np.array([[0.0, 2.0], [0.0, 0.0]])
    assert spectral_radius(nilpotent) == pytest.approx(0.0, abs=1e-12)
    assert largest_singular_value(nilpotent) == pytest.approx(2.0, rel=RTOL)


def test_largest_singular_value_bounds_spectral_radius(reservoir: Reservoir) -> None:
    # 任意の正方行列で rho(A) <= sigma_max(A)。
    assert spectral_radius(reservoir.W) <= largest_singular_value(reservoir.W) + 1e-12


@pytest.mark.parametrize(
    "matrix",
    [np.zeros((2, 3)), np.zeros(3)],
    ids=["non_square", "one_dimensional"],
)
def test_spectral_radius_rejects_non_square_matrix(
    matrix: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="matrix"):
        spectral_radius(matrix)


def test_largest_singular_value_rejects_one_dimensional_array() -> None:
    with pytest.raises(ValueError, match="2 次元"):
        largest_singular_value(np.zeros(3))


@pytest.mark.parametrize("leak_rate", [0.0, -0.1, 1.5])
def test_effective_update_matrix_rejects_out_of_range_leak_rate(
    diagonal: NDArray[np.float64], leak_rate: float
) -> None:
    with pytest.raises(ValueError, match="leak_rate"):
        effective_update_matrix(diagonal, leak_rate)
