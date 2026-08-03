"""`esn_vla_uq.esn.config` のテスト (Sprint 1 T3)。"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest

from esn_vla_uq.esn import ESNConfig

# (違反フィールド名, 実値の文字列表現, 生成関数) の異常系ケース。
INVALID_CASES: list[tuple[str, str, Callable[[], ESNConfig]]] = [
    ("n_reservoir", "0", lambda: ESNConfig(n_reservoir=0)),
    ("n_reservoir", "-5", lambda: ESNConfig(n_reservoir=-5)),
    ("spectral_radius", "0.0", lambda: ESNConfig(spectral_radius=0.0)),
    ("spectral_radius", "-0.5", lambda: ESNConfig(spectral_radius=-0.5)),
    ("leak_rate", "0.0", lambda: ESNConfig(leak_rate=0.0)),
    ("leak_rate", "1.5", lambda: ESNConfig(leak_rate=1.5)),
    ("density", "0.0", lambda: ESNConfig(density=0.0)),
    ("density", "1.5", lambda: ESNConfig(density=1.5)),
    ("ridge_alpha", "-1e-09", lambda: ESNConfig(ridge_alpha=-1e-9)),
    ("washout", "-1", lambda: ESNConfig(washout=-1)),
]


def test_defaults_are_within_valid_ranges() -> None:
    config = ESNConfig()
    assert config.n_reservoir >= 1
    assert config.spectral_radius > 0.0
    assert 0.0 < config.leak_rate <= 1.0
    assert 0.0 < config.density <= 1.0
    assert config.ridge_alpha >= 0.0
    assert config.washout >= 0
    # 入力パススルーは既定で有効 (仕様 §3-10)。
    assert config.input_passthrough is True


def test_config_is_frozen() -> None:
    config = ESNConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, "n_reservoir", 200)  # noqa: B010  # frozen 検証のため setattr


def test_config_equality_is_value_based() -> None:
    assert ESNConfig(seed=3) == ESNConfig(seed=3)
    assert ESNConfig(seed=3) != ESNConfig(seed=4)


@pytest.mark.parametrize(("field_name", "value_repr", "factory"), INVALID_CASES)
def test_invalid_parameters_raise_value_error_with_name_and_value(
    field_name: str, value_repr: str, factory: Callable[[], ESNConfig]
) -> None:
    with pytest.raises(ValueError) as excinfo:
        factory()
    message = str(excinfo.value)
    assert field_name in message
    assert value_repr in message


def test_invalid_cases_cover_at_least_five_parameters() -> None:
    # 受け入れ基準「不正パラメータ 5 種以上が ValueError」。
    assert len({field_name for field_name, _, _ in INVALID_CASES}) >= 5


@pytest.mark.parametrize("leak_rate", [1e-6, 0.5, 1.0])
def test_leak_rate_boundaries_are_accepted(leak_rate: float) -> None:
    assert ESNConfig(leak_rate=leak_rate).leak_rate == leak_rate


@pytest.mark.parametrize("density", [1e-6, 0.5, 1.0])
def test_density_boundaries_are_accepted(density: float) -> None:
    assert ESNConfig(density=density).density == density


def test_zero_ridge_alpha_is_accepted() -> None:
    assert ESNConfig(ridge_alpha=0.0).ridge_alpha == 0.0
