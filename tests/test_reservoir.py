"""`esn_vla_uq.esn.reservoir` のテスト (Sprint 1 T3)。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.esn import ESNConfig, Reservoir, discard_washout
from esn_vla_uq.esn.reservoir import (
    _make_bias,
    _make_input_matrix,
    _make_recurrent_matrix,
)

N_INPUTS = 2
SPECTRAL_RADIUS_RTOL = 1e-8


def _measured_spectral_radius(matrix: NDArray[np.float64]) -> float:
    """テスト側で独立にスペクトル半径を実測する。"""
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


@pytest.fixture
def config() -> ESNConfig:
    # bias_scaling は既定 0.0 (design.md §3.2)。既定のままだと b が恒等的に
    # 零ベクトルになりシード差分の検証が空振りするため、ここでは明示的に有効化する。
    return ESNConfig(
        n_reservoir=100,
        density=0.1,
        spectral_radius=0.9,
        bias_scaling=0.1,
        seed=0,
    )


@pytest.fixture
def reservoir(config: ESNConfig) -> Reservoir:
    return Reservoir(config, N_INPUTS)


@pytest.fixture
def inputs() -> NDArray[np.float64]:
    rng = np.random.default_rng(20240802)
    return rng.uniform(-0.8, 0.8, size=(50, N_INPUTS))


def test_matrix_shapes(reservoir: Reservoir, config: ESNConfig) -> None:
    assert reservoir.W_in.shape == (config.n_reservoir, N_INPUTS)
    assert reservoir.b.shape == (config.n_reservoir,)
    assert reservoir.W.shape == (config.n_reservoir, config.n_reservoir)
    assert reservoir.n_reservoir == config.n_reservoir


def test_measured_spectral_radius_matches_configuration(
    reservoir: Reservoir, config: ESNConfig
) -> None:
    # 受け入れ基準: N=100, density=0.1, rho=0.9 で相対誤差 < 1e-8。
    measured = _measured_spectral_radius(reservoir.W)
    relative_error = abs(measured - config.spectral_radius) / config.spectral_radius
    assert relative_error < SPECTRAL_RADIUS_RTOL


@pytest.mark.parametrize("spectral_radius", [0.1, 0.5, 1.2])
def test_spectral_radius_scaling_for_various_targets(spectral_radius: float) -> None:
    config = ESNConfig(
        n_reservoir=60, density=0.2, spectral_radius=spectral_radius, seed=1
    )
    measured = _measured_spectral_radius(Reservoir(config, 1).W)
    assert abs(measured - spectral_radius) / spectral_radius < SPECTRAL_RADIUS_RTOL


def test_input_and_bias_scaling_bound_the_entries() -> None:
    config = ESNConfig(n_reservoir=80, input_scaling=0.5, bias_scaling=0.25, seed=2)
    built = Reservoir(config, N_INPUTS)
    assert float(np.max(np.abs(built.W_in))) <= config.input_scaling
    assert float(np.max(np.abs(built.b))) <= config.bias_scaling


def test_density_controls_the_fraction_of_nonzero_entries() -> None:
    config = ESNConfig(n_reservoir=200, density=0.1, seed=3)
    built = Reservoir(config, 1)
    fraction = float(np.count_nonzero(built.W)) / float(built.W.size)
    assert abs(fraction - config.density) < 0.02


def test_same_seed_reproduces_identical_matrices(config: ESNConfig) -> None:
    first = Reservoir(config, N_INPUTS)
    second = Reservoir(config, N_INPUTS)
    assert np.array_equal(first.W_in, second.W_in)
    assert np.array_equal(first.b, second.b)
    assert np.array_equal(first.W, second.W)


def test_default_bias_scaling_yields_zero_bias() -> None:
    """既定 `bias_scaling = 0.0` (design.md §3.2) ではバイアス項が無効になる。"""
    reservoir = Reservoir(ESNConfig(n_reservoir=20), N_INPUTS)
    assert np.array_equal(reservoir.b, np.zeros(20))


def test_different_seed_changes_matrices(config: ESNConfig) -> None:
    other_config = dataclasses.replace(config, seed=config.seed + 1)
    other = Reservoir(other_config, N_INPUTS)
    base = Reservoir(config, N_INPUTS)
    assert not np.array_equal(base.W_in, other.W_in)
    assert not np.array_equal(base.b, other.b)
    assert not np.array_equal(base.W, other.W)


def test_same_seed_reproduces_identical_state_sequence(
    config: ESNConfig, inputs: NDArray[np.float64]
) -> None:
    first = Reservoir(config, N_INPUTS).run(inputs)
    second = Reservoir(config, N_INPUTS).run(inputs)
    assert np.array_equal(first, second)


def test_different_seed_changes_state_sequence(
    config: ESNConfig, inputs: NDArray[np.float64]
) -> None:
    base = Reservoir(config, N_INPUTS).run(inputs)
    other_config = dataclasses.replace(config, seed=config.seed + 1)
    other = Reservoir(other_config, N_INPUTS).run(inputs)
    assert not np.array_equal(base, other)


def test_run_returns_state_matrix_without_discarding_washout(
    reservoir: Reservoir, inputs: NDArray[np.float64], config: ESNConfig
) -> None:
    states = reservoir.run(inputs)
    assert states.shape == (inputs.shape[0], config.n_reservoir)
    assert states.dtype == np.float64
    assert np.all(np.isfinite(states))


def test_initial_state_is_zero_vector(reservoir: Reservoir, config: ESNConfig) -> None:
    assert np.array_equal(
        reservoir.initial_state(), np.zeros(config.n_reservoir, dtype=np.float64)
    )


def test_first_step_matches_the_update_equation_from_zero_state(
    reservoir: Reservoir, inputs: NDArray[np.float64], config: ESNConfig
) -> None:
    states = reservoir.run(inputs)
    zero_state = np.zeros(config.n_reservoir, dtype=np.float64)
    expected = (1.0 - config.leak_rate) * zero_state + config.leak_rate * np.tanh(
        reservoir.W_in @ inputs[0] + reservoir.W @ zero_state + reservoir.b
    )
    assert np.array_equal(states[0], expected)


def test_leak_rate_one_degenerates_to_non_leaky_update(
    inputs: NDArray[np.float64],
) -> None:
    # 受け入れ基準: leak_rate=1.0 で非リーク型 x[t] = tanh(...) と atol=0 で一致。
    config = ESNConfig(n_reservoir=40, density=0.2, leak_rate=1.0, seed=5)
    built = Reservoir(config, N_INPUTS)
    states = built.run(inputs)

    reference = np.empty_like(states)
    x = np.zeros(config.n_reservoir, dtype=np.float64)
    for t in range(inputs.shape[0]):
        x = np.tanh(built.W_in @ inputs[t] + built.W @ x + built.b)
        reference[t] = x
    assert np.allclose(states, reference, atol=0.0)


def test_leaky_update_matches_manual_recursion(inputs: NDArray[np.float64]) -> None:
    config = ESNConfig(n_reservoir=30, density=0.3, leak_rate=0.4, seed=6)
    built = Reservoir(config, N_INPUTS)
    states = built.run(inputs)

    reference = np.empty_like(states)
    x = np.zeros(config.n_reservoir, dtype=np.float64)
    for t in range(inputs.shape[0]):
        x = (1.0 - config.leak_rate) * x + config.leak_rate * np.tanh(
            built.W_in @ inputs[t] + built.W @ x + built.b
        )
        reference[t] = x
    assert np.allclose(states, reference, atol=0.0)


def test_custom_initial_state_is_used_and_not_mutated(
    reservoir: Reservoir, inputs: NDArray[np.float64], config: ESNConfig
) -> None:
    initial = np.full(config.n_reservoir, 0.3, dtype=np.float64)
    original = initial.copy()
    states = reservoir.run(inputs, initial_state=initial)
    assert np.array_equal(initial, original)
    assert not np.array_equal(states, reservoir.run(inputs))


def test_zero_spectral_radius_raises_value_error() -> None:
    # n_reservoir=2 / density=0.2 / seed=0 では非零要素が冪零配置になり rho(W)=0。
    with pytest.raises(ValueError, match="スペクトル半径が 0"):
        Reservoir(ESNConfig(n_reservoir=2, density=0.2, seed=0), 1)


def test_invalid_n_inputs_raises_value_error(config: ESNConfig) -> None:
    with pytest.raises(ValueError, match="n_inputs"):
        Reservoir(config, 0)


def test_run_rejects_wrong_input_dimension(reservoir: Reservoir) -> None:
    with pytest.raises(ValueError, match="入力次元"):
        reservoir.run(np.zeros((10, N_INPUTS + 1)))


def test_run_rejects_non_2d_inputs(reservoir: Reservoir) -> None:
    with pytest.raises(ValueError, match="2 次元"):
        reservoir.run(np.zeros(10))


def test_run_rejects_empty_inputs(reservoir: Reservoir) -> None:
    with pytest.raises(ValueError, match="系列長"):
        reservoir.run(np.zeros((0, N_INPUTS)))


def test_run_rejects_initial_state_with_wrong_shape(
    reservoir: Reservoir, inputs: NDArray[np.float64], config: ESNConfig
) -> None:
    with pytest.raises(ValueError, match="initial_state"):
        reservoir.run(inputs, initial_state=np.zeros(config.n_reservoir + 1))


def test_discard_washout_drops_leading_steps() -> None:
    states = np.arange(20, dtype=np.float64).reshape(10, 2)
    trimmed = discard_washout(states, 3)
    assert trimmed.shape == (7, 2)
    assert np.array_equal(trimmed, states[3:])


def test_discard_washout_with_zero_is_identity() -> None:
    states = np.arange(20, dtype=np.float64).reshape(10, 2)
    assert np.array_equal(discard_washout(states, 0), states)


@pytest.mark.parametrize("washout", [10, 11])
def test_discard_washout_rejects_too_large_washout(washout: int) -> None:
    states = np.zeros((10, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="washout"):
        discard_washout(states, washout)


def test_discard_washout_rejects_negative_washout() -> None:
    with pytest.raises(ValueError, match="washout"):
        discard_washout(np.zeros((10, 2), dtype=np.float64), -1)


def test_discard_washout_rejects_non_2d_states() -> None:
    with pytest.raises(ValueError, match="2 次元"):
        discard_washout(np.zeros(10, dtype=np.float64), 1)


# ゴールデン値: `np.random.default_rng(42)` から `_make_input_matrix` ->
# `_make_bias` -> `_make_recurrent_matrix` の順で 1 度だけ呼んで得た値を、実装を
# 信頼できる基準として固定したもの (n_reservoir=4, n_inputs=2, density=0.5,
# input_scaling=0.7, bias_scaling=0.3, spectral_radius=0.9, seed=42)。
#
# `_make_recurrent_matrix` は `mask = rng.random(size) < density` を先に、
# `raw = rng.uniform(-1, 1, size)` を後に呼ぶ。docs/design.md §3.3 は逆順
# (uniform -> mask) を規範として書いているが、ユーザー判断により「実装を正とし
# design.md を改訂する」で確定した (このタスクでは実装を変更しない)。本テストは
# 生成順序 (W_in -> b -> W、W 内部では mask -> uniform) の以後の回帰を、既知
# seed に対する rng 消費順序の結果を突き合わせることで機械的に検知する。
_GOLDEN_N_RESERVOIR = 4
_GOLDEN_N_INPUTS = 2
_GOLDEN_SEED = 42

_GOLDEN_W_IN = np.array(
    [
        [0.3835384679783487, -0.08557018434712675],
        [0.5020370878759354, 0.2763152406831094],
        [-0.5681517129572906, 0.6658712922914582],
        [0.3655955827864941, 0.4004900273877353],
    ],
    dtype=np.float64,
)
_GOLDEN_B = np.array(
    [
        -0.22313182039467247,
        -0.02976843726265972,
        -0.07752118546045125,
        0.2560589933091611,
    ],
    dtype=np.float64,
)
_GOLDEN_W = np.array(
    [
        [0.0, 0.0, 1.1740233290049482, 2.242451780909053],
        [0.0, -0.6213514773031367, 0.0, 0.0],
        [0.0, -0.11653347015538, 0.0, 0.0],
        [0.0, 1.5957203913920468, 0.9605892743678311, -0.9],
    ],
    dtype=np.float64,
)


@pytest.fixture
def golden_config() -> ESNConfig:
    return ESNConfig(
        n_reservoir=_GOLDEN_N_RESERVOIR,
        density=0.5,
        input_scaling=0.7,
        bias_scaling=0.3,
        spectral_radius=0.9,
        seed=_GOLDEN_SEED,
    )


def test_rng_consumption_order_matches_golden_values(golden_config: ESNConfig) -> None:
    rng = np.random.default_rng(golden_config.seed)
    w_in = _make_input_matrix(rng, golden_config, _GOLDEN_N_INPUTS)
    b = _make_bias(rng, golden_config)
    w = _make_recurrent_matrix(rng, golden_config)

    assert np.array_equal(w_in, _GOLDEN_W_IN)
    assert np.array_equal(b, _GOLDEN_B)
    assert np.array_equal(w, _GOLDEN_W)


def test_reservoir_matches_golden_values_end_to_end(golden_config: ESNConfig) -> None:
    # 公開 API (`Reservoir`) 経由でも同じ順序・同じ値になることを確認する。
    reservoir = Reservoir(golden_config, _GOLDEN_N_INPUTS)
    assert np.array_equal(reservoir.W_in, _GOLDEN_W_IN)
    assert np.array_equal(reservoir.b, _GOLDEN_B)
    assert np.array_equal(reservoir.W, _GOLDEN_W)


def test_zero_input_scaling_makes_the_reservoir_ignore_its_input() -> None:
    """`input_scaling=0.0` は入力を無視するリザバーになる (T1)。

    `docs/design.md` 3.2 節がこの挙動を保証しているのにテストが無かった。
    `W_in` が恒等的に 0 になるため、どんな入力でも同じ状態列が出る。
    """
    config = ESNConfig(n_reservoir=20, input_scaling=0.0, bias_scaling=0.1, seed=0)
    reservoir = Reservoir(config, n_inputs=3)
    assert not reservoir.W_in.any()

    rng = np.random.default_rng(0)
    first = reservoir.run(rng.normal(size=(15, 3)))
    second = reservoir.run(rng.normal(size=(15, 3)) * 100.0)
    np.testing.assert_allclose(first, second)
