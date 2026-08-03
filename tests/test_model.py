"""`esn_vla_uq.esn.model` のテスト (Sprint 1 T3)。"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.esn import ESN, ESNConfig

DELAY = 5
N_TRAIN = 1500
N_TEST = 500
NRMSE_THRESHOLD = 0.15

# 線形な記憶課題では tanh の飽和を避けるため入力スケールを小さく取る
# (Lukosevicius 2012 の実務ガイド)。既定値は ESNConfig 側の 1.0 のまま変えない。
DELAY_TASK_CONFIG = ESNConfig(
    n_reservoir=100,
    spectral_radius=0.9,
    input_scaling=0.5,
    leak_rate=1.0,
    density=0.1,
    ridge_alpha=1e-6,
    washout=100,
    seed=0,
)


def _nrmse(predictions: NDArray[np.float64], targets: NDArray[np.float64]) -> float:
    """NRMSE = RMSE / std(target)。"""
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    return rmse / float(np.std(targets))


def _delay_task(
    n_steps: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`y[t] = u[t-DELAY]` の遅延再現課題を生成する (先頭 DELAY ステップは 0)。"""
    inputs = rng.uniform(-0.8, 0.8, size=(n_steps, 1))
    targets = np.concatenate([np.zeros((DELAY, 1)), inputs[:-DELAY]], axis=0)
    return inputs, targets


@pytest.fixture
def config() -> ESNConfig:
    return ESNConfig(n_reservoir=30, density=0.2, washout=10, seed=0)


@pytest.fixture
def inputs() -> NDArray[np.float64]:
    rng = np.random.default_rng(21)
    return rng.uniform(-0.8, 0.8, size=(120, 2))


@pytest.fixture
def targets(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([np.zeros((2, 2)), inputs[:-2]], axis=0)


def test_transform_returns_state_matrix(
    config: ESNConfig, inputs: NDArray[np.float64]
) -> None:
    states = ESN(config).transform(inputs)
    assert states.shape == (inputs.shape[0], config.n_reservoir)


def test_transform_matches_reservoir_run(
    config: ESNConfig, inputs: NDArray[np.float64]
) -> None:
    model = ESN(config)
    states = model.transform(inputs)
    assert np.array_equal(states, model.reservoir.run(inputs))


def test_fit_returns_self_and_marks_fitted(
    config: ESNConfig, inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    model = ESN(config)
    assert model.fit(inputs, targets) is model
    assert model.is_fitted


def test_predict_shape_matches_inputs(
    config: ESNConfig, inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    predictions = ESN(config).fit(inputs, targets).predict(inputs)
    assert predictions.shape == targets.shape


def test_one_dimensional_targets_yield_one_dimensional_predictions(
    config: ESNConfig, inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    predictions = ESN(config).fit(inputs, targets[:, 0]).predict(inputs)
    assert predictions.shape == (inputs.shape[0],)


def test_predict_before_fit_raises_runtime_error(
    config: ESNConfig, inputs: NDArray[np.float64]
) -> None:
    with pytest.raises(RuntimeError, match="未 fit"):
        ESN(config).predict(inputs)


def test_reservoir_before_use_raises_runtime_error(config: ESNConfig) -> None:
    with pytest.raises(RuntimeError, match="未構築"):
        _ = ESN(config).reservoir


def test_same_seed_reproduces_identical_predictions(
    config: ESNConfig, inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    first = ESN(config).fit(inputs, targets).predict(inputs)
    second = ESN(config).fit(inputs, targets).predict(inputs)
    assert np.array_equal(first, second)


def test_different_seed_changes_predictions(
    config: ESNConfig, inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    base = ESN(config).fit(inputs, targets).predict(inputs)
    other_config = dataclasses.replace(config, seed=config.seed + 1)
    other = ESN(other_config).fit(inputs, targets).predict(inputs)
    assert not np.array_equal(base, other)


def test_reservoir_is_independent_of_call_order(
    config: ESNConfig, inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    transform_first = ESN(config)
    transform_first.transform(inputs)
    transform_first.fit(inputs, targets)

    fit_first = ESN(config)
    fit_first.fit(inputs, targets)

    assert np.array_equal(fit_first.reservoir.W, transform_first.reservoir.W)
    assert np.array_equal(fit_first.predict(inputs), transform_first.predict(inputs))


def test_delay_task_reaches_target_nrmse() -> None:
    # 受け入れ基準: y[t] = u[t-5]、N=100 で test NRMSE < 0.15。
    rng = np.random.default_rng(123)
    train_inputs, train_targets = _delay_task(N_TRAIN, rng)
    test_inputs, test_targets = _delay_task(N_TEST, rng)

    model = ESN(DELAY_TASK_CONFIG).fit(train_inputs, train_targets)
    predictions = model.predict(test_inputs)

    washout = DELAY_TASK_CONFIG.washout
    assert _nrmse(predictions[washout:], test_targets[washout:]) < NRMSE_THRESHOLD


def test_delay_task_without_passthrough_also_learns() -> None:
    config = dataclasses.replace(DELAY_TASK_CONFIG, input_passthrough=False)
    rng = np.random.default_rng(123)
    train_inputs, train_targets = _delay_task(N_TRAIN, rng)
    test_inputs, test_targets = _delay_task(N_TEST, rng)

    model = ESN(config).fit(train_inputs, train_targets)
    predictions = model.predict(test_inputs)
    washout = config.washout
    assert _nrmse(predictions[washout:], test_targets[washout:]) < NRMSE_THRESHOLD


def test_input_passthrough_flag_is_forwarded_to_readout() -> None:
    assert ESN(ESNConfig()).readout.input_passthrough is True
    assert ESN(ESNConfig(input_passthrough=False)).readout.input_passthrough is False


def test_fit_rejects_washout_longer_than_sequence(
    inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    config = ESNConfig(n_reservoir=10, washout=inputs.shape[0])
    with pytest.raises(ValueError, match="washout"):
        ESN(config).fit(inputs, targets)


def test_fit_rejects_mismatched_lengths(
    config: ESNConfig, inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError, match="系列長"):
        ESN(config).fit(inputs, targets[:-1])


def test_fit_rejects_non_2d_inputs(
    config: ESNConfig, targets: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError, match="2 次元"):
        ESN(config).fit(np.zeros(targets.shape[0]), targets)


def test_fit_rejects_3d_targets(config: ESNConfig, inputs: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="targets"):
        ESN(config).fit(inputs, np.zeros((inputs.shape[0], 2, 2)))


def test_predict_rejects_changed_input_dimension(
    config: ESNConfig, inputs: NDArray[np.float64], targets: NDArray[np.float64]
) -> None:
    model = ESN(config).fit(inputs, targets)
    with pytest.raises(ValueError, match="入力次元"):
        model.predict(np.zeros((inputs.shape[0], inputs.shape[1] + 1)))
