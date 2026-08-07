"""`esn_vla_uq.esn.readout` のテスト (Sprint 1 T3)。"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.esn import ESNConfig, Reservoir, RidgeReadout

N_STEPS = 400
N_INPUTS = 2
N_RESERVOIR = 20
N_OUTPUTS = 3
# 単調性は理論上厳密だが、浮動小数点誤差の分だけ緩めて判定する。
MONOTONICITY_TOLERANCE = 1e-9


@pytest.fixture
def states() -> NDArray[np.float64]:
    rng = np.random.default_rng(11)
    return rng.normal(size=(N_STEPS, N_RESERVOIR))


@pytest.fixture
def inputs() -> NDArray[np.float64]:
    rng = np.random.default_rng(12)
    return rng.normal(size=(N_STEPS, N_INPUTS))


@pytest.fixture
def targets() -> NDArray[np.float64]:
    rng = np.random.default_rng(13)
    return rng.normal(size=(N_STEPS, N_OUTPUTS))


def _frobenius_norm(matrix: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(matrix, ord="fro"))


def test_design_matrix_with_passthrough_is_ones_inputs_states(
    states: NDArray[np.float64], inputs: NDArray[np.float64]
) -> None:
    design = RidgeReadout(1e-6).design_matrix(states, inputs)
    assert design.shape == (N_STEPS, 1 + N_INPUTS + N_RESERVOIR)
    assert np.array_equal(design[:, 0], np.ones(N_STEPS))
    assert np.array_equal(design[:, 1 : 1 + N_INPUTS], inputs)
    assert np.array_equal(design[:, 1 + N_INPUTS :], states)


def test_design_matrix_without_passthrough_omits_inputs(
    states: NDArray[np.float64], inputs: NDArray[np.float64]
) -> None:
    readout = RidgeReadout(1e-6, input_passthrough=False)
    design = readout.design_matrix(states, inputs)
    assert design.shape == (N_STEPS, 1 + N_RESERVOIR)
    assert np.array_equal(design[:, 0], np.ones(N_STEPS))
    assert np.array_equal(design[:, 1:], states)


def test_design_matrix_without_states_omits_reservoir(
    states: NDArray[np.float64], inputs: NDArray[np.float64]
) -> None:
    """リザバー無し baseline は `[1, u]` になる (アブレーションの対照条件)。"""
    readout = RidgeReadout(1e-6, use_states=False)
    design = readout.design_matrix(states, inputs)
    assert design.shape == (N_STEPS, 1 + N_INPUTS)
    assert np.array_equal(design[:, 0], np.ones(N_STEPS))
    assert np.array_equal(design[:, 1:], inputs)


def test_design_matrix_without_states_accepts_zero_width_states(
    inputs: NDArray[np.float64],
) -> None:
    """状態を使わないときは列数 0 の `[T, 0]` を渡してよい。

    `uncertainty/conformal.py` がリザバーを駆動せずにこの形を渡す。
    """
    empty = np.zeros((N_STEPS, 0))
    readout = RidgeReadout(1e-6, use_states=False)
    assert readout.design_matrix(empty, inputs).shape == (N_STEPS, 1 + N_INPUTS)


def test_both_features_disabled_is_rejected() -> None:
    """設計行列がバイアス列だけになる組は作らせない。"""
    with pytest.raises(ValueError, match="同時に False"):
        RidgeReadout(1e-6, input_passthrough=False, use_states=False)


@pytest.mark.parametrize(
    ("passthrough", "use_states"), [(True, True), (False, True), (True, False)]
)
def test_n_features_matches_design_matrix(
    states: NDArray[np.float64],
    inputs: NDArray[np.float64],
    passthrough: bool,
    use_states: bool,
) -> None:
    readout = RidgeReadout(1e-6, input_passthrough=passthrough, use_states=use_states)
    design = readout.design_matrix(states, inputs)
    assert readout.n_features(N_INPUTS, N_RESERVOIR) == design.shape[1]


def test_fit_shapes_and_prediction_shape(
    states: NDArray[np.float64],
    inputs: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> None:
    readout = RidgeReadout(1e-6).fit(states, inputs, targets)
    assert readout.is_fitted
    assert readout.w_out.shape == (1 + N_INPUTS + N_RESERVOIR, N_OUTPUTS)
    assert readout.predict(states, inputs).shape == (N_STEPS, N_OUTPUTS)


def test_closed_form_matches_lstsq_for_tiny_ridge(
    states: NDArray[np.float64],
    inputs: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> None:
    # 受け入れ基準: alpha=1e-10 の閉形式解が np.linalg.lstsq と rtol=1e-6 で一致。
    readout = RidgeReadout(1e-10).fit(states, inputs, targets)
    design = readout.design_matrix(states, inputs)
    lstsq_solution = np.linalg.lstsq(design, targets, rcond=None)[0]
    assert np.allclose(readout.w_out, lstsq_solution, rtol=1e-6)


def test_frobenius_norm_is_monotonically_non_increasing_in_alpha(
    states: NDArray[np.float64],
    inputs: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> None:
    # 列を中心化するとバイアス項は常に 0 になり、係数ノルムの単調非増加性が
    # 厳密に成り立つ (ridge の標準的な性質)。
    centered_states = states - states.mean(axis=0, keepdims=True)
    centered_inputs = inputs - inputs.mean(axis=0, keepdims=True)
    centered_targets = targets - targets.mean(axis=0, keepdims=True)

    alphas = [0.0, 1e-4, 1e-2, 1.0, 10.0, 1e3]
    norms = [
        _frobenius_norm(
            RidgeReadout(alpha)
            .fit(centered_states, centered_inputs, centered_targets)
            .w_out
        )
        for alpha in alphas
    ]
    assert len(norms) >= 3
    for previous, current in itertools.pairwise(norms):
        assert current <= previous * (1.0 + MONOTONICITY_TOLERANCE)
    assert norms[-1] < norms[0]


def test_coefficient_norm_is_non_increasing_on_reservoir_states() -> None:
    # 中心化しない実運用相当の設計行列でも、非正則化のバイアス行を除いた
    # 係数ブロックのノルムは単調非増加になる。
    config = ESNConfig(n_reservoir=N_RESERVOIR, density=0.2, seed=7)
    reservoir = Reservoir(config, 1)
    rng = np.random.default_rng(14)
    driving_inputs = rng.uniform(-0.8, 0.8, size=(N_STEPS, 1))
    reservoir_states = reservoir.run(driving_inputs)
    delayed_targets = np.concatenate([np.zeros((3, 1)), driving_inputs[:-3]], axis=0)

    alphas = [0.0, 1e-4, 1e-2, 1.0, 100.0]
    norms = [
        _frobenius_norm(
            RidgeReadout(alpha)
            .fit(reservoir_states, driving_inputs, delayed_targets)
            .w_out[1:]
        )
        for alpha in alphas
    ]
    for previous, current in itertools.pairwise(norms):
        assert current <= previous * (1.0 + MONOTONICITY_TOLERANCE)


def test_bias_term_is_not_regularized(
    states: NDArray[np.float64], inputs: NDArray[np.float64]
) -> None:
    # 定数目標に対しては、alpha を極端に大きくしてもバイアスが定数を再現する。
    constant = 2.5
    constant_targets = np.full((N_STEPS, 1), constant, dtype=np.float64)
    readout = RidgeReadout(1e8).fit(states, inputs, constant_targets)
    predictions = readout.predict(states, inputs)
    assert np.allclose(predictions, constant, rtol=1e-6)


def test_predict_before_fit_raises_runtime_error(
    states: NDArray[np.float64], inputs: NDArray[np.float64]
) -> None:
    readout = RidgeReadout(1e-6)
    assert not readout.is_fitted
    with pytest.raises(RuntimeError, match="未 fit"):
        readout.predict(states, inputs)


def test_w_out_before_fit_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="未 fit"):
        _ = RidgeReadout(1e-6).w_out


def test_negative_alpha_raises_value_error() -> None:
    with pytest.raises(ValueError, match="alpha"):
        RidgeReadout(-1e-12)


def test_mismatched_lengths_raise_value_error(
    states: NDArray[np.float64], inputs: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError, match="系列長"):
        RidgeReadout(1e-6).design_matrix(states, inputs[:-1])


def test_mismatched_target_length_raises_value_error(
    states: NDArray[np.float64],
    inputs: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="系列長"):
        RidgeReadout(1e-6).fit(states, inputs, targets[:-1])


def test_non_2d_arrays_raise_value_error(inputs: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="2 次元"):
        RidgeReadout(1e-6).design_matrix(np.zeros(N_STEPS), inputs)


def test_predict_rejects_design_matrix_with_wrong_width(
    states: NDArray[np.float64],
    inputs: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> None:
    readout = RidgeReadout(1e-6).fit(states, inputs, targets)
    with pytest.raises(ValueError, match="列数"):
        readout.predict(states[:, :-1], inputs)
