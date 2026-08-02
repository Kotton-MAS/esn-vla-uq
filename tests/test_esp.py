"""`esn_vla_uq.diagnostics.esp` のテスト (Sprint 1 T4)。

判定表は `docs/design.md` 4.2 節が唯一の真実。ここでは実装が表どおりであること、
および 3 指標が常に併記されることを検証する。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.diagnostics import EspResult, EspVerdict, check_esp, default_test_inputs
from esn_vla_uq.diagnostics.esp import _decay_rate, _decide_verdict
from esn_vla_uq.esn import ESNConfig, Reservoir

N_RESERVOIR = 100
N_STEPS = 500


def _reservoir(spectral_radius: float, leak_rate: float = 1.0) -> Reservoir:
    return Reservoir(
        ESNConfig(
            n_reservoir=N_RESERVOIR,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            seed=0,
        ),
        1,
    )


@pytest.fixture
def zero_inputs() -> NDArray[np.float64]:
    """零入力系列 (必要条件の検証に近い最も厳しいテスト入力)。"""
    return np.zeros((N_STEPS, 1))


@pytest.fixture
def stable_result(zero_inputs: NDArray[np.float64]) -> EspResult:
    return check_esp(_reservoir(0.9), inputs=zero_inputs, seed=0)


@pytest.fixture
def unstable_result(zero_inputs: NDArray[np.float64]) -> EspResult:
    return check_esp(_reservoir(1.5), inputs=zero_inputs, seed=0)


def test_stable_reservoir_holds_esp_with_negative_decay_rate(
    stable_result: EspResult,
) -> None:
    # 受け入れ基準: rho=0.9 / leak=1.0 / 零入力で esp_holds かつ decay_rate < 0。
    assert stable_result.verdict == "esp_holds"
    assert stable_result.decay_rate < 0.0


def test_unstable_reservoir_violates_esp(unstable_result: EspResult) -> None:
    # 受け入れ基準: rho=1.5 で必要条件不成立かつ esp_violated。
    assert unstable_result.necessary_condition_met is False
    assert unstable_result.verdict == "esp_violated"
    assert unstable_result.empirical_converged is False


@pytest.mark.parametrize("fixture_name", ["stable_result", "unstable_result"])
def test_three_indicators_are_always_reported(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    # 3 指標は必ず併記し、各真偽値は報告した実測値と整合していること。
    result: EspResult = request.getfixturevalue(fixture_name)
    assert result.sufficient_condition_met == (result.largest_singular_value < 1.0)
    assert result.necessary_condition_met == (result.effective_spectral_radius < 1.0)
    assert result.empirical_converged == (result.final_distance < result.tolerance)
    assert result.n_initial_states == 8
    assert result.n_steps == N_STEPS
    assert result.zero_input is True


def test_default_spectral_radius_does_not_meet_sufficient_condition(
    stable_result: EspResult,
) -> None:
    # 既定設定 (rho=0.9) では sigma_max >= 1 となり十分条件は満たされない。
    # 判定表 #2 の経路であることを実測値で明示する (想定リスク 3)。
    assert stable_result.sufficient_condition_met is False
    assert stable_result.largest_singular_value > 1.0
    assert stable_result.necessary_condition_met is True
    assert stable_result.empirical_converged is True


def test_default_random_test_inputs_are_used_when_omitted() -> None:
    result = check_esp(_reservoir(0.9), n_steps=200, seed=1)
    assert result.zero_input is False
    assert result.n_steps == 200
    assert result.verdict == "esp_holds"


def test_leaky_reservoir_is_diagnosed(zero_inputs: NDArray[np.float64]) -> None:
    result = check_esp(_reservoir(0.9, leak_rate=0.3), inputs=zero_inputs, seed=0)
    assert result.effective_spectral_radius < 1.0
    assert result.verdict == "esp_holds"
    assert result.decay_rate < 0.0


def test_same_seed_gives_identical_result() -> None:
    first = check_esp(_reservoir(0.9), n_steps=100, seed=3)
    second = check_esp(_reservoir(0.9), n_steps=100, seed=3)
    assert first == second


def test_different_seed_changes_test_inputs() -> None:
    first = check_esp(_reservoir(0.9), n_steps=100, seed=3)
    second = check_esp(_reservoir(0.9), n_steps=100, seed=4)
    assert first.final_distance != second.final_distance


@pytest.mark.parametrize(
    ("sufficient", "necessary", "empirical", "expected"),
    [
        (True, True, True, "esp_holds"),
        (True, True, False, "esp_holds"),
        (True, False, True, "esp_likely"),
        (True, False, False, "esp_likely"),
        (False, True, True, "esp_holds"),
        (False, True, False, "esp_likely"),
        (False, False, True, "esp_likely"),
        (False, False, False, "esp_violated"),
    ],
)
def test_verdict_table(
    sufficient: bool, necessary: bool, empirical: bool, expected: EspVerdict
) -> None:
    # docs/design.md 4.2 節の判定表 #1-#6 を網羅する。
    verdict = _decide_verdict(
        sufficient=sufficient, necessary=necessary, empirical=empirical
    )
    assert verdict == expected


def test_impossible_combination_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # #6: sigma_max < 1 かつ rho >= 1 は理論上生じない。警告して esp_likely。
    with caplog.at_level("WARNING"):
        verdict = _decide_verdict(sufficient=True, necessary=False, empirical=True)
    assert verdict == "esp_likely"
    assert any("理論上生じない" in record.message for record in caplog.records)


def test_decay_rate_is_negative_for_geometrically_decaying_distance() -> None:
    steps = np.arange(1, 101, dtype=np.float64)
    assert _decay_rate(0.9**steps) == pytest.approx(float(np.log(0.9)), rel=1e-6)


def test_decay_rate_falls_back_to_zero_when_undefined(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 正の距離が 2 点未満だと傾きが定まらない。0.0 として報告し警告を残す。
    with caplog.at_level("WARNING"):
        rate = _decay_rate(np.zeros(10))
    assert rate == 0.0
    assert any("decay_rate" in record.message for record in caplog.records)


def test_default_test_inputs_shape_and_range() -> None:
    inputs = default_test_inputs(np.random.default_rng(0), 32, 3)
    assert inputs.shape == (32, 3)
    assert np.all(np.abs(inputs) <= 1.0)


def test_default_test_inputs_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        default_test_inputs(np.random.default_rng(0), 0, 1)


def test_rejects_single_initial_state() -> None:
    with pytest.raises(ValueError, match="n_initial_states"):
        check_esp(_reservoir(0.9), n_steps=10, n_initial_states=1)


def test_rejects_non_positive_tolerance() -> None:
    with pytest.raises(ValueError, match="tol"):
        check_esp(_reservoir(0.9), n_steps=10, tol=0.0)


def test_rejects_inputs_with_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="入力次元"):
        check_esp(_reservoir(0.9), inputs=np.zeros((10, 2)))
