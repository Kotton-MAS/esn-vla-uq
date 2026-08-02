"""`esn_vla_uq.diagnostics.memory_capacity` のテスト (Sprint 1 T4)。"""

from __future__ import annotations

import numpy as np
import pytest

from esn_vla_uq.diagnostics import (
    DEFAULT_MC_RIDGE_ALPHA,
    MemoryCapacityResult,
    default_max_delay,
    linear_memory_capacity,
)
from esn_vla_uq.diagnostics.memory_capacity import _squared_correlation
from esn_vla_uq.esn import ESNConfig, Reservoir, RidgeReadout

SMALL_N = 20
LARGE_N = 40
MIN_TOTAL_MC_AT_LARGE_N = 5.0


def _reservoir(n_reservoir: int, n_inputs: int = 1) -> Reservoir:
    return Reservoir(ESNConfig(n_reservoir=n_reservoir, seed=0), n_inputs)


@pytest.fixture(scope="module")
def small_result() -> MemoryCapacityResult:
    return linear_memory_capacity(_reservoir(SMALL_N), seed=0)


@pytest.fixture(scope="module")
def large_result() -> MemoryCapacityResult:
    return linear_memory_capacity(_reservoir(LARGE_N), seed=0)


@pytest.mark.parametrize(
    ("fixture_name", "n_reservoir"),
    [("small_result", SMALL_N), ("large_result", LARGE_N)],
)
def test_total_mc_respects_theoretical_upper_bound(
    fixture_name: str, n_reservoir: int, request: pytest.FixtureRequest
) -> None:
    # 受け入れ基準: total_mc <= N (理論上界)。
    result: MemoryCapacityResult = request.getfixturevalue(fixture_name)
    assert result.total_mc <= float(n_reservoir)
    assert result.total_mc > 0.0


def test_total_mc_is_substantial_at_forty_neurons(
    large_result: MemoryCapacityResult,
) -> None:
    # 受け入れ基準: N=40 で total_mc > 5.0。
    assert large_result.total_mc > MIN_TOTAL_MC_AT_LARGE_N


def test_per_delay_length_matches_default_max_delay(
    large_result: MemoryCapacityResult,
) -> None:
    expected = default_max_delay(LARGE_N)
    assert len(large_result.per_delay) == expected
    assert large_result.n_delays == expected


def test_short_delay_is_remembered_better_than_long_delay(
    large_result: MemoryCapacityResult,
) -> None:
    # 受け入れ基準: MC_1 > MC_K。
    assert large_result.per_delay[0] > large_result.per_delay[-1]


def test_memory_horizon_is_first_delay_below_threshold(
    large_result: MemoryCapacityResult,
) -> None:
    horizon = large_result.memory_horizon
    assert 1 <= horizon <= large_result.n_delays
    if horizon < large_result.n_delays:
        assert large_result.per_delay[horizon - 1] < 0.1
        assert all(value >= 0.1 for value in large_result.per_delay[: horizon - 1])


def test_mc_per_neuron_is_total_divided_by_reservoir_size(
    large_result: MemoryCapacityResult,
) -> None:
    assert large_result.mc_per_neuron == pytest.approx(
        large_result.total_mc / LARGE_N, rel=1e-12
    )


def test_total_mc_equals_sum_of_per_delay(large_result: MemoryCapacityResult) -> None:
    assert large_result.total_mc == pytest.approx(
        sum(large_result.per_delay), rel=1e-12
    )


def test_larger_reservoir_remembers_more(
    small_result: MemoryCapacityResult, large_result: MemoryCapacityResult
) -> None:
    assert large_result.total_mc > small_result.total_mc


def test_same_seed_gives_identical_result() -> None:
    first = linear_memory_capacity(_reservoir(SMALL_N), max_delay=10, seed=0)
    second = linear_memory_capacity(_reservoir(SMALL_N), max_delay=10, seed=0)
    assert first == second


def test_different_seed_changes_measurement() -> None:
    first = linear_memory_capacity(_reservoir(SMALL_N), max_delay=10, seed=0)
    second = linear_memory_capacity(_reservoir(SMALL_N), max_delay=10, seed=1)
    assert first.total_mc != second.total_mc


def test_joint_readout_matches_per_delay_readouts() -> None:
    """多出力 1 回解と遅延ごとの独立解が一致することを確認する。

    実装は計算量削減のため設計行列を共有した多出力リッジで解く。リッジ回帰の
    閉形式解は出力列ごとに独立に決まるため、遅延ごとに個別学習した場合と同じ
    ``MC_k`` になるはずである。
    """
    n_train, n_test, washout, max_delay = 300, 200, 20, 5
    reservoir = _reservoir(10)
    result = linear_memory_capacity(
        reservoir,
        n_train=n_train,
        n_test=n_test,
        washout=washout,
        max_delay=max_delay,
        seed=0,
    )

    rng = np.random.default_rng(0)
    inputs = rng.uniform(-0.8, 0.8, size=(washout + n_train + n_test, 1))
    states = reservoir.run(inputs)
    train = slice(washout, washout + n_train)
    test = slice(washout + n_train, inputs.shape[0])
    for delay in range(1, max_delay + 1):
        target = np.roll(inputs[:, 0], delay).reshape(-1, 1)
        readout = RidgeReadout(DEFAULT_MC_RIDGE_ALPHA, input_passthrough=False).fit(
            states[train], inputs[train], target[train]
        )
        predicted = readout.predict(states[test], inputs[test])[:, 0]
        expected = float(np.corrcoef(predicted, target[test, 0])[0, 1] ** 2)
        assert result.per_delay[delay - 1] == pytest.approx(expected, rel=1e-8)


def test_squared_correlation_is_zero_for_constant_series() -> None:
    # 分散 0 では相関が定義できないため 0.0 とする (クリップではない)。
    constant = np.ones(5)
    varying = np.arange(5, dtype=np.float64)
    assert _squared_correlation(constant, varying) == 0.0
    assert _squared_correlation(varying, varying) == pytest.approx(1.0, rel=1e-12)


def test_rejects_multi_dimensional_input_reservoir() -> None:
    with pytest.raises(ValueError, match="n_inputs"):
        linear_memory_capacity(_reservoir(SMALL_N, n_inputs=2))


def test_rejects_washout_shorter_than_max_delay() -> None:
    with pytest.raises(ValueError, match="washout"):
        linear_memory_capacity(_reservoir(SMALL_N), washout=5, max_delay=10)


@pytest.mark.parametrize(
    ("n_train", "n_test", "washout", "max_delay", "ridge_alpha", "match"),
    [
        (0, 200, 10, 5, 1e-8, "n_train"),
        (100, 1, 10, 5, 1e-8, "n_test"),
        (100, 200, -1, 5, 1e-8, "washout"),
        (100, 200, 10, 0, 1e-8, "max_delay"),
        (100, 200, 10, 5, -1.0, "ridge_alpha"),
    ],
)
def test_rejects_out_of_range_parameters(
    n_train: int,
    n_test: int,
    washout: int,
    max_delay: int,
    ridge_alpha: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        linear_memory_capacity(
            _reservoir(SMALL_N),
            n_train=n_train,
            n_test=n_test,
            washout=washout,
            max_delay=max_delay,
            ridge_alpha=ridge_alpha,
        )
