"""`esn_vla_uq.diagnostics.trajectory` のテスト。

特徴量が「反復している軌道」と「新規な軌道」を意図どおりに区別することを、
合成した軌道で固定する。実データでの結論は `docs/design.md` 14 節。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.diagnostics import (
    state_autocorrelation,
    state_novelty,
    state_participation_ratio,
)


class _WindowedFeature(Protocol):
    """3 特徴に共通の呼び出し形 (parametrize で型を付けるため)。"""

    def __call__(
        self, states: NDArray[np.float64], /, *, window: int
    ) -> NDArray[np.float64]: ...


N_STEPS = 200
N_UNITS = 12
WINDOW = 20


def _repeating() -> NDArray[np.float64]:
    """短い周期で同じ状態へ戻り続ける軌道 (= 動作の反復)。"""
    rng = np.random.default_rng(0)
    cycle = rng.normal(size=(5, N_UNITS))
    return np.concatenate([cycle] * (N_STEPS // 5), axis=0)


def _novel() -> NDArray[np.float64]:
    """毎ステップ独立な軌道 (= 反復していない)。"""
    rng = np.random.default_rng(1)
    return rng.normal(size=(N_STEPS, N_UNITS))


def test_autocorrelation_is_higher_for_a_smooth_trajectory() -> None:
    """滑らかに進む軌道のほうが、独立な軌道より自己相関が高い。"""
    smooth = np.cumsum(np.random.default_rng(2).normal(size=(N_STEPS, N_UNITS)), axis=0)
    smooth_value = np.nanmean(state_autocorrelation(smooth, window=WINDOW))
    novel_value = np.nanmean(state_autocorrelation(_novel(), window=WINDOW))
    assert smooth_value > novel_value


def test_participation_ratio_collapses_on_a_repeating_trajectory() -> None:
    """周期 5 の反復は実効次元が窓幅よりずっと小さくなる。"""
    repeating = np.nanmean(state_participation_ratio(_repeating(), window=WINDOW))
    novel = np.nanmean(state_participation_ratio(_novel(), window=WINDOW))
    assert repeating < novel


def test_participation_ratio_of_a_frozen_trajectory_is_one() -> None:
    """窓内で全く動かない軌道は実効次元 1 (潰れの極限)。"""
    frozen = np.tile(np.arange(N_UNITS, dtype=np.float64), (N_STEPS, 1))
    values = state_participation_ratio(frozen, window=WINDOW)
    assert np.allclose(values[WINDOW - 1 :], 1.0)


def test_novelty_is_lower_for_a_repeating_trajectory() -> None:
    """直近に同じ状態が現れる軌道では新規性が下がる。"""
    repeating = np.nanmean(state_novelty(_repeating(), window=WINDOW))
    novel = np.nanmean(state_novelty(_novel(), window=WINDOW))
    assert repeating < novel


def test_novelty_of_a_repeating_trajectory_is_zero() -> None:
    """周期が窓に収まっていれば、同一の状態が必ず窓内にあるので 0 になる。"""
    values = state_novelty(_repeating(), window=WINDOW)
    assert np.allclose(values[WINDOW:], 0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("function", "leading"),
    [
        (state_participation_ratio, WINDOW - 1),
        (state_novelty, WINDOW),
    ],
)
def test_leading_steps_are_nan_not_zero(
    function: _WindowedFeature, leading: int
) -> None:
    """窓が埋まらない先頭は NaN で返す (0 埋めすると値 0 と区別できない)。"""
    values = function(_novel(), window=WINDOW)
    assert np.all(np.isnan(values[:leading]))
    assert np.all(np.isfinite(values[leading:]))


def test_autocorrelation_leading_steps_are_nan() -> None:
    values = state_autocorrelation(_novel(), window=WINDOW, lag=3)
    assert np.all(np.isnan(values[: WINDOW + 3 - 1]))
    assert np.all(np.isfinite(values[WINDOW + 3 - 1 :]))


def test_features_are_causal() -> None:
    """末尾を書き換えても、それより前の値は変わらない (未来を見ていない)。"""
    states = _novel()
    tampered = states.copy()
    tampered[150:] = 0.0
    for function in (
        state_autocorrelation,
        state_participation_ratio,
        state_novelty,
    ):
        original = function(states, window=WINDOW)
        modified = function(tampered, window=WINDOW)
        assert np.allclose(original[:150], modified[:150], equal_nan=True)


@pytest.mark.parametrize(
    "function",
    [state_autocorrelation, state_participation_ratio, state_novelty],
)
def test_rejects_non_2d_states(
    function: _WindowedFeature,
) -> None:
    with pytest.raises(ValueError, match="2 次元"):
        function(np.zeros(N_STEPS), window=WINDOW)


@pytest.mark.parametrize(
    "function",
    [state_autocorrelation, state_participation_ratio, state_novelty],
)
def test_rejects_window_below_two(
    function: _WindowedFeature,
) -> None:
    with pytest.raises(ValueError, match="window"):
        function(_novel(), window=1)


def test_rejects_lag_below_one() -> None:
    with pytest.raises(ValueError, match="lag"):
        state_autocorrelation(_novel(), window=WINDOW, lag=0)
