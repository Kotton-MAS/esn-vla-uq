"""`esn_vla_uq.data.features` と `run_episodes` のテスト (A8)。"""

from __future__ import annotations

import numpy as np
import pytest

from esn_vla_uq.data.features import (
    DEFAULT_FEATURE_SET,
    SUPPORTED_FEATURE_SETS,
    DatasetInputs,
    dataset_inputs,
)
from esn_vla_uq.data.schema import ACTION_DIM, STATE_DIM, RolloutDataset
from esn_vla_uq.data.synthetic import generate_dataset
from esn_vla_uq.esn import ESNConfig, Reservoir, run_episodes

N_EPISODES = 3
MIN_STEPS = 20
MAX_STEPS = 30
N_RESERVOIR = 25


@pytest.fixture
def dataset() -> RolloutDataset:
    return generate_dataset(
        seed=0, n_episodes=N_EPISODES, min_steps=MIN_STEPS, max_steps=MAX_STEPS
    )


@pytest.fixture
def inputs(dataset: RolloutDataset) -> DatasetInputs:
    return dataset_inputs(dataset)


def test_default_feature_is_state(inputs: DatasetInputs) -> None:
    assert inputs.feature == DEFAULT_FEATURE_SET == "state"
    assert inputs.n_inputs == STATE_DIM


@pytest.mark.parametrize(
    ("feature", "expected_dim"),
    [
        ("state", STATE_DIM),
        ("action", ACTION_DIM),
        ("state_action", STATE_DIM + ACTION_DIM),
    ],
)
def test_feature_sets_produce_expected_dimension(
    dataset: RolloutDataset, feature: str, expected_dim: int
) -> None:
    result = dataset_inputs(dataset, feature=feature)  # type: ignore[arg-type]
    assert result.n_inputs == expected_dim


def test_all_supported_feature_sets_are_covered_by_tests() -> None:
    """`FeatureSet` に値を足したらパラメータ化テストも増やすことを強制する。"""
    assert set(SUPPORTED_FEATURE_SETS) == {"state", "action", "state_action"}


def test_values_are_float64(inputs: DatasetInputs) -> None:
    """`Reservoir.run` が要求する dtype にこの層で一度だけ変換する。"""
    assert inputs.values.dtype == np.float64


def test_shape_matches_dataset(dataset: RolloutDataset, inputs: DatasetInputs) -> None:
    assert inputs.total_steps == dataset.total_steps
    assert inputs.n_episodes == dataset.n_episodes
    assert inputs.is_inference_step.shape == (dataset.total_steps,)


def test_values_contain_no_nan(inputs: DatasetInputs) -> None:
    """`state`/`action` は有限性が保証されているため NaN は入らない。"""
    assert bool(np.isfinite(inputs.values).all())


def test_segments_match_episode_boundaries(
    dataset: RolloutDataset, inputs: DatasetInputs
) -> None:
    segments = inputs.segments
    assert len(segments) == dataset.n_episodes
    for segment, episode in zip(segments, dataset.episodes, strict=True):
        assert segment.shape == (episode.n_steps, STATE_DIM)
        np.testing.assert_allclose(segment, episode.state.astype(np.float64))


def test_unknown_feature_is_rejected(dataset: RolloutDataset) -> None:
    with pytest.raises(ValueError, match="未知の特徴量"):
        dataset_inputs(dataset, feature="action_chunk")  # type: ignore[arg-type]


def test_run_episodes_resets_state_at_boundaries(inputs: DatasetInputs) -> None:
    """各区間が初期状態から駆動されること (A8 の核心)。

    区間を個別に `Reservoir.run` した結果と一致することを確認する。
    """
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=1), inputs.n_inputs)
    states = run_episodes(reservoir, inputs.segments)
    expected = np.concatenate(
        [reservoir.run(segment) for segment in inputs.segments], axis=0
    )
    np.testing.assert_allclose(states, expected)


def test_run_episodes_differs_from_running_concatenated_input(
    inputs: DatasetInputs,
) -> None:
    """連結配列をそのまま流すのとは**別の結果**になること。

    両者が一致するなら「境界でリセットする」という選択に意味が無いことに
    なる。この差こそが A8 で分散させたくない判断そのものなので、差がある
    ことを固定する。最初のエピソードは同一で、2 本目以降がずれる。
    """
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=1), inputs.n_inputs)
    per_episode = run_episodes(reservoir, inputs.segments)
    continuous = reservoir.run(inputs.values)

    first_length = int(inputs.episode_lengths[0])
    np.testing.assert_allclose(per_episode[:first_length], continuous[:first_length])
    assert not np.allclose(per_episode[first_length:], continuous[first_length:])


def test_run_episodes_returns_one_row_per_step(inputs: DatasetInputs) -> None:
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=1), inputs.n_inputs)
    states = run_episodes(reservoir, inputs.segments)
    assert states.shape == (inputs.total_steps, N_RESERVOIR)


def test_run_episodes_rejects_empty_segments() -> None:
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=1), STATE_DIM)
    with pytest.raises(ValueError, match="1 区間以上"):
        run_episodes(reservoir, [])


def test_dataset_inputs_round_trips_through_reservoir(dataset: RolloutDataset) -> None:
    """`dataset_inputs` -> `Reservoir` -> `run_episodes` が連結できること。

    `n_inputs` をそのまま `Reservoir` へ渡せる、という A8 の目的
    (呼び出し側が次元合わせを自前でやらない) を確認する。
    """
    inputs = dataset_inputs(dataset, feature="state_action")
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=2), inputs.n_inputs)
    states = run_episodes(reservoir, inputs.segments)
    assert states.shape == (dataset.total_steps, N_RESERVOIR)
