"""合成ロールアウト生成のテスト (Sprint 1 T5)。

「失敗区間でチャンク分散が上がること」と「単純ベースラインでは成否を完全分離できない
こと (AUROC < 1.0)」を同時に検証する。これは仕様書 §8 リスク 2 の受け入れ条件であり、
片方だけを満たす合成データは Sprint 2 の較正評価を無意味にする。

AUROC は scipy を導入せず、順位ベースの Mann-Whitney U 統計量から numpy だけで求める。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.data.schema import (
    ACTION_DIM,
    CHUNK_HORIZON,
    SCHEMA_VERSION,
    STATE_DIM,
    Episode,
    RolloutDataset,
)
from esn_vla_uq.data.source import RolloutSource, SyntheticRolloutSource
from esn_vla_uq.data.synthetic import (
    CONTROL_HZ,
    DEFAULT_MAX_STEPS,
    DEFAULT_MIN_STEPS,
    DEFAULT_N_EPISODES,
    POLICY_NAME,
    generate_dataset,
    validate_synthetic_dataset,
)

MIN_VARIANCE_RATIO = 1.5


def chunk_dispersion(episode: Episode) -> NDArray[np.float64]:
    """推論ステップごとの行動チャンク分散 (単純ベースラインの信号)。

    ホライズン方向の 2 階差分の平均二乗を使う。滑らかなトレンド成分が落ち、
    チャンク内のばらつき (flow matching のサンプリング分散に相当) が残る。
    """
    chunks = episode.action_chunk[episode.is_inference_step].astype(np.float64)
    roughness = np.diff(chunks, n=2, axis=1)
    return np.mean(roughness**2, axis=(1, 2))


def _rankdata(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """同順位を平均順位で扱う順位付け (scipy.stats.rankdata 相当)。"""
    sorter = np.argsort(values, kind="stable")
    inverse = np.empty_like(sorter)
    inverse[sorter] = np.arange(values.shape[0])
    sorted_values = values[sorter]
    is_new = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    dense = is_new.cumsum()[inverse]
    boundaries = np.concatenate((np.nonzero(is_new)[0], [values.shape[0]]))
    ranks: NDArray[np.float64] = 0.5 * (
        boundaries[dense] + boundaries[dense - 1] + 1
    ).astype(np.float64)
    return ranks


def auroc(scores: NDArray[np.float64], positive: NDArray[np.bool_]) -> float:
    """Mann-Whitney U 統計量から AUROC を計算する (numpy のみ)。"""
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC には両クラスのサンプルが必要です")
    ranks = _rankdata(scores)
    u_statistic = float(ranks[positive].sum()) - n_positive * (n_positive + 1) / 2.0
    return u_statistic / (n_positive * n_negative)


def episode_scores(dataset: RolloutDataset) -> NDArray[np.float64]:
    """エピソード単位の単純ベースラインスコア (平均チャンク分散)。"""
    return np.array(
        [float(chunk_dispersion(episode).mean()) for episode in dataset.episodes],
        dtype=np.float64,
    )


def failure_labels(dataset: RolloutDataset) -> NDArray[np.bool_]:
    """失敗を陽性としたラベル。"""
    return np.array(
        [not episode.success for episode in dataset.episodes], dtype=np.bool_
    )


@pytest.fixture(scope="module")
def dataset() -> RolloutDataset:
    return generate_dataset(seed=0)


def test_auroc_helper_matches_known_values() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    perfect = np.array([False, False, True, True], dtype=np.bool_)
    assert auroc(scores, perfect) == pytest.approx(1.0)
    assert auroc(scores, ~perfect) == pytest.approx(0.0)
    tied = np.ones(4, dtype=np.float64)
    assert auroc(tied, perfect) == pytest.approx(0.5)


def test_defaults_match_specification(dataset: RolloutDataset) -> None:
    assert dataset.n_episodes == DEFAULT_N_EPISODES
    assert dataset.source == "synthetic"
    assert dataset.policy == POLICY_NAME
    assert dataset.control_hz == CONTROL_HZ
    assert dataset.schema_version == SCHEMA_VERSION
    assert dataset.seed == 0


def test_generated_dataset_is_valid(dataset: RolloutDataset) -> None:
    dataset.validate()


def test_episode_shapes_and_lengths(dataset: RolloutDataset) -> None:
    for episode in dataset.episodes:
        assert DEFAULT_MIN_STEPS <= episode.n_steps <= DEFAULT_MAX_STEPS
        assert episode.state.shape == (episode.n_steps, STATE_DIM)
        assert episode.action.shape == (episode.n_steps, ACTION_DIM)
        assert episode.action_chunk.shape == (
            episode.n_steps,
            CHUNK_HORIZON,
            ACTION_DIM,
        )
        expected = np.zeros(episode.n_steps, dtype=np.bool_)
        expected[::CHUNK_HORIZON] = True
        assert np.array_equal(episode.is_inference_step, expected)


def test_contains_both_success_and_failure(dataset: RolloutDataset) -> None:
    successes = [episode.success for episode in dataset.episodes]
    assert any(successes)
    assert not all(successes)
    assert sum(successes) == 28  # round(0.7 * 40)


def test_failure_onset_is_recorded_only_for_failures(dataset: RolloutDataset) -> None:
    for episode in dataset.episodes:
        if episode.success:
            assert episode.failure_onset is None
        else:
            assert episode.failure_onset is not None
            assert 0 < episode.failure_onset < episode.n_steps


def test_validate_synthetic_dataset_accepts_generated_dataset(
    dataset: RolloutDataset,
) -> None:
    # M2: generate_dataset は末尾で自動的にこれを実行済みだが、公開関数として
    # 単体でも正常系を確認する。
    validate_synthetic_dataset(dataset)


def test_validate_synthetic_dataset_rejects_missing_failure_onset(
    dataset: RolloutDataset,
) -> None:
    # M2: 「失敗エピソードには failure_onset が必須」という合成データ生成器
    # 固有の不変条件を、生成後のデータセットに対しても直接検証できる
    # (`data/io.py` の読み込み境界でも同じ関数を再利用する)。
    failure_index = next(
        index for index, episode in enumerate(dataset.episodes) if not episode.success
    )
    episodes = list(dataset.episodes)
    episodes[failure_index] = replace(episodes[failure_index], failure_onset=None)
    corrupted = replace(dataset, episodes=episodes)

    with pytest.raises(ValueError, match="failure_onset"):
        validate_synthetic_dataset(corrupted)


def test_same_seed_reproduces_identical_dataset() -> None:
    first = generate_dataset(seed=3, n_episodes=4)
    second = generate_dataset(seed=3, n_episodes=4)
    assert first.to_metadata() == second.to_metadata()
    for left, right in zip(first.episodes, second.episodes, strict=True):
        assert np.array_equal(left.state, right.state)
        assert np.array_equal(left.action, right.action)
        assert np.array_equal(left.is_inference_step, right.is_inference_step)
        assert np.array_equal(left.action_chunk, right.action_chunk, equal_nan=True)


def test_different_seed_produces_different_dataset() -> None:
    first = generate_dataset(seed=3, n_episodes=4)
    other = generate_dataset(seed=4, n_episodes=4)
    assert not np.array_equal(first.episodes[0].state, other.episodes[0].state)
    assert not np.array_equal(first.episodes[0].action, other.episodes[0].action)


def test_arrays_are_float32(dataset: RolloutDataset) -> None:
    episode = dataset.episodes[0]
    assert episode.state.dtype == np.float32
    assert episode.action.dtype == np.float32
    assert episode.action_chunk.dtype == np.float32


def test_chunk_variance_increases_after_failure_onset(dataset: RolloutDataset) -> None:
    ratios: list[float] = []
    for episode in dataset.episodes:
        if episode.success:
            continue
        onset = episode.failure_onset
        assert onset is not None
        inference_steps = np.nonzero(episode.is_inference_step)[0]
        dispersion = chunk_dispersion(episode)
        before = dispersion[inference_steps < onset]
        after = dispersion[inference_steps >= onset]
        assert before.size > 0
        assert after.size > 0
        ratios.append(float(after.mean() / before.mean()))

    assert len(ratios) > 0
    # 個々のエピソードでも、失敗区間全体をプールしても比 > 1.5 であること。
    assert min(ratios) > MIN_VARIANCE_RATIO
    assert float(np.median(np.array(ratios))) > MIN_VARIANCE_RATIO


def test_simple_variance_baseline_is_not_perfect(dataset: RolloutDataset) -> None:
    scores = episode_scores(dataset)
    labels = failure_labels(dataset)
    score = auroc(scores, labels)
    # 信号は存在する (ランダム以上) が、単純しきい値では完全分離できない。
    assert 0.5 < score < 1.0


def test_synthetic_source_satisfies_protocol() -> None:
    source = SyntheticRolloutSource(seed=5, n_episodes=3)
    assert isinstance(source, RolloutSource)
    loaded = source.load()
    expected = generate_dataset(seed=5, n_episodes=3)
    assert loaded.to_metadata() == expected.to_metadata()
    assert np.array_equal(loaded.episodes[0].state, expected.episodes[0].state)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: generate_dataset(seed=0, n_episodes=0), "n_episodes: 1 以上"),
        (
            lambda: generate_dataset(seed=0, success_rate=1.5),
            r"success_rate: \[0, 1\] の範囲外",
        ),
        (lambda: generate_dataset(seed=0, min_steps=8), "min_steps: 16 以上"),
        (
            lambda: generate_dataset(seed=0, min_steps=200, max_steps=100),
            "max_steps: min_steps 以上",
        ),
    ],
)
def test_invalid_generation_parameters_raise(
    call: Callable[[], RolloutDataset], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call()
