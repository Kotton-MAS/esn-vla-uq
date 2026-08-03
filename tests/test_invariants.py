"""`esn_vla_uq.data.invariants` のテスト (S7)。"""

from __future__ import annotations

import dataclasses

import pytest

from esn_vla_uq.data.invariants import (
    SOURCE_VALIDATORS,
    validate_by_source,
    validate_synthetic_dataset,
)
from esn_vla_uq.data.schema import RolloutDataset
from esn_vla_uq.data.synthetic import generate_dataset
from esn_vla_uq.provenance import SUPPORTED_SOURCES

N_EPISODES = 4
MIN_STEPS = 20
MAX_STEPS = 30


@pytest.fixture
def dataset() -> RolloutDataset:
    return generate_dataset(
        seed=0,
        n_episodes=N_EPISODES,
        success_rate=0.5,
        min_steps=MIN_STEPS,
        max_steps=MAX_STEPS,
    )


def _drop_failure_onset(dataset: RolloutDataset) -> RolloutDataset:
    """失敗エピソードから `failure_onset` を落とした複製を返す。"""
    episodes = [
        dataclasses.replace(episode, failure_onset=None)
        if not episode.success
        else episode
        for episode in dataset.episodes
    ]
    return dataclasses.replace(dataset, episodes=episodes)


def test_registry_keys_are_valid_sources() -> None:
    """レジストリの鍵が `DataSource` の値域に収まっていること。"""
    assert set(SOURCE_VALIDATORS) <= set(SUPPORTED_SOURCES)


def test_validate_by_source_accepts_generated_dataset(dataset: RolloutDataset) -> None:
    validate_by_source(dataset)


def test_validate_by_source_enforces_synthetic_invariant(
    dataset: RolloutDataset,
) -> None:
    """`source == "synthetic"` のとき合成データ固有の契約が掛かること。"""
    with pytest.raises(ValueError, match="failure_onset"):
        validate_by_source(_drop_failure_onset(dataset))


def test_validate_synthetic_dataset_is_the_registered_validator() -> None:
    assert SOURCE_VALIDATORS["synthetic"] is validate_synthetic_dataset


def test_unregistered_source_has_no_extra_contract(dataset: RolloutDataset) -> None:
    """未登録の出所は「追加契約なし」として素通りすること。

    黙って別の出所の契約が適用されると、openpi のログに合成データ生成器固有の
    不変条件 (`failure_onset` 必須) が課され、正当なデータが読めなくなる。
    """
    assert "openpi" not in SOURCE_VALIDATORS
    openpi_like = dataclasses.replace(_drop_failure_onset(dataset), source="openpi")
    validate_by_source(openpi_like)
