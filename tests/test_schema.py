"""ロールアウトデータスキーマ v0.1 の検証テスト (Sprint 1 T5)。"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.data.schema import (
    ACTION_DIM,
    CHUNK_HORIZON,
    MAX_ACTION_DIM,
    MAX_CHUNK_HORIZON,
    MAX_DATASET_BYTES,
    MAX_STATE_DIM,
    SCHEMA_VERSION,
    STATE_DIM,
    Episode,
    RolloutDataset,
    check_dataset_byte_budget,
    check_dimension_limit,
    validate_episode_index,
)

N_STEPS = 32


def make_episode(
    episode_id: str = "synthetic_0000",
    n_steps: int = N_STEPS,
    success: bool = True,
    failure_onset: int | None = None,
) -> Episode:
    """検証を通る最小限のエピソードを組み立てる。"""
    rng = np.random.default_rng(0)
    is_inference_step = np.zeros(n_steps, dtype=np.bool_)
    is_inference_step[::CHUNK_HORIZON] = True
    action_chunk = np.full(
        (n_steps, CHUNK_HORIZON, ACTION_DIM), np.nan, dtype=np.float32
    )
    action_chunk[is_inference_step] = rng.normal(
        size=(int(is_inference_step.sum()), CHUNK_HORIZON, ACTION_DIM)
    ).astype(np.float32)
    return Episode(
        episode_id=episode_id,
        task_name="synthetic_pick_up_bowl",
        success=success,
        n_steps=n_steps,
        state=rng.normal(size=(n_steps, STATE_DIM)).astype(np.float32),
        action=rng.normal(size=(n_steps, ACTION_DIM)).astype(np.float32),
        action_chunk=action_chunk,
        is_inference_step=is_inference_step,
        failure_onset=failure_onset,
    )


def make_dataset(n_episodes: int = 2) -> RolloutDataset:
    """検証を通る最小限のデータセットを組み立てる。"""
    return RolloutDataset(
        episodes=[
            make_episode(episode_id=f"synthetic_{index:04d}")
            for index in range(n_episodes)
        ],
        source="synthetic",
        policy="synthetic-chunked-policy-v0.1",
        seed=0,
        control_hz=20.0,
    )


def test_valid_episode_passes_validation() -> None:
    make_episode().validate()


def test_valid_dataset_passes_validation() -> None:
    make_dataset().validate()


def test_episode_index_properties() -> None:
    dataset = RolloutDataset(
        episodes=[
            make_episode(episode_id="synthetic_0000", n_steps=32),
            make_episode(episode_id="synthetic_0001", n_steps=48),
        ],
        source="synthetic",
        policy="p",
        seed=0,
        control_hz=20.0,
    )
    assert dataset.n_episodes == 2
    assert dataset.total_steps == 80
    assert np.array_equal(dataset.episode_lengths, np.array([32, 48], dtype=np.int64))
    assert np.array_equal(dataset.episode_starts, np.array([0, 32], dtype=np.int64))


def test_metadata_propagates_source_synthetic() -> None:
    metadata = make_dataset().to_metadata()
    assert metadata["source"] == "synthetic"
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["state_dim"] == STATE_DIM
    assert metadata["action_dim"] == ACTION_DIM
    assert metadata["chunk_horizon"] == CHUNK_HORIZON
    episodes = metadata["episodes"]
    assert isinstance(episodes, list)
    assert len(episodes) == 2


def test_dataset_default_dims_match_module_constants() -> None:
    dataset = make_dataset(n_episodes=1)
    assert dataset.state_dim == STATE_DIM
    assert dataset.action_dim == ACTION_DIM
    assert dataset.chunk_horizon == CHUNK_HORIZON


def test_episode_validate_accepts_custom_dims() -> None:
    n_steps, horizon, action_dim, state_dim = 20, 4, 3, 5
    is_inference_step = np.zeros(n_steps, dtype=np.bool_)
    is_inference_step[::horizon] = True
    action_chunk = np.full((n_steps, horizon, action_dim), np.nan, dtype=np.float32)
    action_chunk[is_inference_step] = 0.0
    episode = Episode(
        episode_id="custom_0000",
        task_name="custom_task",
        success=True,
        n_steps=n_steps,
        state=np.zeros((n_steps, state_dim), dtype=np.float32),
        action=np.zeros((n_steps, action_dim), dtype=np.float32),
        action_chunk=action_chunk,
        is_inference_step=is_inference_step,
    )
    # 実際の次元 (5/3/4) を明示すれば検証を通る。
    episode.validate(state_dim=state_dim, action_dim=action_dim, chunk_horizon=horizon)
    # 既定次元 (STATE_DIM=8 等) では shape が一致しないため落ちる。
    with pytest.raises(ValueError, match="state: shape が不正"):
        episode.validate()


def test_rollout_dataset_metadata_reflects_custom_dims() -> None:
    dataset = replace(
        make_dataset(n_episodes=1), state_dim=5, action_dim=3, chunk_horizon=4
    )
    metadata = dataset.to_metadata()
    assert metadata["state_dim"] == 5
    assert metadata["action_dim"] == 3
    assert metadata["chunk_horizon"] == 4


def test_rollout_dataset_rejects_non_positive_state_dim() -> None:
    dataset = replace(make_dataset(n_episodes=1), state_dim=0)
    with pytest.raises(ValueError, match="state_dim: 1 以上である必要があります"):
        dataset.validate()


def test_rollout_dataset_rejects_non_positive_action_dim() -> None:
    dataset = replace(make_dataset(n_episodes=1), action_dim=0)
    with pytest.raises(ValueError, match="action_dim: 1 以上である必要があります"):
        dataset.validate()


def test_rollout_dataset_rejects_non_positive_chunk_horizon() -> None:
    dataset = replace(make_dataset(n_episodes=1), chunk_horizon=0)
    with pytest.raises(ValueError, match="chunk_horizon: 1 以上である必要があります"):
        dataset.validate()


def test_check_dimension_limit_rejects_over_limit() -> None:
    with pytest.raises(ValueError, match="foo: 上限を超えています"):
        check_dimension_limit("foo", 10, 9)


def test_check_dimension_limit_accepts_value_at_limit() -> None:
    check_dimension_limit("foo", 9, 9)  # ちょうど上限は許可される


def test_check_dataset_byte_budget_rejects_over_limit() -> None:
    with pytest.raises(ValueError, match="action_chunk: 復元後配列の推定確保サイズ"):
        check_dataset_byte_budget(
            n_steps=1_000_000,
            chunk_horizon=MAX_CHUNK_HORIZON,
            action_dim=MAX_ACTION_DIM,
        )


def test_rollout_dataset_rejects_state_dim_over_max() -> None:
    # M3 (CWE-789): メタデータ由来の state_dim/action_dim/chunk_horizon は
    # 配列確保の前に上限で拒否される。
    dataset = replace(make_dataset(n_episodes=1), state_dim=MAX_STATE_DIM + 1)
    with pytest.raises(ValueError, match="state_dim: 上限を超えています"):
        dataset.validate()


def test_rollout_dataset_rejects_action_dim_over_max() -> None:
    dataset = replace(make_dataset(n_episodes=1), action_dim=MAX_ACTION_DIM + 1)
    with pytest.raises(ValueError, match="action_dim: 上限を超えています"):
        dataset.validate()


def test_rollout_dataset_rejects_chunk_horizon_over_max() -> None:
    dataset = replace(make_dataset(n_episodes=1), chunk_horizon=MAX_CHUNK_HORIZON + 1)
    with pytest.raises(ValueError, match="chunk_horizon: 上限を超えています"):
        dataset.validate()


def test_rollout_dataset_rejects_byte_budget_over_max_even_within_individual_caps() -> (
    None
):
    # 個々の次元は MAX_ACTION_DIM / MAX_CHUNK_HORIZON の上限内でも、
    # n_steps * chunk_horizon * action_dim * 4 の積が MAX_DATASET_BYTES (2GiB)
    # を超えれば拒否される (M3: 個別上限だけでは積の爆発を防げない)。
    dataset = replace(
        make_dataset(n_episodes=5),
        action_dim=MAX_ACTION_DIM,
        chunk_horizon=MAX_CHUNK_HORIZON,
    )
    assert dataset.total_steps * MAX_CHUNK_HORIZON * MAX_ACTION_DIM * 4 > (
        MAX_DATASET_BYTES
    )
    with pytest.raises(ValueError, match="action_chunk: 復元後配列の推定確保サイズ"):
        dataset.validate()


def test_rollout_dataset_dim_mismatch_with_episodes_raises() -> None:
    # データセット側の次元宣言とエピソードの実配列 shape が食い違えば検出される。
    dataset = replace(make_dataset(n_episodes=1), state_dim=STATE_DIM + 1)
    with pytest.raises(ValueError, match="state: shape が不正"):
        dataset.validate()


def test_state_dtype_mismatch_raises() -> None:
    # 実行時に dtype 不一致を再現するため、意図的に静的型と食い違う配列を渡す。
    wrong_dtype = cast(
        "NDArray[np.float32]", np.zeros((N_STEPS, STATE_DIM), dtype=np.float64)
    )
    episode = replace(make_episode(), state=wrong_dtype)
    with pytest.raises(ValueError, match="state: dtype が不正"):
        episode.validate()


def test_state_shape_mismatch_raises() -> None:
    episode = replace(
        make_episode(), state=np.zeros((N_STEPS, STATE_DIM + 1), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="state: shape が不正"):
        episode.validate()


def test_action_shape_mismatch_raises() -> None:
    episode = replace(
        make_episode(), action=np.zeros((N_STEPS, ACTION_DIM + 1), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="action: shape が不正"):
        episode.validate()


def test_action_chunk_shape_mismatch_raises() -> None:
    episode = replace(
        make_episode(),
        action_chunk=np.zeros((N_STEPS, CHUNK_HORIZON - 1, ACTION_DIM), np.float32),
    )
    with pytest.raises(ValueError, match="action_chunk: shape が不正"):
        episode.validate()


def test_is_inference_step_dtype_mismatch_raises() -> None:
    wrong_dtype = cast("NDArray[np.bool_]", np.zeros(N_STEPS, dtype=np.int8))
    episode = replace(make_episode(), is_inference_step=wrong_dtype)
    with pytest.raises(ValueError, match="is_inference_step: dtype が不正"):
        episode.validate()


def test_n_steps_inconsistent_with_arrays_raises() -> None:
    episode = replace(make_episode(), n_steps=N_STEPS + 1)
    with pytest.raises(ValueError, match="state: shape が不正"):
        episode.validate()


def test_non_finite_state_raises() -> None:
    state = make_episode().state.copy()
    state[3, 2] = np.nan
    episode = replace(make_episode(), state=state)
    with pytest.raises(ValueError, match="state: 有限でない値"):
        episode.validate()


def test_nan_at_inference_step_raises() -> None:
    base = make_episode()
    chunk = base.action_chunk.copy()
    chunk[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="action_chunk: 推論ステップに有限でない値"):
        replace(base, action_chunk=chunk).validate()


def test_non_nan_at_skipped_step_raises() -> None:
    base = make_episode()
    chunk = base.action_chunk.copy()
    chunk[1, 0, 0] = 0.0
    with pytest.raises(ValueError, match="非推論ステップは全要素 NaN"):
        replace(base, action_chunk=chunk).validate()


def test_no_inference_step_raises() -> None:
    base = make_episode()
    episode = replace(
        base,
        is_inference_step=np.zeros(N_STEPS, dtype=np.bool_),
        action_chunk=np.full(base.action_chunk.shape, np.nan, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="推論ステップが 1 つもありません"):
        episode.validate()


def test_failure_onset_on_success_episode_raises() -> None:
    episode = make_episode(success=True, failure_onset=10)
    with pytest.raises(ValueError, match="成功エピソードでは None"):
        episode.validate()


def test_failure_episode_without_failure_onset_is_valid() -> None:
    # failure_onset は合成データ生成器固有の概念であり、実 openpi ログの失敗
    # エピソードには存在しないことがある (Sprint 2 の OpenpiLogSource が
    # Episode を構築できるようにするため、失敗エピソードでも None を許容する)。
    make_episode(success=False, failure_onset=None).validate()


def test_failure_onset_out_of_range_raises() -> None:
    episode = make_episode(success=False, failure_onset=N_STEPS)
    with pytest.raises(ValueError, match=r"failure_onset: \[0, n_steps\) の範囲外"):
        episode.validate()


def test_unknown_schema_version_raises() -> None:
    dataset = replace(make_dataset(), schema_version="9.9.9")
    with pytest.raises(ValueError, match="schema_version: 未知のバージョン"):
        dataset.validate()


def test_unknown_source_raises() -> None:
    # Literal 型に無い値は型検査で弾かれるが、JSON 由来の値を想定して実行時も検証する。
    dataset = RolloutDataset(
        episodes=make_dataset().episodes,
        source="real_libero",  # type: ignore[arg-type]  # 実行時検証の確認用
        policy="p",
        seed=0,
        control_hz=20.0,
    )
    with pytest.raises(ValueError, match="source: 未知の出所"):
        dataset.validate()


def test_empty_policy_raises() -> None:
    with pytest.raises(ValueError, match="policy: 空文字"):
        replace(make_dataset(), policy="").validate()


@pytest.mark.parametrize("control_hz", [0.0, -1.0, float("nan")])
def test_invalid_control_hz_raises(control_hz: float) -> None:
    with pytest.raises(ValueError, match="control_hz: 正の有限値"):
        replace(make_dataset(), control_hz=control_hz).validate()


def test_empty_dataset_raises() -> None:
    with pytest.raises(ValueError, match="episodes: 1 件以上必要です"):
        RolloutDataset(
            episodes=[], source="synthetic", policy="p", seed=0, control_hz=20.0
        ).validate()


def test_duplicate_episode_id_raises() -> None:
    dataset = RolloutDataset(
        episodes=[make_episode(), make_episode()],
        source="synthetic",
        policy="p",
        seed=0,
        control_hz=20.0,
    )
    with pytest.raises(ValueError, match="episode_id: データセット内で重複"):
        dataset.validate()


def test_empty_episode_id_raises() -> None:
    with pytest.raises(ValueError, match="episode_id: 空文字"):
        make_episode(episode_id="").validate()


def _index(lengths: list[int]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    lengths_array = np.array(lengths, dtype=np.int64)
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(lengths_array[:-1], dtype=np.int64))
    )
    return starts, lengths_array


def test_validate_episode_index_accepts_consistent_index() -> None:
    starts, lengths = _index([10, 20, 30])
    validate_episode_index(starts, lengths, 60)


def test_validate_episode_index_rejects_wrong_starts() -> None:
    starts, lengths = _index([10, 20, 30])
    starts[1] = 11
    with pytest.raises(ValueError, match="累積和と一致しません"):
        validate_episode_index(starts, lengths, 60)


def test_validate_episode_index_rejects_total_mismatch() -> None:
    starts, lengths = _index([10, 20, 30])
    with pytest.raises(ValueError, match="総和が連結配列の長さと一致しません"):
        validate_episode_index(starts, lengths, 61)


def test_validate_episode_index_rejects_length_mismatch() -> None:
    starts, lengths = _index([10, 20, 30])
    with pytest.raises(ValueError, match="長さが一致しません"):
        validate_episode_index(starts[:2], lengths, 60)


def test_validate_episode_index_rejects_non_positive_length() -> None:
    starts, lengths = _index([10, 0, 30])
    with pytest.raises(ValueError, match="episode_lengths: 正の値"):
        validate_episode_index(starts, lengths, 40)


def test_validate_episode_index_rejects_wrong_dtype() -> None:
    starts, lengths = _index([10, 20, 30])
    with pytest.raises(ValueError, match="episode_starts: dtype が不正"):
        validate_episode_index(starts.astype(np.int32), lengths, 60)


def test_validate_episode_index_rejects_empty_index() -> None:
    empty = np.zeros(0, dtype=np.int64)
    with pytest.raises(ValueError, match="エピソードが 1 件も含まれていません"):
        validate_episode_index(empty, empty, 0)
