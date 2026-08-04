"""`OpenpiLogSource` のテスト。

フィクスチャは **openpi の実仕様に合わせた形** (state 8 次元、action 7 次元、
`chunk_horizon = 50`、`replan_steps = 5`) で作る。合成データ (H=16、16 ステップ
間隔) とは値が違うので、スキーマが両方を受け入れられることの検証も兼ねる。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esn_vla_uq.data.schema import ACTION_DIM, STATE_DIM
from esn_vla_uq.data.sources.openpi import (
    EPISODES_DIRNAME,
    LOG_SCHEMA_VERSION,
    MANIFEST_NAME,
    OpenpiLogSource,
)
from esn_vla_uq.uncertainty import build_samples

OPENPI_CHUNK_HORIZON = 50
"""pi0 の `action_horizon`。"""

OPENPI_REPLAN_STEPS = 5
"""openpi の LIBERO 評価ループの既定 `replan_steps`。"""

N_STEPS = 40


def _write_log(
    log_dir: Path,
    *,
    n_episodes: int = 3,
    n_steps: int = N_STEPS,
    chunk_horizon: int = OPENPI_CHUNK_HORIZON,
    schema_version: str = LOG_SCHEMA_VERSION,
) -> Path:
    """openpi 形状の収集ログを書き出す。"""
    rng = np.random.default_rng(0)
    episodes_dir = log_dir / EPISODES_DIRNAME
    episodes_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    # 推論は replan_steps ごと (openpi の評価ループと同じ)。
    inference_steps = np.arange(0, n_steps, OPENPI_REPLAN_STEPS, dtype=np.int64)
    for index in range(n_episodes):
        episode_id = f"openpi_000_{index:03d}"
        np.savez_compressed(
            episodes_dir / f"{episode_id}.npz",
            state=rng.normal(size=(n_steps, STATE_DIM)).astype(np.float32),
            action=rng.normal(size=(n_steps, ACTION_DIM)).astype(np.float32),
            action_chunk=rng.normal(
                size=(inference_steps.size, chunk_horizon, ACTION_DIM)
            ).astype(np.float32),
            inference_steps=inference_steps,
        )
        entries.append(
            {
                "episode_id": episode_id,
                "task_name": "pick up the black bowl",
                "success": index % 2 == 0,
                "n_steps": n_steps,
            }
        )

    (log_dir / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "policy": "pi0_libero",
                "task_suite": "libero_spatial",
                "seed": 7,
                "control_hz": 20.0,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "chunk_horizon": chunk_horizon,
                "replan_steps": OPENPI_REPLAN_STEPS,
                "episodes": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return log_dir


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    return _write_log(tmp_path / "openpi_logs")


def test_loads_a_validated_dataset(log_dir: Path) -> None:
    dataset = OpenpiLogSource(log_dir).load()
    assert dataset.source == "openpi"
    assert dataset.policy == "pi0_libero"
    assert dataset.n_episodes == 3
    dataset.validate()


def test_chunk_horizon_of_fifty_is_accepted(log_dir: Path) -> None:
    """openpi の H=50 が合成データの H=16 と同じスキーマで通ること。

    `RolloutDataset` が `chunk_horizon` をフィールドで持つ設計 (Sprint 1) の
    おかげで、モジュール定数を変えずに両方を扱える。
    """
    dataset = OpenpiLogSource(log_dir).load()
    assert dataset.chunk_horizon == OPENPI_CHUNK_HORIZON
    assert dataset.episodes[0].action_chunk.shape == (
        N_STEPS,
        OPENPI_CHUNK_HORIZON,
        ACTION_DIM,
    )


def test_non_inference_steps_are_nan(log_dir: Path) -> None:
    """非推論ステップが NaN で埋まること (スキーマの契約)。"""
    episode = OpenpiLogSource(log_dir).load().episodes[0]
    skipped = episode.action_chunk[~episode.is_inference_step]
    assert skipped.size > 0
    assert bool(np.isnan(skipped).all())
    inferred = episode.action_chunk[episode.is_inference_step]
    assert bool(np.isfinite(inferred).all())


def test_inference_interval_matches_replan_steps(log_dir: Path) -> None:
    """推論間隔が `replan_steps` (5) になっていること。"""
    episode = OpenpiLogSource(log_dir).load().episodes[0]
    steps = np.nonzero(episode.is_inference_step)[0]
    assert np.all(np.diff(steps) == OPENPI_REPLAN_STEPS)


def test_failure_onset_is_absent(log_dir: Path) -> None:
    """openpi の評価ループには失敗開始時刻の概念が無い。

    `Episode.validate()` が `failure_onset` を要求しない設計 (Sprint 1 の判断)
    が、実仕様に照らして正しかったことをここで固定する。
    """
    dataset = OpenpiLogSource(log_dir).load()
    assert all(episode.failure_onset is None for episode in dataset.episodes)
    assert any(not episode.success for episode in dataset.episodes)


def test_synthetic_invariant_is_not_applied_to_openpi(log_dir: Path) -> None:
    """合成データ固有の不変条件が openpi データに掛からないこと (S7 の意図)。

    `failure_onset` 必須という合成データの契約が適用されると、正当な openpi
    ログが読めなくなる。
    """
    from esn_vla_uq.data.invariants import validate_by_source

    validate_by_source(OpenpiLogSource(log_dir).load())


def test_feeds_the_prediction_task(log_dir: Path) -> None:
    """読み込んだデータがそのまま予測タスクへ流せること。"""
    dataset = OpenpiLogSource(log_dir).load()
    samples = build_samples(dataset)
    assert len(samples) == dataset.n_episodes
    assert samples[0].n_inputs == STATE_DIM + ACTION_DIM + 2
    assert samples[0].n_samples == N_STEPS - 1


def test_unknown_log_schema_version_is_rejected(tmp_path: Path) -> None:
    directory = _write_log(tmp_path / "logs", schema_version="9.9.9")
    with pytest.raises(ValueError, match="未知のログ形式"):
        OpenpiLogSource(directory).load()


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="マニフェスト"):
        OpenpiLogSource(tmp_path).load()


def test_missing_episode_file_is_reported(log_dir: Path) -> None:
    (log_dir / EPISODES_DIRNAME / "openpi_000_000.npz").unlink()
    with pytest.raises(FileNotFoundError, match="エピソード"):
        OpenpiLogSource(log_dir).load()


def test_out_of_range_inference_steps_are_rejected(tmp_path: Path) -> None:
    directory = _write_log(tmp_path / "logs", n_episodes=1)
    path = directory / EPISODES_DIRNAME / "openpi_000_000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive.items())
    arrays["inference_steps"] = np.array([N_STEPS + 5], dtype=np.int64)
    arrays["action_chunk"] = arrays["action_chunk"][:1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="範囲外"):
        OpenpiLogSource(directory).load()


def test_chunk_count_mismatch_is_rejected(tmp_path: Path) -> None:
    directory = _write_log(tmp_path / "logs", n_episodes=1)
    path = directory / EPISODES_DIRNAME / "openpi_000_000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive.items())
    arrays["action_chunk"] = arrays["action_chunk"][:-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="件数が一致しません"):
        OpenpiLogSource(directory).load()


def test_oversized_chunk_horizon_is_rejected(tmp_path: Path) -> None:
    """メタデータ由来の次元に上限を掛けること (CWE-789)。"""
    directory = _write_log(tmp_path / "logs", n_episodes=1, chunk_horizon=4)
    manifest_path = directory / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_horizon"] = 10_000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="上限を超えています"):
        OpenpiLogSource(directory).load()
