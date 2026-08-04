"""`esn_vla_uq.demo` と CLI `demo` のテスト (Sprint 3 T1)。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esn_vla_uq.cli import main
from esn_vla_uq.data.schema import STATE_DIM, RolloutDataset
from esn_vla_uq.data.synthetic import generate_dataset
from esn_vla_uq.demo import DemoFrames, build_demo_frames, write_demo_animation
from esn_vla_uq.demo.frames import PANEL_LABEL
from esn_vla_uq.esn import ESNConfig
from esn_vla_uq.uncertainty import build_samples, split_samples

N_RESERVOIR = 60
N_EPISODES = 24


@pytest.fixture(scope="module")
def dataset() -> RolloutDataset:
    return generate_dataset(seed=0, n_episodes=N_EPISODES)


@pytest.fixture(scope="module")
def config() -> ESNConfig:
    return ESNConfig(n_reservoir=N_RESERVOIR, seed=0)


@pytest.fixture(scope="module")
def frames(dataset: RolloutDataset, config: ESNConfig) -> DemoFrames:
    return build_demo_frames(dataset, config, split_seed=0)


def test_demo_uses_a_failure_episode_from_the_test_split(
    dataset: RolloutDataset, frames: DemoFrames
) -> None:
    """学習・較正に使ったエピソードを描かないこと。

    fit や calibrate のエピソードで不確実性を見せるのはデモとして誠実でない。
    """
    assert not frames.success
    assert frames.failure_onset is not None
    split = split_samples(build_samples(dataset), seed=0)
    test_ids = {sample.episode_id for sample in split.test}
    assert frames.episode_id in test_ids


def test_panel_carries_the_stand_in_label(frames: DemoFrames) -> None:
    """実映像でないことを図の中に残すこと。"""
    assert frames.panel_label == PANEL_LABEL
    assert "stand-in" in frames.panel_label


def test_panel_is_the_proprioceptive_block(frames: DemoFrames) -> None:
    assert frames.panel.shape == (frames.n_steps, STATE_DIM)


def test_uncertainty_rises_after_failure_onset(frames: DemoFrames) -> None:
    """「バーが跳ねる」が演出ではなく実測であること。"""
    ratio = frames.uncertainty_ratio_after_onset()
    assert ratio is not None
    assert ratio > 1.5


def test_detection_lag_is_measured_and_non_negative(frames: DemoFrames) -> None:
    """遅れを隠さず数値で持つこと。

    不確実性は失敗の**あと**に立ち上がる。予兆ではないという事実をレポートと
    図に出すため、フレームデータの側で測る。
    """
    lag = frames.detection_lag_steps()
    assert lag is not None
    assert lag >= 0


def test_absolute_score_cannot_produce_a_rising_bar(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    """`absolute` では区間幅が定数なのでバーが跳ねない (比が 1.0)。"""
    absolute = build_demo_frames(dataset, config, score_kind="absolute", split_seed=0)
    ratio = absolute.uncertainty_ratio_after_onset()
    assert ratio == pytest.approx(1.0)
    assert float(np.std(absolute.uncertainty)) == pytest.approx(0.0, abs=1e-12)


def test_explicit_episode_id_outside_the_test_split_is_rejected(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    split = split_samples(build_samples(dataset), seed=0)
    fit_episode = split.fit[0].episode_id
    with pytest.raises(ValueError, match="テスト集合にありません"):
        build_demo_frames(dataset, config, split_seed=0, episode_id=fit_episode)


def test_animation_is_written(frames: DemoFrames, tmp_path: Path) -> None:
    path = write_demo_animation(frames, tmp_path / "anim" / "demo.gif", max_frames=6)
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.parametrize(("fps", "max_frames"), [(0, 10), (10, 0)])
def test_animation_rejects_non_positive_settings(
    frames: DemoFrames, tmp_path: Path, fps: int, max_frames: int
) -> None:
    with pytest.raises(ValueError):
        write_demo_animation(
            frames,
            tmp_path / "demo.gif",
            fps=max(fps, 1) if fps else 0,
            max_frames=max_frames,
        )


def test_cli_demo_writes_a_gif(tmp_path: Path) -> None:
    exit_code = main(
        [
            "demo",
            "--output-dir",
            str(tmp_path),
            "--n-reservoir",
            str(N_RESERVOIR),
            "--max-frames",
            "6",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "demo" / "uncertainty_demo.gif").exists()
