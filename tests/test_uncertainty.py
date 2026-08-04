"""`esn_vla_uq.uncertainty` のテスト (Sprint 2 T1-T3)。"""

from __future__ import annotations

import numpy as np
import pytest

from esn_vla_uq.data.features import CHUNK_FEATURE_NAMES
from esn_vla_uq.data.schema import ACTION_DIM, STATE_DIM, RolloutDataset
from esn_vla_uq.data.synthetic import generate_dataset
from esn_vla_uq.esn import ESNConfig
from esn_vla_uq.uncertainty import (
    ACROSS_TASK_WARNING,
    SplitConformalPredictor,
    build_samples,
    conformal_quantile_index,
    split_samples,
    stack_failure_labels,
)

N_RESERVOIR = 60
N_EPISODES = 24


@pytest.fixture(scope="module")
def dataset() -> RolloutDataset:
    return generate_dataset(seed=0, n_episodes=N_EPISODES)


@pytest.fixture(scope="module")
def config() -> ESNConfig:
    return ESNConfig(n_reservoir=N_RESERVOIR, seed=0)


# --- T1: 予測タスクの構築 ---------------------------------------------------


def test_samples_drop_one_step_per_episode(dataset: RolloutDataset) -> None:
    """目標が存在しない最終ステップを落とし `T_i - 1` 標本になる。"""
    samples = build_samples(dataset)
    assert len(samples) == dataset.n_episodes
    for sample, episode in zip(samples, dataset.episodes, strict=True):
        assert sample.n_samples == episode.n_steps - 1


def test_input_includes_state_action_and_chunk_features(
    dataset: RolloutDataset,
) -> None:
    """既定入力は固有受容感覚 + 行動 + チャンク由来の要約量 (要件書の入力定義)。"""
    samples = build_samples(dataset)
    assert samples[0].n_inputs == STATE_DIM + ACTION_DIM + len(CHUNK_FEATURE_NAMES)
    assert samples[0].n_targets == ACTION_DIM
    assert samples[0].difficulty_column is not None


def test_state_action_feature_has_no_difficulty_column(
    dataset: RolloutDataset,
) -> None:
    """チャンクを入れない構成では区間幅を変調する観測量が無い。"""
    samples = build_samples(dataset, feature="state_action")
    assert samples[0].n_inputs == STATE_DIM + ACTION_DIM
    assert samples[0].difficulty_column is None


def test_target_is_the_next_action_within_the_same_episode(
    dataset: RolloutDataset,
) -> None:
    """目標が同一エピソードの次ステップの action であること (境界を跨がない)。"""
    samples = build_samples(dataset)
    for sample, episode in zip(samples, dataset.episodes, strict=True):
        np.testing.assert_allclose(
            sample.targets, episode.action[1:].astype(np.float64)
        )
        # 入力側の action 部分は 1 つ前のステップ。
        action_block = sample.inputs[:, STATE_DIM : STATE_DIM + ACTION_DIM]
        np.testing.assert_allclose(action_block, episode.action[:-1].astype(np.float64))


def test_failure_labels_start_at_failure_onset(dataset: RolloutDataset) -> None:
    samples = build_samples(dataset)
    failing = next(sample for sample in samples if sample.failure_onset is not None)
    labels = failing.after_failure_onset()
    onset = failing.failure_onset
    assert onset is not None
    assert not labels[failing.target_steps < onset].any()
    assert labels[failing.target_steps >= onset].all()


def test_success_episodes_have_no_failure_labels(dataset: RolloutDataset) -> None:
    samples = build_samples(dataset)
    for sample in samples:
        if sample.success:
            assert not sample.after_failure_onset().any()


# --- T2: 較正データ分割 -----------------------------------------------------


def test_split_parts_are_disjoint(dataset: RolloutDataset) -> None:
    """fit/calibrate/test がエピソード単位で重ならないこと。

    重なると残差が楽観的になり、区間が過小になって被覆率が名目を下回る。
    """
    split = split_samples(build_samples(dataset), seed=0)
    ids = [
        {sample.episode_id for sample in part}
        for part in (split.fit, split.calibrate, split.test)
    ]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert sum(len(group) for group in ids) == dataset.n_episodes


def test_within_task_split_keeps_every_task_in_every_part(
    dataset: RolloutDataset,
) -> None:
    """タスク内 split では各パートのタスク構成が揃う (交換可能性の根拠)。"""
    split = split_samples(build_samples(dataset), strategy="within_task", seed=0)
    tasks = [
        {sample.task_name for sample in part}
        for part in (split.fit, split.calibrate, split.test)
    ]
    assert tasks[0] == tasks[1] == tasks[2]
    assert split.warning is None


def test_across_task_split_separates_tasks_and_warns(
    dataset: RolloutDataset,
) -> None:
    """タスク間 split ではタスクが重ならず、警告が付くこと。"""
    split = split_samples(build_samples(dataset), strategy="across_task", seed=0)
    calibrate_tasks = {sample.task_name for sample in split.calibrate}
    test_tasks = {sample.task_name for sample in split.test}
    assert calibrate_tasks.isdisjoint(test_tasks)
    assert split.warning == ACROSS_TASK_WARNING
    assert split.to_dict()["exchangeability_warning"] == ACROSS_TASK_WARNING


def test_same_seed_reproduces_the_split(dataset: RolloutDataset) -> None:
    samples = build_samples(dataset)
    first = split_samples(samples, seed=3)
    second = split_samples(samples, seed=3)
    assert [s.episode_id for s in first.test] == [s.episode_id for s in second.test]


def test_unknown_strategy_is_rejected(dataset: RolloutDataset) -> None:
    with pytest.raises(ValueError, match="未知の分割方針"):
        split_samples(build_samples(dataset), strategy="by_vibes")  # type: ignore[arg-type]


def test_ratios_that_empty_the_test_part_are_rejected(
    dataset: RolloutDataset,
) -> None:
    with pytest.raises(ValueError, match="1 未満"):
        split_samples(build_samples(dataset), fit_ratio=0.8, calibrate_ratio=0.3)


# --- T3: conformal ----------------------------------------------------------


@pytest.mark.parametrize(
    ("n_calibration", "alpha", "expected"),
    [(9, 0.1, 9), (19, 0.1, 18), (99, 0.05, 95), (10, 0.5, 6)],
)
def test_quantile_index_uses_the_finite_sample_formula(
    n_calibration: int, alpha: float, expected: int
) -> None:
    """``ceil((n+1)(1-alpha))``。単純な経験分位点では被覆率保証が出ない。"""
    assert conformal_quantile_index(n_calibration, alpha) == expected


def test_insufficient_calibration_samples_raise_instead_of_infinite_interval() -> None:
    """水準に対し標本が足りないときは黙って無限区間にせずエラーにする。

    無限区間を返すとレポート上は被覆率 100% の「良い」結果に見えてしまう。
    """
    with pytest.raises(ValueError, match="少なすぎて"):
        conformal_quantile_index(5, 0.1)


def test_absolute_score_gives_constant_interval_width(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    """`absolute` の区間幅は入力に依存しない定数。"""
    split = split_samples(build_samples(dataset), seed=0)
    predictor = SplitConformalPredictor(config, alpha=0.1, score_kind="absolute")
    predictor.fit(split.fit).calibrate(split.calibrate)
    intervals = predictor.predict_intervals(split.test)
    assert float(intervals.uncertainty.std()) == pytest.approx(0.0, abs=1e-12)


def test_normalized_score_gives_varying_interval_width(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    """`normalized` の区間幅はステップごとに変わる (不確実性スコアになる)。"""
    split = split_samples(build_samples(dataset), seed=0)
    predictor = SplitConformalPredictor(config, alpha=0.1, score_kind="normalized")
    predictor.fit(split.fit).calibrate(split.calibrate)
    intervals = predictor.predict_intervals(split.test)
    assert float(intervals.uncertainty.std()) > 0.0


def test_normalized_interval_width_stays_on_the_scale_of_the_data(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    """スケール推定の外挿で区間幅が発散しないこと。

    `log sigma` の丸め込みが無いと、`exp` が発散して平均幅が行動の実スケール
    (0.01 程度) の 1000 倍以上になった (実測: 平均幅 44)。
    """
    split = split_samples(build_samples(dataset), seed=0)
    predictor = SplitConformalPredictor(config, alpha=0.1, score_kind="normalized")
    predictor.fit(split.fit).calibrate(split.calibrate)
    intervals = predictor.predict_intervals(split.test)
    targets = predictor.stacked_targets(split.test)
    target_scale = float(np.abs(targets).max())
    assert float(intervals.uncertainty.mean()) < 100.0 * target_scale


def test_coverage_matches_the_nominal_level_on_average(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    """複数分割の平均被覆率が名目値の近傍に入ること。

    **単一分割では検証できない。** 同一エピソード内のステップは強く相関する
    ため、被覆率の分散を決めるのはエピソード数であってステップ数ではない。
    単一分割の実測値は 0.63〜1.00 まで振れる。
    """
    samples = build_samples(dataset)
    alpha = 0.1
    coverages = []
    for seed in range(15):
        split = split_samples(samples, seed=seed)
        predictor = SplitConformalPredictor(config, alpha=alpha)
        predictor.fit(split.fit).calibrate(split.calibrate)
        intervals = predictor.predict_intervals(split.test)
        targets = predictor.stacked_targets(split.test)
        coverages.append(float(intervals.covers(targets).mean()))
    assert float(np.mean(coverages)) == pytest.approx(1.0 - alpha, abs=0.06)


def test_predict_before_calibrate_is_an_error(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    split = split_samples(build_samples(dataset), seed=0)
    predictor = SplitConformalPredictor(config)
    with pytest.raises(RuntimeError, match="calibrate"):
        predictor.predict_intervals(split.test)


def test_failure_labels_stack_in_row_order(dataset: RolloutDataset) -> None:
    """連結したラベルが区間の連結順と一致すること。"""
    samples = build_samples(dataset)
    labels = stack_failure_labels(samples)
    assert labels.shape[0] == sum(sample.n_samples for sample in samples)


# --- washout と入力検証 -----------------------------------------------------


def test_washout_drops_the_leading_samples_of_every_episode(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    """washout は区間ごとに先頭から落とす (連結後の先頭だけではない)。"""
    split = split_samples(build_samples(dataset), seed=0)
    washout = 5
    predictor = SplitConformalPredictor(config, washout=washout)
    predictor.fit(split.fit).calibrate(split.calibrate)
    targets = predictor.stacked_targets(split.test)
    expected = sum(sample.n_samples - washout for sample in split.test)
    assert targets.shape[0] == expected


def test_washout_longer_than_an_episode_is_rejected(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    split = split_samples(build_samples(dataset), seed=0)
    shortest = min(sample.n_samples for sample in split.fit)
    predictor = SplitConformalPredictor(config, washout=shortest)
    with pytest.raises(ValueError, match="washout が標本数以上"):
        predictor.fit(split.fit)


def test_negative_washout_is_rejected(config: ESNConfig) -> None:
    with pytest.raises(ValueError, match="washout"):
        SplitConformalPredictor(config, washout=-1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_outside_the_open_unit_interval_is_rejected(
    config: ESNConfig, alpha: float
) -> None:
    with pytest.raises(ValueError, match="alpha"):
        SplitConformalPredictor(config, alpha=alpha)


def test_quantile_before_calibrate_is_an_error(config: ESNConfig) -> None:
    predictor = SplitConformalPredictor(config)
    with pytest.raises(RuntimeError, match="calibrate"):
        _ = predictor.quantile
    with pytest.raises(RuntimeError, match="calibrate"):
        _ = predictor.n_calibration


def test_fit_on_empty_samples_is_rejected(config: ESNConfig) -> None:
    predictor = SplitConformalPredictor(config)
    with pytest.raises(ValueError, match="1 件以上"):
        predictor.fit([])


def test_unknown_score_kind_is_rejected() -> None:
    from esn_vla_uq.uncertainty.nonconformity import fit_score_model

    residuals = np.zeros((4, 2), dtype=np.float64)
    states = np.zeros((4, 3), dtype=np.float64)
    inputs = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="未知のスコア"):
        fit_score_model("magic", residuals, states, inputs)  # type: ignore[arg-type]


def test_across_task_split_requires_at_least_three_tasks() -> None:
    single_task = generate_dataset(seed=1, n_episodes=6)
    samples = [
        sample
        for sample in build_samples(single_task)
        if sample.task_name == single_task.episodes[0].task_name
    ]
    with pytest.raises(ValueError, match="3 つ以上のタスク"):
        split_samples(samples, strategy="across_task", seed=0)


def test_stack_helpers_reject_empty_input() -> None:
    from esn_vla_uq.uncertainty.targets import stack_targets

    with pytest.raises(ValueError, match="1 件以上"):
        stack_targets([])
    with pytest.raises(ValueError, match="1 件以上"):
        stack_failure_labels([])


# --- 難易度の有界性 (実データで踏んだ問題への対処) -------------------------


def test_difficulty_is_bounded_by_construction(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    """難易度 `g(x)` の値域が観測量の分布に依存しないこと。

    以前は `g = exp(中心化した log 観測量)` で、値域が観測量のレンジ次第だった。
    合成データでは log 分散のレンジが約 83 倍で収まっていたが、実 openpi ログでは
    約 17,000 倍あり、`g` が 528 まで振れて平均区間幅が行動スケールの 1,858 倍に
    なった。順位へ写すことで値域を構造的に閉じる。
    """
    from esn_vla_uq.uncertainty.nonconformity import DIFFICULTY_SPREAD

    split = split_samples(build_samples(dataset), seed=0)
    predictor = SplitConformalPredictor(config, score_kind="normalized")
    predictor.fit(split.fit).calibrate(split.calibrate)

    states, inputs, _targets = predictor._design(split.test)
    score_model = predictor._score_model
    assert score_model is not None
    difficulty = score_model.difficulty(states, inputs)
    assert float(difficulty.min()) >= DIFFICULTY_SPREAD**-0.5 - 1e-12
    assert float(difficulty.max()) <= DIFFICULTY_SPREAD**0.5 + 1e-12


def test_detection_auroc_is_invariant_to_the_difficulty_spread(
    dataset: RolloutDataset, config: ESNConfig
) -> None:
    """`spread` を変えても失敗検知 AUROC が変わらないこと。

    順位への写像は単調変換であり、AUROC は順位だけで決まる。したがって幅の
    値域をどう選んでも検知性能は 1 ビットも変わらない。**この性質があるから
    こそ、spread は被覆率だけを見て決めてよい。**
    """
    from esn_vla_uq.calibration.metrics import detection_auroc
    from esn_vla_uq.uncertainty.nonconformity import fit_score_model
    from esn_vla_uq.uncertainty.targets import detection_labels

    split = split_samples(build_samples(dataset), seed=0)
    labels, _kind = detection_labels(split.test)

    scores = []
    for spread in (2.0, 8.0):
        predictor = SplitConformalPredictor(config, score_kind="normalized")
        predictor.fit(split.fit)
        states, inputs, targets = predictor._design(split.fit)
        readout = predictor._readout
        assert readout is not None
        residuals = targets - readout.predict(states, inputs)
        predictor._score_model = fit_score_model(
            "normalized",
            residuals,
            states,
            inputs,
            difficulty_column=split.fit[0].difficulty_column,
            spread=spread,
        )
        predictor.calibrate(split.calibrate)
        intervals = predictor.predict_intervals(split.test)
        scores.append(detection_auroc(intervals.uncertainty, labels))

    assert scores[0] == pytest.approx(scores[1], abs=1e-12)
