"""`esn_vla_uq.calibration` と CLI `calibrate` のテスト (Sprint 2 T4-T5)。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esn_vla_uq.calibration import (
    ECE_DEFINITION,
    ReliabilityCurve,
    conformal_coverage,
    detection_auroc,
    rank_data,
    reliability_curve,
    write_reliability_diagram,
)
from esn_vla_uq.calibration.report import REPORT_SUBDIR, CalibrationReport
from esn_vla_uq.calibration.runner import (
    ABSOLUTE_SCORE_CAVEAT,
    SYNTHETIC_DATA_CAVEAT,
    run_calibration,
)
from esn_vla_uq.cli import main
from esn_vla_uq.data.schema import RolloutDataset
from esn_vla_uq.data.synthetic import generate_dataset
from esn_vla_uq.esn import ESNConfig

N_RESERVOIR = 60
N_EPISODES = 24
N_SPLITS = 8


@pytest.fixture(scope="module")
def dataset() -> RolloutDataset:
    return generate_dataset(seed=0, n_episodes=N_EPISODES)


@pytest.fixture(scope="module")
def report(dataset: RolloutDataset) -> CalibrationReport:
    return run_calibration(
        dataset,
        ESNConfig(n_reservoir=N_RESERVOIR, seed=0),
        alpha=0.1,
        n_splits=N_SPLITS,
    )


# --- 順位付けと AUROC -------------------------------------------------------


def test_rank_data_averages_ties() -> None:
    ranks = rank_data(np.array([1.0, 1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(ranks, [1.5, 1.5, 3.0])


def test_auroc_of_constant_scores_is_exactly_one_half() -> None:
    """定数スコアの AUROC は 0.5。

    同順位を平均順位で扱わないと、**元の並び順**に依存した意味のない値が出る
    (実測: 本来 0.5 のところ 0.67)。`absolute` スコアは常にこの状況になるため、
    ここが崩れると「absolute でも失敗を検知できている」という誤った結論になる。
    """
    scores = np.ones(100, dtype=np.float64)
    positive = np.zeros(100, dtype=np.bool_)
    positive[:20] = True
    assert detection_auroc(scores, positive) == pytest.approx(0.5)
    # 並び順を変えても 0.5 のまま。
    assert detection_auroc(scores, positive[::-1]) == pytest.approx(0.5)


def test_auroc_is_one_for_perfectly_separating_scores() -> None:
    scores = np.arange(10, dtype=np.float64)
    positive = np.array([False] * 5 + [True] * 5, dtype=np.bool_)
    assert detection_auroc(scores, positive) == pytest.approx(1.0)


def test_auroc_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="両クラス"):
        detection_auroc(np.arange(4, dtype=np.float64), np.zeros(4, dtype=np.bool_))


# --- 被覆率と reliability curve ---------------------------------------------


def test_coverage_from_scores_matches_the_definition() -> None:
    """較正スコアの分位点以下である割合が被覆率になる。"""
    calibration = np.arange(1, 21, dtype=np.float64)
    test = np.array([1.0, 5.0, 100.0], dtype=np.float64)
    # n=20, alpha=0.1 -> index = ceil(21*0.9) = 19 -> 19 番目の値 = 19.0
    assert conformal_coverage(calibration, test, 0.1) == pytest.approx(2.0 / 3.0)


def test_reliability_curve_is_monotone_in_the_nominal_level() -> None:
    """名目水準を上げれば経験被覆率は下がらない。"""
    rng = np.random.default_rng(0)
    calibration = rng.normal(size=200)
    test = rng.normal(size=300)
    curve = reliability_curve(calibration, test)
    empirical = np.asarray(curve.empirical)
    assert np.all(np.diff(empirical) >= -1e-12)


def test_ece_is_zero_for_a_perfectly_calibrated_curve() -> None:
    curve = ReliabilityCurve(nominal=(0.5, 0.9), empirical=(0.5, 0.9))
    assert curve.expected_calibration_error() == pytest.approx(0.0)


def test_curve_dict_records_the_ece_definition() -> None:
    """分類の ECE と混同されないよう定義を JSON に残す。"""
    curve = ReliabilityCurve(nominal=(0.9,), empirical=(0.8,))
    assert curve.to_dict()["ece_definition"] == ECE_DEFINITION


# --- レポート ---------------------------------------------------------------


def test_report_coverage_matches_nominal_on_average(
    report: CalibrationReport,
) -> None:
    assert report.coverage.nominal == pytest.approx(0.9)
    assert report.coverage.mean == pytest.approx(0.9, abs=0.07)
    assert report.coverage.n_splits == N_SPLITS
    assert len(report.coverage.per_split) == N_SPLITS


def test_report_records_the_spread_not_just_the_mean(
    report: CalibrationReport,
) -> None:
    """単一分割では代表値にならないため散らばりを必ず残す。"""
    assert report.coverage.std > 0.0
    assert report.coverage.minimum < report.coverage.maximum


def test_reliability_curve_agrees_with_the_aggregate_coverage(
    report: CalibrationReport,
) -> None:
    """曲線上の 0.9 の点が集約被覆率と一致すること (同じ集約方法)。"""
    index = report.reliability.nominal.index(0.9)
    assert report.reliability.empirical[index] == pytest.approx(
        report.coverage.mean, abs=1e-9
    )


def test_report_always_states_the_data_source(report: CalibrationReport) -> None:
    assert report.data_source == "synthetic"
    assert SYNTHETIC_DATA_CAVEAT in report.caveats


def test_absolute_score_report_carries_its_own_caveat(
    dataset: RolloutDataset,
) -> None:
    absolute = run_calibration(
        dataset,
        ESNConfig(n_reservoir=N_RESERVOIR, seed=0),
        score_kind="absolute",
        n_splits=2,
    )
    assert ABSOLUTE_SCORE_CAVEAT in absolute.caveats


def test_normalized_detects_failures_while_absolute_cannot(
    dataset: RolloutDataset,
) -> None:
    """Sprint 2 の中心的な主張。

    `absolute` は被覆率としては正しいが区間幅が定数なので、失敗検知 AUROC は
    定義上 0.5 になる。要件書のデモ GIF が求める「失敗直前に跳ねるバー」は
    `normalized` でしか成立しない。
    """
    config = ESNConfig(n_reservoir=N_RESERVOIR, seed=0)
    absolute = run_calibration(
        dataset, config, score_kind="absolute", n_splits=N_SPLITS
    )
    normalized = run_calibration(
        dataset, config, score_kind="normalized", n_splits=N_SPLITS
    )
    # 合成データは failure_onset を持つので細かいラベルが使われる。
    assert absolute.detection.label == "failure_onset"
    assert absolute.detection.mean_auroc == pytest.approx(0.5, abs=1e-12)
    assert absolute.detection.std_auroc == pytest.approx(0.0, abs=1e-12)
    assert normalized.detection.mean_auroc is not None
    # 観測量ベースの難易度に変えて 0.37 -> 0.87 になった。0.75 は実測 0.869 に
    # 対する余裕を見た下限で、学習型の推定に戻すと (0.28-0.61) 必ず落ちる。
    assert normalized.detection.mean_auroc > 0.75


def test_normalized_keeps_coverage_close_to_nominal(
    dataset: RolloutDataset,
) -> None:
    """検知能力と引き換えに被覆率を壊していないこと。

    `normalized` の `sigma(x)` は残差の推定ではなく観測量なので、被覆率は
    `absolute` よりやや名目を下回る (本番規模の実測: 0.864 対 0.903)。
    許容範囲に収まっていることだけを見張る。**どちらが ECE で優れるかは
    データ規模に依存する**ため (テスト規模では両者ほぼ同じ)、大小関係は
    テストで固定しない。
    """
    normalized = run_calibration(
        dataset,
        ESNConfig(n_reservoir=N_RESERVOIR, seed=0),
        score_kind="normalized",
        n_splits=N_SPLITS,
    )
    assert normalized.coverage.mean == pytest.approx(0.9, abs=0.08)


def test_report_is_json_serializable(report: CalibrationReport) -> None:
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["coverage"]["nominal"] == pytest.approx(0.9)
    assert payload["detection"]["mean_auroc"] >= 0.0
    assert payload["detection"]["label"] == "failure_onset"


def test_same_seed_reproduces_the_report(dataset: RolloutDataset) -> None:
    config = ESNConfig(n_reservoir=N_RESERVOIR, seed=0)
    first = run_calibration(dataset, config, n_splits=3).to_dict()
    second = run_calibration(dataset, config, n_splits=3).to_dict()
    del first["generated_at"], second["generated_at"]
    assert first == second


def test_n_splits_must_be_positive(dataset: RolloutDataset) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        run_calibration(dataset, ESNConfig(n_reservoir=N_RESERVOIR), n_splits=0)


# --- 作図 -------------------------------------------------------------------


def test_reliability_diagram_is_written(
    report: CalibrationReport, tmp_path: Path
) -> None:
    path = write_reliability_diagram(report.reliability, tmp_path / "fig" / "r.png")
    assert path.exists()
    assert path.stat().st_size > 0


# --- CLI --------------------------------------------------------------------


def test_cli_calibrate_writes_a_report(tmp_path: Path) -> None:
    exit_code = main(
        [
            "calibrate",
            "--output-dir",
            str(tmp_path),
            "--n-reservoir",
            str(N_RESERVOIR),
            "--n-splits",
            "3",
        ]
    )
    assert exit_code == 0
    reports = list((tmp_path / REPORT_SUBDIR).glob("*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["data_source"] == "synthetic"
    assert payload["conformal"]["score_kind"] == "normalized"
    assert payload["split"]["strategy"] == "within_task"


def test_cli_calibrate_writes_the_diagram_when_requested(tmp_path: Path) -> None:
    exit_code = main(
        [
            "calibrate",
            "--output-dir",
            str(tmp_path),
            "--n-reservoir",
            str(N_RESERVOIR),
            "--n-splits",
            "2",
            "--diagram",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / REPORT_SUBDIR / "reliability.png").exists()


def test_cli_across_task_split_records_the_warning(tmp_path: Path) -> None:
    """タスク間 split を選ぶと保証が弱いことがレポートに残ること。"""
    exit_code = main(
        [
            "calibrate",
            "--output-dir",
            str(tmp_path),
            "--n-reservoir",
            str(N_RESERVOIR),
            "--n-splits",
            "2",
            "--split",
            "across_task",
        ]
    )
    assert exit_code == 0
    payload = json.loads(
        next((tmp_path / REPORT_SUBDIR).glob("*.json")).read_text(encoding="utf-8")
    )
    assert any("交換可能性" in caveat for caveat in payload["caveats"])


def test_inverted_detection_is_reported_as_a_finding(
    dataset: RolloutDataset,
) -> None:
    """AUROC が 0.5 を下回ったら「向きが逆」だと明示すること。

    実 openpi ログでは 0.374 になる。黙って数値だけ出すと「効いていない」と
    読まれるが、実際には**逆向きに効いている**。不具合ではなく結果なので、
    読み手が誤解しないよう注意書きを付ける (docs/design.md 10.11 節)。
    """
    from esn_vla_uq.calibration.runner import (
        INVERTED_DETECTION_CAVEAT,
        INVERTED_DETECTION_THRESHOLD,
        _caveats,
    )

    inverted = _caveats(None, "normalized", "episode_success", "openpi", 0.374)
    assert INVERTED_DETECTION_CAVEAT in inverted

    healthy = _caveats(None, "normalized", "failure_onset", "synthetic", 0.87)
    assert INVERTED_DETECTION_CAVEAT not in healthy

    # 閾値のすぐ上は「偶然の揺らぎ」として扱い、注意書きを出さない。
    borderline = _caveats(
        None, "normalized", "episode_success", "openpi", INVERTED_DETECTION_THRESHOLD
    )
    assert INVERTED_DETECTION_CAVEAT not in borderline


def test_synthetic_report_does_not_claim_inverted_detection(
    report: CalibrationReport,
) -> None:
    """合成データ (AUROC 0.87) では反相関の注意書きが出ないこと。"""
    from esn_vla_uq.calibration.runner import INVERTED_DETECTION_CAVEAT

    assert INVERTED_DETECTION_CAVEAT not in report.caveats
