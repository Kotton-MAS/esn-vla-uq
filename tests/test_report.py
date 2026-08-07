"""`esn_vla_uq.diagnostics.report` と CLI `diagnose` のテスト (Sprint 1 T4)。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from esn_vla_uq import __version__
from esn_vla_uq.cli import main
from esn_vla_uq.cli.app import EXIT_ERROR
from esn_vla_uq.diagnostics import (
    REPORT_SCHEMA_VERSION,
    REPORT_SUBDIR,
    DiagnosticsReport,
    MemoryCapacityMeasurement,
    MemoryCapacityResult,
    linear_memory_capacity,
    run_diagnostics,
    spectral_radius,
    summarize_spectral,
    utc_timestamp,
    write_report,
)
from esn_vla_uq.esn import ESNConfig, Reservoir

N_RESERVOIR = 30
CLI_N_RESERVOIR = 50
TOP_LEVEL_KEYS = (
    "schema_version",
    "generated_at",
    "package_version",
    "numpy_version",
    "data_source",
    "seed",
    "n_inputs",
    "esn_config",
    "spectral",
    "esp",
    "memory_capacity",
)
ESP_KEYS = (
    "verdict",
    "sufficient_condition_met",
    "necessary_condition_met",
    "empirical_converged",
    "decay_rate",
    "largest_singular_value",
    "effective_spectral_radius",
    "final_distance",
    "tolerance",
    "n_initial_states",
    "n_steps",
    "zero_input",
)
MEMORY_CAPACITY_KEYS = (
    "total_mc",
    "mc_per_neuron",
    "memory_horizon",
    "n_delays",
    "per_delay",
    "n_inputs",
    "reservoir",
)


@pytest.fixture
def config() -> ESNConfig:
    return ESNConfig(n_reservoir=N_RESERVOIR, seed=0)


@pytest.fixture(scope="module")
def report() -> DiagnosticsReport:
    return run_diagnostics(ESNConfig(n_reservoir=N_RESERVOIR, seed=0), seed=0)


def test_report_metadata_fields(report: DiagnosticsReport) -> None:
    assert report.schema_version == REPORT_SCHEMA_VERSION
    assert report.package_version == __version__
    assert report.numpy_version == np.__version__
    assert report.data_source == "synthetic"
    assert report.seed == 0
    assert report.generated_at.endswith("Z")


def test_n_inputs_defaults_to_one(config: ESNConfig) -> None:
    report = run_diagnostics(config, seed=0, skip_memory_capacity=True)
    assert report.n_inputs == 1


def test_memory_capacity_n_inputs_matches_report_when_default(
    report: DiagnosticsReport,
) -> None:
    # 既定 n_inputs=1 ではメモリ容量も同じリザバー (D_u=1) で測る。
    assert report.memory_capacity is not None
    assert report.memory_capacity.n_inputs == 1
    assert report.memory_capacity.n_inputs == report.n_inputs
    assert report.memory_capacity.reservoir_label(report.n_inputs) == "shared"


def test_memory_capacity_n_inputs_is_none_when_skipped(config: ESNConfig) -> None:
    report = run_diagnostics(config, seed=0, skip_memory_capacity=True)
    assert report.memory_capacity is None


def test_memory_capacity_uses_separate_reservoir_when_n_inputs_not_one(
    config: ESNConfig,
) -> None:
    # n_inputs != 1 のとき、spectral/esp のリザバー (D_u=n_inputs) とは別の
    # D_u=1 リザバーでメモリ容量を測る。同一 seed でも n_inputs が違えば
    # 別の行列になるため (docs/design.md 3.3 節)、独立に構築した D_u=1
    # リザバーでの測定結果と一致するはずである。
    n_inputs = 3
    report = run_diagnostics(config, n_inputs=n_inputs, seed=0)
    assert report.n_inputs == n_inputs
    assert report.memory_capacity is not None
    assert report.memory_capacity.n_inputs == 1
    assert report.memory_capacity.reservoir_label(report.n_inputs) == "separate"

    reference_reservoir = Reservoir(config, 1)
    reference = linear_memory_capacity(reference_reservoir, seed=0)
    assert report.memory_capacity.result == reference

    # spectral/esp が D_u=n_inputs のリザバーを見ていることの裏付け。
    main_reservoir = Reservoir(config, n_inputs)
    assert report.spectral.spectral_radius == pytest.approx(
        spectral_radius(main_reservoir.W)
    )


def test_memory_capacity_measurement_rejects_non_scalar_n_inputs() -> None:
    # `MemoryCapacityMeasurement` は `diagnostics/__init__.py` で公開エクス
    # ポートされており、`run_diagnostics` を介さず単体で組み立てられる。
    # `linear_memory_capacity` の契約上 n_inputs は MEMORY_CAPACITY_INPUT_DIM
    # (=1) 以外を取れないため、それ以外を渡すと嘘の reservoir_label を出せて
    # しまう (元 finding)。構築時点で ValueError にする。
    dummy_result = MemoryCapacityResult(
        total_mc=0.0, per_delay=[0.0], memory_horizon=1, mc_per_neuron=0.0
    )
    with pytest.raises(ValueError, match="n_inputs"):
        MemoryCapacityMeasurement(result=dummy_result, n_inputs=2)


def test_to_dict_contains_all_sections(report: DiagnosticsReport) -> None:
    payload = report.to_dict()
    assert tuple(payload) == TOP_LEVEL_KEYS
    assert set(_section(payload, "esp")) == set(ESP_KEYS)
    assert set(_section(payload, "memory_capacity")) == set(MEMORY_CAPACITY_KEYS)
    assert set(_section(payload, "spectral")) == {
        "spectral_radius",
        "effective_spectral_radius",
    }
    assert set(_section(payload, "esn_config")) == {
        "n_reservoir",
        "spectral_radius",
        "input_scaling",
        "bias_scaling",
        "leak_rate",
        "density",
        "ridge_alpha",
        "washout",
        "input_passthrough",
        "use_reservoir",
        "seed",
    }


def test_to_dict_is_json_serializable(report: DiagnosticsReport) -> None:
    restored = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert restored["esp"]["verdict"] in {"esp_holds", "esp_likely", "esp_violated"}


def test_spectral_summary_matches_reservoir(config: ESNConfig) -> None:
    summary = summarize_spectral(Reservoir(config, 1))
    assert summary.spectral_radius == pytest.approx(config.spectral_radius, rel=1e-8)
    # leak_rate=1.0 では実効更新行列は W に退化する。
    assert summary.effective_spectral_radius == pytest.approx(
        summary.spectral_radius, rel=1e-12
    )


def test_same_seed_reproduces_report_except_timestamp(config: ESNConfig) -> None:
    # 受け入れ基準: 同一 seed の 2 回実行がタイムスタンプ以外完全一致。
    first = run_diagnostics(config, seed=0).to_dict()
    second = run_diagnostics(config, seed=0).to_dict()
    del first["generated_at"], second["generated_at"]
    assert first == second


def test_skip_memory_capacity_yields_null_section(config: ESNConfig) -> None:
    report = run_diagnostics(config, seed=0, skip_memory_capacity=True)
    assert report.memory_capacity is None
    assert report.to_dict()["memory_capacity"] is None


def test_seed_defaults_to_config_seed() -> None:
    report = run_diagnostics(
        ESNConfig(n_reservoir=N_RESERVOIR, seed=5), skip_memory_capacity=True
    )
    assert report.seed == 5


def test_log_summary_reports_all_three_esp_indicators(
    report: DiagnosticsReport, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        report.log_summary()
    messages = [record.message for record in caplog.records]
    assert len(messages) == 4
    esp_line = next(message for message in messages if message.startswith("esp:"))
    for indicator in ("sufficient", "necessary", "empirical", "decay_rate"):
        assert indicator in esp_line
    assert any(message.startswith("memory_capacity:") for message in messages)


def test_write_report_creates_json_under_output_dir(
    report: DiagnosticsReport, tmp_path: Path
) -> None:
    path = write_report(report, tmp_path)
    assert path.parent == tmp_path / REPORT_SUBDIR
    assert path.suffix == ".json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["generated_at"] == report.generated_at


def test_utc_timestamp_is_iso8601_utc() -> None:
    timestamp = utc_timestamp()
    assert timestamp.endswith("Z")
    assert "+00:00" not in timestamp
    assert timestamp[4] == "-" and timestamp[10] == "T"


def test_cli_diagnose_writes_loadable_report(tmp_path: Path) -> None:
    # 受け入れ基準: diagnose --n-reservoir 50 --output-dir <tmp> --seed 0 が exit 0、
    # JSON が生成され json.load 可能でキー欠落なし。
    exit_code = main(
        [
            "diagnose",
            "--n-reservoir",
            str(CLI_N_RESERVOIR),
            "--output-dir",
            str(tmp_path),
            "--seed",
            "0",
        ]
    )
    assert exit_code == 0
    reports = sorted((tmp_path / REPORT_SUBDIR).glob("*.json"))
    assert len(reports) == 1
    with reports[0].open(encoding="utf-8") as stream:
        payload = json.load(stream)
    assert tuple(payload) == TOP_LEVEL_KEYS
    assert set(_section(payload, "esp")) == set(ESP_KEYS)
    assert set(_section(payload, "memory_capacity")) == set(MEMORY_CAPACITY_KEYS)
    assert _section(payload, "esn_config")["n_reservoir"] == CLI_N_RESERVOIR
    assert payload["data_source"] == "synthetic"


def test_cli_diagnose_honours_skip_memory_capacity(tmp_path: Path) -> None:
    exit_code = main(
        [
            "diagnose",
            "--n-reservoir",
            "20",
            "--spectral-radius",
            "0.5",
            "--leak-rate",
            "0.5",
            "--output-dir",
            str(tmp_path),
            "--skip-memory-capacity",
        ]
    )
    assert exit_code == 0
    report_path = next((tmp_path / REPORT_SUBDIR).glob("*.json"))
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["memory_capacity"] is None
    assert _section(payload, "esn_config")["spectral_radius"] == 0.5
    assert _section(payload, "esn_config")["leak_rate"] == 0.5


def test_cli_diagnose_rejects_invalid_configuration(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """不正な設定は終了コード 1 とエラーログになる (S3)。

    以前は `ValueError` が `main()` を素通りし、トレースバックが stderr に出て
    いた。トレースバックはソースの絶対パスを含むため CLI 境界で捕まえる。
    どのパラメータが不正かはメッセージに残す。
    """
    with caplog.at_level(logging.ERROR):
        exit_code = main(
            ["diagnose", "--n-reservoir", "0", "--output-dir", str(tmp_path)]
        )
    assert exit_code == EXIT_ERROR
    assert "n_reservoir" in caplog.text
    assert "Traceback" not in caplog.text


def test_cli_diagnose_accepts_n_inputs(tmp_path: Path) -> None:
    exit_code = main(
        [
            "diagnose",
            "--n-reservoir",
            "20",
            "--n-inputs",
            "8",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "0",
        ]
    )
    assert exit_code == 0
    report_path = next((tmp_path / REPORT_SUBDIR).glob("*.json"))
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["n_inputs"] == 8
    # メモリ容量診断は D_u=1 を要求するため、report 全体の n_inputs (8) とは
    # 別のリザバーで測定したことが memory_capacity オブジェクト自身から
    # 自己記述的に分かる (M1)。
    memory_capacity_section = _section(payload, "memory_capacity")
    assert memory_capacity_section["n_inputs"] == 1
    assert memory_capacity_section["reservoir"] == "separate"


def _section(payload: dict[str, object], key: str) -> dict[str, object]:
    """JSON ペイロードから辞書セクションを型付きで取り出す。"""
    section = payload[key]
    assert isinstance(section, dict)
    return section
