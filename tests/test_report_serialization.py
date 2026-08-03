"""各結果型の `to_dict()` がフィールドを取りこぼさないことの検証 (A2)。

A2 の指摘は「辞書化が他モジュールのフィールドを手書き列挙しているため、
`EspResult` にフィールドを足しても JSON から黙って欠落する」だった。`to_dict()`
を各結果型へ移しただけでは、**列挙の場所が変わるだけで取りこぼしは起きうる**。
実装は `dataclasses.asdict` に寄せてあるが、それを人手で書き直せてしまう以上、
「全フィールドが辞書に現れる」ことをテストで固定しておく必要がある。

`dataclasses.fields` から期待値を導くため、フィールドを足せば自動的にこの
テストの対象になる。
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from esn_vla_uq.diagnostics import (
    DiagnosticsReport,
    MemoryCapacityMeasurement,
    MemoryCapacityResult,
    SpectralSummary,
    check_esp,
    linear_memory_capacity,
    run_diagnostics,
)
from esn_vla_uq.esn import ESNConfig, Reservoir

N_RESERVOIR = 25


@pytest.fixture
def report() -> DiagnosticsReport:
    return run_diagnostics(ESNConfig(n_reservoir=N_RESERVOIR, seed=0), seed=0)


def test_config_to_dict_covers_every_field() -> None:
    config = ESNConfig(n_reservoir=N_RESERVOIR, seed=0)
    assert set(config.to_dict()) == {field.name for field in fields(config)}


def test_esp_to_dict_covers_every_field() -> None:
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=0), 1)
    result = check_esp(reservoir, seed=0)
    assert set(result.to_dict()) == {field.name for field in fields(result)}


def test_memory_capacity_to_dict_covers_every_field() -> None:
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=0), 1)
    result = linear_memory_capacity(reservoir, seed=0)
    payload = result.to_dict()
    assert {field.name for field in fields(result)} <= set(payload)


def test_memory_capacity_to_dict_includes_derived_n_delays() -> None:
    """`n_delays` は property なので `asdict` には現れない。明示的に足している。"""
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=0), 1)
    result = linear_memory_capacity(reservoir, seed=0)
    assert result.to_dict()["n_delays"] == result.n_delays


def test_spectral_summary_to_dict_covers_every_field() -> None:
    summary = SpectralSummary(spectral_radius=0.9, effective_spectral_radius=0.9)
    assert set(summary.to_dict()) == {field.name for field in fields(summary)}


def test_measurement_to_dict_adds_context_to_result_fields(
    report: DiagnosticsReport,
) -> None:
    """測定コンテキストが測定値と同じオブジェクトに載ること。"""
    measurement = report.memory_capacity
    assert measurement is not None
    payload = measurement.to_dict(report.n_inputs)
    assert {field.name for field in fields(MemoryCapacityResult)} <= set(payload)
    assert payload["n_inputs"] == measurement.n_inputs
    assert payload["reservoir"] == "shared"


def test_measurement_reports_separate_reservoir() -> None:
    """`spectral`/`esp` と別のリザバーで測ったときにそう記録されること。"""
    reservoir = Reservoir(ESNConfig(n_reservoir=N_RESERVOIR, seed=0), 1)
    measurement = MemoryCapacityMeasurement(
        result=linear_memory_capacity(reservoir, seed=0), n_inputs=1
    )
    assert measurement.to_dict(report_n_inputs=8)["reservoir"] == "separate"


def test_report_to_dict_covers_every_field(report: DiagnosticsReport) -> None:
    """レポート自身のフィールドも取りこぼさないこと。

    `esn_config` / `spectral` / `esp` / `memory_capacity` は入れ子の辞書に
    なるが、キーとしては全フィールドが現れる。
    """
    assert set(report.to_dict()) == {field.name for field in fields(report)}


def test_report_to_dict_is_json_serializable(report: DiagnosticsReport) -> None:
    """`asdict` 由来の値がそのまま JSON にできること (numpy 型の混入検出)。"""
    round_tripped = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert round_tripped["esn_config"]["n_reservoir"] == N_RESERVOIR
