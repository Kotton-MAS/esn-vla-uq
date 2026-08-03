"""`esn_vla_uq.provenance` のテストと、定義が一本化されていることの検証 (A4)。"""

from __future__ import annotations

from types import ModuleType
from typing import cast, get_args

from esn_vla_uq.data import schema
from esn_vla_uq.diagnostics import report
from esn_vla_uq.provenance import (
    SUPPORTED_SOURCES,
    SYNTHETIC_DATA_SOURCE,
    DataSource,
)


def _module_attribute(module: ModuleType, name: str) -> object:
    """モジュール属性を取り出す。

    `mypy --strict` の `no_implicit_reexport` は「他モジュールから import した
    だけの名前」への属性アクセスを禁じる。本テストが確認したいのはまさに
    「再エクスポートされた名前が同一オブジェクトを指すこと」なので、
    `__dict__` 経由で取り出す。
    """
    return cast("object", module.__dict__[name])


def test_supported_sources_matches_type_arguments() -> None:
    """実行時の検証に使うタプルが型注釈からずれないこと。"""
    assert get_args(DataSource) == SUPPORTED_SOURCES


def test_synthetic_is_a_supported_source() -> None:
    assert SYNTHETIC_DATA_SOURCE in SUPPORTED_SOURCES


def test_data_and_diagnostics_share_one_definition() -> None:
    """データ層と診断層が**同一の** `DataSource` を指すこと (A4)。

    以前は両者が独立に同じ Literal を定義しており、出所を 1 つ足したときに
    片方だけ古いまま残っても型検査は何も言わなかった。同一性を固定する。
    """
    assert _module_attribute(schema, "DataSource") is DataSource
    assert _module_attribute(report, "DataSource") is DataSource


def test_schema_validation_uses_the_shared_tuple() -> None:
    assert _module_attribute(schema, "SUPPORTED_SOURCES") is SUPPORTED_SOURCES
