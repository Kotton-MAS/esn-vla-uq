"""モジュール間の依存の向きを固定するテスト (A1 / S7)。

「openpi をランタイム依存に含めない」という設計は、Sprint 2 で openpi ログ
パーサを足したときに、それを import してよいモジュールとしてはいけない
モジュールの線引きが守られてはじめて成立する。docstring に書いた規約は
守られたかどうかを機械的に確認できないため、依存の向きそのものをテストする。

**保証の範囲について。** `esn_vla_uq.data.io` のようなサブモジュールを import
すると、Python は先に `esn_vla_uq/data/__init__.py` を実行する。その
`__init__.py` は公開 API として `SyntheticRolloutSource` などを再エクスポート
しているため、`data` 配下のどれか 1 つを import した時点で合成データ生成器も
ロードされる。したがって「Protocol だけを import すれば具象は一切ロード
されない」という保証は**成立しない**。A1 / S7 の分離が実際に守っているのは
次の 2 点であり、テストもその範囲で書く。

1. `io.py` / `invariants.py` / `sources/base.py` が具象供給元を **import 文と
   して持たない**。これらに手を入れる際、具象へ依存を足せば静的検査で落ちる。
2. **任意依存を持つ具象 (openpi) はどこからも巻き込まれない。** 合成データ
   生成器は numpy と本パッケージだけに依存するためロードされても実害が無く、
   問題になるのは openpi のような外部依存を持つ供給元だけである。この区別を
   テストの対象にする。
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import esn_vla_uq

PACKAGE_ROOT = Path(esn_vla_uq.__file__).parent

CONCRETE_SOURCE_MODULES = (
    "esn_vla_uq.data.synthetic",
    "esn_vla_uq.data.sources.synthetic",
    "esn_vla_uq.data.sources.openpi",
)
"""具象の供給元 (実データ形式を知っているモジュール)。"""

OPTIONAL_DEPENDENCY_MODULES = ("esn_vla_uq.data.sources.openpi",)
"""任意依存 (openpi) を必要とする供給元。

Sprint 2 で追加される。存在しないうちからここに書いておくことで、追加された
時点で本テストが自動的に適用される。それまでは意図的に空振りする番人であり、
現時点で通ることを「検証済み」とは読まないこと。
"""


def _imported_modules(relative_path: str) -> set[str]:
    """モジュールがトップレベルで import している絶対モジュール名を集める。"""
    tree = ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


@pytest.mark.parametrize(
    "relative_path",
    ["data/io.py", "data/invariants.py", "data/sources/base.py"],
)
def test_module_does_not_import_concrete_sources(relative_path: str) -> None:
    """入出力・不変条件・Protocol は具象供給元を import しない。

    `io.py` は以前 `data/synthetic.py` を直接 import しており、
    `source == "openpi"` の分岐を足せば openpi パーサに依存する構造だった
    (S7)。`sources/base.py` は Protocol だけを提供し、具象を巻き込まない
    (A1)。
    """
    forbidden = _imported_modules(relative_path).intersection(CONCRETE_SOURCE_MODULES)
    assert not forbidden, (
        f"{relative_path} が具象供給元を import しています: {forbidden}"
    )


@pytest.mark.parametrize(
    "relative_path",
    ["data/__init__.py", "data/sources/__init__.py"],
)
def test_package_init_does_not_import_optional_dependency_sources(
    relative_path: str,
) -> None:
    """パッケージ `__init__` は任意依存を持つ供給元を eager import しない。

    `__init__.py` での再エクスポートは、その配下のどれか 1 つを import した
    だけで対象モジュールをロードさせる。openpi 供給元をここに足すと、
    `esn_vla_uq.data.schema` を触るだけのコードにまで openpi が必要になる
    (A1)。`SyntheticRolloutSource` は任意依存を持たないため対象外。
    """
    forbidden = _imported_modules(relative_path).intersection(
        OPTIONAL_DEPENDENCY_MODULES
    )
    assert not forbidden, (
        f"{relative_path} が任意依存の供給元を再エクスポートしています: {forbidden}"
    )


def test_invariants_depends_only_on_schema_and_provenance() -> None:
    """`invariants.py` の依存を最小に保つ (S7)。

    ここが具象を引き込み始めると、`io.py` が `invariants.py` 経由で間接的に
    具象へ依存することになり、切り出した意味が無くなる。
    """
    package_imports = {
        name
        for name in _imported_modules("data/invariants.py")
        if name.startswith("esn_vla_uq")
    }
    allowed = {
        "esn_vla_uq.data.schema",
        "esn_vla_uq.data.schema.RolloutDataset",
        "esn_vla_uq.provenance",
        "esn_vla_uq.provenance.DataSource",
    }
    assert package_imports <= allowed, f"想定外の依存: {package_imports - allowed}"


def _modules_loaded_by(import_statement: str) -> set[str]:
    """新しいインタプリタで import し、ロードされたモジュール名を返す。"""
    script = (
        f"{import_statement}\n"
        "import sys, json\n"
        "print(json.dumps([m for m in sys.modules if m.startswith('esn_vla_uq')]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded: list[str] = json.loads(completed.stdout)
    return set(loaded)


def test_importing_data_does_not_load_optional_dependency_sources() -> None:
    """`esn_vla_uq.data` を import しても openpi 供給元がロードされないこと。

    静的な import 検査と違い、再エクスポートの連鎖など間接的な経路も捉える。
    Sprint 2 で openpi 供給元が加わるまでは空振りする (本モジュール docstring
    の `OPTIONAL_DEPENDENCY_MODULES` を参照)。
    """
    loaded = _modules_loaded_by("import esn_vla_uq.data")
    assert "esn_vla_uq.data.io" in loaded
    assert not loaded.intersection(OPTIONAL_DEPENDENCY_MODULES)


def test_diagnostics_does_not_load_data_layer() -> None:
    """診断層はデータ層に依存しない。

    `DataSource` を最下層の `provenance.py` へ移した結果 (A4)、`diagnostics`
    から `data` への辺は無くなった。`data/schema.py` から import して済ませて
    いたらこの辺が生まれていた。
    """
    loaded = _modules_loaded_by("import esn_vla_uq.diagnostics")
    assert "esn_vla_uq.provenance" in loaded
    assert not {name for name in loaded if name.startswith("esn_vla_uq.data")}
