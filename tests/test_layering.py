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
2. **本パッケージの import が openpi / LIBERO を要求しない。** 要件書の
   「openpi をランタイム依存に含めない」の直接検査であり、`EXTERNAL_PACKAGES`
   の docstring に経緯を書いた。
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

EXTERNAL_PACKAGES = ("openpi", "openpi_client", "libero", "robosuite")
"""本パッケージが**決して要求してはならない**外部パッケージ。

要件書の「openpi をランタイム依存に含めない」を直接検査するための一覧。

当初は「`esn_vla_uq.data.sources.openpi` をロードしないこと」を代理指標にして
いた。実装してみると `sources/openpi.py` は収集済みログのファイルを読むだけで
**openpi のパッケージを一切 import しない**ことが分かったため、その代理指標は
実態と合わなくなった (自前モジュールのロードを禁じても意味が無い)。守りたいのは
「インストールしていなくても動く」ことなので、外部パッケージが要求されないことを
直接見る。こちらのほうが強く、かつ今日から効く。
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
    ["data/sources/openpi.py", "data/io.py", "data/schema.py"],
)
def test_modules_do_not_import_external_packages(relative_path: str) -> None:
    """外部パッケージ (openpi / LIBERO) を import しないこと。

    `sources/openpi.py` が読むのは収集スクリプトが書き出したファイルだけで、
    openpi の policy server にも LIBERO 環境にも触らない。この性質が崩れると、
    ログを読むだけの利用者にまで重い依存が要求される。
    """
    imported = {name.split(".")[0] for name in _imported_modules(relative_path)}
    forbidden = imported.intersection(EXTERNAL_PACKAGES)
    assert not forbidden, (
        f"{relative_path} が外部パッケージを import しています: {forbidden}"
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


@pytest.mark.parametrize(
    "statement",
    [
        "import esn_vla_uq.data",
        "import esn_vla_uq.cli",
        "import esn_vla_uq.calibration",
    ],
)
def test_importing_the_package_does_not_require_external_packages(
    statement: str,
) -> None:
    """本パッケージの import が openpi / LIBERO を要求しないこと。

    要件書の「openpi をランタイム依存に含めない」の直接検査。CLI は
    `--input` の解決で `OpenpiLogSource` を使うため、そこからも外部パッケージが
    引き込まれないことを確認する。静的な import 検査と違い、間接的な経路も
    捉える。
    """
    script = (
        f"{statement}\n"
        "import sys, json\n"
        "print(json.dumps(sorted(m.split('.')[0] for m in sys.modules)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    top_level = set(json.loads(completed.stdout))
    forbidden = top_level.intersection(EXTERNAL_PACKAGES)
    assert not forbidden, f"{statement} が外部パッケージを引き込みました: {forbidden}"


def test_diagnostics_does_not_load_data_layer() -> None:
    """診断層はデータ層に依存しない。

    `DataSource` を最下層の `provenance.py` へ移した結果 (A4)、`diagnostics`
    から `data` への辺は無くなった。`data/schema.py` から import して済ませて
    いたらこの辺が生まれていた。
    """
    loaded = _modules_loaded_by("import esn_vla_uq.diagnostics")
    assert "esn_vla_uq.provenance" in loaded
    assert not {name for name in loaded if name.startswith("esn_vla_uq.data")}
