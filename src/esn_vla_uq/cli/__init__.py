"""CLI パッケージ。

`[project.scripts]` のエントリポイント `esn_vla_uq.cli:main` を公開する。
実体は `esn_vla_uq.cli.app` にある。
"""

from esn_vla_uq.cli.app import build_parser, main

__all__ = ["build_parser", "main"]
