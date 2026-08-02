"""esn-vla-uq: Echo State Network による VLA ポリシーの不確実性定量化.

パッケージのレイヤ構成は `docs/design.md` に従う (data -> esn -> diagnostics ->
uncertainty -> calibration)。本パッケージが提供する数値は既定では同梱の合成
データ (`source: "synthetic"`) 由来であり、実 LIBERO 評価の結果ではない。
"""

from importlib.metadata import version

__version__ = version("esn-vla-uq")

__all__ = ["__version__"]
