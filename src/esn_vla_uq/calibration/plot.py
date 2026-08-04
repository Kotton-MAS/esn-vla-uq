"""reliability diagram の作図 (matplotlib は任意依存)。

matplotlib は `esn-vla-uq[viz]` でのみ入る任意依存であり、**関数の内側で**
import する。コア (ESN・診断・conformal・較正の数値) は numpy だけで動き続ける。
未インストールなら、何をすればよいかが分かるメッセージ付きの `ImportError` にする。

数値 (被覆率・ECE・reliability curve) は `calibration/metrics.py` が matplotlib
無しで計算する。図は同じ数値の見せ方に過ぎず、図が出せないことで評価そのものが
止まることはない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from esn_vla_uq.calibration.metrics import ReliabilityCurve

VIZ_EXTRA_HINT: Final[str] = (
    "reliability diagram の作図には matplotlib が必要です。"
    "`uv sync --extra viz` または `pip install 'esn-vla-uq[viz]'` を実行してください。"
    "数値 (被覆率・ECE・reliability curve) は matplotlib 無しでも "
    "レポート JSON に出力されています。"
)
"""matplotlib が無いときのメッセージ。"""

FIGURE_DPI: Final[int] = 150
"""保存する PNG の DPI。"""

DIAGRAM_TITLE_TEMPLATE: Final[str] = "Reliability diagram (source: {data_source})"
"""図タイトルの雛形。数値の出所を図の中にも残す (図だけが独り歩きするため)。"""

DEFAULT_TITLE: Final[str] = DIAGRAM_TITLE_TEMPLATE.format(data_source="synthetic")
"""既定の図タイトル。"""


def write_reliability_diagram(
    curve: ReliabilityCurve, path: Path, *, title: str = DEFAULT_TITLE
) -> Path:
    """reliability diagram を PNG で書き出す。

    Args:
        curve: 描画する reliability curve。
        path: 出力する PNG のパス。親ディレクトリは自動で作る。
        title: 図のタイトル。

    Returns:
        書き出した PNG のパス。

    Raises:
        ImportError: matplotlib が入っていない場合 (`VIZ_EXTRA_HINT`)。
    """
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - 依存の有無でしか通らない
        raise ImportError(VIZ_EXTRA_HINT) from error

    # 画面を持たない環境 (CI・サーバ) で動かすためバックエンドを固定する。
    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axes = pyplot.subplots(figsize=(4.5, 4.5))
    axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, label="ideal")
    axes.plot(
        curve.nominal, curve.empirical, marker="o", linewidth=1.5, label="empirical"
    )
    axes.set_xlabel("nominal coverage (1 - alpha)")
    axes.set_ylabel("empirical coverage")
    axes.set_title(title)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.grid(visible=True, linewidth=0.3, alpha=0.5)
    axes.legend(loc="lower right")
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI)
    pyplot.close(figure)
    return path
