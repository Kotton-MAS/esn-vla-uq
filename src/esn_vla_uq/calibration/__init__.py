"""較正評価 (被覆率 / reliability diagram / ECE / 失敗検知)。

`metrics.py` が数値、`plot.py` が作図を担う。**数値は numpy だけで計算でき**、
matplotlib は任意依存 (`esn-vla-uq[viz]`) として作図にのみ使う。

本パッケージは `uncertainty` 層に依存する (逆方向は無い)。
"""

from esn_vla_uq.calibration.metrics import (
    DEFAULT_NOMINAL_LEVELS,
    ECE_DEFINITION,
    ReliabilityCurve,
    conformal_coverage,
    detection_auroc,
    rank_data,
    reliability_curve,
)
from esn_vla_uq.calibration.plot import VIZ_EXTRA_HINT, write_reliability_diagram

__all__ = [
    "DEFAULT_NOMINAL_LEVELS",
    "ECE_DEFINITION",
    "VIZ_EXTRA_HINT",
    "ReliabilityCurve",
    "conformal_coverage",
    "detection_auroc",
    "rank_data",
    "reliability_curve",
    "write_reliability_diagram",
]
