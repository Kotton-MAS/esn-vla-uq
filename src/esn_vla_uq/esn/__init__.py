"""ESN コア (リザバー生成・状態更新・リッジ read-out)。

数学仕様の唯一の真実は `docs/design.md` の「ESN の数学仕様」節。
公開 API は本モジュールから再エクスポートする。
"""

from esn_vla_uq.esn.config import ESNConfig
from esn_vla_uq.esn.model import ESN
from esn_vla_uq.esn.readout import RidgeReadout
from esn_vla_uq.esn.reservoir import (
    Activation,
    Reservoir,
    discard_washout,
    tanh_activation,
)

__all__ = [
    "ESN",
    "Activation",
    "ESNConfig",
    "Reservoir",
    "RidgeReadout",
    "discard_washout",
    "tanh_activation",
]
