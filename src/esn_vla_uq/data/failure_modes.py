"""タイムアウト失敗の内訳を物体の軌道から推定する。

## なぜ要るのか

LIBERO の `step` は `done = self._check_success()` であり、**成功以外に終了条件が
無い**。したがって失敗は定義上すべてタイムアウトになり、試行数を増やしても難しい
スイートに変えても失敗様式は増えない (実測: `pi05_libero` の 6 本も `pi0_libero` の
23 本もすべてタイムアウト。`docs/design.md` 10.13 節)。

失敗検知を評価するには「失敗した / しなかった」以上の情報が要る。同じタイムアウト
でも、一度も物体に触れられなかったのか、掴んだあとに落としたのかでは、不確実性が
現れるべき時刻も様式も違う。

収集ログに残した物体の状態 (`object_state`)から、この内訳を**事後に**推定する。
収集側は生の値だけを残し、解釈はここで行う (分類の基準を変えても再収集が要らない)。

## 推定でしかないことについて

ここで返すのは**ヒューリスティックな推定**であって、シミュレータが報告した事実では
ない。LIBERO は「掴んだ」「落とした」を出力しないため、物体の高さの変化から
読み取るしかない。閾値は `GRASP_HEIGHT_MARGIN` などに定数として置き、根拠を
docstring に書く。**この分類を使った結論は、閾値の妥当性に依存する。**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

FailureMode = Literal[
    "success",
    "never_lifted",
    "dropped",
    "held_but_unplaced",
    "unknown",
]
"""タイムアウト失敗の内訳。

- ``"success"``: 成功エピソード (失敗の分類対象外)。
- ``"never_lifted"``: どの物体も持ち上がらなかった。掴めていない可能性が高い。
- ``"dropped"``: 一度持ち上がった物体が元の高さ付近まで戻った。落としたとみなす。
- ``"held_but_unplaced"``: 持ち上げたまま終了した。運べたが置けなかった。
- ``"unknown"``: 物体の状態が記録されていない、または判定できない。
"""

GRASP_HEIGHT_MARGIN: Final[float] = 0.02
"""「持ち上がった」とみなす高さ [m]。

LIBERO の物体はテーブル上に置かれており、把持して持ち上げると z が数 cm 上がる。
2 cm はシミュレータの数値誤差や微小な揺れより十分大きく、意図的な持ち上げより
十分小さい値として選んだ。**実測で校正した値ではない。**
"""

DROP_RETURN_RATIO: Final[float] = 0.5
"""「落とした」とみなす戻り具合。

最大到達高さのうち、初期高さからの上昇分の半分以下まで戻ったら落下とみなす。
"""


@dataclass(frozen=True)
class ObjectTrace:
    """1 エピソード分の物体の高さ軌跡。

    Attributes:
        heights: `float64[T, n_objects]`。各物体の z 座標。
    """

    heights: NDArray[np.float64]

    @property
    def n_objects(self) -> int:
        """追跡している物体の数。"""
        return int(self.heights.shape[1])


def object_heights(
    object_state: NDArray[np.float32], object_keys: tuple[str, ...]
) -> ObjectTrace | None:
    """`object-state` から各物体の z 座標を切り出す。

    `object-state` は各物体の `pos`(3) と `quat`(4)、および eef との相対
    `pos`(3) / `quat`(4) を連結したものである。キー名の並びから z の位置を求める。

    Args:
        object_state: `float32[T, D]` の連結ベクトル。
        object_keys: 連結の順序を表すキー名 (`<object>_pos` / `<object>_quat`)。

    Returns:
        高さの軌跡。切り出せない場合は `None`。
    """
    if object_state.size == 0 or not object_keys:
        return None
    offsets: list[int] = []
    cursor = 0
    for key in object_keys:
        width = 3 if key.endswith("_pos") else 4
        if key.endswith("_pos") and "_to_robot0" not in key:
            # pos は (x, y, z) なので z は 3 番目。
            offsets.append(cursor + 2)
        cursor += width
    if not offsets or cursor > object_state.shape[1]:
        return None
    return ObjectTrace(heights=object_state[:, offsets].astype(np.float64))


def classify_failure(trace: ObjectTrace | None, *, success: bool) -> FailureMode:
    """失敗の内訳を推定する。

    Args:
        trace: 物体の高さ軌跡。`None` なら判定できない。
        success: エピソードが成功したか。

    Returns:
        `FailureMode`。成功エピソードは常に ``"success"``。
    """
    if success:
        return "success"
    if trace is None or trace.n_objects == 0:
        return "unknown"

    initial = trace.heights[0]
    peak = trace.heights.max(axis=0)
    final = trace.heights[-1]
    lifted = peak - initial > GRASP_HEIGHT_MARGIN
    if not bool(lifted.any()):
        return "never_lifted"

    # 持ち上がった物体のうち、最も高く上がったものを見る。
    index = int(np.argmax(peak - initial))
    rise = float(peak[index] - initial[index])
    remaining = float(final[index] - initial[index])
    if remaining < rise * DROP_RETURN_RATIO:
        return "dropped"
    return "held_but_unplaced"


def drop_onset(trace: ObjectTrace | None, mode: FailureMode) -> int | None:
    """落下が始まったステップを推定する。

    `"dropped"` と分類されたエピソードでのみ意味を持つ。最も高く上がった物体が
    ピークに達した時刻を返す。**ここから不確実性が上がるべき、という仮説の
    検証に使うための時刻**であり、シミュレータが報告した事実ではない。

    Returns:
        ステップ番号。判定できない場合は `None`。
    """
    if mode != "dropped" or trace is None or trace.n_objects == 0:
        return None
    initial = trace.heights[0]
    rise = trace.heights.max(axis=0) - initial
    index = int(np.argmax(rise))
    return int(np.argmax(trace.heights[:, index]))
