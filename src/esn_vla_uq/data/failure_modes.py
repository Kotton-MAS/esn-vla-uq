"""タイムアウト失敗の内訳を物体の軌道から推定する。

## なぜ要るのか

LIBERO の `step` は `done = self._check_success()` であり、**成功以外に終了条件が
無い**。したがって失敗は定義上すべてタイムアウトになり、試行数を増やしても難しい
スイートに変えても失敗様式は増えない (実測: `pi05_libero` の 6 本も `pi0_libero` の
23 本もすべてタイムアウト。`docs/design.md` 10.13 節)。

失敗検知を評価するには「失敗した / しなかった」以上の情報が要る。同じタイムアウト
でも、一度も物体に触れられなかったのか、掴んだあとに落としたのかでは、不確実性が
現れるべき時刻も様式も違う。

収集ログに残した物体の位置 (`object_pos`) から、この内訳を**事後に**推定する。
収集側は生の値だけを残し、解釈はここで行う (分類の基準を変えても再収集が要らない)。

## 結論: どの様式も検知できない

実測で決着した (`docs/design.md` 10.16 節)。内訳ごとに分けても不確実性は失敗を
判別しない。

| 対象 | AUROC | 失敗/成功 | 95% CI |
| --- | --- | --- | --- |
| 全失敗 | 0.475 | 157 / 458 | [0.418, 0.533] |
| `dropped` | 0.469 | 63 / 287 | [0.382, 0.556] |
| `never_lifted` | 0.479 | 49 / 287 | [0.385, 0.573] |
| `held_but_unplaced` | 0.525 | 15 / 282 | [0.373, 0.677] |

上 3 つは実用水準 (0.6) を区間で排除する。`held_but_unplaced` は 615 本中 15 本と
稀で、区間が閉じていない。

**この分類自体は否定されていない。** 内訳は実際に分かれており、`dropped` からは
落下開始時刻も取れる (中央値でエピソード長の 0.19〜0.38)。使い道は残る。

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


def object_heights(object_pos: NDArray[np.float32]) -> ObjectTrace | None:
    """記録した物体位置から z 座標だけを取り出す。

    収集側は `<object>_pos` を個別に取り出して積んである。観測にある
    `object-state` (全物体の量を連結した 1 本のベクトル) は**使わない**。連結順を
    推測して切り出すと、タスクごとに物体数が違う (libero_10 では 28〜112 次元)
    ため位置がずれる。

    Args:
        object_pos: `float32[T, n_objects, 3]`。各物体の (x, y, z)。

    Returns:
        高さの軌跡。形が合わない、または空なら `None`。
    """
    if object_pos.ndim != 3 or object_pos.shape[2] != 3 or object_pos.size == 0:
        return None
    return ObjectTrace(heights=object_pos[:, :, 2].astype(np.float64))


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
