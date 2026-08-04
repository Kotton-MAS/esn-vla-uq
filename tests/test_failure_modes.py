"""タイムアウト失敗の内訳推定のテスト。

分類はヒューリスティックなので、境界のふるまいを明示的に固定しておく。閾値を
動かしたときにどのケースが変わるかがテストから読めることを狙う。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from esn_vla_uq.data.failure_modes import (
    DROP_RETURN_RATIO,
    GRASP_HEIGHT_MARGIN,
    FailureMode,
    ObjectTrace,
    classify_failure,
    drop_onset,
    object_heights,
)


def _trace(*columns: list[float]) -> ObjectTrace:
    """各物体の高さ列から軌跡を作る。"""
    return ObjectTrace(heights=np.asarray(columns, dtype=np.float64).T)


def test_success_is_never_classified_as_a_failure() -> None:
    """成功エピソードは物体が落ちていても ``"success"``。

    分類対象は失敗だけである。成功したのに落下と報告されると、内訳の集計が
    そのまま壊れる。
    """
    fallen = _trace([1.0, 1.5, 1.0])
    assert classify_failure(fallen, success=True) == "success"


def test_object_never_lifted() -> None:
    """持ち上がらなければ ``"never_lifted"``。"""
    flat = _trace([1.0, 1.0 + GRASP_HEIGHT_MARGIN / 2, 1.0])
    assert classify_failure(flat, success=False) == "never_lifted"


def test_lifted_then_returned_is_a_drop() -> None:
    """持ち上げてから元の高さに戻れば ``"dropped"``。"""
    dropped = _trace([1.0, 1.0 + GRASP_HEIGHT_MARGIN * 5, 1.0])
    assert classify_failure(dropped, success=False) == "dropped"


def test_still_held_at_the_end_is_unplaced() -> None:
    """持ち上げたまま終われば ``"held_but_unplaced"``。

    「落とした」と「運べたが置けなかった」は不確実性が現れる時刻が違うため、
    同じタイムアウトでも区別する意味がある。
    """
    held = _trace([1.0, 1.0 + GRASP_HEIGHT_MARGIN * 5, 1.0 + GRASP_HEIGHT_MARGIN * 5])
    assert classify_failure(held, success=False) == "held_but_unplaced"


def test_the_drop_threshold_is_the_documented_ratio() -> None:
    """戻り具合が `DROP_RETURN_RATIO` を跨ぐところで判定が変わる。

    閾値を動かしたときに影響するケースを明示しておく。
    """
    rise = GRASP_HEIGHT_MARGIN * 10
    just_above = 1.0 + rise * (DROP_RETURN_RATIO + 0.1)
    just_below = 1.0 + rise * (DROP_RETURN_RATIO - 0.1)
    assert classify_failure(_trace([1.0, 1.0 + rise, just_above]), success=False) == (
        "held_but_unplaced"
    )
    assert classify_failure(_trace([1.0, 1.0 + rise, just_below]), success=False) == (
        "dropped"
    )


def test_the_most_lifted_object_decides() -> None:
    """複数物体があるときは最も持ち上がったものを見る。

    LIBERO のシーンには操作対象でない物体も置かれている。動かなかった物体に
    引きずられて ``"never_lifted"`` にならないことを確認する。
    """
    scene = _trace(
        [1.0, 1.0, 1.0],  # 触っていない物体
        [1.0, 1.0 + GRASP_HEIGHT_MARGIN * 5, 1.0],  # 掴んで落とした物体
    )
    assert classify_failure(scene, success=False) == "dropped"


@pytest.mark.parametrize("trace", [None, ObjectTrace(heights=np.zeros((3, 0)))])
def test_missing_object_state_is_unknown(trace: ObjectTrace | None) -> None:
    """物体の状態が無ければ ``"unknown"``。

    古いログには `object_state` が無い。判定できないことを ``"never_lifted"``
    と混ぜると、内訳がデータの有無に依存して歪む。
    """
    assert classify_failure(trace, success=False) == "unknown"


def test_drop_onset_is_the_peak_step() -> None:
    """落下開始はピーク到達時刻。"""
    dropped = _trace([1.0, 1.2, 1.3, 1.1, 1.0])
    assert drop_onset(dropped, "dropped") == 2


@pytest.mark.parametrize(
    "mode", ["success", "never_lifted", "held_but_unplaced", "unknown"]
)
def test_drop_onset_only_applies_to_drops(mode: FailureMode) -> None:
    """落下以外では時刻を返さない。

    落下していないエピソードに時刻が付くと、そこを起点に不確実性を測る後段が
    無意味な区間を見ることになる。
    """
    assert drop_onset(_trace([1.0, 1.2, 1.0]), mode) is None


def test_object_heights_extracts_z_from_the_concatenated_vector() -> None:
    """`object-state` の連結から z だけを切り出す。

    連結の順序はキー名の並びで決まる。`pos` は 3、`quat` は 4 を占める。
    """
    # milk_1_pos(3) + milk_1_quat(4) + milk_1_to_robot0_eef_pos(3)
    frame = [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.1]
    state = np.asarray([frame], dtype=np.float32)
    keys = ("milk_1_pos", "milk_1_quat", "milk_1_to_robot0_eef_pos")
    trace = object_heights(state, keys)
    assert trace is not None
    assert trace.n_objects == 1
    assert trace.heights[0, 0] == pytest.approx(1.5)


def test_relative_positions_are_not_treated_as_object_heights() -> None:
    """eef との相対位置は物体の高さではない。

    `_to_robot0_eef_pos` を混ぜると、グリッパが動くだけで「持ち上がった」と
    誤判定する。
    """
    keys = ("milk_1_pos", "milk_1_quat", "milk_1_to_robot0_eef_pos")
    state = np.zeros((2, 10), dtype=np.float32)
    trace = object_heights(state, keys)
    assert trace is not None
    assert trace.n_objects == 1


@pytest.mark.parametrize(
    ("state", "keys"),
    [
        (np.zeros((0, 0), dtype=np.float32), ("milk_1_pos",)),
        (np.zeros((2, 3), dtype=np.float32), ()),
        (np.zeros((2, 2), dtype=np.float32), ("milk_1_pos",)),
    ],
)
def test_object_heights_returns_none_when_it_cannot_slice(
    state: NDArray[np.float32], keys: tuple[str, ...]
) -> None:
    """切り出せない入力では `None` を返す (例外にしない)。

    古いログや想定外のキー構成でも、分類が ``"unknown"`` に落ちるだけで
    集計自体は続けられるようにする。
    """
    assert object_heights(state, keys) is None
