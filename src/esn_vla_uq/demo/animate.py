"""デモアニメーションの描画 (matplotlib は任意依存)。

`demo/frames.py` が作った `DemoFrames` を GIF にする。**フレームデータの作り方は
知らない**ので、実 LIBERO 映像へ差し替えるときも本モジュールは変更しなくてよい
(`docs/design.md` 6.4 節)。

matplotlib は `esn-vla-uq[viz]` でのみ入る任意依存であり、関数の内側で import
する。GIF の書き出しには matplotlib 同梱の Pillow ライタを使うので、追加の依存は
要らない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from esn_vla_uq.calibration.plot import VIZ_EXTRA_HINT
from esn_vla_uq.demo.frames import DemoFrames

DEFAULT_FPS: Final[int] = 20
"""GIF のフレームレート。"""

DEFAULT_MAX_FRAMES: Final[int] = 160
"""GIF に含める最大フレーム数。

エピソードは 150〜250 ステップある。全ステップを 20 fps で描くと 10 秒を超え、
README に貼る GIF としては重くなる。上限を超える場合は等間隔に間引く。
"""

FIGURE_DPI: Final[int] = 96
"""保存する GIF の DPI。README 埋め込みを想定した控えめな値。"""


def write_demo_animation(
    frames: DemoFrames,
    path: Path,
    *,
    fps: int = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> Path:
    """デモアニメーションを GIF で書き出す。

    上段に映像パネル (v0.1 では関節軌道の代替プロット)、下段に不確実性バーを
    描き、失敗開始位置に縦線を入れる。

    Args:
        frames: 描画するフレームデータ。
        path: 出力する GIF のパス。親ディレクトリは自動で作る。
        fps: フレームレート。
        max_frames: 含める最大フレーム数。超える分は等間隔に間引く。

    Returns:
        書き出した GIF のパス。

    Raises:
        ImportError: matplotlib が入っていない場合 (`VIZ_EXTRA_HINT`)。
        ValueError: `fps` または `max_frames` が 1 未満の場合。
    """
    if fps < 1:
        raise ValueError(f"fps: 1 以上が必要です (actual={fps})")
    if max_frames < 1:
        raise ValueError(f"max_frames: 1 以上が必要です (actual={max_frames})")

    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - 依存の有無でしか通らない
        raise ImportError(VIZ_EXTRA_HINT) from error

    matplotlib.use("Agg")
    from matplotlib import pyplot
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.artist import Artist

    indices = _frame_indices(frames.n_steps, max_frames)
    figure, (panel_axes, bar_axes) = pyplot.subplots(
        2, 1, figsize=(6.0, 4.6), height_ratios=(2.0, 1.0)
    )

    steps = np.arange(frames.n_steps)
    panel_axes.plot(steps, frames.panel, linewidth=0.8, alpha=0.75)
    panel_axes.set_xlim(0, frames.n_steps - 1)
    panel_axes.set_ylabel("joint / gripper state")
    panel_axes.set_title(
        f"{frames.task_name}  ({frames.episode_id}, "
        f"{'success' if frames.success else 'failure'})",
        fontsize=10,
    )
    panel_axes.text(
        0.01,
        0.02,
        frames.panel_label,
        transform=panel_axes.transAxes,
        fontsize=7,
        alpha=0.7,
    )
    panel_cursor = panel_axes.axvline(0.0, color="k", linewidth=1.2)

    bar_axes.set_xlim(0, frames.n_steps - 1)
    bar_axes.set_ylim(0.0, float(frames.uncertainty.max()) * 1.1)
    bar_axes.set_xlabel("step")
    bar_axes.set_ylabel("uncertainty\n(interval half-width)")
    (bar_line,) = bar_axes.plot([], [], linewidth=1.4)

    if frames.failure_onset is not None:
        for axes in (panel_axes, bar_axes):
            axes.axvline(
                frames.failure_onset,
                color="crimson",
                linestyle="--",
                linewidth=1.2,
            )
        lag = frames.detection_lag_steps()
        label = " failure onset" if lag is None else f" failure onset (+{lag} steps)"
        bar_axes.text(
            frames.failure_onset,
            bar_axes.get_ylim()[1] * 0.92,
            label,
            color="crimson",
            fontsize=8,
            va="top",
        )

    figure.suptitle(
        f"conformal prediction interval  "
        f"(nominal coverage {frames.nominal_coverage:.0%}, "
        f"score={frames.score_kind}, source=synthetic) "
        "— uncertainty reacts to the failure, it does not anticipate it",
        fontsize=9,
    )
    figure.tight_layout()

    def _update(frame_index: int) -> list[Artist]:
        upto = int(frame_index) + 1
        panel_cursor.set_xdata([frame_index, frame_index])
        bar_line.set_data(steps[:upto], frames.uncertainty[:upto])
        return [panel_cursor, bar_line]

    animation = FuncAnimation(
        figure, _update, frames=indices, interval=1000 // fps, blit=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=FIGURE_DPI)
    pyplot.close(figure)
    return path


def _frame_indices(n_steps: int, max_frames: int) -> list[int]:
    """描画するフレーム番号を等間隔に間引いて返す。"""
    if n_steps <= max_frames:
        return list(range(n_steps))
    return [int(value) for value in np.linspace(0, n_steps - 1, max_frames)]
