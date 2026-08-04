"""較正データの分割 (タスク内 split / タスク間 split)。

split conformal prediction は、較正集合とテスト集合が**交換可能 (exchangeable)**
であることを根拠に有限標本の周辺被覆率保証を導く (`docs/design.md` 6.3 節)。
分割の切り方がその仮定を満たすかどうかを決めるため、この判断をここ 1 箇所に
集める。

**3 分割にする理由**: split conformal は較正集合が read-out の学習に使われていない
ことを要求する。学習集合と較正集合が重なると残差が楽観的になり、区間が過小に
なって被覆率が名目値を下回る。したがって fit (read-out 学習) / calibrate (分位点) /
test (評価) の 3 つに分ける。

分割はエピソード単位で行う。同一エピソード内のステップは強く相関しているため、
ステップ単位で分割すると較正集合とテスト集合に同じエピソードの隣接ステップが
入り、実質的に情報が漏れる。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np

from esn_vla_uq.uncertainty.targets import EpisodeSamples

SplitStrategy = Literal["within_task", "across_task"]
"""較正データの分割方針。

- ``"within_task"``: 同一タスクのエピソード群を fit/calibrate/test に分ける。
  同一タスク内のエピソードは概ね同質な生成過程から得られると仮定でき、
  交換可能性が比較的妥当。**既定**。
- ``"across_task"``: タスクごと分ける (あるタスク群で較正し別のタスク群で評価)。
  タスクごとに残差分布が系統的に異なりうるため交換可能性が崩れ、標準的な
  被覆率保証はそのままは成立しない。
"""

SUPPORTED_SPLIT_STRATEGIES: Final[tuple[SplitStrategy, ...]] = get_args(SplitStrategy)
"""`SplitStrategy` が許可する値の実行時タプル。"""

DEFAULT_SPLIT_STRATEGY: Final[SplitStrategy] = "within_task"
"""既定の分割方針 (`docs/design.md` 6.3 節の確定事項)。"""

DEFAULT_FIT_RATIO: Final[float] = 0.5
"""fit (read-out 学習) に割り当てる割合。"""

DEFAULT_CALIBRATE_RATIO: Final[float] = 0.25
"""calibrate (conformal 分位点) に割り当てる割合。残りが test。"""

ACROSS_TASK_WARNING: Final[str] = (
    "across_task split: 較正集合とテスト集合が異なるタスクから来ているため、"
    "交換可能性の仮定が崩れる。split conformal の被覆率保証はそのままは成立しない"
    " (docs/design.md 6.3 節)。報告する被覆率は保証値ではなく実測値として扱うこと。"
)
"""タスク間 split を選んだときに出力へ載せる警告。

「オプションとして提供するが、保証が弱いことを出力に明記する」という 6.3 節の
確定事項の実装。レポート JSON とログの両方に出す。
"""

MIN_EPISODES_PER_PART: Final[int] = 1
"""各パートに必要な最小エピソード数。"""


@dataclass(frozen=True)
class CalibrationSplit:
    """fit / calibrate / test の 3 分割。

    Attributes:
        fit: read-out の学習に使う標本。
        calibrate: conformal の分位点を決めるのに使う標本。fit と重ならない。
        test: 評価に使う標本。
        strategy: 使った分割方針。
        warning: 交換可能性に関する警告。`across_task` のときのみ非 None。
    """

    fit: tuple[EpisodeSamples, ...]
    calibrate: tuple[EpisodeSamples, ...]
    test: tuple[EpisodeSamples, ...]
    strategy: SplitStrategy
    warning: str | None

    def to_dict(self) -> dict[str, object]:
        """レポート用の辞書。"""
        return {
            "strategy": self.strategy,
            "n_episodes_fit": len(self.fit),
            "n_episodes_calibrate": len(self.calibrate),
            "n_episodes_test": len(self.test),
            "n_samples_fit": sum(sample.n_samples for sample in self.fit),
            "n_samples_calibrate": sum(sample.n_samples for sample in self.calibrate),
            "n_samples_test": sum(sample.n_samples for sample in self.test),
            "exchangeability_warning": self.warning,
        }


def split_samples(
    samples: Sequence[EpisodeSamples],
    *,
    strategy: SplitStrategy = DEFAULT_SPLIT_STRATEGY,
    seed: int = 0,
    fit_ratio: float = DEFAULT_FIT_RATIO,
    calibrate_ratio: float = DEFAULT_CALIBRATE_RATIO,
) -> CalibrationSplit:
    """標本を fit / calibrate / test に分ける。

    Args:
        samples: `build_samples` が返したエピソード単位の標本。
        strategy: 分割方針 (`SplitStrategy`)。
        seed: シャッフルの乱数種。同じ seed なら常に同じ分割になる。
        fit_ratio: fit に割り当てる割合。
        calibrate_ratio: calibrate に割り当てる割合。残りが test。

    Returns:
        3 分割と、方針に応じた警告を持つ `CalibrationSplit`。

    Raises:
        ValueError: 方針が未知、割合が不正、またはどれかのパートが空になる場合。
    """
    if strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise ValueError(
            f"strategy: 未知の分割方針です (actual={strategy!r}, "
            f"supported={list(SUPPORTED_SPLIT_STRATEGIES)})"
        )
    _validate_ratios(fit_ratio, calibrate_ratio)

    rng = np.random.default_rng(seed)
    if strategy == "within_task":
        parts = _split_within_task(samples, rng, fit_ratio, calibrate_ratio)
    else:
        parts = _split_across_task(samples, rng, fit_ratio, calibrate_ratio)

    fit, calibrate, test = parts
    _validate_parts(fit, calibrate, test, strategy)
    return CalibrationSplit(
        fit=tuple(fit),
        calibrate=tuple(calibrate),
        test=tuple(test),
        strategy=strategy,
        warning=ACROSS_TASK_WARNING if strategy == "across_task" else None,
    )


def _validate_ratios(fit_ratio: float, calibrate_ratio: float) -> None:
    """割合が (0, 1) に収まり、合計が 1 未満であることを検証する。"""
    for name, value in (("fit_ratio", fit_ratio), ("calibrate_ratio", calibrate_ratio)):
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"{name}: 0 < 値 < 1 である必要があります (actual={value})"
            )
    if fit_ratio + calibrate_ratio >= 1.0:
        raise ValueError(
            "fit_ratio + calibrate_ratio は 1 未満である必要があります "
            f"(test が空になる: {fit_ratio} + {calibrate_ratio})"
        )


def _validate_parts(
    fit: list[EpisodeSamples],
    calibrate: list[EpisodeSamples],
    test: list[EpisodeSamples],
    strategy: SplitStrategy,
) -> None:
    """どのパートも空でないことを検証する。"""
    empty = [
        name
        for name, part in (("fit", fit), ("calibrate", calibrate), ("test", test))
        if len(part) < MIN_EPISODES_PER_PART
    ]
    if empty:
        raise ValueError(
            f"分割の結果 {', '.join(empty)} が空になりました "
            f"(strategy={strategy}, "
            f"エピソード数={len(fit) + len(calibrate) + len(test)})。"
            "エピソード数を増やすか割合を調整してください"
        )


def _partition(
    items: list[EpisodeSamples],
    rng: np.random.Generator,
    fit_ratio: float,
    calibrate_ratio: float,
) -> tuple[list[EpisodeSamples], list[EpisodeSamples], list[EpisodeSamples]]:
    """1 つの集団をシャッフルして 3 分割する。"""
    order = rng.permutation(len(items))
    shuffled = [items[int(index)] for index in order]
    n_fit = max(MIN_EPISODES_PER_PART, int(len(shuffled) * fit_ratio))
    n_calibrate = max(MIN_EPISODES_PER_PART, int(len(shuffled) * calibrate_ratio))
    return (
        shuffled[:n_fit],
        shuffled[n_fit : n_fit + n_calibrate],
        shuffled[n_fit + n_calibrate :],
    )


def _split_within_task(
    samples: Sequence[EpisodeSamples],
    rng: np.random.Generator,
    fit_ratio: float,
    calibrate_ratio: float,
) -> tuple[list[EpisodeSamples], list[EpisodeSamples], list[EpisodeSamples]]:
    """タスクごとに 3 分割し、各パートへ混ぜる。

    どのタスクも 3 つのパートすべてに現れるため、較正集合とテスト集合の
    タスク構成が揃う。
    """
    fit: list[EpisodeSamples] = []
    calibrate: list[EpisodeSamples] = []
    test: list[EpisodeSamples] = []
    for task_name in sorted({sample.task_name for sample in samples}):
        task_samples = [sample for sample in samples if sample.task_name == task_name]
        task_fit, task_calibrate, task_test = _partition(
            task_samples, rng, fit_ratio, calibrate_ratio
        )
        fit.extend(task_fit)
        calibrate.extend(task_calibrate)
        test.extend(task_test)
    return fit, calibrate, test


def _split_across_task(
    samples: Sequence[EpisodeSamples],
    rng: np.random.Generator,
    fit_ratio: float,
    calibrate_ratio: float,
) -> tuple[list[EpisodeSamples], list[EpisodeSamples], list[EpisodeSamples]]:
    """タスク単位で 3 分割する (1 つのタスクは 1 つのパートにのみ属する)。

    テスト集合には較正時に一度も見ていないタスクが入るため、交換可能性が
    崩れる。`ACROSS_TASK_WARNING` を参照。
    """
    task_names = sorted({sample.task_name for sample in samples})
    if len(task_names) < 3:
        raise ValueError(
            "across_task split には 3 つ以上のタスクが必要です "
            f"(actual={len(task_names)}: {task_names})"
        )
    order = rng.permutation(len(task_names))
    shuffled = [task_names[int(index)] for index in order]
    n_fit = max(MIN_EPISODES_PER_PART, int(len(shuffled) * fit_ratio))
    n_calibrate = max(MIN_EPISODES_PER_PART, int(len(shuffled) * calibrate_ratio))
    fit_tasks = set(shuffled[:n_fit])
    calibrate_tasks = set(shuffled[n_fit : n_fit + n_calibrate])
    test_tasks = set(shuffled[n_fit + n_calibrate :])

    def _select(task_group: set[str]) -> list[EpisodeSamples]:
        return [sample for sample in samples if sample.task_name in task_group]

    return _select(fit_tasks), _select(calibrate_tasks), _select(test_tasks)
