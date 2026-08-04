"""デモアニメーションのフレームデータ。

**描画には一切関与しない。** 「各時刻に何を見せるか」だけを組み立て、
matplotlib での描画は `demo/animate.py` が担う (`docs/design.md` 6.4 節が求める
「描画ロジックと映像入力を分離した構造」)。

v0.1 では実 LIBERO の操作映像が無いため、映像パネルは同梱の合成データから作った
プロット (関節軌道) で代替する。`DemoFrames.panel` はその代替パネルであり、
実映像が入手できた時点で**このモジュールだけを差し替えれば** `animate.py` は
変更せずに済む。

要件書のデモは「操作映像 + 不確実性バー + 失敗直前にバーが跳ねる」。3 つ目は
演出ではなく実測に基づく必要があるため、失敗開始位置 (`failure_onset`) を
フレームデータに持たせ、描画側がそこに印を付けられるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.data.schema import STATE_DIM, RolloutDataset
from esn_vla_uq.esn.config import ESNConfig
from esn_vla_uq.uncertainty.conformal import DEFAULT_ALPHA, SplitConformalPredictor
from esn_vla_uq.uncertainty.nonconformity import DEFAULT_SCORE_KIND, ScoreKind
from esn_vla_uq.uncertainty.split import (
    DEFAULT_SPLIT_STRATEGY,
    SplitStrategy,
    split_samples,
)
from esn_vla_uq.uncertainty.targets import EpisodeSamples, build_samples

PANEL_LABEL: Final[str] = "synthetic joint trajectory (stand-in for LIBERO video)"
"""映像パネルの説明。実映像でないことを図の中に必ず残す。"""


@dataclass(frozen=True)
class DemoFrames:
    """1 エピソード分のアニメーション素材。

    Attributes:
        episode_id: 対象エピソードの識別子。
        task_name: タスク名。
        success: エピソードの成否。
        panel: 映像パネルに描く時系列 `[T, D_panel]`。v0.1 では関節軌道。
            実 LIBERO 映像へ差し替える際はここをフレーム列に置き換える。
        panel_label: パネルの説明 (実映像でないことを明示する)。
        uncertainty: ステップ単位の不確実性スコア `[T]`。予測区間の半幅。
        failure_onset: 失敗が始まったステップ。成功エピソードでは `None`。
        nominal_coverage: 予測区間の名目被覆率 (図の説明に使う)。
        score_kind: 使った非適合度スコア。
    """

    episode_id: str
    task_name: str
    success: bool
    panel: NDArray[np.float64]
    panel_label: str
    uncertainty: NDArray[np.float64]
    failure_onset: int | None
    nominal_coverage: float
    score_kind: ScoreKind

    @property
    def n_steps(self) -> int:
        """フレーム数。"""
        return int(self.uncertainty.shape[0])

    def detection_lag_steps(self) -> int | None:
        """失敗開始から不確実性が立ち上がるまでの遅れ (ステップ)。

        開始前の不確実性の 95 パーセンタイルを初めて超えたステップまでの距離。
        負にはならない (開始前を超える点は定義上探索対象外)。超える点が無ければ
        `None` (= この閾値では検知できなかった)。
        """
        if self.failure_onset is None:
            return None
        before = self.uncertainty[: self.failure_onset]
        after = self.uncertainty[self.failure_onset :]
        if before.size == 0 or after.size == 0:
            return None
        threshold = float(np.quantile(before, 0.95))
        exceeded = np.nonzero(after > threshold)[0]
        if exceeded.size == 0:
            return None
        return int(exceeded[0])

    def uncertainty_ratio_after_onset(self) -> float | None:
        """失敗開始以降と以前の不確実性の比。

        「失敗直前にバーが跳ねる」が演出ではなく実測であることを数値で示す。
        成功エピソードや、失敗開始が端に寄っていて比較できない場合は `None`。
        """
        if self.failure_onset is None:
            return None
        before = self.uncertainty[: self.failure_onset]
        after = self.uncertainty[self.failure_onset :]
        if before.size == 0 or after.size == 0:
            return None
        baseline = float(np.median(before))
        if baseline <= 0.0:
            return None
        return float(np.median(after) / baseline)


def build_demo_frames(
    dataset: RolloutDataset,
    config: ESNConfig,
    *,
    alpha: float = DEFAULT_ALPHA,
    score_kind: ScoreKind = DEFAULT_SCORE_KIND,
    split_strategy: SplitStrategy = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = 0,
    episode_id: str | None = None,
) -> DemoFrames:
    """テスト集合の失敗エピソードを 1 本選び、アニメーション素材を作る。

    conformal は `calibrate` と同じ手順で当てる (fit / calibrate / test の
    3 分割)。**描画対象はテスト集合のエピソードに限る。** fit や calibrate の
    エピソードを描くと、学習に使ったデータの上で不確実性を見せることになり、
    デモとして誠実でない。

    Args:
        dataset: 対象データセット。
        config: ESN のハイパーパラメータ。
        alpha: 有意水準。名目被覆率は ``1 - alpha``。
        score_kind: 非適合度スコア。`absolute` は区間幅が定数になり、バーが
            跳ねないため実質的にデモにならない。
        split_strategy: 較正データの分割方針。
        split_seed: 分割の乱数種。
        episode_id: 描画するエピソードを明示指定する。省略時はテスト集合の
            失敗エピソードのうち、失敗開始以降の不確実性の上がり方が最も
            大きいものを選ぶ。

    Returns:
        アニメーション素材。

    Raises:
        ValueError: 指定したエピソードがテスト集合に無い場合、またはテスト集合に
            失敗エピソードが 1 つも無い場合。
    """
    samples = build_samples(dataset)
    split = split_samples(samples, strategy=split_strategy, seed=split_seed)
    predictor = SplitConformalPredictor(config, alpha=alpha, score_kind=score_kind)
    predictor.fit(split.fit).calibrate(split.calibrate)

    candidates = _candidates(split.test, episode_id)

    scored = [
        _frames_for(sample, predictor, alpha, score_kind) for sample in candidates
    ]
    if episode_id is not None:
        return scored[0]
    return max(scored, key=lambda frames: frames.uncertainty_ratio_after_onset() or 0.0)


def _candidates(
    test_samples: tuple[EpisodeSamples, ...], episode_id: str | None
) -> list[EpisodeSamples]:
    """描画候補のエピソードを選ぶ。"""
    if episode_id is not None:
        chosen = [sample for sample in test_samples if sample.episode_id == episode_id]
        if not chosen:
            raise ValueError(
                f"episode_id: テスト集合にありません (actual={episode_id!r})。"
                "学習・較正に使ったエピソードは描画対象にできません"
            )
        return chosen

    failures = [sample for sample in test_samples if not sample.success]
    if not failures:
        raise ValueError(
            "テスト集合に失敗エピソードがありません。--split-seed を変えるか "
            "エピソード数を増やしてください"
        )
    return failures


def _frames_for(
    sample: EpisodeSamples,
    predictor: SplitConformalPredictor,
    alpha: float,
    score_kind: ScoreKind,
) -> DemoFrames:
    """1 エピソード分のフレームデータを組み立てる。"""
    intervals = predictor.predict_intervals([sample])
    # 予測タスクは 1 ステップ先を当てるため、フレーム t の不確実性は
    # 元エピソードのステップ t+1 に対応する (`target_steps`)。
    panel = sample.inputs[:, : _panel_width(sample)]
    return DemoFrames(
        episode_id=sample.episode_id,
        task_name=sample.task_name,
        success=sample.success,
        panel=panel,
        panel_label=PANEL_LABEL,
        uncertainty=intervals.uncertainty,
        failure_onset=_onset_in_frames(sample),
        nominal_coverage=1.0 - alpha,
        score_kind=score_kind,
    )


def _panel_width(sample: EpisodeSamples) -> int:
    """映像パネルに使う入力の列数 (先頭の関節状態のみ)。

    入力は `[state, action, (チャンク要約)]` の順なので、先頭 `STATE_DIM` 列が
    固有受容感覚にあたる。
    """
    return min(STATE_DIM, sample.n_inputs)


def _onset_in_frames(sample: EpisodeSamples) -> int | None:
    """失敗開始をフレーム番号 (標本の添字) へ変換する。

    `target_steps` は元エピソードのステップ番号なので、そこから探す。
    """
    if sample.failure_onset is None:
        return None
    matches = np.nonzero(sample.target_steps >= sample.failure_onset)[0]
    if matches.size == 0:
        return None
    return int(matches[0])
