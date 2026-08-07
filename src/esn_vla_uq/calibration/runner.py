"""較正評価の実行オーケストレーション。

データセットから `CalibrationReport` を組み立てるところまでを担う。レポートの
表現 (辞書化・JSON 書き出し・ログ整形) には関与しない (`calibration/report.py`)。
`diagnostics/runner.py` と `diagnostics/report.py` の分担に揃えてある。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq import __version__
from esn_vla_uq.calibration.metrics import (
    DEFAULT_NOMINAL_LEVELS,
    ReliabilityCurve,
    detection_auroc,
    reliability_curve,
)
from esn_vla_uq.calibration.report import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationReport,
    CoverageSummary,
    DetectionSummary,
    utc_timestamp,
)
from esn_vla_uq.data.schema import RolloutDataset
from esn_vla_uq.esn.config import ESNConfig
from esn_vla_uq.provenance import DataSource
from esn_vla_uq.uncertainty.conformal import (
    DEFAULT_ALPHA,
    DEFAULT_WASHOUT,
    SplitConformalPredictor,
)
from esn_vla_uq.uncertainty.nonconformity import DEFAULT_SCORE_KIND, ScoreKind
from esn_vla_uq.uncertainty.split import (
    DEFAULT_SPLIT_STRATEGY,
    SplitStrategy,
    split_samples,
)
from esn_vla_uq.uncertainty.targets import (
    EpisodeSamples,
    build_samples,
    detection_labels,
)

logger = logging.getLogger(__name__)

EFFECTIVE_SAMPLE_SIZE_CAVEAT: Final[str] = (
    "被覆率はステップ単位で数えているが、同一エピソード内のステップは強く相関する。"
    "被覆率の分散を決めるのはテストエピソード数であってステップ数ではない。"
    "同梱の合成データ規模では、名目 90% に対する実測被覆率が分割の乱数種によって"
    "0.63〜1.00 まで振れる (30 分割の平均は 0.896 で名目どおり)。"
    "単一の分割の被覆率をもって較正のずれと判断しないこと。"
)
"""被覆率の読み方に関する注意。単一分割の数値を過大評価させないため。"""

SYNTHETIC_DATA_CAVEAT: Final[str] = (
    "数値の出所は合成データ (source=synthetic) であり、実 LIBERO 評価の結果ではない。"
    "失敗検知の成績は合成データ生成器が失敗区間に注入した分布シフトを"
    "検出しているにすぎず、実ロールアウトへの転移は検証されていない。"
)
"""合成データであることの明示 (`docs/design.md` 7 節の誠実性宣言)。

**出所が合成データのときだけ付ける。** 実ログに対して「これは合成データだ」と
書いたレポートを出すと、誠実性宣言のつもりが逆に事実と違う注記になる。
"""

DETECTION_UNAVAILABLE_REASON: Final[str] = (
    "どの分割でも陽性か陰性のどちらかが 0 件で、AUROC が定義できなかった。"
    "失敗エピソードが含まれていないか、全て失敗している可能性がある。"
)
"""失敗検知を計算できなかったときの理由。"""

EPISODE_LABEL_CAVEAT: Final[str] = (
    "失敗検知のラベルは episode_success (失敗エピソードの全ステップを陽性) を"
    "使っている。失敗開始時刻を持たない出所 (openpi) では失敗開始以降だけを"
    "陽性にする細かいラベルが作れないため。合成データでの failure_onset ベースの"
    "数値とは粒度が違うので、直接比較しないこと。"
)
"""粗いラベルを使ったときの注意。"""

DETECTION_IS_EXPLORATORY_CAVEAT: Final[str] = (
    "失敗検知 AUROC は**探索的な診断値であり、v0.1 の成果ではない**。"
    "実 openpi ログでは 0.457〜0.477 と無相関で、代替の観測量もタスク内で見ると"
    "判定不能だった (docs/design.md 10.14 節)。合成データで高い値が出るのは生成器が"
    "チャンク分散と失敗を結びつけて作っているためである。"
    "不確実性スコアは「予測が難しいステップ」を表す量であり、それが「失敗する"
    "ステップ」と一致するかは未検証。"
)
"""失敗検知の位置づけ。

**どのデータでも必ず付ける。** 数値だけを見た読み手が「失敗検知できる」と読むのを
防ぐ。合成データで 0.87 が出るのは生成器の作りによるものであり、実データでは
再現しない。
"""

INVERTED_DETECTION_THRESHOLD: Final[float] = 0.45
"""この値を下回ったら「不確実性が失敗と反相関している」とみなす閾値。

0.5 ちょうどではなく余裕を持たせる。分割間のばらつき (実測で標準偏差 0.07〜0.09)
があるため、0.5 のわずか下を反相関と呼ぶと偶然を拾う。
"""

INVERTED_DETECTION_CAVEAT: Final[str] = (
    "失敗検知 AUROC が 0.5 を下回っている。**不確実性が高いステップほど成功して"
    "いる**ことを意味し、信号が無いのではなく向きが逆である。実 openpi ログでは"
    "タイムアウト型の失敗 (ポリシーが動けなくなり同じ行動を繰り返す) で"
    "チャンク分散が下がるため、この向きになる (docs/design.md 10.11 節)。"
    "符号を反転すれば検知に使えるが、失敗様式によって向きが変わる可能性があるため"
    "自動では反転しない。"
)
"""AUROC が 0.5 を下回ったときの注意。

**これは不具合ではなく結果である。** 黙って 0.4 という数値だけを出すと「効いて
いない」と読まれるが、実際には「逆向きに効いている」。読み手が誤解しないよう
明示する。
"""

ABSOLUTE_SCORE_CAVEAT: Final[str] = (
    "score_kind=absolute の予測区間は全ステップで同じ幅になるため、"
    "不確実性スコアはステップを区別しない (失敗検知 AUROC は定義上 0.5)。"
    "ステップ単位の不確実性が要る用途では score_kind=normalized を使うこと。"
)
"""`absolute` を選んだときの注意。"""


DEFAULT_N_SPLITS: Final[int] = 20
"""既定の分割回数。

単一分割の被覆率は同梱データ規模で 0.63〜1.00 まで振れるため、代表値として
使えない。複数の分割で評価して平均と散らばりを報告する。
"""


@dataclass(frozen=True)
class _SplitOutcome:
    """1 分割分の評価結果。"""

    coverage: float
    auroc: float | None
    label_kind: str
    mean_width: float
    n_test_samples: int
    n_test_episodes: int
    n_positive: int
    n_negative: int
    calibration_scores: NDArray[np.float64]
    test_scores: NDArray[np.float64]
    warning: str | None


def _evaluate_split(
    samples: Sequence[EpisodeSamples],
    config: ESNConfig,
    *,
    alpha: float,
    score_kind: ScoreKind,
    split_strategy: SplitStrategy,
    split_seed: int,
    washout: int,
) -> _SplitOutcome:
    """1 つの分割で fit -> calibrate -> evaluate を行う。"""
    split = split_samples(samples, strategy=split_strategy, seed=split_seed)
    predictor = SplitConformalPredictor(
        config, alpha=alpha, score_kind=score_kind, washout=washout
    )
    predictor.fit(split.fit).calibrate(split.calibrate)

    intervals = predictor.predict_intervals(split.test)
    targets = predictor.stacked_targets(split.test)
    # ラベルは washout 前の行数で作られるので、区間と行を合わせてから使う。
    labels, label_kind = detection_labels(split.test)
    labels = labels[predictor.retained_mask(split.test)]
    return _SplitOutcome(
        coverage=float(intervals.covers(targets).mean()),
        auroc=_maybe_auroc(intervals.uncertainty, labels),
        label_kind=label_kind,
        mean_width=float(intervals.uncertainty.mean()),
        n_test_samples=int(targets.shape[0]),
        n_test_episodes=len(split.test),
        n_positive=int(labels.sum()),
        n_negative=int((~labels).sum()),
        calibration_scores=predictor.nonconformity_scores(split.calibrate),
        test_scores=predictor.nonconformity_scores(split.test),
        warning=split.warning,
    )


def run_calibration(
    dataset: RolloutDataset,
    config: ESNConfig,
    *,
    alpha: float = DEFAULT_ALPHA,
    score_kind: ScoreKind = DEFAULT_SCORE_KIND,
    split_strategy: SplitStrategy = DEFAULT_SPLIT_STRATEGY,
    split_seed: int = 0,
    n_splits: int = DEFAULT_N_SPLITS,
    washout: int = DEFAULT_WASHOUT,
    generated_at: str | None = None,
) -> CalibrationReport:
    """データセットに split conformal を掛けて較正レポートを組み立てる。

    ``n_splits`` 個の異なる分割 (乱数種 ``split_seed`` から連番) で評価し、
    被覆率と失敗検知 AUROC は**その平均と散らばり**を報告する。単一分割の値は
    テストエピソード数が少ないため代表値にならない (`CoverageSummary`)。

    reliability curve も分割をまたいで平均する。各名目水準における経験被覆率を
    分割ごとに求めて平均するので、``alpha`` における値は `coverage.mean` と
    一致する。最初の分割だけから引くと、たまたま悪い分割を引いたときに ECE が
    実勢より大きく出て、集約した被覆率と食い違って見える (実測: 集約被覆率
    0.890 に対し単一分割の ECE が 0.20)。

    Args:
        dataset: 評価対象。`validate()` 済みであることを前提とする。
        config: ESN のハイパーパラメータ。
        alpha: 有意水準。名目被覆率は ``1 - alpha``。
        score_kind: 非適合度スコアの種類。
        split_strategy: 較正データの分割方針。
        split_seed: 分割の乱数種 (先頭)。
        n_splits: 評価する分割の数。1 なら単一分割。
        washout: エピソードごとに先頭から捨てる標本数。**`ESNConfig.washout`
            ではない。** あちらは `ESN.fit` の経路だけに効き、この経路は通らない
            (`uncertainty/conformal.py` の `DEFAULT_WASHOUT`)。既定は 0 で、
            初期過渡も「予測しづらい区間」として評価に含める。
        generated_at: タイムスタンプの明示指定 (テスト用)。

    Returns:
        被覆率・reliability curve・失敗検知をまとめた `CalibrationReport`。

    Raises:
        ValueError: `n_splits` が 1 未満、または下層の検証に失敗した場合。
    """
    if n_splits < 1:
        raise ValueError(f"n_splits: 1 以上が必要です (actual={n_splits})")
    samples = build_samples(dataset)
    outcomes = [
        _evaluate_split(
            samples,
            config,
            alpha=alpha,
            score_kind=score_kind,
            split_strategy=split_strategy,
            split_seed=split_seed + offset,
            washout=washout,
        )
        for offset in range(n_splits)
    ]

    coverages = np.asarray([outcome.coverage for outcome in outcomes])
    measured = [outcome.auroc for outcome in outcomes if outcome.auroc is not None]
    aurocs = np.asarray(measured)
    coverage = CoverageSummary(
        nominal=1.0 - alpha,
        mean=float(coverages.mean()),
        std=float(coverages.std()),
        minimum=float(coverages.min()),
        maximum=float(coverages.max()),
        per_split=tuple(float(value) for value in coverages),
        n_splits=n_splits,
        n_test_samples=int(np.mean([outcome.n_test_samples for outcome in outcomes])),
        n_test_episodes=int(np.mean([outcome.n_test_episodes for outcome in outcomes])),
        mean_interval_width=float(
            np.mean([outcome.mean_width for outcome in outcomes])
        ),
        std_interval_width=float(np.std([outcome.mean_width for outcome in outcomes])),
        per_split_interval_width=tuple(
            float(outcome.mean_width) for outcome in outcomes
        ),
    )
    detection = DetectionSummary(
        mean_auroc=float(aurocs.mean()) if measured else None,
        std_auroc=float(aurocs.std()) if measured else None,
        per_split=tuple(float(value) for value in aurocs),
        label=outcomes[0].label_kind,
        n_positive=int(np.mean([outcome.n_positive for outcome in outcomes])),
        n_negative=int(np.mean([outcome.n_negative for outcome in outcomes])),
        unavailable_reason=None if measured else DETECTION_UNAVAILABLE_REASON,
    )

    first = outcomes[0]
    curve = _mean_reliability_curve(outcomes)

    predictor_settings: dict[str, object] = {
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "score_kind": score_kind,
        "n_splits": n_splits,
        "split_seed": split_seed,
        # **この経路で実際に効く washout はこちらである。** レポートに載る
        # `esn_config.washout` は `ESN.fit` 用の値で、較正では使われない (A3)。
        "washout": washout,
    }
    return CalibrationReport(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        generated_at=utc_timestamp() if generated_at is None else generated_at,
        package_version=__version__,
        esn_config=config,
        conformal=predictor_settings,
        split={"strategy": split_strategy, "n_splits": n_splits},
        coverage=coverage,
        reliability=curve,
        detection=detection,
        caveats=_caveats(
            first.warning,
            score_kind,
            first.label_kind,
            dataset.source,
            detection.mean_auroc,
        ),
        data_source=dataset.source,
    )


def _maybe_auroc(
    scores: NDArray[np.float64], labels: NDArray[np.bool_]
) -> float | None:
    """両クラスが揃っているときだけ AUROC を返す。

    openpi のログのように失敗エピソードが 1 つも含まれない分割では陽性が 0 に
    なり AUROC が定義できない。**その場合に例外で止めない。** 被覆率や ECE は
    問題なく計算できるので、検知だけを `None` にして理由を残す。
    """
    if not labels.any() or labels.all():
        return None
    return detection_auroc(scores, labels)


def _mean_reliability_curve(outcomes: Sequence[_SplitOutcome]) -> ReliabilityCurve:
    """分割ごとの reliability curve を名目水準ごとに平均する。

    集約した被覆率と同じ集約方法にすることで、``alpha`` における曲線上の値が
    `CoverageSummary.mean` と一致する。
    """
    curves = [
        reliability_curve(
            outcome.calibration_scores, outcome.test_scores, DEFAULT_NOMINAL_LEVELS
        )
        for outcome in outcomes
    ]
    # 分割ごとに評価できる水準が違いうるので、全分割で共通して評価できた
    # 水準だけを平均する。1 つでも落ちた水準は unsupported に回す。
    supported = sorted(set.intersection(*(set(curve.nominal) for curve in curves)))
    if not supported:
        raise ValueError("reliability curve: 全分割で共通の名目水準がありません")
    empirical = [
        float(
            np.mean([curve.empirical[curve.nominal.index(level)] for curve in curves])
        )
        for level in supported
    ]
    dropped = tuple(
        level for level in DEFAULT_NOMINAL_LEVELS if level not in set(supported)
    )
    return ReliabilityCurve(
        nominal=tuple(supported),
        empirical=tuple(empirical),
        unsupported=dropped,
    )


def _caveats(
    split_warning: str | None,
    score_kind: ScoreKind,
    label_kind: str,
    data_source: DataSource,
    mean_auroc: float | None,
) -> tuple[str, ...]:
    """レポートに載せる注意書きを組み立てる。"""
    caveats = [EFFECTIVE_SAMPLE_SIZE_CAVEAT, DETECTION_IS_EXPLORATORY_CAVEAT]
    if data_source == "synthetic":
        caveats.insert(0, SYNTHETIC_DATA_CAVEAT)
    if label_kind == "episode_success":
        caveats.append(EPISODE_LABEL_CAVEAT)
    if split_warning is not None:
        caveats.append(split_warning)
    if score_kind == "absolute":
        caveats.append(ABSOLUTE_SCORE_CAVEAT)
    elif mean_auroc is not None and mean_auroc < INVERTED_DETECTION_THRESHOLD:
        caveats.append(INVERTED_DETECTION_CAVEAT)
    return tuple(caveats)
