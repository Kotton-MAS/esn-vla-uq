"""不確実性定量化 (split conformal prediction)。

ESN の 1 ステップ先 action 予測に split conformal を掛け、ステップ単位の予測区間と
不確実性スコアを出す。定義と設計判断は `docs/plans/sprint2_v0.1.md` と
`docs/design.md` 6.3 節を参照。

本パッケージは `data` 層と `esn` 層に依存し、`calibration` 層からは依存される
(逆方向は無い)。
"""

from esn_vla_uq.uncertainty.conformal import (
    DEFAULT_ALPHA,
    PredictionIntervals,
    SplitConformalPredictor,
    conformal_quantile_index,
)
from esn_vla_uq.uncertainty.nonconformity import (
    DEFAULT_SCORE_KIND,
    SUPPORTED_SCORE_KINDS,
    ScoreKind,
    ScoreModel,
    fit_score_model,
)
from esn_vla_uq.uncertainty.split import (
    ACROSS_TASK_WARNING,
    DEFAULT_SPLIT_STRATEGY,
    SUPPORTED_SPLIT_STRATEGIES,
    CalibrationSplit,
    SplitStrategy,
    split_samples,
)
from esn_vla_uq.uncertainty.targets import (
    EpisodeSamples,
    build_samples,
    stack_failure_labels,
    stack_targets,
)

__all__ = [
    "ACROSS_TASK_WARNING",
    "DEFAULT_ALPHA",
    "DEFAULT_SCORE_KIND",
    "DEFAULT_SPLIT_STRATEGY",
    "SUPPORTED_SCORE_KINDS",
    "SUPPORTED_SPLIT_STRATEGIES",
    "CalibrationSplit",
    "EpisodeSamples",
    "PredictionIntervals",
    "ScoreKind",
    "ScoreModel",
    "SplitConformalPredictor",
    "SplitStrategy",
    "build_samples",
    "conformal_quantile_index",
    "fit_score_model",
    "split_samples",
    "stack_failure_labels",
    "stack_targets",
]
