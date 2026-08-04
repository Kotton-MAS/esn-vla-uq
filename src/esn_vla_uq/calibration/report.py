"""較正レポートの構築と書き出し。

`diagnostics/report.py` と同じ分担にする。実行の組み立ては `runner.py`、
ここは結果の表現 (レポート型・JSON 化・ログ整形) のみを担う。

数値の出所は Sprint 2 時点では常に合成データであり、実 LIBERO 評価の結果では
ない (`docs/design.md` 7 節の誠実性宣言)。`data_source` を必ず記録する。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from esn_vla_uq.calibration.metrics import ReliabilityCurve
from esn_vla_uq.esn.config import ESNConfig
from esn_vla_uq.logging_paths import display_path
from esn_vla_uq.provenance import SYNTHETIC_DATA_SOURCE, DataSource

logger = logging.getLogger(__name__)

CALIBRATION_SCHEMA_VERSION: Final[str] = "0.1.0"
"""較正レポート JSON のスキーマバージョン。"""

REPORT_SUBDIR: Final[str] = "calibration"
"""`--output-dir` 配下の書き出し先サブディレクトリ。"""

DIAGRAM_FILENAME: Final[str] = "reliability.png"
"""reliability diagram のファイル名。"""


@dataclass(frozen=True)
class DetectionSummary:
    """失敗検知の成績。

    Attributes:
        mean_auroc: 分割をまたいだ AUROC の平均。不確実性スコアが「失敗開始
            以降のステップ」をどれだけ順位付けできるか。`absolute` スコアでは
            区間幅が定数 (全て同順位) になるため**定義上厳密に 0.5**。
        std_auroc: AUROC の標準偏差。
        per_split: 分割ごとの AUROC。
        label: 使ったラベルの種類。`"failure_onset"` (細かい) と
            `"episode_success"` (粗い) では数値の意味が違うため必ず記録する。
        n_positive: 陽性ステップ数 (1 分割あたり平均)。
        n_negative: 陰性ステップ数 (1 分割あたり平均)。
        unavailable_reason: 計算できなかった理由。計算できた場合は `None`。
    """

    mean_auroc: float | None
    std_auroc: float | None
    per_split: tuple[float, ...]
    label: str
    n_positive: int
    n_negative: int
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON シリアライズ可能な辞書へ変換する。"""
        payload = asdict(self)
        payload["per_split"] = list(self.per_split)
        return payload


@dataclass(frozen=True)
class CoverageSummary:
    """要求水準における被覆率 (複数分割の集約)。

    **単一分割の被覆率は報告しない。** 被覆率の分散を決めるのはテスト
    エピソード数であってステップ数ではなく、同梱データ規模では単一分割の値が
    0.63〜1.00 まで振れる。1 つの数字だけを見せると、たまたま悪い分割を引いた
    ときに「較正が壊れている」、良い分割を引いたときに「完璧だ」と誤読される。
    平均と散らばりを併記する。

    Attributes:
        nominal: 名目被覆率 ``1 - alpha``。
        mean: 分割をまたいだ実測被覆率の平均。**これが代表値**。
        std: 実測被覆率の標準偏差。
        minimum: 最小値。
        maximum: 最大値。
        per_split: 分割ごとの実測被覆率。
        n_splits: 評価した分割数。
        n_test_samples: 1 分割あたりのテスト標本数 (ステップ、平均)。
        n_test_episodes: 1 分割あたりのテストエピソード数 (平均)。
        mean_interval_width: 区間半幅の平均 (次元方向 max を取ったもの)。
    """

    nominal: float
    mean: float
    std: float
    minimum: float
    maximum: float
    per_split: tuple[float, ...]
    n_splits: int
    n_test_samples: int
    n_test_episodes: int
    mean_interval_width: float

    def to_dict(self) -> dict[str, object]:
        """JSON シリアライズ可能な辞書へ変換する。"""
        payload = asdict(self)
        payload["per_split"] = list(self.per_split)
        return payload


@dataclass(frozen=True)
class CalibrationReport:
    """較正評価の結果一式。

    Attributes:
        schema_version: レポート JSON のスキーマバージョン。
        generated_at: UTC ISO8601 タイムスタンプ。
        package_version: パッケージのバージョン。
        data_source: 数値の出所。Sprint 2 では常に ``"synthetic"``。
        esn_config: 使った ESN のハイパーパラメータ。
        conformal: conformal の設定と較正結果 (`SplitConformalPredictor.to_dict`)。
        split: 較正データ分割の内訳 (`CalibrationSplit.to_dict`)。
        coverage: 要求水準における被覆率。
        reliability: reliability curve と ECE。
        detection: 失敗検知 AUROC。
        caveats: 読み手が数値を過大評価しないための注意書き。
    """

    schema_version: str
    generated_at: str
    package_version: str
    esn_config: ESNConfig
    conformal: dict[str, object]
    split: dict[str, object]
    coverage: CoverageSummary
    reliability: ReliabilityCurve
    detection: DetectionSummary
    caveats: tuple[str, ...]
    data_source: DataSource = SYNTHETIC_DATA_SOURCE

    def to_dict(self) -> dict[str, object]:
        """JSON シリアライズ可能な辞書へ変換する。"""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "package_version": self.package_version,
            "data_source": self.data_source,
            "esn_config": self.esn_config.to_dict(),
            "conformal": self.conformal,
            "split": self.split,
            "coverage": self.coverage.to_dict(),
            "reliability": self.reliability.to_dict(),
            "detection": self.detection.to_dict(),
            "caveats": list(self.caveats),
        }

    def log_summary(self) -> None:
        """1 指標 1 行の人間可読サマリを `logging.info` で出す。"""
        logger.info(
            "calibration: schema_version=%s data_source=%s score_kind=%s "
            "split=%s generated_at=%s",
            self.schema_version,
            self.data_source,
            self.conformal["score_kind"],
            self.split["strategy"],
            self.generated_at,
        )
        logger.info(
            "coverage: nominal=%.4f mean=%.4f std=%.4f range=[%.4f, %.4f] "
            "n_splits=%d n_test_episodes=%d mean_half_width=%.6f",
            self.coverage.nominal,
            self.coverage.mean,
            self.coverage.std,
            self.coverage.minimum,
            self.coverage.maximum,
            self.coverage.n_splits,
            self.coverage.n_test_episodes,
            self.coverage.mean_interval_width,
        )
        logger.info(
            "reliability: ece=%.4f max_calibration_error=%.4f n_levels=%d",
            self.reliability.expected_calibration_error(),
            self.reliability.max_calibration_error(),
            len(self.reliability.nominal),
        )
        if self.detection.mean_auroc is None:
            logger.warning(
                "detection: 計算できませんでした (%s)",
                self.detection.unavailable_reason,
            )
        else:
            logger.info(
                "detection: auroc=%.4f (std=%.4f) label=%s n_positive=%d n_negative=%d",
                self.detection.mean_auroc,
                self.detection.std_auroc,
                self.detection.label,
                self.detection.n_positive,
                self.detection.n_negative,
            )
        for caveat in self.caveats:
            logger.warning("caveat: %s", caveat)


def utc_timestamp(moment: datetime | None = None) -> str:
    """UTC の ISO8601 タイムスタンプ (マイクロ秒まで) を返す。"""
    now = moment if moment is not None else datetime.now(UTC)
    return now.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _filename_stem(generated_at: str) -> str:
    """ISO8601 タイムスタンプをファイル名に使える形へ整える。"""
    return generated_at.replace("-", "").replace(":", "").replace(".", "")


def write_report(report: CalibrationReport, output_dir: Path) -> Path:
    """レポートを ``<output_dir>/calibration/<timestamp>.json`` へ書き出す。

    Args:
        report: 書き出す較正レポート。
        output_dir: 共通オプション `--output-dir` の値。

    Returns:
        書き出した JSON のパス。
    """
    directory = Path(output_dir) / REPORT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_filename_stem(report.generated_at)}.json"
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # 絶対パスはユーザー名を含みうるため INFO には出さない (S4)。
    logger.info("saved calibration report: path=%s", display_path(path))
    logger.debug("saved calibration report: abs_path=%s", path)
    return path
