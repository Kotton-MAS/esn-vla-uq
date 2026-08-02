"""診断レポートの構築・JSON 書き出し・人間可読サマリ。

収録内容は `docs/design.md` の 4.4 節に従う (ただし `n_inputs` は当初の設計に
なかった追加フィールド。詳細は `run_diagnostics` の docstring を参照)。JSON
本体はファイルへ書き出し、`logging.info` では 1 指標 1 行のサマリのみを出す
(`print()` は使わない)。

`data_source` は Sprint 1 では常に ``"synthetic"``。本レポートの数値は同梱の
合成データと同じ合成的な設定に由来し、実 LIBERO 評価の結果ではない
(`docs/design.md` 7 節の誠実性宣言)。同じ理由で `n_inputs` を必ず記録する:
リザバー行列の生成は `seed` と入力次元 `D_u` の両方に依存するため、`n_inputs`
を書かないと `spectral`/`esp` の数値がどの `D_u` のリザバーのものか事後に
判別できず、実データを fit したリザバーの性質だと誤読されうる。

同じ理由で `memory_capacity` は測定に使ったリザバーの入力次元 (`n_inputs`) と
`spectral`/`esp` と同じリザバーで測ったか (`reservoir`) を **自身のオブジェクト
の中に** 埋め込む (`MemoryCapacityMeasurement`)。以前はこれをトップレベルの
`memory_capacity_n_inputs` という別フィールドに分けていたが、`memory_capacity`
が `None` の場合にどちらか片方だけ `None` にする実装ミスを型で防げず、JSON を
単体で読んでも `memory_capacity` オブジェクトだけでは測定コンテキストが分から
なかった (`docs/design.md` 4.4 節、`REPORT_SCHEMA_VERSION` 0.2.0 で解消)。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np

from esn_vla_uq import __version__
from esn_vla_uq.diagnostics.esp import EspResult, check_esp
from esn_vla_uq.diagnostics.memory_capacity import (
    MEMORY_CAPACITY_INPUT_DIM,
    MemoryCapacityResult,
    linear_memory_capacity,
)
from esn_vla_uq.diagnostics.spectral import effective_spectral_radius, spectral_radius
from esn_vla_uq.esn.config import ESNConfig
from esn_vla_uq.esn.reservoir import Reservoir

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION: Final[str] = "0.2.0"
"""診断レポート JSON のスキーマバージョン。

0.2.0 での変更: `memory_capacity` の測定コンテキスト (`n_inputs` / `reservoir`)
を `memory_capacity` オブジェクトの内側へ移し、トップレベルの
`memory_capacity_n_inputs` フィールドを廃止した (M1、`docs/design.md` 4.4 節)。
"""

REPORT_SUBDIR: Final[str] = "diagnostics"
"""`--output-dir` 配下の書き出し先サブディレクトリ。"""

DataSource = Literal["synthetic", "openpi"]
"""数値の出所。Sprint 1 は常に ``"synthetic"``。"""

SYNTHETIC_DATA_SOURCE: Final[DataSource] = "synthetic"


def utc_timestamp(moment: datetime | None = None) -> str:
    """UTC の ISO8601 タイムスタンプ (マイクロ秒まで) を返す。"""
    now = moment if moment is not None else datetime.now(UTC)
    return now.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _filename_stem(generated_at: str) -> str:
    """ISO8601 タイムスタンプをファイル名に使える形へ整える。"""
    return generated_at.replace("-", "").replace(":", "").replace(".", "")


@dataclass(frozen=True)
class MemoryCapacityMeasurement:
    """メモリ容量の測定結果と、その測定コンテキストを 1 つにまとめたもの。

    `MemoryCapacityResult` 自体はどのリザバーで測ったかを知らない (測定関数
    `linear_memory_capacity` はリザバーを引数に取るだけ)。レポート JSON を
    自己記述的にするには「どの `D_u` のリザバーで測ったか」「`spectral`/`esp`
    と同じリザバーか別か」を結果と**同じオブジェクトの中**に持たせる必要が
    ある (別々の nullable フィールドに分けると、`memory_capacity` を省略した
    ときに片方だけ `None` にする実装ミスを型で防げない)。この情報はレポート
    (`DiagnosticsReport`) の文脈でのみ意味を持つため、`diagnostics/
    memory_capacity.py` の `MemoryCapacityResult` 自体には持たせず、
    `report.py` 側でラップする。

    Attributes:
        result: `linear_memory_capacity` の測定結果。
        n_inputs: 測定に使ったリザバーの入力次元 `D_u`
            (`MEMORY_CAPACITY_INPUT_DIM` に固定)。

    `n_inputs` は `run_diagnostics` からは常に `MEMORY_CAPACITY_INPUT_DIM` で
    構築されるため整合するが、この型自体は `diagnostics/__init__.py` で公開
    エクスポートされており、外部から単体で組み立てられうる。
    `linear_memory_capacity` の契約上 `n_inputs != MEMORY_CAPACITY_INPUT_DIM`
    はそもそも測定条件として無効であり、`reservoir_label` は「同じ文脈で
    正しく組み立てられている」ことに暗黙に依存して "shared"/"separate" を
    返す。呼び出し側が誤った `n_inputs` を渡すと、実際には測定していない
    次元のリザバーと同じ/別扱いされた嘘のラベルを出しうるため、
    `__post_init__` でこの契約を型ではなく実行時に強制する
    (`reservoir_label` を廃して `reservoir: Literal["shared","separate"]` を
    フィールド化する代替案もあったが、`n_inputs` 自体を検証するほうが
    `MemoryCapacityResult` との対応が直接的で、既存の呼び出し側
    (`run_diagnostics`/`_memory_capacity_to_dict`) の変更が不要なため採用した)。
    """

    result: MemoryCapacityResult
    n_inputs: int

    def __post_init__(self) -> None:
        if self.n_inputs != MEMORY_CAPACITY_INPUT_DIM:
            raise ValueError(
                "n_inputs: メモリ容量測定はスカラー入力のみ対応します "
                f"(actual={self.n_inputs}, expected={MEMORY_CAPACITY_INPUT_DIM})"
            )

    def reservoir_label(self, report_n_inputs: int) -> Literal["shared", "separate"]:
        """`spectral`/`esp` (入力次元 `report_n_inputs`) と同じリザバーで
        測ったかを返す。"""
        return "shared" if self.n_inputs == report_n_inputs else "separate"


@dataclass(frozen=True)
class SpectralSummary:
    """スペクトル指標のまとめ。

    Attributes:
        spectral_radius: 再帰行列 ``W`` のスペクトル半径。
        effective_spectral_radius: 実効更新行列 ``(1 - a) I + a W`` の
            スペクトル半径 (``a = leak_rate``)。
    """

    spectral_radius: float
    effective_spectral_radius: float


@dataclass(frozen=True)
class DiagnosticsReport:
    """リザバー診断の結果一式 (JSON 化してファイルへ書き出す)。

    Attributes:
        n_inputs: `spectral` / `esp` の計算に使ったリザバーの入力次元
            `D_u`。`Reservoir.__init__` は seed が同じでも `n_inputs` が
            変われば `W_in` -> `b` -> `W` の消費順序の結果として全く別の行列を
            生成するため、この値を記録しないとスペクトル半径・ESP の数値が
            どのリザバーのものか事後に再現できない。
        memory_capacity: メモリ容量の測定結果と測定コンテキスト
            (`MemoryCapacityMeasurement`)。メモリ容量診断は仕様上スカラー入力
            (`D_u=1`) を要求するため、`n_inputs != 1` のときは `spectral`/
            `esp` を計算したリザバーとは **別の** `D_u=1` リザバーで測って
            いる。`--skip-memory-capacity` 指定時は `None`。
    """

    schema_version: str
    generated_at: str
    package_version: str
    numpy_version: str
    esn_config: ESNConfig
    seed: int
    n_inputs: int
    spectral: SpectralSummary
    esp: EspResult
    memory_capacity: MemoryCapacityMeasurement | None
    data_source: DataSource = SYNTHETIC_DATA_SOURCE

    def to_dict(self) -> dict[str, object]:
        """JSON シリアライズ可能な辞書へ変換する。"""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "package_version": self.package_version,
            "numpy_version": self.numpy_version,
            "data_source": self.data_source,
            "seed": self.seed,
            "n_inputs": self.n_inputs,
            "esn_config": _config_to_dict(self.esn_config),
            "spectral": {
                "spectral_radius": self.spectral.spectral_radius,
                "effective_spectral_radius": self.spectral.effective_spectral_radius,
            },
            "esp": _esp_to_dict(self.esp),
            "memory_capacity": (
                None
                if self.memory_capacity is None
                else _memory_capacity_to_dict(self.memory_capacity, self.n_inputs)
            ),
        }

    def log_summary(self) -> None:
        """1 指標 1 行の人間可読サマリを `logging.info` で出す。"""
        logger.info(
            "diagnostics: schema_version=%s data_source=%s seed=%d "
            "n_reservoir=%d n_inputs=%d generated_at=%s",
            self.schema_version,
            self.data_source,
            self.seed,
            self.esn_config.n_reservoir,
            self.n_inputs,
            self.generated_at,
        )
        logger.info(
            "spectral: spectral_radius=%.6f effective_spectral_radius=%.6f",
            self.spectral.spectral_radius,
            self.spectral.effective_spectral_radius,
        )
        # ESP は 3 指標を必ず併記する (design.md 4.2 節)。
        logger.info(
            "esp: verdict=%s sufficient(sigma_max<1)=%s[%.6f] "
            "necessary(rho<1)=%s[%.6f] empirical(d(T)<%.1e)=%s[%.3e] decay_rate=%.6f",
            self.esp.verdict,
            self.esp.sufficient_condition_met,
            self.esp.largest_singular_value,
            self.esp.necessary_condition_met,
            self.esp.effective_spectral_radius,
            self.esp.tolerance,
            self.esp.empirical_converged,
            self.esp.final_distance,
            self.esp.decay_rate,
        )
        if self.memory_capacity is None:
            logger.info("memory_capacity: skipped (--skip-memory-capacity)")
            return
        # reservoir_label が "separate" のとき、spectral/esp とは別の
        # (D_u=1 の) リザバーで測定したことを明記する (誠実性方針)。
        result = self.memory_capacity.result
        reservoir_note: str = self.memory_capacity.reservoir_label(self.n_inputs)
        if reservoir_note == "separate":
            reservoir_note = f"separate(n_inputs={self.memory_capacity.n_inputs})"
        logger.info(
            "memory_capacity: total_mc=%.4f mc_per_neuron=%.4f "
            "memory_horizon=%d n_delays=%d reservoir=%s",
            result.total_mc,
            result.mc_per_neuron,
            result.memory_horizon,
            result.n_delays,
            reservoir_note,
        )


def _config_to_dict(config: ESNConfig) -> dict[str, object]:
    """`ESNConfig` の全フィールドを辞書にする。"""
    return {
        "n_reservoir": config.n_reservoir,
        "spectral_radius": config.spectral_radius,
        "input_scaling": config.input_scaling,
        "bias_scaling": config.bias_scaling,
        "leak_rate": config.leak_rate,
        "density": config.density,
        "ridge_alpha": config.ridge_alpha,
        "washout": config.washout,
        "input_passthrough": config.input_passthrough,
        "seed": config.seed,
    }


def _esp_to_dict(result: EspResult) -> dict[str, object]:
    """`EspResult` の全フィールドを辞書にする (3 指標を必ず併記)。"""
    return {
        "verdict": result.verdict,
        "sufficient_condition_met": result.sufficient_condition_met,
        "necessary_condition_met": result.necessary_condition_met,
        "empirical_converged": result.empirical_converged,
        "decay_rate": result.decay_rate,
        "largest_singular_value": result.largest_singular_value,
        "effective_spectral_radius": result.effective_spectral_radius,
        "final_distance": result.final_distance,
        "tolerance": result.tolerance,
        "n_initial_states": result.n_initial_states,
        "n_steps": result.n_steps,
        "zero_input": result.zero_input,
    }


def _memory_capacity_to_dict(
    measurement: MemoryCapacityMeasurement, report_n_inputs: int
) -> dict[str, object]:
    """`MemoryCapacityMeasurement` を、測定コンテキストを埋め込んだ自己記述的な
    辞書にする (`n_inputs` / `reservoir`)。"""
    result = measurement.result
    return {
        "total_mc": result.total_mc,
        "mc_per_neuron": result.mc_per_neuron,
        "memory_horizon": result.memory_horizon,
        "n_delays": result.n_delays,
        "per_delay": list(result.per_delay),
        "n_inputs": measurement.n_inputs,
        "reservoir": measurement.reservoir_label(report_n_inputs),
    }


def summarize_spectral(reservoir: Reservoir) -> SpectralSummary:
    """リザバーからスペクトル指標を計算する。"""
    return SpectralSummary(
        spectral_radius=spectral_radius(reservoir.W),
        effective_spectral_radius=effective_spectral_radius(
            reservoir.W, reservoir.config.leak_rate
        ),
    )


DEFAULT_DIAGNOSTICS_N_INPUTS: Final[int] = 1
"""`run_diagnostics` の既定入力次元 (`--n-inputs` の既定値)。"""


def run_diagnostics(
    config: ESNConfig,
    *,
    n_inputs: int = DEFAULT_DIAGNOSTICS_N_INPUTS,
    seed: int | None = None,
    skip_memory_capacity: bool = False,
    generated_at: str | None = None,
) -> DiagnosticsReport:
    """スペクトル / ESP / メモリ容量を実行して `DiagnosticsReport` を組み立てる。

    ``spectral`` / ``esp`` は入力次元 ``n_inputs`` のリザバー 1 個で計算する
    (行列の生成は `ESNConfig.seed` と入力次元の両方に依存するため、指標間で
    別のリザバーを見てしまわないよう同じリザバーを使い回す)。既定
    ``n_inputs=1`` は互換のための既定値であり、実データを fit する ESN の
    実際の `D_u` (例: `state` なら 8) に近づけたい場合は明示的に指定する。

    メモリ容量診断はスカラー入力 (`D_u=1`) を要求する。``n_inputs == 1`` の
    ときは上記と同じリザバーをそのまま使うが、``n_inputs != 1`` のときは
    `spectral`/`esp` のリザバーとは別に `D_u=1` のリザバーを ``config`` から
    改めて構築して測る (同じ `seed` でも `n_inputs` が違えば `W_in`/`b`/`W` は
    別物になるため、一診断の都合でレポート全体のリザバーを決めない)。
    どちらのリザバーで測ったかは `MemoryCapacityMeasurement.n_inputs`
    (`DiagnosticsReport.memory_capacity.n_inputs`) に必ず記録する。

    Args:
        config: 診断対象の ESN ハイパーパラメータ。
        n_inputs: `spectral`/`esp` を計算するリザバーの入力次元 `D_u`。
        seed: 診断の乱数種 (テスト入力・初期状態・メモリ容量入力)。省略時は
            `ESNConfig.seed` を使う。
        skip_memory_capacity: True なら K 本の read-out 学習を伴うメモリ容量
            診断を省略する (レポートでは ``null``)。
        generated_at: タイムスタンプの明示指定 (テスト用)。省略時は現在時刻。

    Returns:
        3 指標を収録した `DiagnosticsReport`。
    """
    diagnostics_seed = config.seed if seed is None else seed
    reservoir = Reservoir(config, n_inputs)

    memory_capacity: MemoryCapacityMeasurement | None = None
    if not skip_memory_capacity:
        memory_capacity_reservoir = (
            reservoir
            if n_inputs == MEMORY_CAPACITY_INPUT_DIM
            else Reservoir(config, MEMORY_CAPACITY_INPUT_DIM)
        )
        memory_capacity = MemoryCapacityMeasurement(
            result=linear_memory_capacity(
                memory_capacity_reservoir, seed=diagnostics_seed
            ),
            n_inputs=MEMORY_CAPACITY_INPUT_DIM,
        )

    return DiagnosticsReport(
        schema_version=REPORT_SCHEMA_VERSION,
        generated_at=utc_timestamp() if generated_at is None else generated_at,
        package_version=__version__,
        numpy_version=np.__version__,
        esn_config=config,
        seed=diagnostics_seed,
        n_inputs=n_inputs,
        spectral=summarize_spectral(reservoir),
        esp=check_esp(reservoir, seed=diagnostics_seed),
        memory_capacity=memory_capacity,
        data_source=SYNTHETIC_DATA_SOURCE,
    )


def write_report(report: DiagnosticsReport, output_dir: Path) -> Path:
    """レポートを ``<output_dir>/diagnostics/<timestamp>.json`` へ書き出す。

    Args:
        report: 書き出す診断レポート。
        output_dir: 共通オプション `--output-dir` の値 (既定 ``outputs/``)。

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
    logger.info("saved diagnostics report: path=%s", path)
    return path
