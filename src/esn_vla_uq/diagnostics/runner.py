"""診断の実行オーケストレーション。

どのリザバーで何を測るかを決め、`diagnostics/report.py` の
`DiagnosticsReport` を組み立てるところまでを担う。レポートの表現
(辞書化・JSON 書き出し・ログ整形) には関与しない。

以前は `report.py` が「実行 / 辞書化 / ファイル書き出し / ログ整形」の 4 責務を
すべて持っていた (A2)。診断を 1 つ追加すると、実行順の決定・レポート型の
フィールド追加・辞書化の列挙・ログ行の追加が 1 ファイルの中で混ざり、どの
変更がレポートの**内容**を変え、どれが**表現**を変えるのかが読み取れなく
なっていた。ここでは実行だけを扱う。

依存の向きは `runner.py` -> `report.py` の一方向。`report.py` はこのモジュールを
import しない。
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

from esn_vla_uq import __version__
from esn_vla_uq.diagnostics.esp import check_esp
from esn_vla_uq.diagnostics.memory_capacity import (
    MEMORY_CAPACITY_INPUT_DIM,
    linear_memory_capacity,
)
from esn_vla_uq.diagnostics.report import (
    REPORT_SCHEMA_VERSION,
    DiagnosticsReport,
    MemoryCapacityMeasurement,
    SpectralSummary,
    utc_timestamp,
)
from esn_vla_uq.diagnostics.spectral import effective_spectral_radius, spectral_radius
from esn_vla_uq.esn.config import ESNConfig
from esn_vla_uq.esn.reservoir import Reservoir
from esn_vla_uq.provenance import SYNTHETIC_DATA_SOURCE

logger = logging.getLogger(__name__)

DEFAULT_DIAGNOSTICS_N_INPUTS: Final[int] = 1
"""`run_diagnostics` の既定入力次元 (`--n-inputs` の既定値)。"""


def summarize_spectral(reservoir: Reservoir) -> SpectralSummary:
    """リザバーからスペクトル指標を計算する。

    `spectral_radius(reservoir.W)` は `reservoir.config.spectral_radius` で
    代替しない。この診断の目的は「設定値どおりのスペクトル半径が実際に達成
    されているか」を実測で検証することであり、設定値をそのまま出力しては
    検証にならない (`docs/next-pr-candidates.md` の「不採用」節)。
    """
    return SpectralSummary(
        spectral_radius=spectral_radius(reservoir.W),
        effective_spectral_radius=effective_spectral_radius(
            reservoir.W, reservoir.config.leak_rate
        ),
    )


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
    spectral = summarize_spectral(reservoir)

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
        spectral=spectral,
        # `rho(A)` は `summarize_spectral` が計算済み。同じ行列の固有値を
        # 2 度求めない (P2。N=500 で diagnose 全体の約 18%)。
        esp=check_esp(
            reservoir,
            seed=diagnostics_seed,
            effective_spectral_radius=spectral.effective_spectral_radius,
        ),
        memory_capacity=memory_capacity,
        data_source=SYNTHETIC_DATA_SOURCE,
    )
