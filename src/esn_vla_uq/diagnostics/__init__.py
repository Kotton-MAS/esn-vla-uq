"""リザバー診断 (スペクトル半径 / ESP / メモリ容量) とレポート出力。

各指標の定義式・判定閾値・依拠文献は `docs/design.md` の「診断指標の定義」節
(4 節) に従う。ESP は 3 指標 (十分条件 / 必要条件 / 経験的収束) を必ず併記し、
単一の指標だけで結論を出さない。

本パッケージは `esn` 層が生成したリザバー (`W` / `W_in` / `b` と `ESNConfig`) を
入力に取り、`esn` 層へ書き戻すことはしない。
"""

from esn_vla_uq.diagnostics.esp import (
    DEFAULT_ESP_N_INITIAL_STATES,
    DEFAULT_ESP_N_STEPS,
    DEFAULT_ESP_TOL,
    EspResult,
    EspVerdict,
    check_esp,
    default_test_inputs,
)
from esn_vla_uq.diagnostics.memory_capacity import (
    DEFAULT_MC_N_TEST,
    DEFAULT_MC_N_TRAIN,
    DEFAULT_MC_RIDGE_ALPHA,
    DEFAULT_MC_WASHOUT,
    MEMORY_CAPACITY_INPUT_DIM,
    MemoryCapacityResult,
    default_max_delay,
    linear_memory_capacity,
)
from esn_vla_uq.diagnostics.report import (
    DEFAULT_DIAGNOSTICS_N_INPUTS,
    REPORT_SCHEMA_VERSION,
    REPORT_SUBDIR,
    DiagnosticsReport,
    MemoryCapacityMeasurement,
    SpectralSummary,
    run_diagnostics,
    summarize_spectral,
    utc_timestamp,
    write_report,
)
from esn_vla_uq.diagnostics.spectral import (
    effective_spectral_radius,
    effective_update_matrix,
    largest_singular_value,
    spectral_radius,
)

__all__ = [
    "DEFAULT_DIAGNOSTICS_N_INPUTS",
    "DEFAULT_ESP_N_INITIAL_STATES",
    "DEFAULT_ESP_N_STEPS",
    "DEFAULT_ESP_TOL",
    "DEFAULT_MC_N_TEST",
    "DEFAULT_MC_N_TRAIN",
    "DEFAULT_MC_RIDGE_ALPHA",
    "DEFAULT_MC_WASHOUT",
    "MEMORY_CAPACITY_INPUT_DIM",
    "REPORT_SCHEMA_VERSION",
    "REPORT_SUBDIR",
    "DiagnosticsReport",
    "EspResult",
    "EspVerdict",
    "MemoryCapacityMeasurement",
    "MemoryCapacityResult",
    "SpectralSummary",
    "check_esp",
    "default_max_delay",
    "default_test_inputs",
    "effective_spectral_radius",
    "effective_update_matrix",
    "largest_singular_value",
    "linear_memory_capacity",
    "run_diagnostics",
    "spectral_radius",
    "summarize_spectral",
    "utc_timestamp",
    "write_report",
]
