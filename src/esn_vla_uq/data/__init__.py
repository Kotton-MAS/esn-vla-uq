"""ロールアウトデータのスキーマ・合成生成・入出力。

`schema.py` がデータ契約 (v0.1)、`synthetic.py` が決定論的な合成生成、`io.py` が
`.npz` + `metadata.json` の入出力、`source.py` が供給元の Protocol を担う。openpi の
ロールアウトログは `RolloutSource` Protocol 経由で差し替え可能にし、openpi を
ランタイム依存に含めない。

Sprint 1 で提供するデータはすべて合成データ (`source: "synthetic"`) であり、実 LIBERO の
ロールアウトではない。
"""

from esn_vla_uq.data.io import (
    BUNDLED_SAMPLE_ARCHIVE,
    BUNDLED_SAMPLE_METADATA,
    bundled_sample_size_bytes,
    load_bundled_sample,
    load_dataset,
    metadata_path_for,
    save_dataset,
)
from esn_vla_uq.data.schema import (
    ACTION_DIM,
    CHUNK_HORIZON,
    SCHEMA_VERSION,
    STATE_DIM,
    SUPPORTED_SCHEMA_VERSIONS,
    DataSource,
    Episode,
    RolloutDataset,
    validate_episode_index,
)
from esn_vla_uq.data.source import RolloutSource, SyntheticRolloutSource
from esn_vla_uq.data.synthetic import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MIN_STEPS,
    DEFAULT_N_EPISODES,
    DEFAULT_SUCCESS_RATE,
    generate_dataset,
)

__all__ = [
    "ACTION_DIM",
    "BUNDLED_SAMPLE_ARCHIVE",
    "BUNDLED_SAMPLE_METADATA",
    "CHUNK_HORIZON",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MIN_STEPS",
    "DEFAULT_N_EPISODES",
    "DEFAULT_SUCCESS_RATE",
    "SCHEMA_VERSION",
    "STATE_DIM",
    "SUPPORTED_SCHEMA_VERSIONS",
    "DataSource",
    "Episode",
    "RolloutDataset",
    "RolloutSource",
    "SyntheticRolloutSource",
    "bundled_sample_size_bytes",
    "generate_dataset",
    "load_bundled_sample",
    "load_dataset",
    "metadata_path_for",
    "save_dataset",
    "validate_episode_index",
]
