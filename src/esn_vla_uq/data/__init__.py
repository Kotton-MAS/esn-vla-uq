"""ロールアウトデータのスキーマ・合成生成・入出力・特徴量。

モジュール構成:

- `schema.py`: データ契約 (v0.1)。他の data モジュールが依存する最下層。
- `invariants.py`: 出所ごとの追加不変条件。依存は `schema.py` のみ。
- `synthetic.py`: 決定論的な合成生成。
- `io.py`: `.npz` + `metadata.json` の入出力。
- `sources/`: 供給元の Protocol (`base.py`) と具象 (`synthetic.py`)。
- `features.py`: `RolloutDataset` から ESN 入力を取り出す変換。

openpi のロールアウトログは `RolloutSource` Protocol 経由で差し替え可能にし、
openpi をランタイム依存に含めない。この分離は規律ではなく import 構造で保つ
(`io.py` と `sources/base.py` はどちらも具象供給元を import しない)。

Sprint 1 で提供するデータはすべて合成データ (`source: "synthetic"`) であり、実 LIBERO の
ロールアウトではない。
"""

from esn_vla_uq.data.features import (
    DatasetInputs,
    FeatureSet,
    dataset_inputs,
)
from esn_vla_uq.data.invariants import validate_by_source, validate_synthetic_dataset
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
    Episode,
    RolloutDataset,
    validate_episode_index,
)
from esn_vla_uq.data.sources import RolloutSource, SyntheticRolloutSource
from esn_vla_uq.data.synthetic import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MIN_STEPS,
    DEFAULT_N_EPISODES,
    DEFAULT_SUCCESS_RATE,
    generate_dataset,
)
from esn_vla_uq.provenance import DataSource

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
    "DatasetInputs",
    "Episode",
    "FeatureSet",
    "RolloutDataset",
    "RolloutSource",
    "SyntheticRolloutSource",
    "bundled_sample_size_bytes",
    "dataset_inputs",
    "generate_dataset",
    "load_bundled_sample",
    "load_dataset",
    "metadata_path_for",
    "save_dataset",
    "validate_by_source",
    "validate_episode_index",
    "validate_synthetic_dataset",
]
