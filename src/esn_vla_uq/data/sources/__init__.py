"""ロールアウトデータの供給元 (Protocol と具象実装)。

- `base.RolloutSource`: 抽象。依存は `data/schema.py` のみ。
- `synthetic.SyntheticRolloutSource`: 合成データの具象。

**任意依存を持つ具象供給元をここで eager import しないこと。** Sprint 2 の
`OpenpiLogSource` は openpi のログ形式に依存するため、
`esn_vla_uq.data.sources.openpi` を利用側が明示的に import する形にする。
ここで再エクスポートすると、`RolloutSource` を型注釈に使いたいだけの
コードが openpi 側のモジュールまでロードすることになり、抽象と具象を
分けた意味 (A1) が失われる。

`SyntheticRolloutSource` は任意依存を持たない (numpy と本パッケージのみ) ため
ここから再エクスポートする。この線引きは import 構造では強制できないので、
規約として本 docstring に置く。
"""

from esn_vla_uq.data.sources.base import RolloutSource
from esn_vla_uq.data.sources.synthetic import SyntheticRolloutSource

__all__ = [
    "RolloutSource",
    "SyntheticRolloutSource",
]
