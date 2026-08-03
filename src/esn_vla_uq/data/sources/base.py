"""ロールアウトデータの供給元を抽象化する Protocol。

**依存は `data/schema.py` のみ**。具象の供給元は import しない。これが本
モジュールを `data/sources/synthetic.py` から分けている理由である (A1)。

以前は Protocol と具象 (`SyntheticRolloutSource`) が 1 つの `data/source.py`
に同居していた。その形のまま Sprint 2 で `OpenpiLogSource` を足すと、
「供給元を受け取るだけ」の利用側 (型注釈に `RolloutSource` を書きたいだけの
コード) が、import しただけで合成データ生成器と openpi ログパーサの両方を
ロードすることになる。「openpi をランタイム依存に入れない」という設計が
import 構造ではなく規律だけで保たれる状態になるため、抽象と具象を分ける。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from esn_vla_uq.data.schema import RolloutDataset


@runtime_checkable
class RolloutSource(Protocol):
    """ロールアウトデータセットの供給元。"""

    def load(self) -> RolloutDataset:
        """検証済みの `RolloutDataset` を返す。"""
        ...
