"""数値とデータの出所 (provenance)。

パッケージ内の**最下層**。標準ライブラリ以外に依存せず、`data` / `diagnostics` /
`esn` のどのモジュールも import しない。

`DataSource` は以前 `data/schema.py` と `diagnostics/report.py` に独立して
定義されていた。出所を 1 つ追加する作業 (Sprint 2 の ``"openpi"`` がまさに
それ) では両方を直す必要があるが、型としては互いに無関係な別の Literal で
あるため、片方だけ更新しても mypy は何も言わない。片方が古いまま残ると、
データは ``"openpi"`` として読めるのに診断レポートには書けない、という
非対称が静かに生まれる。定義をここ 1 箇所にする (A4)。

出所を必須メタデータとして持ち回るのは、合成データに由来する数値を実
ロールアウトの評価結果と誤読させないためである (`docs/design.md` 7 節の
誠実性宣言)。
"""

from __future__ import annotations

from typing import Final, Literal, get_args

DataSource = Literal["synthetic", "openpi"]
"""データおよび数値の出所。

- ``"synthetic"``: `data/synthetic.py` の決定論的な合成生成器に由来する。
  実 LIBERO のロールアウトではない。
- ``"openpi"``: openpi の policy server が出力したロールアウトログに由来する
  (Sprint 2)。
"""

SUPPORTED_SOURCES: Final[tuple[str, ...]] = get_args(DataSource)
"""`DataSource` が許可する値の実行時タプル。

外部入力 (メタデータ JSON) の検証に使うため、型注釈からずれないよう
`get_args` で導出する。
"""

SYNTHETIC_DATA_SOURCE: Final[DataSource] = "synthetic"
"""Sprint 1 の全成果物が持つ出所。"""
