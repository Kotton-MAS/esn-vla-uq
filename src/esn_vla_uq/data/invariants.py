"""出所 (`RolloutDataset.source`) ごとの追加不変条件。

`RolloutDataset.validate()` (`data/schema.py`) は出所に依存しない共通スキーマ
契約だけを検証する。出所固有の追加契約はここに集め、`validate_by_source` が
レジストリを引いて振り分ける。

**依存は `schema.py` と `provenance.py` のみ**。これが本モジュールの存在理由で
ある (S7)。以前は `data/io.py` が `data/synthetic.py` の
`validate_synthetic_dataset` をトップレベル import しており、Sprint 2 で
``source == "openpi"`` の分岐を足すと `io.py` が openpi ログパーサを
import することになっていた。そうなると「openpi をランタイム依存に含めない」
という設計は import 構造ではなく規律だけで支えられる。不変条件の**実装本体**を
ここへ置き、具象パーサ側 (`data/synthetic.py`、Sprint 2 の openpi 読み取り) が
逆にここを import する向きにすれば、`io.py` から具象への辺が消える。

不変条件は `RolloutDataset` の中身だけを見るため、どの出所の分もパーサを
import せずに書ける。Sprint 2 の openpi 固有契約もここに足す。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

from esn_vla_uq.data.schema import RolloutDataset
from esn_vla_uq.provenance import DataSource


def validate_synthetic_dataset(dataset: RolloutDataset) -> None:
    """合成データ生成器固有の不変条件を検証する。

    「失敗エピソードには必ず `failure_onset` が付く」は `Episode.validate()`
    (`data/schema.py`) が課す契約ではない。実 openpi ログの失敗エピソードには
    `failure_onset` という概念自体が存在しないことがあるため、`Episode` レベル
    の検証はこれを要求しない。この不変条件は合成データ生成器固有の追加契約
    としてここで検証する。

    `generate_dataset` の末尾で必ず呼ぶことに加え、`data/io.py` の
    `load_dataset` / `load_bundled_sample` / `save_dataset` が
    `source == "synthetic"` のデータを通す経路でも `validate_by_source` 経由で
    呼ばれる (生成経路にしか掛からないと、保存済みの破損データが読み込み時に
    素通りしてしまうため)。

    Args:
        dataset: 検証対象。`source` を問わず全エピソードを検査するが、通常は
            `validate_by_source` が `source == "synthetic"` のときだけ呼ぶ。

    Raises:
        ValueError: 失敗エピソードに `failure_onset` が無い場合。
    """
    for episode in dataset.episodes:
        if not episode.success and episode.failure_onset is None:
            raise ValueError(
                "failure_onset: 合成データセットの不変条件として、失敗エピソード"
                f"には必須です (episode_id={episode.episode_id!r})"
            )


SOURCE_VALIDATORS: Final[Mapping[DataSource, Callable[[RolloutDataset], None]]] = {
    "synthetic": validate_synthetic_dataset,
}
"""出所ごとの追加バリデータ。

登録の無い出所 (現時点では ``"openpi"``) は追加契約なしとして扱う。
`DataSource` を鍵にしているため、出所を増やしたときにここへ足し忘れても
型としては不正にならない点には注意が必要だが、`validate_by_source` が
未登録を「追加契約なし」と明示的に扱うため、黙って別の出所の契約が適用される
ことはない。
"""


def validate_by_source(dataset: RolloutDataset) -> None:
    """`dataset.source` に対応する追加不変条件を検証する。

    `data/io.py` は読み込み境界 (`_build_dataset`) と書き出し境界
    (`save_dataset`) の両方からこの関数を呼ぶ。片方だけだと、出所固有の契約に
    違反したデータセットが保存はできるのに二度と読み込めない、という非対称な
    成果物を作れてしまう。

    Args:
        dataset: 検証対象。`RolloutDataset.validate()` (共通契約) を先に通して
            いることを前提とする。

    Raises:
        ValueError: 出所固有の不変条件に違反する場合。
    """
    validator = SOURCE_VALIDATORS.get(dataset.source)
    if validator is not None:
        validator(dataset)
