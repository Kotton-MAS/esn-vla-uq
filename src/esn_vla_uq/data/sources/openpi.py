"""openpi の LIBERO ロールアウトログを読む具象 `RolloutSource`。

**openpi をランタイム依存にしない。** 本モジュールが読むのは
`scripts/collect_openpi_rollouts.py` が書き出したログのディレクトリだけであり、
openpi のパッケージも policy server も import しない。収集時にだけ openpi が
要る (`data/sources/__init__.py` がここを eager import しない理由でもある)。

## 実仕様の確認

次元と間隔は openpi の LIBERO 評価ループ (`examples/libero/main.py`) と
`src/openpi/policies/libero_policy.py` を読んで確定させた
(`docs/design.md` 8 節の未解決論点 2 の解消)。

| 量 | 実仕様 | 備考 |
| --- | --- | --- |
| `observation/state` | 8 次元 | eef 位置(3) + 軸角(3) + グリッパ(2) |
| action | 7 次元 | 6 DoF デルタ + グリッパ |
| `action_horizon` | 10 | `pi0_libero` の学習設定による (クラス既定は 50) |
| `replan_steps` | 5 | 実際に実行してから再計画するステップ数 |

合成データ (H=16、16 ステップ間隔) とは値が違うが、`RolloutDataset` は
`chunk_horizon` をフィールドで持つため同じスキーマで共存できる。実収集したログで
H=10 / 間隔 5 を確認済み。

## ログ形式

openpi の評価スクリプトは**ロールアウトを保存しない** (replay 動画だけを書く)。
そのため収集スクリプト側で記録する必要があり、その形式をここで定義する。

```
<log_dir>/
├── manifest.json          # 収集条件とエピソード一覧
└── episodes/<id>.npz      # state / action / action_chunk / inference_steps
```

`manifest.json` の必須キー: `schema_version`, `policy`, `task_suite`,
`control_hz`, `state_dim`, `action_dim`, `chunk_horizon`, `replan_steps`,
`episodes` (各要素に `episode_id`, `task_name`, `success`, `n_steps`)。

エピソード npz の配列:

- `state`: `float32[T, state_dim]`
- `action`: `float32[T, action_dim]`
- `action_chunk`: `float32[n_inference, chunk_horizon, action_dim]`
  (推論したチャンクだけを詰める。非推論ステップの NaN は保存しない)
- `inference_steps`: `int64[n_inference]` (チャンクを推論したステップ番号)

`failure_onset` は持たない。openpi の評価ループは `done` が立てば成功、
`max_steps` 到達なら失敗とするだけで、**失敗が始まった時刻という概念が無い**。
`Episode.validate()` が `failure_onset` を要求しない設計 (5.1 節) はこのために
用意してあり、実ログでその前提が正しかったことが確認できた。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.data.schema import (
    MAX_ACTION_DIM,
    MAX_CHUNK_HORIZON,
    MAX_STATE_DIM,
    Episode,
    RolloutDataset,
    check_dataset_byte_budget,
    check_dimension_limit,
)
from esn_vla_uq.provenance import DataSource

MANIFEST_NAME: Final[str] = "manifest.json"
"""収集条件とエピソード一覧を書いたファイル名。"""

EPISODES_DIRNAME: Final[str] = "episodes"
"""エピソード npz を置くサブディレクトリ名。"""

LOG_SCHEMA_VERSION: Final[str] = "0.1.0"
"""収集ログの形式バージョン。読み込み時に一致を要求する。"""

OPENPI_SOURCE: Final[DataSource] = "openpi"
"""この供給元が付ける `RolloutDataset.source`。"""


@dataclass(frozen=True)
class OpenpiLogSource:
    """収集済み openpi ロールアウトログを読む供給元。

    Attributes:
        log_dir: `manifest.json` と `episodes/` を含むディレクトリ。
    """

    log_dir: Path

    def load(self) -> RolloutDataset:
        """ログを読んで検証済みの `RolloutDataset` を返す (`source == "openpi"`)。

        Returns:
            検証済みのデータセット。

        Raises:
            FileNotFoundError: マニフェストまたはエピソード npz が無い場合。
            ValueError: 形式バージョンが違う、次元が上限を超える、または
                スキーマ検証に失敗した場合。
        """
        manifest = _read_manifest(self.log_dir / MANIFEST_NAME)
        state_dim = _require_int(manifest, "state_dim")
        action_dim = _require_int(manifest, "action_dim")
        chunk_horizon = _require_int(manifest, "chunk_horizon")

        # 合成データと同じく、配列を確保する前に次元を検証する (CWE-789)。
        check_dimension_limit("state_dim", state_dim, MAX_STATE_DIM)
        check_dimension_limit("action_dim", action_dim, MAX_ACTION_DIM)
        check_dimension_limit("chunk_horizon", chunk_horizon, MAX_CHUNK_HORIZON)

        episodes = [
            _read_episode(
                self.log_dir / EPISODES_DIRNAME,
                entry,
                state_dim=state_dim,
                action_dim=action_dim,
                chunk_horizon=chunk_horizon,
            )
            for entry in _require_episodes(manifest)
        ]
        dataset = RolloutDataset(
            episodes=episodes,
            source=OPENPI_SOURCE,
            policy=_require_str(manifest, "policy"),
            seed=_require_int(manifest, "seed"),
            control_hz=_require_float(manifest, "control_hz"),
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
        )
        dataset.validate()
        return dataset


def _read_manifest(path: Path) -> dict[str, object]:
    """マニフェストを読み、形式バージョンを検証する。"""
    if not path.exists():
        raise FileNotFoundError(f"マニフェストがありません: {path.name}")
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest: オブジェクトである必要があります")
    manifest: dict[str, object] = payload
    version = _require_str(manifest, "schema_version")
    if version != LOG_SCHEMA_VERSION:
        raise ValueError(
            "manifest.schema_version: 未知のログ形式です "
            f"(actual={version!r}, supported={LOG_SCHEMA_VERSION!r})"
        )
    return manifest


def _read_episode(
    episodes_dir: Path,
    entry: dict[str, object],
    *,
    state_dim: int,
    action_dim: int,
    chunk_horizon: int,
) -> Episode:
    """エピソード npz を読み、NaN 埋めの `action_chunk` を復元する。"""
    episode_id = _require_str(entry, "episode_id")
    path = episodes_dir / f"{episode_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"エピソードがありません: {path.name}")

    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=np.float32)
        action = np.asarray(archive["action"], dtype=np.float32)
        inferred = np.asarray(archive["action_chunk"], dtype=np.float32)
        inference_steps = np.asarray(archive["inference_steps"], dtype=np.int64)

    n_steps = int(state.shape[0])
    # 復元後の `action_chunk` を確保する前に見積もる (CWE-789)。
    check_dataset_byte_budget(n_steps, chunk_horizon, action_dim)

    is_inference_step = np.zeros(n_steps, dtype=np.bool_)
    if inference_steps.size > 0:
        if int(inference_steps.min()) < 0 or int(inference_steps.max()) >= n_steps:
            raise ValueError(
                f"inference_steps: [0, {n_steps}) の範囲外の値があります "
                f"(episode_id={episode_id!r})"
            )
        is_inference_step[inference_steps] = True

    action_chunk = np.full(
        (n_steps, chunk_horizon, action_dim), np.nan, dtype=np.float32
    )
    if inferred.shape[0] != inference_steps.shape[0]:
        raise ValueError(
            "action_chunk と inference_steps の件数が一致しません "
            f"(episode_id={episode_id!r}, chunks={inferred.shape[0]}, "
            f"steps={inference_steps.shape[0]})"
        )
    action_chunk[is_inference_step] = inferred

    return Episode(
        episode_id=episode_id,
        task_name=_require_str(entry, "task_name"),
        success=_require_bool(entry, "success"),
        n_steps=n_steps,
        state=_check_shape(state, (n_steps, state_dim), "state", episode_id),
        action=_check_shape(action, (n_steps, action_dim), "action", episode_id),
        action_chunk=action_chunk,
        is_inference_step=is_inference_step,
        # openpi の評価ループには失敗開始時刻の概念が無い (モジュール docstring)。
        failure_onset=None,
    )


def _check_shape(
    array: NDArray[np.float32],
    expected: tuple[int, ...],
    name: str,
    episode_id: str,
) -> NDArray[np.float32]:
    """配列の shape を検証して返す。"""
    if array.shape != expected:
        raise ValueError(
            f"{name}: shape が不正です (episode_id={episode_id!r}, "
            f"expected={expected}, actual={array.shape})"
        )
    return array


def _require_episodes(manifest: dict[str, object]) -> list[dict[str, object]]:
    """マニフェストのエピソード一覧を取り出す。"""
    value = manifest.get("episodes")
    if not isinstance(value, list):
        raise ValueError("manifest.episodes: 配列である必要があります")
    entries: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(
                "manifest.episodes: 各要素はオブジェクトである必要があります"
            )
        entries.append(item)
    if not entries:
        raise ValueError("manifest.episodes: 1 件以上必要です")
    return entries


def _require_str(payload: dict[str, object], key: str) -> str:
    """文字列フィールドを取り出す。"""
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key}: 文字列である必要があります (actual={value!r})")
    return value


def _require_int(payload: dict[str, object], key: str) -> int:
    """整数フィールドを取り出す。"""
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: 整数である必要があります (actual={value!r})")
    return value


def _require_float(payload: dict[str, object], key: str) -> float:
    """実数フィールドを取り出す。"""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key}: 実数である必要があります (actual={value!r})")
    return float(value)


def _require_bool(payload: dict[str, object], key: str) -> bool:
    """真偽値フィールドを取り出す。"""
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key}: 真偽値である必要があります (actual={value!r})")
    return value
