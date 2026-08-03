"""ロールアウトデータのスキーマ v0.1。

`Episode` は 1 回のロールアウト、`RolloutDataset` はその集合とメタデータを表す。
同梱データおよび `SyntheticRolloutSource` が返すデータはすべて合成データであり
(`source == "synthetic"`)、実 LIBERO のロールアウトではない。Sprint 2 で openpi の
ロールアウトログを読む際は `source == "openpi"` を用いる。

配列レイアウト (T = ステップ数):

- `state`: `float32[T, 8]` — 7 関節 + グリッパ
- `action`: `float32[T, 7]` — 6 DoF デルタ + グリッパ
- `action_chunk`: `float32[T, 16, 7]` — 推論ステップのみ有効。非推論ステップは NaN
- `is_inference_step`: `bool[T]` — 行動チャンクを推論したステップ
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.provenance import SUPPORTED_SOURCES, DataSource

SCHEMA_VERSION: Final[str] = "0.1.0"
"""現在のスキーマバージョン。"""

SUPPORTED_SCHEMA_VERSIONS: Final[tuple[str, ...]] = (SCHEMA_VERSION,)
"""読み込みを許可するスキーマバージョン。"""

STATE_DIM: Final[int] = 8
"""状態次元 (7 関節 + グリッパ)。"""

ACTION_DIM: Final[int] = 7
"""行動次元 (6 DoF デルタ + グリッパ)。"""

CHUNK_HORIZON: Final[int] = 16
"""行動チャンクの予測ホライズン H。"""

MAX_STATE_DIM: Final[int] = 1024
"""`state_dim` の上限。

`state_dim`/`action_dim`/`chunk_horizon` はサイドカー JSON のメタデータから
読み戻す値であり、`data/io.py` が npz を読み込む際は攻撃者が制御できる入力
として扱う必要がある (CWE-789: Memory Allocation with Excessive Size Value)。
上限を課さずに `np.full` 等へ直接渡すと、圧縮率の高い (=ディスク上は小さい)
npz でも読み込み時に巨大な配列確保を誘発できてしまう。`data/io.py` は次元を
メタデータから読んだ直後・配列を確保する前にこの上限を検証し、
`RolloutDataset.validate()` (書き出し側) も同じ上限を使う。
"""

MAX_ACTION_DIM: Final[int] = 1024
"""`action_dim` の上限。理由は `MAX_STATE_DIM` を参照。"""

MAX_CHUNK_HORIZON: Final[int] = 4096
"""`chunk_horizon` の上限。理由は `MAX_STATE_DIM` を参照。"""

MAX_DATASET_BYTES: Final[int] = 2 * 1024**3
"""`action_chunk` 復元後配列 (`float32[n_steps, chunk_horizon, action_dim]`) の
推定バイト数の上限 (2 GiB)。

`MAX_STATE_DIM`/`MAX_ACTION_DIM`/`MAX_CHUNK_HORIZON` は各次元を個別に制限
するが、3 次元の積は依然大きくなりうる (例: 上限いっぱいの
`action_dim=1024, chunk_horizon=4096` でも `n_steps` が数千あれば数十 GiB に
達する)。そのため `data/io.py` は
`n_steps * chunk_horizon * action_dim * 4` (float32 の itemsize) の推定バイト
数をこの上限とも比較し、配列を確保する前に `ValueError` を送出する
(`check_dataset_byte_budget`)。

上記はいずれもメタデータ JSON 由来の次元 (`state_dim`/`action_dim`/
`chunk_horizon`) を経由する検証であり、npz 内の `.npy` ヘッダが自己申告する
配列 shape そのもの (特に `n_steps` 方向) は別途 `check_npz_uncompressed_budget`
で npz アーカイブ全体の非圧縮サイズ合計を同じ上限と比較して検証する。
"""

# `DataSource` / `SUPPORTED_SOURCES` の実体は `esn_vla_uq.provenance` にある
# (A4)。このモジュールは両者を使うため import しており、結果として
# `esn_vla_uq.data.schema` 経由でも従来どおり参照できる。


# `np.generic` は型スタブ上のみ総称型で実行時には添字を取れない。PEP 695 の
# 型エイリアスは遅延評価されるため、実行時に評価されず安全に書ける。
type NpzArray = np.ndarray[tuple[int, ...], np.dtype[np.generic[object]]]
"""dtype 未確定の配列。npz から読んだ直後の配列を表すために使う。"""

type ScalarType = type[np.float32] | type[np.bool_] | type[np.int64]
"""本スキーマで許可する要素型。"""


def check_shape(name: str, array: NpzArray, expected: tuple[int, ...]) -> None:
    """配列の shape が期待どおりかを検証する。"""
    if array.shape != expected:
        raise ValueError(
            f"{name}: shape が不正です (expected={expected}, actual={array.shape})"
        )


def check_dtype(name: str, array: NpzArray, expected: ScalarType) -> None:
    """配列の dtype が期待どおりかを検証する。"""
    expected_dtype = np.dtype(expected)
    if array.dtype != expected_dtype:
        raise ValueError(
            f"{name}: dtype が不正です "
            f"(expected={expected_dtype}, actual={array.dtype})"
        )


def check_all_finite(name: str, array: NDArray[np.float32]) -> None:
    """配列に NaN / inf が含まれていないことを検証する。"""
    if not bool(np.isfinite(array).all()):
        n_bad = int((~np.isfinite(array)).sum())
        raise ValueError(f"{name}: 有限でない値が {n_bad} 個含まれています")


def check_dimension_limit(name: str, value: int, maximum: int) -> None:
    """次元が上限を超えていないことを検証する (CWE-789 対策)。

    `MAX_STATE_DIM` 等の docstring を参照。呼び出し側は、この値を使って配列を
    確保する**前**に呼ぶこと。

    Raises:
        ValueError: `value` が `maximum` を超える場合。
    """
    if value > maximum:
        raise ValueError(f"{name}: 上限を超えています (actual={value}, max={maximum})")


def check_dataset_byte_budget(
    n_steps: int, chunk_horizon: int, action_dim: int
) -> None:
    """`action_chunk` を `float32[n_steps, chunk_horizon, action_dim]` で確保した
    ときの推定バイト数が `MAX_DATASET_BYTES` を超えないことを検証する
    (CWE-789 対策。`MAX_DATASET_BYTES` の docstring を参照)。

    呼び出し側は、この値を使って配列を確保する**前**に呼ぶこと。

    Raises:
        ValueError: 推定バイト数が `MAX_DATASET_BYTES` を超える場合。
    """
    itemsize = np.dtype(np.float32).itemsize
    estimated_bytes = n_steps * chunk_horizon * action_dim * itemsize
    if estimated_bytes > MAX_DATASET_BYTES:
        raise ValueError(
            "action_chunk: 復元後配列の推定確保サイズが上限を超えています "
            f"(estimated_bytes={estimated_bytes}, max_bytes={MAX_DATASET_BYTES}, "
            f"n_steps={n_steps}, chunk_horizon={chunk_horizon}, "
            f"action_dim={action_dim})"
        )


def check_npz_uncompressed_budget(total_bytes: int, context: str) -> None:
    """npz アーカイブ全エントリの非圧縮サイズ合計が `MAX_DATASET_BYTES` を
    超えないことを検証する (CWE-789 対策)。

    `check_dataset_byte_budget` はメタデータ由来の次元 (`chunk_horizon` /
    `action_dim`) と、読み込み側で `state.shape[0]` から求めた `n_steps` の
    積で見積もる。しかし `state`/`action` 等の配列自体の shape (特に第 1 軸
    `n_steps`) は npz 内の `.npy` ヘッダが自己申告する値であり、メタデータの
    どのフィールドからも検証されない。圧縮率の高い (=ディスク上は小さい)
    npz でも、展開後の配列が巨大になるよう作れてしまう
    (`data/io.py` の `_build_dataset` が実測した PoC: 状態次元・行動次元・
    チャンクホライズンを正規値に保ったまま `state` の `n_steps` 方向だけを
    巨大化した ~4.6MB の npz が、`state = npz["state"]` の時点
    (`np.load` の遅延ロードが実際に配列を確保するタイミング) で
    `MemoryError` を起こす)。

    呼び出し側は、`npz["state"]` 等で個々の配列を実体化する**前**に、
    以下 2 通りの見積もりをそれぞれこの関数に渡すこと。どちらも配列を
    実体化しない。

    1. `zipfile.ZipFile.infolist()` (セントラルディレクトリのヘッダのみを
       読み、エントリを解凍しない) から求めた非圧縮サイズの合計。
       実データを伴う decompression bomb を塞ぐ。
    2. 各エントリの `.npy` ヘッダが宣言する shape と dtype から求めた
       バイト数の合計。**実データをほとんど伴わずヘッダだけが巨大な shape を
       騙る細工**は (1) を素通りするため、こちらが必要になる
       (実測: 1,105 バイトの npz で 4.47 GiB の確保を誘発できた)。

    Args:
        total_bytes: 見積もった総バイト数。
        context: どちらの見積もりかを示すラベル。エラーメッセージに含め、
            原因の切り分けを可能にする。

    Raises:
        ValueError: 総バイト数が `MAX_DATASET_BYTES` を超える場合。
    """
    if total_bytes > MAX_DATASET_BYTES:
        raise ValueError(
            f"npz: {context}が上限を超えています "
            f"(total_bytes={total_bytes}, max_bytes={MAX_DATASET_BYTES})"
        )


def validate_episode_index(
    episode_starts: NDArray[np.int64],
    episode_lengths: NDArray[np.int64],
    total_steps: int,
) -> None:
    """連結表現のエピソード索引 (`episode_starts` / `episode_lengths`) を検証する。

    Args:
        episode_starts: 各エピソードの開始インデックス。
        episode_lengths: 各エピソードのステップ数。
        total_steps: 連結後の総ステップ数。

    Raises:
        ValueError: 索引が連結表現と整合しない場合。
    """
    check_dtype("episode_starts", episode_starts, np.int64)
    check_dtype("episode_lengths", episode_lengths, np.int64)
    if episode_starts.ndim != 1 or episode_lengths.ndim != 1:
        raise ValueError(
            "episode_starts / episode_lengths: 1 次元配列である必要があります "
            f"(actual={episode_starts.ndim} / {episode_lengths.ndim})"
        )
    if episode_starts.shape != episode_lengths.shape:
        raise ValueError(
            "episode_starts と episode_lengths の長さが一致しません "
            f"(starts={episode_starts.shape[0]}, lengths={episode_lengths.shape[0]})"
        )
    if episode_starts.shape[0] == 0:
        raise ValueError("episode_starts: エピソードが 1 件も含まれていません")
    if bool((episode_lengths <= 0).any()):
        raise ValueError(
            "episode_lengths: 正の値である必要があります "
            f"(min={int(episode_lengths.min())})"
        )
    expected_starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(episode_lengths[:-1], dtype=np.int64))
    )
    if not np.array_equal(episode_starts, expected_starts):
        raise ValueError(
            "episode_starts が episode_lengths の累積和と一致しません "
            f"(expected={expected_starts.tolist()}, actual={episode_starts.tolist()})"
        )
    total_from_lengths = int(episode_lengths.sum())
    if total_from_lengths != total_steps:
        raise ValueError(
            "episode_lengths の総和が連結配列の長さと一致しません "
            f"(sum={total_from_lengths}, total_steps={total_steps})"
        )


@dataclass(frozen=True, eq=False)
class Episode:
    """1 回のロールアウト。

    Attributes:
        episode_id: エピソード識別子 (データセット内で一意)。
        task_name: タスク名。
        success: タスク成功可否。
        n_steps: ステップ数 T。
        state: `float32[T, 8]`。
        action: `float32[T, 7]`。
        action_chunk: `float32[T, 16, 7]`。非推論ステップは全要素 NaN。
        is_inference_step: `bool[T]`。
        failure_onset: 失敗が始まったステップ。成功エピソードでは `None`。
            失敗エピソードでも `None` を許容する (`failure_onset` は合成データ
            生成器固有の概念であり、実 openpi ログには存在しないことがある)。
    """

    episode_id: str
    task_name: str
    success: bool
    n_steps: int
    state: NDArray[np.float32]
    action: NDArray[np.float32]
    action_chunk: NDArray[np.float32]
    is_inference_step: NDArray[np.bool_]
    failure_onset: int | None = None

    def validate(
        self,
        *,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_horizon: int = CHUNK_HORIZON,
    ) -> None:
        """shape / dtype / NaN 配置 / 失敗メタデータの整合を検証する。

        次元 (`state_dim` / `action_dim` / `chunk_horizon`) は既定でモジュール
        定数を使うが、`RolloutDataset.validate()` からはデータセット自身が
        保持する次元 (メタデータから読み戻した値) が渡される。npz を単体で
        自己記述的に検証できるようにするための引数であり、単体で `Episode` を
        組み立てて検証する場合は既定値のままでよい。

        Args:
            state_dim: `state` の第 2 軸の期待次元。
            action_dim: `action` の第 3 軸 (末尾) の期待次元。
            chunk_horizon: `action_chunk` の第 2 軸の期待次元 (予測ホライズン)。

        Raises:
            ValueError: どのフィールドがどう不正かを含むメッセージで送出する。
        """
        if not self.episode_id:
            raise ValueError("episode_id: 空文字は許可されません")
        if self.n_steps < 1:
            raise ValueError(
                f"n_steps: 1 以上である必要があります (actual={self.n_steps})"
            )

        check_shape("state", self.state, (self.n_steps, state_dim))
        check_dtype("state", self.state, np.float32)
        check_all_finite("state", self.state)

        check_shape("action", self.action, (self.n_steps, action_dim))
        check_dtype("action", self.action, np.float32)
        check_all_finite("action", self.action)

        check_shape(
            "action_chunk",
            self.action_chunk,
            (self.n_steps, chunk_horizon, action_dim),
        )
        check_dtype("action_chunk", self.action_chunk, np.float32)

        check_shape("is_inference_step", self.is_inference_step, (self.n_steps,))
        check_dtype("is_inference_step", self.is_inference_step, np.bool_)

        self._validate_chunk_nan_layout()
        self._validate_failure_onset()

    def _validate_chunk_nan_layout(self) -> None:
        """`action_chunk` の NaN 配置が `is_inference_step` と一致するか検証する。"""
        n_inference = int(self.is_inference_step.sum())
        if n_inference == 0:
            raise ValueError("is_inference_step: 推論ステップが 1 つもありません")

        inferred = self.action_chunk[self.is_inference_step]
        if not bool(np.isfinite(inferred).all()):
            raise ValueError(
                "action_chunk: 推論ステップに有限でない値が含まれています "
                f"({int((~np.isfinite(inferred)).sum())} 要素)"
            )

        skipped = self.action_chunk[~self.is_inference_step]
        if skipped.size > 0 and not bool(np.isnan(skipped).all()):
            raise ValueError(
                "action_chunk: 非推論ステップは全要素 NaN である必要があります "
                f"({int((~np.isnan(skipped)).sum())} 要素が非 NaN)"
            )

    def _validate_failure_onset(self) -> None:
        """`failure_onset` と `success` の整合を検証する。

        「失敗エピソードには必ず `failure_onset` が付く」という制約はここでは
        課さない。それは合成データ生成器 (`data/synthetic.py`) 固有の不変条件で
        あり、実 openpi ログの失敗エピソードには存在しない概念のため
        (Sprint 2 の `OpenpiLogSource` は `failure_onset=None` の失敗エピソードを
        構築できる必要がある)。
        """
        if self.success and self.failure_onset is not None:
            raise ValueError(
                "failure_onset: 成功エピソードでは None である必要があります "
                f"(actual={self.failure_onset})"
            )
        if self.failure_onset is not None and not (
            0 <= self.failure_onset < self.n_steps
        ):
            raise ValueError(
                "failure_onset: [0, n_steps) の範囲外です "
                f"(actual={self.failure_onset}, n_steps={self.n_steps})"
            )

    def to_metadata(self) -> dict[str, object]:
        """サイドカー JSON に書き出すエピソード単位のメタデータ。"""
        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "success": self.success,
            "n_steps": self.n_steps,
            "failure_onset": self.failure_onset,
        }


@dataclass(frozen=True, eq=False)
class RolloutDataset:
    """ロールアウトの集合とデータセット単位のメタデータ。

    Attributes:
        episodes: エピソード列。
        source: データの出所 (`"synthetic"` / `"openpi"`)。
        policy: 行動を生成したポリシー名。
        seed: 生成に用いた乱数シード。
        control_hz: 制御周波数 [Hz]。
        schema_version: スキーマバージョン。
        state_dim: `state` の次元。永続化フォーマットを自己記述的にするため
            メタデータに書き出し、読み戻し時もこの値で検証する。
        action_dim: `action` / `action_chunk` 末尾の次元。
        chunk_horizon: `action_chunk` の予測ホライズン H。
    """

    episodes: Sequence[Episode]
    source: DataSource
    policy: str
    seed: int
    control_hz: float
    schema_version: str = SCHEMA_VERSION
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    chunk_horizon: int = CHUNK_HORIZON

    @property
    def n_episodes(self) -> int:
        """エピソード数。"""
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        """全エピソードの総ステップ数。"""
        return int(sum(episode.n_steps for episode in self.episodes))

    @property
    def episode_lengths(self) -> NDArray[np.int64]:
        """各エピソードのステップ数。"""
        return np.asarray(
            [episode.n_steps for episode in self.episodes], dtype=np.int64
        )

    @property
    def episode_starts(self) -> NDArray[np.int64]:
        """連結表現における各エピソードの開始インデックス。"""
        lengths = self.episode_lengths
        return np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64))
        )

    def validate(self) -> None:
        """メタデータと全エピソードを検証する。

        Raises:
            ValueError: どのフィールドがどう不正かを含むメッセージで送出する。
        """
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                "schema_version: 未知のバージョンです "
                f"(actual={self.schema_version!r}, "
                f"supported={list(SUPPORTED_SCHEMA_VERSIONS)})"
            )
        if self.source not in SUPPORTED_SOURCES:
            raise ValueError(
                f"source: 未知の出所です (actual={self.source!r}, "
                f"supported={list(SUPPORTED_SOURCES)})"
            )
        if not self.policy:
            raise ValueError("policy: 空文字は許可されません")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError(
                f"control_hz: 正の有限値である必要があります (actual={self.control_hz})"
            )
        for name, value, maximum in (
            ("state_dim", self.state_dim, MAX_STATE_DIM),
            ("action_dim", self.action_dim, MAX_ACTION_DIM),
            ("chunk_horizon", self.chunk_horizon, MAX_CHUNK_HORIZON),
        ):
            if value < 1:
                raise ValueError(f"{name}: 1 以上である必要があります (actual={value})")
            # 書き出し側 (このメソッド) と読み込み側 (`data/io.py`) で同一の
            # 不変条件を保つため、`MAX_STATE_DIM` 等と同じ上限を使う (CWE-789 対策)。
            check_dimension_limit(name, value, maximum)
        check_dataset_byte_budget(self.total_steps, self.chunk_horizon, self.action_dim)
        if self.n_episodes == 0:
            raise ValueError("episodes: 1 件以上必要です")

        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("episode_id: データセット内で重複しています")

        for episode in self.episodes:
            episode.validate(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                chunk_horizon=self.chunk_horizon,
            )

        validate_episode_index(
            self.episode_starts, self.episode_lengths, self.total_steps
        )

    def to_metadata(self) -> dict[str, object]:
        """サイドカー JSON に書き出すデータセット単位のメタデータ。"""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "policy": self.policy,
            "seed": self.seed,
            "control_hz": self.control_hz,
            "n_episodes": self.n_episodes,
            "total_steps": self.total_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_horizon": self.chunk_horizon,
            "episodes": [episode.to_metadata() for episode in self.episodes],
        }
