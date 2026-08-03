"""ロールアウトデータセットの保存・読み込み。

保存形式は 2 ファイル 1 組:

- `<name>.npz`: 全エピソードを連結した数値配列と
  `episode_starts` / `episode_lengths`。
  `action_chunk` は推論ステップ分のみを詰めて保存し (非推論ステップの NaN は
  保存しない)、読み込み時に `is_inference_step` から NaN 埋めの `[T, H, D]` を戻す。
- `<name>.json`: メタデータのサイドカー。`source` を含み、合成データを実ロールアウト
  として誤読させないための情報を保持する。`state_dim` / `action_dim` /
  `chunk_horizon` も収録し、読み込み時はこれらの値で shape を検証する
  (`schema.py` のモジュール定数を暗黙に前提しない自己記述的な永続化形式)。

配列はすべて float32 で保存する。

`state_dim` / `action_dim` / `chunk_horizon` はサイドカー JSON から読み戻す
攻撃者制御可能な値であり、検証前に配列確保へ使うと圧縮率の高い小さな npz から
巨大な確保を誘発できる (CWE-789)。`_build_dataset` は次元を読んだ直後・
`action_chunk` を復元する前に `schema.py` の `MAX_STATE_DIM` /
`MAX_ACTION_DIM` / `MAX_CHUNK_HORIZON` / `MAX_DATASET_BYTES` で検証する。

上記はメタデータ由来の次元を経由する検証であり、npz 内の `.npy` ヘッダが
自己申告する配列 shape 自体 (特に `state`/`action` 等の第 1 軸 `n_steps`) は
別経路の攻撃者制御可能な入力で、メタデータのどのフィールドからも検証されない。
`_build_dataset` は `npz["state"]` 等で最初の配列を実体化するより前に、
`zipfile.ZipFile.infolist()` (エントリを解凍せずセントラルディレクトリの
ヘッダのみ読む) から求めた npz 全エントリの非圧縮サイズ合計を
`check_npz_uncompressed_budget` で検証する。

出所ごとの追加不変条件は `data/invariants.py` の `validate_by_source` に委ねる。
`save_dataset` (書き出し境界) と `_build_dataset` (読み込み境界) の両方から
呼ぶことで、書けるのに読めない非対称な成果物を作れないようにする。

本モジュールは具象の供給元 (`data/synthetic.py`、Sprint 2 の openpi ログ
パーサ) を import しない。以前は `validate_synthetic_dataset` を
`data/synthetic.py` から直接 import しており、`source == "openpi"` の分岐を
足した時点で `io.py` が openpi パーサに依存する構造だった (S7)。
"""

from __future__ import annotations

import io
import json
import logging
import math
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Final, NamedTuple, cast

import numpy as np
from numpy.lib import format as npy_format
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

from esn_vla_uq.data.invariants import validate_by_source
from esn_vla_uq.data.schema import (
    MAX_ACTION_DIM,
    MAX_CHUNK_HORIZON,
    MAX_STATE_DIM,
    SUPPORTED_SCHEMA_VERSIONS,
    Episode,
    NpzArray,
    RolloutDataset,
    check_dataset_byte_budget,
    check_dimension_limit,
    check_dtype,
    check_npz_uncompressed_budget,
    validate_episode_index,
)
from esn_vla_uq.logging_paths import display_path
from esn_vla_uq.provenance import DataSource

logger = logging.getLogger(__name__)

SAMPLES_PACKAGE: Final[str] = "esn_vla_uq.assets.samples"
"""同梱サンプルデータを収めたパッケージ。"""

BUNDLED_SAMPLE_ARCHIVE: Final[str] = "libero_synthetic_v0.1.npz"
"""同梱サンプルデータの配列ファイル名。"""

BUNDLED_SAMPLE_METADATA: Final[str] = "libero_synthetic_v0.1.json"
"""同梱サンプルデータのメタデータファイル名。"""

ARCHIVE_SUFFIX: Final[str] = ".npz"
METADATA_SUFFIX: Final[str] = ".json"


def metadata_path_for(path: Path) -> Path:
    """`.npz` のパスからサイドカー JSON のパスを導く。"""
    return path.with_suffix(METADATA_SUFFIX)


def _ensure_archive_path(path: Path) -> Path:
    """パスが `.npz` であることを検証する。"""
    if path.suffix != ARCHIVE_SUFFIX:
        raise ValueError(
            f"path: 拡張子は {ARCHIVE_SUFFIX} である必要があります "
            f"(actual={path.name!r})"
        )
    return path


class _ConcatenatedArrays(NamedTuple):
    """npz に書き出す連結済み配列一式。"""

    state: NDArray[np.float32]
    action: NDArray[np.float32]
    action_chunk: NDArray[np.float32]
    is_inference_step: NDArray[np.bool_]
    episode_starts: NDArray[np.int64]
    episode_lengths: NDArray[np.int64]


def _concatenate(dataset: RolloutDataset) -> _ConcatenatedArrays:
    """エピソードを連結して npz に書く配列一式を作る。"""
    state = np.concatenate([episode.state for episode in dataset.episodes], axis=0)
    action = np.concatenate([episode.action for episode in dataset.episodes], axis=0)
    is_inference_step = np.concatenate(
        [episode.is_inference_step for episode in dataset.episodes], axis=0
    )
    action_chunk = np.concatenate(
        [
            episode.action_chunk[episode.is_inference_step]
            for episode in dataset.episodes
        ],
        axis=0,
    )
    return _ConcatenatedArrays(
        state=state.astype(np.float32),
        action=action.astype(np.float32),
        action_chunk=action_chunk.astype(np.float32),
        is_inference_step=is_inference_step,
        episode_starts=dataset.episode_starts,
        episode_lengths=dataset.episode_lengths,
    )


def _ensure_writable(archive_path: Path, *, overwrite: bool) -> None:
    """書き出し先 2 ファイルが既存でないことを確認する (CWE-73 対策)。

    `save_dataset` は `.npz` と**同名の `.json`** を書く。利用者が意識するのは
    前者だけなので、`--output notes.npz` のような指定で、まったく無関係な既存の
    `notes.json` が黙って壊れうる。書き出し前に両方を検査し、既存なら
    `FileExistsError` にする。

    Args:
        archive_path: 書き出す `.npz` のパス。
        overwrite: True なら既存ファイルの上書きを許可する。

    Raises:
        FileExistsError: `overwrite` が False で、いずれかが既に存在する場合。
            サイドカーだけが存在するケース (本文の危険な例) も検出する。
    """
    if overwrite:
        return
    existing = [
        target
        for target in (archive_path, metadata_path_for(archive_path))
        if target.exists()
    ]
    if existing:
        names = ", ".join(sorted(target.name for target in existing))
        raise FileExistsError(
            f"出力先が既に存在します ({names})。上書きするには overwrite=True "
            "(CLI では --force) を指定してください"
        )


def save_dataset(
    dataset: RolloutDataset, path: Path, *, overwrite: bool = False
) -> Path:
    """データセットを `.npz` + `.json` サイドカーに保存する。

    Args:
        dataset: 保存対象。保存前に `validate()` と出所別の追加検証
            (`invariants.validate_by_source`) を実行する。
        path: 出力する `.npz` のパス。サイドカーは同名の `.json`。
        overwrite: 既存の `.npz` / サイドカー `.json` を上書きするか。既定は
            False で、どちらかが存在すれば書き出す前に `FileExistsError`。
            サイドカーの名前は `.npz` から導出されるため、利用者が意図して
            いない既存ファイルを壊しうる (`_ensure_writable`)。

    Returns:
        書き出した `.npz` のパス。

    Raises:
        ValueError: データセットが不正、出所別の不変条件に違反、または
            パスが `.npz` でない場合。
        FileExistsError: `overwrite=False` で出力先が既に存在する場合。
    """
    archive_path = _ensure_archive_path(path)
    _ensure_writable(archive_path, overwrite=overwrite)
    dataset.validate()
    # 出所別の検証は元々 `_build_dataset` (読み込み側) からのみ
    # 呼ばれており、save 側は `dataset.validate()` (共通スキーマ契約) しか
    # 経由しなかった。そのため出所固有の不変条件 (例: 合成データの失敗
    # エピソードには failure_onset が必須) に違反したデータセットが
    # save_dataset には受理されるのに load_dataset では ValueError になる、
    # という非対称な成果物 (書けるが二度と読めない) を作れてしまっていた。
    # save/load の両境界で同じ検証を通すことでこの非対称性を解消する。
    validate_by_source(dataset)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = _concatenate(dataset)
    np.savez_compressed(
        archive_path,
        state=arrays.state,
        action=arrays.action,
        action_chunk=arrays.action_chunk,
        is_inference_step=arrays.is_inference_step,
        episode_starts=arrays.episode_starts,
        episode_lengths=arrays.episode_lengths,
    )
    metadata_path = metadata_path_for(archive_path)
    metadata_path.write_text(
        json.dumps(dataset.to_metadata(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # 絶対パスはユーザー名を含みうるため INFO には出さない (S4)。
    logger.info(
        "saved rollout dataset: source=%s n_episodes=%d total_steps=%d path=%s",
        dataset.source,
        dataset.n_episodes,
        dataset.total_steps,
        display_path(archive_path),
    )
    logger.debug("saved rollout dataset: abs_path=%s", archive_path)
    return archive_path


def _require_mapping(payload: object, context: str) -> Mapping[str, object]:
    """JSON から読んだ値が辞書であることを検証する。"""
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: オブジェクトである必要があります")
    mapping: Mapping[str, object] = payload
    return mapping


def _require_str(payload: Mapping[str, object], key: str) -> str:
    """メタデータから文字列フィールドを取り出す。"""
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key}: 文字列である必要があります (actual={value!r})")
    return value


def _require_int(payload: Mapping[str, object], key: str) -> int:
    """メタデータから整数フィールドを取り出す。"""
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: 整数である必要があります (actual={value!r})")
    return value


def _require_float(payload: Mapping[str, object], key: str) -> float:
    """メタデータから実数フィールドを取り出す。"""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key}: 実数である必要があります (actual={value!r})")
    return float(value)


def _require_bool(payload: Mapping[str, object], key: str) -> bool:
    """メタデータから真偽値フィールドを取り出す。"""
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key}: 真偽値である必要があります (actual={value!r})")
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    """メタデータから `int | None` フィールドを取り出す。"""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{key}: 整数または null である必要があります (actual={value!r})"
        )
    return value


def _as_data_source(value: str) -> DataSource:
    """文字列を `DataSource` に変換する。"""
    if value == "synthetic":
        return "synthetic"
    if value == "openpi":
        return "openpi"
    raise ValueError(f"source: 未知の出所です (actual={value!r})")


def _require_schema_version(payload: Mapping[str, object]) -> str:
    """メタデータの `schema_version` を検証して返す。"""
    schema_version = _require_str(payload, "schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "schema_version: 未知のバージョンです "
            f"(actual={schema_version!r}, supported={list(SUPPORTED_SCHEMA_VERSIONS)})"
        )
    return schema_version


def _episode_metadata(payload: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    """メタデータからエピソード単位のレコード列を取り出す。"""
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("episodes: 配列である必要があります")
    return [
        _require_mapping(record, f"episodes[{index}]")
        for index, record in enumerate(episodes)
    ]


def _restore_action_chunk(
    packed: NDArray[np.float32],
    is_inference_step: NDArray[np.bool_],
    *,
    action_dim: int,
    chunk_horizon: int,
) -> NDArray[np.float32]:
    """推論ステップ分のみのチャンクから NaN 埋めの `[T, H, D]` を復元する。

    `action_dim` / `chunk_horizon` はメタデータ (`RolloutDataset.to_metadata()`)
    から読み戻した値を渡す。スキーマのモジュール定数を暗黙に使わないことで、
    次元の異なるデータを読んだときに無関係な shape エラーにならないようにする。
    """
    n_steps = int(is_inference_step.shape[0])
    n_inference = int(is_inference_step.sum())
    expected = (n_inference, chunk_horizon, action_dim)
    if packed.shape != expected:
        raise ValueError(
            "action_chunk: shape が is_inference_step と整合しません "
            f"(expected={expected}, actual={packed.shape})"
        )
    restored = np.full((n_steps, chunk_horizon, action_dim), np.nan, dtype=np.float32)
    restored[is_inference_step] = packed
    return restored


def _require_array(npz: Mapping[str, NpzArray], key: str) -> NpzArray:
    """npz から配列を取り出す。欠落は `ValueError` にする。"""
    if key not in npz:
        raise ValueError(f"{key}: npz に必要な配列がありません")
    return npz[key]


def _as_float32(npz: Mapping[str, NpzArray], key: str) -> NDArray[np.float32]:
    """npz の配列を dtype 検証したうえで float32 配列として返す。"""
    array = _require_array(npz, key)
    check_dtype(key, array, np.float32)
    # dtype を実行時に検証済みのため、静的型の絞り込みを cast で補う。
    return cast("NDArray[np.float32]", array)


def _as_bool(npz: Mapping[str, NpzArray], key: str) -> NDArray[np.bool_]:
    """npz の配列を dtype 検証したうえで bool 配列として返す。"""
    array = _require_array(npz, key)
    check_dtype(key, array, np.bool_)
    return cast("NDArray[np.bool_]", array)


def _as_int64(npz: Mapping[str, NpzArray], key: str) -> NDArray[np.int64]:
    """npz の配列を dtype 検証したうえで int64 配列として返す。"""
    array = _require_array(npz, key)
    check_dtype(key, array, np.int64)
    return cast("NDArray[np.int64]", array)


def _npz_uncompressed_total_bytes(npz: NpzFile[np.generic[object]]) -> int:
    """npz の全エントリの非圧縮サイズ合計を返す (CWE-789 対策)。

    `NpzFile.zip` は numpy 自身が公開している属性 (numpy の `NpzFile`
    docstring に「the ZipFile object itself using ``obj.zip``」と明記された
    public API であり、`_zip` のような private 属性ではない)。
    `zipfile.ZipFile.infolist()` はセントラルディレクトリのヘッダのみを
    読み、各エントリを解凍しないため、この関数はどの配列も実体化せずに
    合計サイズを見積もれる。
    """
    zip_file = npz.zip
    if zip_file is None:
        # np.load(..., allow_pickle=False) が返す NpzFile は `with` ブロック
        # 内で常に開いた ZipFile を保持する。None になるのは close 後だけ
        # であり、`_build_dataset` は呼び出し側が npz を開いたまま呼ぶため
        # 実運用では到達しない防御的分岐 (mypy 上 `zip` は `ZipFile | None`)。
        raise ValueError("npz: ZipFile ハンドルを取得できません")
    return sum(info.file_size for info in zip_file.infolist())


def _npz_declared_nbytes(npz: NpzFile[np.generic[object]]) -> int:
    """各エントリの `.npy` ヘッダが宣言する shape から総バイト数を求める。

    `_npz_uncompressed_total_bytes` (zip のセントラルディレクトリ) だけでは
    不十分である。numpy の `read_array` は `.npy` ヘッダの shape を読んだ
    直後、実データを読む前に `np.empty(shape, dtype)` を確保するため、
    「ヘッダだけが巨大な shape を自己申告し、実データはほとんど無い」
    エントリは zip 上のサイズが小さく、非圧縮サイズ合計の検証を素通りする
    (実測: 1,104 バイトの npz で 4.47 GiB の確保を誘発できた)。

    ここでは各エントリのヘッダのみを読んで shape と dtype を取得し、
    確保されうるバイト数を実データを読まずに見積もる。

    対象エントリの判定に**ファイル名を使わない**こと。numpy の `NpzFile` は
    `name.removesuffix(".npy")` でキーを解決しており `.npy` サフィックスは
    **任意**である。つまり `state` という名前のエントリも `npz["state"]` と
    して配列に読まれる。ファイル名で `.npy` を要求すると、拡張子を外した
    エントリ名でこの検証を丸ごと迂回できてしまう
    (実測: 1,099 バイトの npz で 32 GiB の確保を誘発できた)。
    代わりに `.npy` の magic バイトで判定する。
    """
    zip_file = npz.zip
    if zip_file is None:
        # np.load(..., allow_pickle=False) が返す NpzFile は `with` ブロック
        # 内で常に開いた ZipFile を保持する。None になるのは close 後だけ
        # であり、`_build_dataset` は呼び出し側が npz を開いたまま呼ぶため
        # 実運用では到達しない防御的分岐 (mypy 上 `zip` は `ZipFile | None`)。
        raise ValueError("npz: ZipFile ハンドルを取得できません")

    total = 0
    for info in zip_file.infolist():
        with zip_file.open(info) as probe:
            if probe.read(len(npy_format.MAGIC_PREFIX)) != npy_format.MAGIC_PREFIX:
                # `.npy` ではないエントリ (numpy は配列として読まない)。
                continue
        with zip_file.open(info) as entry:
            version = npy_format.read_magic(entry)
            if version == (1, 0):
                shape, _fortran, dtype = npy_format.read_array_header_1_0(entry)
            elif version == (2, 0):
                shape, _fortran, dtype = npy_format.read_array_header_2_0(entry)
            else:
                raise ValueError(
                    f"npz: 未対応の .npy フォーマットです "
                    f"(entry={info.filename}, version={version})"
                )

        # numpy のヘッダ検証は shape の要素が int であることしか見ておらず、
        # **負の次元を拒否しない**。合計だけを見ると、numpy が決して読まない
        # ダミーエントリに負の次元を宣言させて本命エントリの巨大な寄与を
        # 相殺できてしまう (実測: 1,283 バイトで 32 GiB、shape 次第で 512 TiB)。
        # したがって負の次元をここで潰し、かつ**エントリ単位でも**上限を見る。
        if any(dim < 0 for dim in shape):
            raise ValueError(
                f"npz: .npy ヘッダの shape に負の次元があります "
                f"(entry={info.filename}, shape={shape})"
            )
        nbytes = math.prod(shape) * dtype.itemsize
        check_npz_uncompressed_budget(nbytes, f"エントリ {info.filename} の宣言サイズ")
        total += nbytes
    return total


def _build_dataset(
    npz: NpzFile[np.generic[object]], metadata_text: str
) -> RolloutDataset:
    """npz とメタデータ JSON から `RolloutDataset` を復元する。"""
    # メタデータ由来の次元 (state_dim 等) の検証より前に、npz 自体が自己申告
    # する配列 shape (state.npy 等の `.npy` ヘッダの n_steps 方向) を縛る。
    # メタデータのどのフィールドもこれを検証しないため、ここで検証しないと
    # `npz["state"]` (下の `_as_float32`) が最初の配列実体化として巨大な
    # 確保を試みてしまう (CWE-789)。
    #
    # zip の非圧縮サイズ合計とヘッダ宣言サイズの両方を見る。前者は
    # 実データを伴う decompression bomb を、後者はヘッダだけが巨大な
    # shape を騙る細工を塞ぐ。どちらも配列を実体化しない。
    check_npz_uncompressed_budget(
        _npz_uncompressed_total_bytes(npz), "全エントリの非圧縮サイズ合計"
    )
    check_npz_uncompressed_budget(
        _npz_declared_nbytes(npz), ".npy ヘッダが宣言する配列サイズ合計"
    )

    payload = _require_mapping(json.loads(metadata_text), "metadata")
    schema_version = _require_schema_version(payload)
    state_dim = _require_int(payload, "state_dim")
    action_dim = _require_int(payload, "action_dim")
    chunk_horizon = _require_int(payload, "chunk_horizon")
    # メタデータ由来の次元は攻撃者制御可能な入力として扱う。これらを使って
    # 配列を確保する (`_restore_action_chunk` の `np.full`) 前に上限を検証する
    # (CWE-789 対策。`schema.py` の `MAX_STATE_DIM` 等の docstring を参照)。
    check_dimension_limit("state_dim", state_dim, MAX_STATE_DIM)
    check_dimension_limit("action_dim", action_dim, MAX_ACTION_DIM)
    check_dimension_limit("chunk_horizon", chunk_horizon, MAX_CHUNK_HORIZON)

    state = _as_float32(npz, "state")
    action = _as_float32(npz, "action")
    packed_chunk = _as_float32(npz, "action_chunk")
    is_inference_step = _as_bool(npz, "is_inference_step")
    episode_starts = _as_int64(npz, "episode_starts")
    episode_lengths = _as_int64(npz, "episode_lengths")

    # `state` が 0 次元 shape `()` を宣言していると `state.shape[0]` が
    # IndexError になり、ValueError を約束している公開 API の契約を破る。
    if state.ndim != 2:
        raise ValueError(
            f"state: 2 次元配列である必要があります (actual_ndim={state.ndim})"
        )
    total_steps = int(state.shape[0])
    # 個々の次元は上限内でも、3 次元の積 (= action_chunk 復元後配列の推定
    # バイト数) はまだ大きくなりうるため、復元ループ (`_restore_action_chunk`
    # の `np.full`) の前にあわせて検証する。
    check_dataset_byte_budget(total_steps, chunk_horizon, action_dim)
    validate_episode_index(episode_starts, episode_lengths, total_steps)
    for name, array in (("action", action), ("is_inference_step", is_inference_step)):
        if array.shape[0] != total_steps:
            raise ValueError(
                f"{name}: 先頭次元が state と一致しません "
                f"(expected={total_steps}, actual={array.shape[0]})"
            )

    records = _episode_metadata(payload)
    if len(records) != int(episode_lengths.shape[0]):
        raise ValueError(
            "episodes: メタデータの件数が episode_lengths と一致しません "
            f"(metadata={len(records)}, arrays={int(episode_lengths.shape[0])})"
        )

    chunk_offset = 0
    episodes: list[Episode] = []
    for record, start, length in zip(
        records, episode_starts.tolist(), episode_lengths.tolist(), strict=True
    ):
        stop = start + length
        episode_inference = is_inference_step[start:stop]
        n_inference = int(episode_inference.sum())
        episodes.append(
            Episode(
                episode_id=_require_str(record, "episode_id"),
                task_name=_require_str(record, "task_name"),
                success=_require_bool(record, "success"),
                n_steps=_require_int(record, "n_steps"),
                state=state[start:stop],
                action=action[start:stop],
                action_chunk=_restore_action_chunk(
                    packed_chunk[chunk_offset : chunk_offset + n_inference],
                    episode_inference,
                    action_dim=action_dim,
                    chunk_horizon=chunk_horizon,
                ),
                is_inference_step=episode_inference,
                failure_onset=_optional_int(record, "failure_onset"),
            )
        )
        chunk_offset += n_inference

    if chunk_offset != int(packed_chunk.shape[0]):
        raise ValueError(
            "action_chunk: 保存されたチャンク数が推論ステップ数と一致しません "
            f"(chunks={int(packed_chunk.shape[0])}, inference_steps={chunk_offset})"
        )

    dataset = RolloutDataset(
        episodes=episodes,
        source=_as_data_source(_require_str(payload, "source")),
        policy=_require_str(payload, "policy"),
        seed=_require_int(payload, "seed"),
        control_hz=_require_float(payload, "control_hz"),
        schema_version=schema_version,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
    )
    dataset.validate()
    validate_by_source(dataset)
    return dataset


def load_dataset(path: Path) -> RolloutDataset:
    """`.npz` + `.json` サイドカーからデータセットを読み込む。

    Args:
        path: 読み込む `.npz` のパス。

    Returns:
        検証済みの `RolloutDataset`。

    Raises:
        FileNotFoundError: `.npz` またはサイドカー `.json` が無い場合。
        ValueError: スキーマ違反・索引不整合・未知の `schema_version` の場合。
    """
    archive_path = _ensure_archive_path(path)
    metadata_path = metadata_path_for(archive_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"サイドカー JSON が見つかりません: {metadata_path}")
    metadata_text = metadata_path.read_text(encoding="utf-8")
    # allow_pickle=False を明示する。numpy の既定値は False (バージョン間で
    # 変わらない) だが、任意コード実行を招きうる pickle 逆シリアライズを
    # 公開ライブラリ API として明示的に禁止しておく。
    with np.load(archive_path, allow_pickle=False) as npz:
        return _build_dataset(npz, metadata_text)


def load_bundled_sample() -> RolloutDataset:
    """同梱の合成サンプルデータを読み込む。

    Returns:
        `source == "synthetic"` の検証済み `RolloutDataset`。実 LIBERO の
        ロールアウトではない。
    """
    package = files(SAMPLES_PACKAGE)
    metadata_text = package.joinpath(BUNDLED_SAMPLE_METADATA).read_text(
        encoding="utf-8"
    )
    archive_bytes = package.joinpath(BUNDLED_SAMPLE_ARCHIVE).read_bytes()
    # allow_pickle=False: 上記 load_dataset と同じ理由で明示する。
    with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as npz:
        return _build_dataset(npz, metadata_text)


def bundled_sample_size_bytes() -> int:
    """同梱サンプル `.npz` のバイト数 (500kB 上限の監視用)。"""
    return len(files(SAMPLES_PACKAGE).joinpath(BUNDLED_SAMPLE_ARCHIVE).read_bytes())
