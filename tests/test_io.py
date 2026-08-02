"""データセット入出力・同梱サンプル・`gen-sample-data` のテスト (Sprint 1 T5)。"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import IO, Final

import numpy as np
import numpy.lib.format as npy_format
import pytest

from esn_vla_uq.cli import main
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
    MAX_ACTION_DIM,
    MAX_CHUNK_HORIZON,
    MAX_DATASET_BYTES,
    SCHEMA_VERSION,
    Episode,
    NpzArray,
    RolloutDataset,
)
from esn_vla_uq.data.synthetic import DEFAULT_N_EPISODES, generate_dataset

MAX_BUNDLED_SIZE_BYTES = 500_000
"""pre-commit の check-added-large-files 既定値 (500kB) に合わせた上限。"""

BUNDLED_SEED = 0
"""同梱サンプルの生成シード (assets/samples/__init__.py の再生成手順と一致)。"""


@pytest.fixture(scope="module")
def dataset() -> RolloutDataset:
    return generate_dataset(seed=1, n_episodes=4)


@pytest.fixture
def saved_archive(dataset: RolloutDataset, tmp_path: Path) -> Path:
    return save_dataset(dataset, tmp_path / "rollouts.npz")


def assert_datasets_equal(left: RolloutDataset, right: RolloutDataset) -> None:
    """メタデータと全配列が一致することを検証する。"""
    assert left.to_metadata() == right.to_metadata()
    for lhs, rhs in zip(left.episodes, right.episodes, strict=True):
        assert lhs.episode_id == rhs.episode_id
        assert lhs.task_name == rhs.task_name
        assert lhs.success == rhs.success
        assert lhs.n_steps == rhs.n_steps
        assert lhs.failure_onset == rhs.failure_onset
        assert np.array_equal(lhs.state, rhs.state)
        assert np.array_equal(lhs.action, rhs.action)
        assert np.array_equal(lhs.is_inference_step, rhs.is_inference_step)
        assert np.array_equal(lhs.action_chunk, rhs.action_chunk, equal_nan=True)


def rewrite_archive(source: Path, destination: Path, **overrides: NpzArray) -> Path:
    """npz の一部配列を差し替えた壊れたアーカイブを作る (異常系テスト用)。"""
    with np.load(source) as npz:
        arrays: dict[str, NpzArray] = {name: npz[name] for name in npz.files}
    arrays.update(overrides)
    np.savez_compressed(
        destination,
        state=arrays["state"],
        action=arrays["action"],
        action_chunk=arrays["action_chunk"],
        is_inference_step=arrays["is_inference_step"],
        episode_starts=arrays["episode_starts"],
        episode_lengths=arrays["episode_lengths"],
    )
    shutil.copyfile(metadata_path_for(source), metadata_path_for(destination))
    return destination


def test_roundtrip_preserves_arrays_and_metadata(
    dataset: RolloutDataset, saved_archive: Path
) -> None:
    loaded = load_dataset(saved_archive)
    assert_datasets_equal(loaded, dataset)


def test_save_writes_metadata_sidecar(saved_archive: Path) -> None:
    metadata_path = metadata_path_for(saved_archive)
    assert metadata_path.is_file()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["source"] == "synthetic"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert len(payload["episodes"]) == 4


def test_saved_arrays_are_float32(saved_archive: Path) -> None:
    with np.load(saved_archive) as npz:
        assert npz["state"].dtype == np.float32
        assert npz["action"].dtype == np.float32
        assert npz["action_chunk"].dtype == np.float32
        assert npz["is_inference_step"].dtype == np.bool_


def test_non_npz_path_raises(dataset: RolloutDataset, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"拡張子は \.npz"):
        save_dataset(dataset, tmp_path / "rollouts.bin")


def test_missing_metadata_sidecar_raises(saved_archive: Path) -> None:
    metadata_path_for(saved_archive).unlink()
    with pytest.raises(FileNotFoundError, match="サイドカー JSON"):
        load_dataset(saved_archive)


def test_dtype_mismatch_in_archive_raises(saved_archive: Path, tmp_path: Path) -> None:
    with np.load(saved_archive) as npz:
        state = npz["state"].astype(np.float64)
    broken = rewrite_archive(saved_archive, tmp_path / "broken.npz", state=state)
    with pytest.raises(ValueError, match="state: dtype が不正"):
        load_dataset(broken)


def test_shape_mismatch_in_archive_raises(saved_archive: Path, tmp_path: Path) -> None:
    with np.load(saved_archive) as npz:
        action = npz["action"][:, :-1]
    broken = rewrite_archive(saved_archive, tmp_path / "broken.npz", action=action)
    with pytest.raises(ValueError, match="action: shape が不正"):
        load_dataset(broken)


def test_inconsistent_episode_starts_raises(
    saved_archive: Path, tmp_path: Path
) -> None:
    with np.load(saved_archive) as npz:
        starts = npz["episode_starts"].copy()
    starts[1] += 3
    broken = rewrite_archive(
        saved_archive, tmp_path / "broken.npz", episode_starts=starts
    )
    with pytest.raises(ValueError, match="累積和と一致しません"):
        load_dataset(broken)


def test_unknown_schema_version_raises(saved_archive: Path) -> None:
    metadata_path = metadata_path_for(saved_archive)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "0.0.0-unknown"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version: 未知のバージョン"):
        load_dataset(saved_archive)


def test_unknown_source_raises(saved_archive: Path) -> None:
    metadata_path = metadata_path_for(saved_archive)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["source"] = "real_libero"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source: 未知の出所"):
        load_dataset(saved_archive)


def test_episode_count_mismatch_raises(saved_archive: Path) -> None:
    metadata_path = metadata_path_for(saved_archive)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["episodes"] = payload["episodes"][:-1]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="episode_lengths と一致しません"):
        load_dataset(saved_archive)


def test_chunk_count_mismatch_raises(saved_archive: Path, tmp_path: Path) -> None:
    with np.load(saved_archive) as npz:
        chunks = npz["action_chunk"][:-1]
    broken = rewrite_archive(
        saved_archive, tmp_path / "broken.npz", action_chunk=chunks
    )
    with pytest.raises(ValueError, match="action_chunk"):
        load_dataset(broken)


def test_load_dataset_rejects_synthetic_dataset_missing_failure_onset(
    saved_archive: Path, dataset: RolloutDataset
) -> None:
    # M2: 合成データ生成器固有の不変条件 (失敗エピソードには failure_onset が
    # 必須) は生成直後だけでなく読み込み境界でも検証される。保存済みメタ
    # データから failure_onset を抜いた「破損した合成データ」を load_dataset
    # が ValueError で弾くことを確認する。
    failure_episode = next(
        episode for episode in dataset.episodes if not episode.success
    )
    assert failure_episode.failure_onset is not None

    metadata_path = metadata_path_for(saved_archive)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    record = next(
        record
        for record in payload["episodes"]
        if record["episode_id"] == failure_episode.episode_id
    )
    assert record["failure_onset"] is not None
    record["failure_onset"] = None
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="failure_onset"):
        load_dataset(saved_archive)


def test_save_dataset_rejects_synthetic_dataset_missing_failure_onset(
    dataset: RolloutDataset, tmp_path: Path
) -> None:
    # F2 (3 周目レビュー): 検証フックが読み込み側 (_build_dataset) にしか
    # 無く、save_dataset は dataset.validate() (出所に依存しない共通スキーマ
    # 契約) しか経由しなかった。そのため failure_onset=None の失敗エピソード
    # を持つ synthetic データセットが save_dataset には受理されるのに、同じ
    # ファイルを load_dataset で読むと ValueError で弾かれる非対称な成果物
    # (書けるが二度と読めない) を作れてしまっていた。save 側でも
    # validate_synthetic_dataset 相当の検証が通ることを確認する。
    failure_episode = next(ep for ep in dataset.episodes if not ep.success)
    assert failure_episode.failure_onset is not None
    broken_episode = replace(failure_episode, failure_onset=None)
    broken_dataset = replace(
        dataset,
        episodes=[
            broken_episode if episode is failure_episode else episode
            for episode in dataset.episodes
        ],
    )

    archive_path = tmp_path / "broken_synthetic.npz"
    with pytest.raises(ValueError, match="failure_onset"):
        save_dataset(broken_dataset, archive_path)

    # mkdir より前に検証することの裏付け: 拒否されたファイルは作られない。
    assert not archive_path.exists()
    assert not metadata_path_for(archive_path).exists()


def test_load_dataset_rejects_action_dim_over_max(saved_archive: Path) -> None:
    # M3 (CWE-789): メタデータ由来の action_dim は、それを使って
    # action_chunk の復元配列を確保する前に上限で拒否される。実際に巨大な
    # 配列を確保させないよう、メタデータだけを細工した小さいファイルで検証する。
    metadata_path = metadata_path_for(saved_archive)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["action_dim"] = MAX_ACTION_DIM + 1
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="action_dim: 上限を超えています"):
        load_dataset(saved_archive)


def test_load_dataset_rejects_chunk_horizon_over_max(saved_archive: Path) -> None:
    metadata_path = metadata_path_for(saved_archive)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["chunk_horizon"] = MAX_CHUNK_HORIZON + 1
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_horizon: 上限を超えています"):
        load_dataset(saved_archive)


def test_load_dataset_rejects_byte_budget_over_max_even_within_individual_caps(
    saved_archive: Path, dataset: RolloutDataset
) -> None:
    # reviewer PoC 相当: 個々の次元は MAX_ACTION_DIM / MAX_CHUNK_HORIZON の
    # 上限内でも、n_steps * chunk_horizon * action_dim * 4 の推定確保サイズが
    # MAX_DATASET_BYTES (2GiB) を超えれば、action_chunk 復元 (`np.full`) の
    # 前に拒否される。ディスク上の npz は変更しない (実際には確保させない)。
    metadata_path = metadata_path_for(saved_archive)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["action_dim"] = MAX_ACTION_DIM
    payload["chunk_horizon"] = MAX_CHUNK_HORIZON
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    assert dataset.total_steps * MAX_CHUNK_HORIZON * MAX_ACTION_DIM * 4 > (
        MAX_DATASET_BYTES
    )
    with pytest.raises(ValueError, match="action_chunk: 復元後配列の推定確保サイズ"):
        load_dataset(saved_archive)


_POC_STATE_SHAPE: Final[tuple[int, int]] = (150_000_000, 8)
"""reviewer PoC (3 周目レビュー F1) を再現する `state` の自己申告 shape。

`state_dim` (第 2 軸, 8) はメタデータ上も正規値のままにし、メタデータの
どのフィールドからも検証されない `n_steps` 方向 (第 1 軸) だけを膨らませる。
"""


def _write_lying_state_header(fp: IO[bytes], dtype: np.dtype[np.float32]) -> None:
    """`state.npy` の `.npy` ヘッダのみを書く (実データはこの後 zero-fill する)。"""
    header = {
        "descr": npy_format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": _POC_STATE_SHAPE,
    }
    npy_format.write_array_header_1_0(fp, header)


def _build_oversized_state_archive(path: Path, *, npy_suffix: bool = True) -> None:
    """reviewer PoC 相当の細工 npz を構築する (CWE-789 リグレッションテスト用)。

    `npy_suffix=False` のとき、zip エントリ名から `.npy` を外す。numpy の
    `NpzFile` は `name.removesuffix(".npy")` でキーを解決するため拡張子は
    **任意**であり、`state` という名前でも `npz["state"]` として読まれる。
    検証対象を拡張子で絞ると、この名前でヘッダ検証を丸ごと迂回できる
    (3 周目の敵対的レビューで実測: 1,099 バイトで 32 GiB の確保を誘発)。

    `state.npy` は shape=`_POC_STATE_SHAPE` (float32) を自己申告するが、実体は
    全ゼロで極めて高圧縮率なため、ディスク上のファイルは 4〜5MB に収まる
    (`state_dim`/`action_dim`/`chunk_horizon` はメタデータ上は正規値のまま)。
    展開後は `150_000_000 * 8 * 4` バイト (約 4.47 GiB) の確保を要求するため、
    `check_npz_uncompressed_budget` による事前拒否が無いと `state = npz["state"]`
    (最初の配列実体化) で巨大な確保が起きる。

    zero-fill データを実際に deflate 圧縮するため、この関数自体の実行には
    数秒かかる (検証対象の `load_dataset` 呼び出しそのものではない)。
    """
    dtype = np.dtype(np.float32)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # `.npy` ヘッダだけが巨大な shape を自己申告し、実データはほぼ伴わない。
        # numpy の read_array はヘッダの shape を読んだ直後・実データを読む前に
        # np.empty(shape, dtype) を確保するため、この 1KB 程度のファイルでも
        # 検証が無ければ約 4.47 GiB の確保を誘発できる。実データを 4.5GB 書く
        # 必要は無く (むしろ zip の非圧縮サイズ合計チェックを素通りする分だけ
        # 攻撃として強い)、テストも数ミリ秒で構築できる。
        suffix = ".npy" if npy_suffix else ""
        with archive.open(f"state{suffix}", "w") as entry:
            _write_lying_state_header(entry, dtype)
            entry.write(bytes(4))
        for name, small_array in (
            ("action", np.zeros((1, 7), dtype=np.float32)),
            ("action_chunk", np.zeros((1, 16, 7), dtype=np.float32)),
            ("is_inference_step", np.zeros((1,), dtype=np.bool_)),
            ("episode_starts", np.zeros((1,), dtype=np.int64)),
            ("episode_lengths", np.ones((1,), dtype=np.int64)),
        ):
            buffer = io.BytesIO()
            np.save(buffer, small_array)
            archive.writestr(f"{name}{suffix}", buffer.getvalue())

    _write_poc_metadata(path)


def _write_poc_metadata(path: Path) -> None:
    """PoC アーカイブ用の正規メタデータを書く (次元はすべて正規値)。"""
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": "synthetic",
        "policy": "poc-oversized-state",
        "seed": 0,
        "control_hz": 20.0,
        "n_episodes": 1,
        "total_steps": 1,
        "state_dim": 8,
        "action_dim": 7,
        "chunk_horizon": 16,
        "episodes": [
            {
                "episode_id": "e0",
                "task_name": "poc",
                "success": True,
                "n_steps": 1,
                "failure_onset": None,
            }
        ],
    }
    metadata_path_for(path).write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture
def oversized_state_archive(tmp_path: Path) -> Path:
    path = tmp_path / "oversized_state.npz"
    _build_oversized_state_archive(path)
    return path


def test_load_dataset_rejects_negative_dimension_offset(tmp_path: Path) -> None:
    """負の次元を宣言したダミーエントリで合計を相殺できないこと。

    numpy のヘッダ検証は shape の要素が int であることしか見ておらず負の値を
    通す。宣言サイズを**合計してから 1 回だけ**上限と比べる実装だと、numpy が
    決して読まないエントリに負の次元を宣言させて本命の巨大な寄与を相殺でき、
    1KB 強の npz で 32 GiB (shape 次第で 512 TiB) の確保を誘発できた。
    負の次元の拒否と per-entry 検証のどちらを外してもこのテストは落ちる。
    """
    big = 2**30
    dtype = np.dtype(np.float32)
    path = tmp_path / "negative_offset.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, shape in (
            ("state.npy", (big, 8)),
            ("pad.npy", (-big, 8)),  # 合計を 0 に相殺させるためのダミー
        ):
            buffer = io.BytesIO()
            npy_format.write_array_header_1_0(
                buffer,
                {
                    "descr": npy_format.dtype_to_descr(dtype),
                    "fortran_order": False,
                    "shape": shape,
                },
            )
            buffer.write(bytes(4))
            archive.writestr(name, buffer.getvalue())
        for name, small_array in (
            ("action.npy", np.zeros((1, 7), dtype=np.float32)),
            ("action_chunk.npy", np.zeros((1, 16, 7), dtype=np.float32)),
            ("is_inference_step.npy", np.zeros((1,), dtype=np.bool_)),
            ("episode_starts.npy", np.zeros((1,), dtype=np.int64)),
            ("episode_lengths.npy", np.ones((1,), dtype=np.int64)),
        ):
            buffer = io.BytesIO()
            np.save(buffer, small_array)
            archive.writestr(name, buffer.getvalue())
    _write_poc_metadata(path)

    # 相殺が成立していることを明示する: 符号付き合計は上限を大きく下回る。
    signed_total = (big * 8 - big * 8) * dtype.itemsize
    assert signed_total < MAX_DATASET_BYTES

    # **確保前のガードが出すメッセージ**を要求する。単に ValueError を待つだけ
    # では不十分: ガードを外すと 32 GiB の確保が overcommit で遅延成立し、
    # 後段の shape 検証が別の ValueError を出すためテストが素通りしてしまう。
    with pytest.raises(
        ValueError, match=r"エントリ .* の宣言サイズが上限を超えています|負の次元"
    ):
        load_dataset(path)


def test_load_dataset_rejects_oversized_npz_without_npy_suffix(
    tmp_path: Path,
) -> None:
    """`.npy` を持たないエントリ名でもヘッダ検証を迂回できないこと。

    numpy の `NpzFile` は `.npy` サフィックスを任意として扱うため、検証対象を
    ファイル名の拡張子で絞ると `state` (拡張子なし) で丸ごと迂回できた。
    magic バイトで判定することでこの経路を塞いでいる。
    """
    path = tmp_path / "oversized_noext.npz"
    _build_oversized_state_archive(path, npy_suffix=False)

    start = time.perf_counter()
    with pytest.raises(ValueError, match=r"npz: .*が上限を超えています"):
        load_dataset(path)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.01, (
        f"想定より遅い ({elapsed:.6f}s): 割り当て前に拒否できていない可能性があります"
    )


def test_load_dataset_rejects_oversized_npz_before_allocating(
    oversized_state_archive: Path,
) -> None:
    # reviewer PoC 実測 (3 周目レビュー F1)。修正前は npz 内の `.npy` ヘッダが
    # 自己申告する `state` の shape (n_steps 方向) を検証しておらず、
    # `state = npz["state"]` (最初の配列実体化) で RLIMIT_AS=1GiB 下なら
    # MemoryError、無制限環境では約 4.47 GiB の実確保が起きた
    # (state_dim/action_dim/chunk_horizon はメタデータ上は正規値のまま)。
    # 修正後は配列を 1 つも実体化する前に ValueError で弾かれる。
    #
    # ディスク上のサイズと確保されうるサイズの比 (増幅率) こそが攻撃の本質。
    # 数 KB のファイルが GB 級の確保を要求できることを固定する。
    size_bytes = oversized_state_archive.stat().st_size
    assert size_bytes < 100_000, (
        f"PoC は小さなファイルである必要があります: {size_bytes} bytes"
    )
    declared_bytes = _POC_STATE_SHAPE[0] * _POC_STATE_SHAPE[1] * 4
    assert declared_bytes / size_bytes > 1000, "増幅率が小さく PoC として不十分です"

    start = time.perf_counter()
    with pytest.raises(ValueError, match=r"npz: .*が上限を超えています"):
        load_dataset(oversized_state_archive)
    elapsed = time.perf_counter() - start

    # 割り当て前 (=配列を実体化する前) に拒否されたことを、所要時間でも
    # 裏付ける。実際に allocate していれば数百ミリ秒〜数秒かかりうる規模。
    assert elapsed < 0.01, (
        f"想定より遅い ({elapsed:.6f}s): 割り当て前に拒否できていない可能性があります"
    )


def _custom_dim_dataset() -> RolloutDataset:
    """既定の次元 (8/7/16) とは異なる自己記述的なデータセットを組み立てる。"""
    state_dim, action_dim, chunk_horizon, n_steps = 5, 3, 4, 20
    rng = np.random.default_rng(7)
    is_inference_step = np.zeros(n_steps, dtype=np.bool_)
    is_inference_step[::chunk_horizon] = True
    action_chunk = np.full(
        (n_steps, chunk_horizon, action_dim), np.nan, dtype=np.float32
    )
    action_chunk[is_inference_step] = rng.normal(
        size=(int(is_inference_step.sum()), chunk_horizon, action_dim)
    ).astype(np.float32)
    episode = Episode(
        episode_id="custom_0000",
        task_name="custom_task",
        success=True,
        n_steps=n_steps,
        state=rng.normal(size=(n_steps, state_dim)).astype(np.float32),
        action=rng.normal(size=(n_steps, action_dim)).astype(np.float32),
        action_chunk=action_chunk,
        is_inference_step=is_inference_step,
    )
    return RolloutDataset(
        episodes=[episode],
        source="synthetic",
        policy="custom-dim-policy",
        seed=7,
        control_hz=10.0,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_horizon=chunk_horizon,
    )


def test_save_load_round_trips_custom_dims_from_metadata(tmp_path: Path) -> None:
    # io._build_dataset がモジュール定数ではなくメタデータの state_dim /
    # action_dim / chunk_horizon を読んで復元することを検証する (次元の違う
    # データを読んでも無関係な shape エラーにならない自己記述性の確認)。
    dataset = _custom_dim_dataset()
    archive = save_dataset(dataset, tmp_path / "custom_dims.npz")
    loaded = load_dataset(archive)
    assert loaded.state_dim == 5
    assert loaded.action_dim == 3
    assert loaded.chunk_horizon == 4
    assert loaded.episodes[0].state.shape == (20, 5)
    assert loaded.episodes[0].action.shape == (20, 3)
    assert loaded.episodes[0].action_chunk.shape == (20, 4, 3)


def test_bundled_sample_loads_and_validates() -> None:
    sample = load_bundled_sample()
    sample.validate()
    assert sample.n_episodes == DEFAULT_N_EPISODES
    assert sample.source == "synthetic"
    assert sample.seed == BUNDLED_SEED
    successes = [episode.success for episode in sample.episodes]
    assert any(successes)
    assert not all(successes)


def test_bundled_sample_matches_documented_generation() -> None:
    # assets/samples/__init__.py に書いた再生成手順で同じ内容が得られること。
    assert_datasets_equal(load_bundled_sample(), generate_dataset(seed=BUNDLED_SEED))


def test_bundled_sample_is_small_enough() -> None:
    size = bundled_sample_size_bytes()
    message = f"{BUNDLED_SAMPLE_ARCHIVE} が大きすぎます: {size} bytes"
    assert size < MAX_BUNDLED_SIZE_BYTES, message


def test_bundled_metadata_declares_synthetic_source() -> None:
    from importlib.resources import files

    text = (
        files("esn_vla_uq.assets.samples")
        .joinpath(BUNDLED_SAMPLE_METADATA)
        .read_text(encoding="utf-8")
    )
    assert json.loads(text)["source"] == "synthetic"


def test_cli_gen_sample_data_writes_files(tmp_path: Path) -> None:
    archive = tmp_path / "s.npz"
    assert main(["gen-sample-data", "--seed", "0", "--output", str(archive)]) == 0
    assert archive.is_file()
    assert metadata_path_for(archive).is_file()
    loaded = load_dataset(archive)
    assert loaded.source == "synthetic"
    assert loaded.n_episodes == DEFAULT_N_EPISODES


def test_cli_gen_sample_data_respects_n_episodes(tmp_path: Path) -> None:
    archive = tmp_path / "small.npz"
    exit_code = main(
        [
            "gen-sample-data",
            "--seed",
            "2",
            "--n-episodes",
            "3",
            "--output",
            str(archive),
        ]
    )
    assert exit_code == 0
    assert load_dataset(archive).n_episodes == 3


def test_cli_gen_sample_data_defaults_to_output_dir(tmp_path: Path) -> None:
    exit_code = main(
        [
            "gen-sample-data",
            "--seed",
            "0",
            "--n-episodes",
            "2",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "outputs" / "synthetic_rollouts.npz").is_file()


def test_console_script_gen_sample_data_exits_zero(tmp_path: Path) -> None:
    executable = shutil.which("esn-vla-uq")
    if executable is None:
        pytest.skip("console script `esn-vla-uq` が PATH 上に無い (未インストール環境)")
    archive = tmp_path / "s.npz"
    result = subprocess.run(
        [executable, "gen-sample-data", "--seed", "0", "--output", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert archive.is_file()
