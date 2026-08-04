"""CLI のエラー処理と出力先保護のテスト (S2 / S3)。"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pytest

from esn_vla_uq.cli import app, main
from esn_vla_uq.cli.app import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_OK
from esn_vla_uq.data.io import save_dataset
from esn_vla_uq.data.schema import RolloutDataset
from esn_vla_uq.data.synthetic import generate_dataset

N_EPISODES = 2


@pytest.fixture
def dataset() -> RolloutDataset:
    return generate_dataset(seed=0, n_episodes=N_EPISODES)


# --- S2: サイドカーの無警告上書き (CWE-73) ---------------------------------


def test_save_refuses_to_overwrite_existing_archive(
    dataset: RolloutDataset, tmp_path: Path
) -> None:
    target = tmp_path / "rollouts.npz"
    save_dataset(dataset, target)
    with pytest.raises(FileExistsError, match="既に存在します"):
        save_dataset(dataset, target)


def test_save_refuses_to_overwrite_unrelated_sidecar(
    dataset: RolloutDataset, tmp_path: Path
) -> None:
    """`.npz` が無くても、同名の `.json` が既にあれば止まること。

    これが S2 の本体。`--output notes.npz` を指定した利用者は `notes.json` が
    書かれることを意識しておらず、無関係な既存ファイルが黙って壊れうる。
    """
    unrelated = tmp_path / "notes.json"
    unrelated.write_text('{"keep": "me"}', encoding="utf-8")

    with pytest.raises(FileExistsError, match=re.escape("notes.json")):
        save_dataset(dataset, tmp_path / "notes.npz")

    assert json.loads(unrelated.read_text(encoding="utf-8")) == {"keep": "me"}
    assert not (tmp_path / "notes.npz").exists()


def test_save_overwrites_when_explicitly_allowed(
    dataset: RolloutDataset, tmp_path: Path
) -> None:
    target = tmp_path / "rollouts.npz"
    save_dataset(dataset, target)
    assert save_dataset(dataset, target, overwrite=True) == target


def test_gen_sample_data_refuses_to_clobber(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """CLI 経由でも止まり、終了コード 1 とエラーログを返すこと。"""
    unrelated = tmp_path / "notes.json"
    unrelated.write_text('{"keep": "me"}', encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        exit_code = main(
            [
                "gen-sample-data",
                "--n-episodes",
                str(N_EPISODES),
                "--output",
                str(tmp_path / "notes.npz"),
            ]
        )

    assert exit_code == EXIT_ERROR
    assert "FileExistsError" in caplog.text
    assert json.loads(unrelated.read_text(encoding="utf-8")) == {"keep": "me"}


def test_gen_sample_data_force_allows_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "rollouts.npz"
    argv = [
        "gen-sample-data",
        "--n-episodes",
        str(N_EPISODES),
        "--output",
        str(target),
    ]
    assert main(argv) == EXIT_OK
    assert main([*argv, "--force"]) == EXIT_OK


# --- S3: トレースバックの漏洩 (CWE-209) ------------------------------------


def test_runtime_error_returns_exit_code_without_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """例外は終了コードとログになり、stderr へ送出されないこと。"""
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        exit_code = main(["gen-sample-data", "--output", str(tmp_path / "notes.npz")])
    assert exit_code == EXIT_ERROR
    # 例外の型名とメッセージのみ。トレースバックは含まれない。
    assert "Traceback" not in caplog.text
    assert 'File "' not in caplog.text


def test_traceback_is_available_at_debug_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """開発時の切り分け手段は残すこと (隠蔽ではなく水準の分離)。"""
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        main(["gen-sample-data", "--output", str(tmp_path / "notes.npz")])
    debug_records = [
        record for record in caplog.records if record.levelno == logging.DEBUG
    ]
    assert any(record.exc_info is not None for record in debug_records)


def test_keyboard_interrupt_maps_to_conventional_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _interrupt(_args: argparse.Namespace) -> int:
        raise KeyboardInterrupt

    # サブコマンドは `set_defaults(handler=...)` で振り分けるため、テーブル側の
    # ハンドラを差し替える (A6 でラッパ関数は廃止した)。
    monkeypatch.setattr(
        "esn_vla_uq.diagnostics.commands.run_diagnose", _interrupt, raising=True
    )
    monkeypatch.setattr(
        "esn_vla_uq.cli.app.SUBCOMMANDS",
        tuple(
            subcommand._replace(handler=_interrupt)
            if subcommand.name == "diagnose"
            else subcommand
            for subcommand in app.SUBCOMMANDS
        ),
    )
    exit_code = main(["diagnose", "--output-dir", str(tmp_path)])
    assert exit_code == EXIT_INTERRUPTED


def test_argument_errors_still_exit_with_code_two() -> None:
    """使い方の誤りは実行時エラーと区別する (argparse の SystemExit(2))。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["no-such-command"])
    assert excinfo.value.code == 2
