"""`esn_vla_uq.logging_paths` と、ログに絶対パスを出さないことの検証 (S4)。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from esn_vla_uq.cli import main
from esn_vla_uq.logging_paths import display_path


def test_path_under_cwd_becomes_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "outputs" / "report.json"
    assert display_path(target) == str(Path("outputs") / "report.json")


def test_path_outside_cwd_and_home_keeps_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ユーザー名を含まない場所は、書き出し先が分かるようそのまま出す。

    消す対象はホームディレクトリの部分であって「場所」ではない。
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outside = tmp_path / "elsewhere" / "report.json"
    assert display_path(outside) == str(outside)


def test_path_under_home_is_abbreviated_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ホーム配下は `~/...` にしてユーザー名だけを落とす。"""
    home = tmp_path / "home" / "someone"
    home.mkdir(parents=True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("HOME", str(home))

    rendered = display_path(home / "notes" / "report.json")
    assert rendered == str(Path("~") / "notes" / "report.json")
    assert "someone" not in rendered


def test_relative_path_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert display_path(Path("outputs/report.json")) == str(
        Path("outputs") / "report.json"
    )


def test_home_directory_is_never_exposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/home/<ユーザー名>/...` の形をログへ出さないこと (S4 の核心)。"""
    monkeypatch.chdir(tmp_path)
    rendered = display_path(Path.home() / "secret_project" / "report.json")
    assert str(Path.home()) not in rendered
    assert "secret_project" in rendered  # 場所は分かる


def test_gen_sample_data_logs_no_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CLI の INFO ログに絶対パスが現れないこと (S4 の実地確認)。"""
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        exit_code = main(
            ["gen-sample-data", "--n-episodes", "2", "--output-dir", "outputs"]
        )
    assert exit_code == 0
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "outputs" in messages
    assert str(tmp_path) not in messages
    assert str(Path.home()) not in messages


def test_debug_level_still_records_the_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """切り分けに必要な絶対パスは DEBUG では残す (隠蔽ではなく水準の分離)。"""
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.DEBUG):
        main(["gen-sample-data", "--n-episodes", "2", "--output-dir", "outputs"])
    debug_messages = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    )
    assert "abs_path=" in debug_messages
