"""CLI 骨格のスモークテスト (Sprint 1 T1)。"""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from esn_vla_uq import __version__
from esn_vla_uq.cli import build_parser, main
from esn_vla_uq.esn.config import DEFAULT_LEAK_RATE


def _pyproject_version() -> str:
    """`pyproject.toml` の version を読む。

    テスト側にバージョンを書き写すと、リリースのたびに手で同期する箇所が
    増える (U3 と同じ drift)。唯一の真実から読む。
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


EXPECTED_VERSION = _pyproject_version()
SUBCOMMANDS = ("diagnose", "gen-sample-data", "calibrate", "demo")


def test_package_version_matches_pyproject() -> None:
    """インストール済みメタデータと pyproject が一致すること。"""
    assert __version__ == EXPECTED_VERSION


def test_version_option_exits_zero_and_prints_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert EXPECTED_VERSION in capsys.readouterr().out


def test_help_lists_subcommands_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    stdout = capsys.readouterr().out
    for subcommand in SUBCOMMANDS:
        assert subcommand in stdout


def test_no_arguments_prints_usage_and_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_unknown_subcommand_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["no-such-command"])
    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_common_option_defaults(subcommand: str) -> None:
    args = build_parser().parse_args([subcommand])
    assert args.command == subcommand
    assert args.seed == 0
    assert args.output_dir == Path("outputs")
    assert args.log_level == "INFO"


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_common_options_are_parsed(subcommand: str, tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            subcommand,
            "--seed",
            "7",
            "--output-dir",
            str(tmp_path),
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.seed == 7
    assert args.output_dir == tmp_path
    assert args.log_level == "DEBUG"


def test_diagnose_option_defaults() -> None:
    # diagnose は T4 で配線済み (実行を伴うスモークテストは tests/test_report.py、
    # gen-sample-data は T5 で配線済みで tests/test_io.py)。
    args = build_parser().parse_args(["diagnose"])
    assert args.n_reservoir == 200
    assert args.spectral_radius == 0.9
    assert args.leak_rate == DEFAULT_LEAK_RATE
    assert args.skip_memory_capacity is False


def test_installed_console_script_reports_version() -> None:
    executable = shutil.which("esn-vla-uq")
    if executable is None:
        pytest.skip("console script `esn-vla-uq` が PATH 上に無い (未インストール環境)")
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert EXPECTED_VERSION in result.stdout
