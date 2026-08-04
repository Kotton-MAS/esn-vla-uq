"""`argparse.Namespace` から型付きの値を取り出すヘルパ。

`Namespace` の属性は無型なので、`args.n_reservoir` のような読み出しは mypy strict
でも検査されない。ハンドラ本体がそれを直接触ると、レイヤ全体が型検査の外に出る
(A7)。各サブコマンドは**入口で 1 度だけ**ここを通して型付きの設定オブジェクトへ
変換し、以降は型の付いた値だけを扱う。

`getattr` の戻り値は `Any` になるが、いったん `object` として受けてから
`isinstance` で絞り込むことで、`disallow_any_explicit` を満たしたまま実行時にも
検証できる。argparse の設定と食い違えば `TypeError` になり、黙って別の型が流れる
ことはない。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import cast


def _raw(args: Namespace, name: str) -> object:
    """属性を `object` として取り出す (未定義なら `AttributeError`)。"""
    value: object = getattr(args, name)
    return value


def _fail(name: str, expected: str, value: object) -> TypeError:
    """型が合わないときの例外を組み立てる。"""
    return TypeError(
        f"--{name.replace('_', '-')}: {expected} が必要です "
        f"(actual={type(value).__name__})"
    )


def get_int(args: Namespace, name: str) -> int:
    """整数オプションを取り出す。"""
    value = _raw(args, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(name, "整数", value)
    return value


def get_float(args: Namespace, name: str) -> float:
    """実数オプションを取り出す (整数も受ける)。"""
    value = _raw(args, name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(name, "実数", value)
    return float(value)


def get_bool(args: Namespace, name: str) -> bool:
    """真偽値オプション (`store_true`) を取り出す。"""
    value = _raw(args, name)
    if not isinstance(value, bool):
        raise _fail(name, "真偽値", value)
    return value


def get_str(args: Namespace, name: str) -> str:
    """文字列オプションを取り出す。"""
    value = _raw(args, name)
    if not isinstance(value, str):
        raise _fail(name, "文字列", value)
    return value


def get_optional_str(args: Namespace, name: str) -> str | None:
    """省略可能な文字列オプションを取り出す。"""
    value = _raw(args, name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _fail(name, "文字列", value)
    return value


def get_path(args: Namespace, name: str) -> Path:
    """パスオプションを取り出す。"""
    value = _raw(args, name)
    if not isinstance(value, Path):
        raise _fail(name, "パス", value)
    return value


def get_optional_path(args: Namespace, name: str) -> Path | None:
    """省略可能なパスオプションを取り出す。"""
    value = _raw(args, name)
    if value is None:
        return None
    if not isinstance(value, Path):
        raise _fail(name, "パス", value)
    return value


def get_choice[T: str](args: Namespace, name: str, allowed: tuple[T, ...]) -> T:
    """`choices` 付きオプションを `Literal` 型として取り出す。

    argparse が `choices` を検証済みなので通常は通るが、`Literal` へ絞り込むには
    実行時にも確認が要る。想定外の値は `TypeError` にする。
    """
    value = get_str(args, name)
    if value not in allowed:
        raise _fail(name, f"{list(allowed)} のいずれか", value)
    # `allowed` は呼び出し側の `Literal` の値域そのもの (`get_args` 由来) なので、
    # 上の検査を通った時点で `T` に属することが実行時に保証されている。
    return cast("T", value)
