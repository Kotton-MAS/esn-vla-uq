"""ログに出すパス表記。

パッケージ内の**最下層**。標準ライブラリ以外に依存せず、本パッケージの他の
モジュールも import しない。

`CLAUDE.md` のログ出力ルールは、ログに氏名などの個人情報を含めないことを求める。
`Path.resolve()` した絶対パスはホームディレクトリを含み、多くの環境で
`/home/<ユーザー名>/...` の形になる。診断レポートやデータセットの書き出し先を
INFO で記録する運用では、そのログが issue に貼られたり共有されたりする経路が
現実にあるため、**INFO 以下の水準では絶対パスを出さない** (S4、CWE-532 系)。

方針:

- `display_path` はカレントディレクトリ配下なら相対パスを返す。
- 配下でなければファイル名のみを返す。上位ディレクトリ名にもユーザー名や
  組織名が含まれうるため、パスの一部を残す折衷は取らない。
- 完全な絶対パスが要るのは開発時の切り分けなので、呼び出し側が DEBUG で別途
  出す (`logger.debug("... abs_path=%s", path)`)。DEBUG を有効にするのは明示的な
  操作であり、そこで何が出るかは利用者が選べる。
"""

from __future__ import annotations

from pathlib import Path


def display_path(path: Path) -> str:
    """ログ用のパス表記を返す (絶対パスを出さない)。

    Args:
        path: 表示したいパス。相対・絶対のどちらでもよい。

    Returns:
        カレントディレクトリからの相対パス。カレントディレクトリの外にある
        場合はファイル名のみ。

    Examples:
        カレントディレクトリが ``/home/user/project`` のとき::

            display_path(Path("/home/user/project/outputs/a.json"))  # "outputs/a.json"
            display_path(Path("/var/tmp/a.json"))                    # "a.json"
    """
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        # ValueError: カレントディレクトリの配下ではない。
        # OSError: カレントディレクトリが削除済み等で resolve できない。
        return path.name
