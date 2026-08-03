"""ログに出すパス表記。

パッケージ内の**最下層**。標準ライブラリ以外に依存せず、本パッケージの他の
モジュールも import しない。

`CLAUDE.md` のログ出力ルールは、ログに氏名などの個人情報を含めないことを求める。
実際に漏れて困るのは**ユーザー名**であり、その典型的な出所はホームディレクトリを
含む絶対パス (`/home/<ユーザー名>/...`) である。診断レポートやデータセットの
書き出し先を INFO で記録する運用では、そのログが issue に貼られたり共有されたり
する経路が現実にあるため、ここを落とす (S4、CWE-532 系)。

方針は「ユーザー名を消す」であって「場所を隠す」ではない。書き出し先が分からない
ログは運用上の役に立たないため、消す対象はホームディレクトリの部分に限る。

- カレントディレクトリ配下 → 相対パス (最も読みやすい)
- ホームディレクトリ配下 → `~/...` に置換 (ユーザー名だけが落ちる)
- それ以外 (`/var/tmp/...`、`/mnt/...` など) → 絶対パスのまま

3 番目はユーザー名を含まないため、そのまま出す。共有ストレージのマウント名などに
組織名が含まれる可能性は残るが、それを避けるために場所を丸ごと落とすのは代償が
大きい (どこに書いたか分からなくなる)。ここは意図的な線引きである。

完全な絶対パスが要るのは開発時の切り分けなので、呼び出し側が DEBUG で別途出す
(`logger.debug("... abs_path=%s", path)`)。
"""

from __future__ import annotations

from pathlib import Path


def display_path(path: Path) -> str:
    """ログ用のパス表記を返す (ホームディレクトリを露出させない)。

    Args:
        path: 表示したいパス。相対・絶対のどちらでもよい。

    Returns:
        カレントディレクトリ配下なら相対パス、ホームディレクトリ配下なら
        ``~/`` 始まりのパス、それ以外は絶対パスのまま。

    Examples:
        カレントディレクトリが ``/home/user/project`` のとき::

            display_path(Path("/home/user/project/outputs/a.json"))  # "outputs/a.json"
            display_path(Path("/home/user/notes/a.json"))            # "~/notes/a.json"
            display_path(Path("/var/tmp/a.json"))                    # "/var/tmp/a.json"
    """
    try:
        resolved = path.resolve()
    except OSError:
        # カレントディレクトリが削除済み等で resolve できない。
        return path.name

    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        pass

    try:
        return str(Path("~") / resolved.relative_to(Path.home()))
    except (ValueError, RuntimeError):
        # ValueError: ホームディレクトリ配下ではない。
        # RuntimeError: ホームディレクトリを特定できない。
        return str(resolved)
