# 次PR候補（Sprint 1 レビューの MEDIUM / INFO findings）

Sprint 1（`feat/sprint1-esn-core`）の 7 観点並列レビューで検出され、
**BLOCKER/HIGH ではないため今回のスコープ外**とした findings。

HIGH 12 件は Phase 4 で修正済み。以下は次スプリント以降の候補。

## アーキテクチャ（Sprint 2 着手前にやると安い）

| #   | 対象                                           | 内容                                                                                                                                                                                                                                                                                                                                                                               |
| --- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | `data/source.py`                               | 抽象（`RolloutSource` Protocol）と具象（`SyntheticRolloutSource`）が同居。Sprint 2 で `OpenpiLogSource` を足すと、Protocol を参照するだけの利用側が合成データ生成器と openpi ログパーサの両方をロードすることになり、「openpi をランタイム依存に入れない」の担保が import 構造ではなく規律だけに依存する。`data/sources/` へ具象を分離する（再エクスポート名を変えなければ非破壊） |
| A2  | `diagnostics/report.py`                        | 4 責務（実行オーケストレーション / 辞書化 / ファイル書き出し / ログ整形）を持つ。特に辞書化が他モジュールのフィールドを手書き列挙しているため、`EspResult` にフィールドを足しても JSON から黙って欠落する。各結果型に `to_dict()` を持たせ、`run_diagnostics` は `runner.py` へ分離                                                                                                |
| A3  | `esn/config.py`                                | `ESNConfig` にリザバー生成系と read-out 学習系が同居。diagnose 経路では後者（`ridge_alpha`/`washout`/`input_passthrough`）が効かないのにレポート JSON に記録され、読者が「この ridge_alpha でこの数値が出た」と誤読しうる。`ReservoirConfig` / `ReadoutConfig` への分割は破壊的変更なので design.md 改訂とセットで                                                                 |
| A4  | `diagnostics/report.py` / `data/schema.py`     | `DataSource = Literal["synthetic", "openpi"]` が二重定義。出所を追加するとき片方だけ更新される危険                                                                                                                                                                                                                                                                                 |
| A5  | `esn/reservoir.py` / `diagnostics/spectral.py` | `_max_abs_eigenvalue` と `spectral_radius` が同一計算の二重実装。N>500 で反復法へ切り替えた瞬間に「W は目標 ρ にスケール済み」を検証しているはずの診断値が無意味になる。最下層 `linalg.py` へ寄せる                                                                                                                                                                                |
| A6  | `cli/app.py`                                   | サブコマンド追加が 4 箇所同時編集を要求。argparse 標準の `set_defaults(handler=...)` を使えば 1 行になる                                                                                                                                                                                                                                                                           |
| A7  | `*/commands.py`                                | ハンドラ引数が無型の `argparse.Namespace`。mypy strict でもここだけ型検査が効かず、ノートブックや uncertainty 層から呼ぶとき Namespace を偽装する羽目になる。型付き関数に切り出す                                                                                                                                                                                                  |
| A8  | `data/schema.py`                               | `RolloutDataset` から ESN 入力配列を取り出す変換が無く呼び出し側責務。Sprint 2 で 3 者が同じ抽出（NaN 行の扱い、エピソード境界を跨がない状態リセット）を各自実装するとコピペになる。**「エピソード境界でリザバー状態をリセットするか」は正しさに直結するので分散させない**                                                                                                         |
| A9  | Phase 1 準備                                   | `check_esp` / `linear_memory_capacity` の引数を最小 Protocol（`ReservoirLike`）に置き換えると物理リザバー移植時の差し替え点になる。実装が 1 つのうちは前倒ししない                                                                                                                                                                                                                 |

## セキュリティ（公開前に対応）

| #   | 対象         | 内容                                                                                                                                                                                                                         |
| --- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1  | `data/io.py` | `.npz` の配列サイズ無制限。zip 圧縮のため小さなファイルで OOM を起こせる（decompression bomb, CWE-409/789）。`MAX_TOTAL_STEPS` 等の上限を設ける                                                                              |
| S2  | `data/io.py` | `save_dataset` が `.npz` だけでなくサイドカー `.json` まで無警告上書き。`--output notes.npz` が既存の無関係な `notes.json` を破壊する（CWE-73）。`overwrite: bool = False` + CLI `--force`                                   |
| S3  | `cli/app.py` | `main()` に例外ハンドリングが無く、トレースバック（絶対パス・ユーザー名を含む）が stderr に出る。CLAUDE.md「エラーレスポンスにスタックトレースを含めない」に反する。公開後は issue への貼り付けで環境情報が漏れる（CWE-209） |
| S4  | ログ全般     | 書き出し先の**絶対パス**を INFO ログに出している（`data/io.py`, `diagnostics/report.py`, `*/commands.py`）。ユーザー名が残る。INFO では相対パス、絶対パスは DEBUG へ                                                         |
| S5  | `LICENSE`    | Apache-2.0 付録の `Copyright [yyyy] [name of copyright owner]` が未記入。権利帰属が不明確                                                                                                                                    |
| S6  | git 履歴     | **push 前ゲート（下記に詳述）。要件書だけでなく `docs/plans/sprint1_v0.1.md` と `docs/design.md` の過去版にも内部語彙が残存**                                                                                                |
| S7  | `data/io.py` | 出所固有バリデータを `io.py` がトップレベル import している。Sprint 2 で `source == "openpi"` の分岐を足すと `io.py` が openpi ログパーサを import することになり、「openpi をランタイム依存に含めない」の担保が import 構造ではなく規律のみになる。`data/invariants.py`（依存は `schema.py` のみ）へ切り出し、`io` はレジストリを引くだけにする |
| S8  | pre-commit   | 秘密情報の防御が `detect-private-key`（PEM のみ）だけで、しかも pre-commit は `make ci` に含まれない。`.gitignore` は `git add -f` で無力化されるため、`gitleaks` を pre-commit に追加し `make ci` からも走らせて `.gitignore` の網羅性への依存を下げる |
| S9  | dev 依存     | `pip-audit` が既知脆弱性 2 件を報告: pygments 2.19.2 (PYSEC-2026-2987, fix 2.20.0) / pytest 9.0.2 (PYSEC-2026-1845, fix 9.0.3)。dev グループのみで本番配布物（numpy のみ）には入らない。`make ci` に pip-audit を追加して自動検出させる |

## パッケージング / uv

| #   | 対象                                 | 内容                                                                                                                                                                                                                  |
| --- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| U1  | `pyproject.toml`                     | `[tool.hatch.build.targets.sdist]` が未定義。sdist に `.claude/`, `.devcontainer/`, `.github/`, `.vscode/`, `CLAUDE.md`, `docs/`（7000行）が同梱されている                                                            |
| U2  | `pyproject.toml`                     | OSS 公開想定なのに `classifiers` / `keywords` が無い。CITATION.cff には keywords があるのに pyproject に無く PyPI の検索性が落ちる                                                                                    |
| U3  | `CITATION.cff`                       | `version` が pyproject と手動二重管理。リリース時に片方だけ更新される drift。`make ci` に整合チェックを足す                                                                                                           |
| U4  | `.devcontainer/postCreateCommand.sh` | `uv sync`（ロック非固定）が Makefile の `uv sync --locked` と食い違う。Sprint 1 で editable package になったため、Dev Container 初回起動で uv.lock が無自覚に書き換わると `make ci` 先頭の `uv lock --check` が落ちる |

## パフォーマンス（実測済み。現状 60 秒要件に対し 0.35〜0.99s で余裕あり）

| #   | 対象                               | 内容                                                                                                                                                                                                                                          |
| --- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | `diagnostics/esp.py`               | `check_esp` が `n_initial_states`（既定8）本の軌道を独立に `reservoir.run` で駆動し、T=500 の Python ループを 8 回（計4000反復）回す。状態を `[K, N]` バッチにすれば反復数を 1/8 に削減できる。cProfile で `reservoir.run` が最大のコスト要素 |
| P2  | `diagnostics/esp.py` / `report.py` | 実効更新行列 `A = (1-a)I + aW` の固有値を独立に 2 回計算（O(N³) × 2）。N=500 で eigvals 合計 0.463s（diagnose 本体の約53%）。`run_diagnostics` で 1 回計算して渡す                                                                            |
| P3  | `esn/reservoir.py`                 | `Reservoir.run` の Python ループは O(T·N²)。design.md §8-4 に N≤500 の制約として記載済み。将来 numba / ベクトル化                                                                                                                             |

> **不採用**: reviewer-performance の「`summarize_spectral` の `spectral_radius(reservoir.W)` を `config.spectral_radius` で置き換える」提案は採用しない。
> この診断は「設定値どおりのスペクトル半径が実際に達成されているか」を**実測で検証する**のが目的であり、
> 設定値をそのまま出力しては診断として無意味になる。O(N³) 1 回分のコストは意図的に払う。

## スタイル / 可読性

| #   | 対象                                                                                                                                         | 内容                                                                                                                  |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| C1  | `data/io.py:_build_dataset` (~70行), `diagnostics/esp.py:check_esp` (~50行), `diagnostics/memory_capacity.py:linear_memory_capacity` (~79行) | CLAUDE.md の「50行超は分割を検討」を超過                                                                              |
| C2  | `diagnostics/esp.py`, `memory_capacity.py`                                                                                                   | `seed: int = 0` がインラインのマジックナンバー。同モジュールの他の既定値は `Final` 定数化されており一貫性が崩れている |
| C3  | `cli/app.py`                                                                                                                                 | `_run_diagnose` / `_run_gen_sample_data` が 1 行の委譲ラッパで間接参照が増えているだけ（A6 と同時に解消）             |

## テスト

| #   | 対象                      | 内容                                                                                                        |
| --- | ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| T1  | `tests/test_reservoir.py` | design.md §3.2 が「`input_scaling=0.0` は入力を無視するリザバーになる」と挙動を保証しているのにテストが無い |
| T2  | `tests/test_model.py`     | `ESN.fit` 経由の `washout=0`（捨てない）が未検証。`discard_washout` 単体では検証済み                        |

## ドキュメント

| #   | 対象                             | 内容                                                                                                                                                                                            |
| --- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | `esn/model.py`, `esn/readout.py` | 公開 API（`ESN.fit/predict/transform`, `RidgeReadout.fit/predict/design_matrix`）が一行 docstring のみ。diagnostics 層は Args/Returns/Raises 完備なので不整合。**利用者が最初に触る層こそ必要** |
| D2  | `CHANGELOG.md`                   | 未作成。v0.1.0 タグ付け（Sprint 3）までに Keep a Changelog 形式で整備                                                                                                                           |
| D3  | コミットメッセージ               | 「何を」の列挙が中心で「なぜ」が薄い。`Ref: docs/plans/sprint1_v0.1.md T4` のような設計文書への参照を入れる                                                                                     |

## push 前ゲート（S6 の詳細・必須）

このブランチはまだ push されていない（`origin/main` は Initial commit のみ、
`git branch -r --contains 94dfd74` は空）。**今なら履歴書き換えのコストはゼロ**だが、
一度 push すると GitHub 側にダングリングオブジェクトが残り `--force-with-lease` でも消えない
（GitHub サポートへの GC 依頼か、リポジトリ削除＋再作成が必要になる）。

非公開情報（第三者の実名、事業意図、特許ポジショニング、GPU 型番、社内でのみ通用する
プログラム/演習の呼称）が複数コミットに残存しており、対象は要件書だけでなく
`docs/plans/sprint1_v0.1.md` と `docs/design.md` にも及ぶ。

**この文書自体に該当語を書かないこと。** 検出パターンは各自の手元で組み立てる
（この文書は公開リポジトリに含まれるため、ここに列挙すると除去の意味がなくなる）。

push 前に以下を完了させること:

1. 履歴を書き換える。いずれかを選ぶ:
   - Sprint 1 全体を 1 コミットへ squash する（最も単純）
   - `git filter-repo --replace-text` で全履歴に対する語彙置換を行う
   - `git rebase -i --root` で該当コミットの内容を公開版に差し替える
2. **要件書だけを対象にしても不十分**。plans / design.md の過去版も対象に含めること
3. 書き換え後に全履歴を再走査して空であることを確認する（`<pattern>` は手元で組み立てる）:
   ```
   git rev-list --all | while read c; do git grep -linE '<pattern>' $c; done
   ```
4. 上記が空になるまで `git push` を実行しない
