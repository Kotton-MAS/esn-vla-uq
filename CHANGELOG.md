# Changelog

このプロジェクトの重要な変更を記録します。

書式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に従い、
バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に従います。

v0.1.0 は未リリースです。それまでの間、公開 API は予告なく変更されることがあります。

## [Unreleased]

### Added

- **conformal 予測区間と較正評価** (`esn-vla-uq calibrate`): ESN の 1 ステップ先
  action 予測に split conformal を掛け、ステップ単位の予測区間・被覆率・
  reliability curve・ECE・失敗検知 AUROC を JSON で出力する

- 非適合度スコア 2 種 (`absolute` / `normalized`)。既定は `normalized`。
  `absolute` は被覆率が正確 (ECE 0.002) だが区間幅が定数になり、失敗検知 AUROC が
  定義上 0.5 になる。`normalized` は失敗検知 AUROC 0.869 と引き換えに被覆率が
  やや名目を下回る (0.864 対 0.903)

- ESN 入力にチャンク由来の要約量 (`log_chunk_dispersion` /
  `steps_since_inference`) を追加。要件書が定める入力「action chunk 系列と
  固有受容感覚」に合わせたもので、平均区間幅が 0.113 から 0.053 へ半減し、
  被覆率の分割間ばらつきも 0.069 から 0.027 に縮んだ

- 較正データ分割 (`within_task` 既定 / `across_task`)。`across_task` は交換可能性が
  崩れるため、被覆率保証が弱いことをレポートとログに明記する

- reliability diagram の PNG 出力 (`--diagram`)。matplotlib は任意依存
  (`esn-vla-uq[viz]`) で、数値は matplotlib 無しでも常に出力される

- ESN コア: リザバー生成 (`Reservoir`)、リッジ read-out (`RidgeReadout`)、
  両者を束ねる `ESN`、ハイパーパラメータ定義 (`ESNConfig`)

- リザバー診断: スペクトル半径、Echo State Property (3 指標を必ず併記)、
  線形メモリ容量、および診断レポートの JSON 出力

- 決定論的な合成ロールアウト生成と同梱サンプルデータ
  (`source: "synthetic"`。実 LIBERO のロールアウトではありません)

- CLI `esn-vla-uq`: `diagnose` / `gen-sample-data`

- `data/features.py`: `RolloutDataset` から ESN 入力への変換。エピソード境界で
  リザバー状態をリセットする方針を確定 (`esn.reservoir.run_episodes`)

- `gen-sample-data --force`: 既存の出力を明示的に上書きする

### Changed

- 診断の実行を `diagnostics/runner.py` へ分離し、`diagnostics/report.py` は
  レポートの表現 (型・JSON・ログ) のみを担うようにした
- 診断レポート JSON の各セクションを、対応する結果型の `to_dict()` から生成する
  ようにした。結果型にフィールドを足せば JSON にも自動的に現れる
- 供給元を `data/sources/` に分割し、抽象 (`base.py`) と具象を分離した
- 出所ごとの不変条件を `data/invariants.py` に集約し、`data/io.py` が具象の
  データ形式を知らないようにした

### Fixed

- `.npz` 読み込み時の過大なメモリ確保 (CWE-409/789)。`.npy` ヘッダが自己申告する
  shape を含めて検証し、負の次元を拒否する
- `save_dataset` がサイドカー `.json` を無警告で上書きしていた問題 (CWE-73)。
  既定で `FileExistsError` を送出する
- CLI の未捕捉例外がトレースバックを stderr に出していた問題 (CWE-209)。
  トレースバックは `--log-level DEBUG` のときのみ出力する
- 書き出し先のパスがホームディレクトリ (= ユーザー名) を含んだまま INFO ログに
  出ていた問題。カレントディレクトリ配下なら相対パス、ホーム配下なら `~/...`、
  それ以外は絶対パスのまま出す。完全な絶対パスは DEBUG へ

### Security

- `make ci` に `gitleaks` (シークレット検出) と `pip-audit` (依存の既知脆弱性)
  を追加した。`gitleaks` は既定の「ステージ済み差分のみ」ではなく作業ツリー
  全体を走査する (クリーンな checkout では前者が何も検査しないため)
- 既知脆弱性の解消: pygments 2.20.0 以上 (PYSEC-2026-2987)、
  pytest 9.0.3 以上 (PYSEC-2026-1845)。いずれも dev 依存であり、配布物
  (numpy のみ) には含まれない

### Removed

- `esn_vla_uq.data.source` モジュール (`esn_vla_uq.data.sources` へ移行)。
  `esn_vla_uq.data` からの再エクスポート名は変更していない

[unreleased]: https://github.com/Kotton-MAS/esn-vla-uq/commits/main
