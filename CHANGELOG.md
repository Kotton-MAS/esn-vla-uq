# Changelog

このプロジェクトの重要な変更を記録します。

書式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に従い、
バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に従います。

v0.1.0 が最初のリリースです。

## [Unreleased]

### Added

- **openpi 接続** (`data/sources/openpi.py` の `OpenpiLogSource`): 収集済みの
  LIBERO ロールアウトログを読んで `RolloutDataset` へ変換する。openpi も
  policy server も import しない
- `calibrate` / `demo` の `--input` が openpi ログのディレクトリを受け取れる
  ようになった。`.npz` ファイルか収集ログのディレクトリかをパスの形で判別する
- **ロールアウト収集スクリプト** (`scripts/collect_openpi_rollouts.py`): openpi の
  評価ループをなぞって state / action / action_chunk を記録する。openpi の評価
  スクリプトはロールアウトを保存しない (replay 動画だけを書く) ため必要

### Fixed

- **実 openpi ログで区間幅が行動スケールの 1,858 倍になっていた。** 難易度
  `sigma(x)` に観測量をそのまま使っていたため、観測量のレンジ (実データでは約
  17,000 倍) がそのまま幅に出ていた。fit 集合における**順位**へ写して値域を
  構造的に閉じる。順位への写像は単調変換なので**失敗検知 AUROC は変わらない**
  (spread 2/4/8/16 で完全一致することを実測)。実 openpi の平均半幅は 139 から
  2.21 へ、合成データも 0.0525 から 0.0486 へ改善した

- **`action_horizon` を 50 と記載していたのは誤り。** `pi0_libero` は
  `Pi0Config(action_horizon=10)` でクラス既定値を上書きしている。実収集した
  ログで 10 であることを確認して訂正した

- **openpi のログに対して「出所は合成データ」と主張するレポートが出ていた。**
  `data_source` をハードコードしていたため。データセット自身の出所を使うようにし、
  合成データ用の注意書きも出所が合成のときだけ付ける

- **`failure_onset` を持たない出所で較正評価そのものが失敗していた。** 陽性が
  0 件になり AUROC が定義できず例外になっていた。失敗開始時刻が無い場合は
  エピソード単位の成否 (`episode_success`) へ落とし、**どちらのラベルを使ったかを
  レポートに記録する** (粒度が違う数値を並べて比較できてしまうため)

- **較正標本が少ないと較正評価が丸ごと失敗していた。** 高い名目水準 (99% 等) を
  有限標本で保証できないと例外にしていたため。評価できた水準だけで曲線を引き、
  **落とした水準を `unsupported_levels` に記録する** (黙って落とすと落ちるのは
  決まって右端なので ECE が実勢より小さく出る)

- **`state` 8 次元の意味の記述が誤っていた。** 「7 関節 + グリッパ」ではなく
  「エンドエフェクタ位置 3 + 姿勢 (軸角) 3 + グリッパ 2」。openpi の実装を読んで
  訂正した (次元数は合っていたため見落としやすい)

- `chunk_horizon` の扱いを明確にした。openpi の pi0 は 50 (推論間隔 5)、同梱の
  合成データは 16。`RolloutDataset` がフィールドで持つため同じスキーマで共存する

### Changed

- CLI ハンドラの引数を型付きの設定オブジェクトへ変換するようにした (A7)。
  `argparse.Namespace` の無型アクセスを各サブコマンドの `from_namespace` 1 箇所に
  閉じ込め、ハンドラ本体を mypy strict の検査対象にした
- `esn` 層の公開 API (`ESN.fit/predict/transform`、`RidgeReadout`) に
  Args/Returns/Raises を書いた。利用者が最初に触る層だが diagnostics 層より
  記述が薄かった (D1)
- Dev Container の `uv sync` を `--locked` にした。初回起動で uv.lock が
  書き換わると `make ci` 先頭の `uv lock --check` が落ちる (U4)

### Performance

- `diagnose` が実効更新行列の固有値を 2 度求めていたのを 1 度にした (P2)。
  N=500 で 0.615 秒から 0.493 秒 (20% 短縮)。報告される値は変わらない

### Added

- テストの穴を 2 件埋めた: `input_scaling=0.0` が入力を無視すること (T1)、
  `ESN.fit` 経由の `washout=0` が全ステップを使うこと (T2)

## [0.1.0] - 2026-08-04

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

- **デモアニメーション** (`esn-vla-uq demo`): 失敗エピソードの不確実性推移を GIF に
  する。フレームデータ (`demo/frames.py`) と描画 (`demo/animate.py`) を分離してあり、
  実 LIBERO 映像が入手できた時点で前者だけを差し替えられる

- README を英語主体に刷新し、`README.ja.md` を併設

### Changed

- 診断の実行を `diagnostics/runner.py` へ分離し、`diagnostics/report.py` は
  レポートの表現 (型・JSON・ログ) のみを担うようにした
- 診断レポート JSON の各セクションを、対応する結果型の `to_dict()` から生成する
  ようにした。結果型にフィールドを足せば JSON にも自動的に現れる
- 供給元を `data/sources/` に分割し、抽象 (`base.py`) と具象を分離した
- 出所ごとの不変条件を `data/invariants.py` に集約し、`data/io.py` が具象の
  データ形式を知らないようにした
- CLI のサブコマンド定義を `cli/app.py` のテーブル 1 箇所へ集約した。以前は
  サブコマンドを 1 つ足すのに 4 箇所を同時編集する必要があった (A6)

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

### 既知の制限

- **不確実性は失敗への反応であって予兆ではない。** 立ち上がりは失敗開始の約 15
  ステップ後で、遅れはチャンク周期 (推論間隔 16 ステップ) で上限が決まる。
- **openpi 接続 (`OpenpiLogSource`) は未実装。** 実ログが入手できず、フィールドの
  マッピングを推測で書くことを避けた。アダプタ境界は用意済み。
- デモの映像パネルは実 LIBERO 映像ではなく合成データによる代替。
- `normalized` は失敗検知と引き換えに被覆率が名目をやや下回る (0.864 対 0.903)。

[0.1.0]: https://github.com/Kotton-MAS/esn-vla-uq/releases/tag/v0.1.0
[unreleased]: https://github.com/Kotton-MAS/esn-vla-uq/compare/v0.1.0...main
