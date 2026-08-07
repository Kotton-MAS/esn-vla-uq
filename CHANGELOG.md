# Changelog

このプロジェクトの重要な変更を記録します。

書式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に従い、
バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に従います。

v0.1.0 が最初のリリースです。

## [Unreleased]

### Added

- **リザバーの寄与を測るアブレーション。** `calibrate --readout` で read-out の設計
  行列を `[1, u, x]` (既定) / `[1, x]` / `[1, u]` から選べる。`[1, u]` はリザバー
  無しの baseline で、この条件では**リザバーを構築も駆動もしない**
  (`uncertainty/conformal.py` が列数 0 の状態を渡す)

- `ESNConfig.use_reservoir` と `RidgeReadout(use_states=)`。入力もリザバーも外す組は
  定数予測に退化するため `ValueError` にする

- 較正レポートに `coverage.std_interval_width` と `coverage.per_split_interval_width`。
  **条件を変えた 2 本は同じ乱数種の同じ分割を見ているので、分割ごとの幅を引き算した
  対応のある差で比較できる。** 平均だけでは差が誤差の内か判定できなかった

- `docs/design.md` 11 節と `docs/next-research-directions.md`。アブレーションの結果と、
  次に測ることの一覧

- `calibrate --washout` と `run_calibration(washout=)`。較正経路で実際に効く washout を
  外から動かせるようにした。**`ESNConfig.washout` とは別物**なので、実際に使った値を
  レポートの `conformal.washout` に残す

### Fixed

- **`washout > 0` で失敗検知のラベルが区間と揃っていなかった。** 区間は washout 後の
  行数で返るのにラベルは washout 前の行数で作られており、`IndexError` になる。長さが
  たまたま一致すれば黙って別のステップと突き合わせていた。マスクを
  `SplitConformalPredictor.retained_mask` として公開し、`calibration/runner.py` が
  ラベルに同じマスクを掛ける。`run_calibration` から washout を渡す経路が無かったため
  露呈していなかった

### Changed

- 診断レポートのスキーマを 0.3.0 に、較正レポートのスキーマを 0.2.0 に上げた
  (`esn_config` への `use_reservoir` 追加と、`coverage` への幅の散らばり追加)

### 分かったこと

- **read-out の仕事の大半はパススルーが担っている。** `[1, x]` はどのデータでも最も
  区間が広く、20 分割すべてで現行に負ける
- **リザバーの寄与は小さく、タスクスイートで向きが変わる。** libero_10 では幅が
  7〜11% 狭まるが、libero_spatial では 17%、合成データでは 77% 広がる。しかも
  寄与は `N` が増えるほど 0 へ縮む (`N=50` で最大、`N=500` でほぼ消える)
- **既定の `ridge_alpha=1e-6` はリザバー有りの条件に対して弱すぎる。** 条件ごとに
  最良化しないと比較の結論が逆に出る。既定値の見直しは別タスク
- **失敗検知 AUROC は設計行列に依存しない** (3 条件で完全一致)。条件が変えるのは
  区間幅の定数倍だけで、AUROC は順位だけで決まるため
- **初期過渡は「予測しづらい区間」ではなく最も予測しやすい区間だった。** エピソード
  先頭 20 標本の非適合度はそれ以降の約半分 (比 0.518〜0.588)。washout を増やすと幅は
  単調に広がる。ESN の定石 (過渡は捨てる) はこの予測タスクには当てはまらない

## [0.2.0] - 2026-08-04

実 openpi ロールアウトとの接続と、それによって判明した事実の反映が中心。**合成
データでの結論が実データで否定された箇所があり、主張を狭めている。**

### Added

- **openpi 接続。** `OpenpiLogSource` (`data/sources/openpi.py`) が収集済みの LIBERO
  ロールアウトログを `RolloutDataset` へ変換する。openpi も policy server も import
  しない
- **ロールアウト収集スクリプト** (`scripts/collect_openpi_rollouts.py`)。openpi の
  評価ループをなぞって state / action / action_chunk を記録する。openpi の評価
  スクリプトはロールアウトを保存しない (replay 動画だけを書く) ため必要。**このファイル
  だけが openpi と LIBERO を必要とし、配布物には含めない**
- `calibrate` / `demo` の `--input` が openpi ログのディレクトリを受け取る。`.npz`
  ファイルかディレクトリかをパスの形で判別する
- `demo` に `--split`。1 タスク 1 エピソードのデータでは `within_task` が 3 分割
  できないため
- テストの穴を 2 件補完 (`input_scaling=0.0` が入力を無視すること、`ESN.fit` 経由の
  `washout=0` が全ステップを使うこと)

### Changed

- **失敗検知を v0.1 のスコープから外した。** 不確実性スコアは出力し続けるが
  「失敗を検知できる」とは主張しない。実 openpi ロールアウトでは AUROC 0.457〜0.477
  と偶然と同水準で、代替の観測量もタスク内で見ると判定不能だった (失敗 23 本を
  8 タスクに分けるとタスクあたり 1〜3 本しかなく、タスクごとの AUROC が 0.000 と
  1.000 の間で振れる)。要件書の「評価軸: 成功率ではなく較正」とも整合する。
  レポートには失敗検知が探索的な診断値である旨を常に付ける
- CLI ハンドラの引数を型付きの設定オブジェクトへ変換 (A7)。`argparse.Namespace` の
  無型アクセスを各サブコマンドの `from_namespace` 1 箇所に閉じ込め、ハンドラ本体を
  mypy strict の検査対象にした
- `esn` 層の公開 API (`ESN.fit/predict/transform`、`RidgeReadout`) に
  Args/Returns/Raises を追加 (D1)
- Dev Container の `uv sync` を `--locked` に (U4)
- README (英日) を実装の現状に同期。収集の手順を追加し、サブコマンド 4 つを明記した

### Fixed

- **実 openpi ログで区間幅が行動スケールの 1,858 倍になっていた。** 難易度
  `sigma(x)` に観測量をそのまま使っていたため、観測量のレンジ (実データで約 17,000
  倍) が幅に直結していた。fit 集合における**順位**へ写して値域を構造的に閉じる。
  順位への写像は単調変換なので**検知の順序は変わらない** (spread 2/4/8/16 で AUROC
  が完全一致することを実測)。実 openpi の平均半幅は 139 から 2.21 へ、合成データも
  0.0525 から 0.0486 へ改善
- **openpi のログに対して「出所は合成データ」と主張するレポートが出ていた。**
  `data_source` をハードコードしていたため。データセット自身の出所を使い、合成
  データ用の注意書きも出所が合成のときだけ付ける
- **`failure_onset` を持たない出所で較正評価そのものが失敗していた。** 陽性が 0 件で
  AUROC が定義できず例外になっていた。失敗開始時刻が無い場合はエピソード単位の成否
  へ落とし、**どちらのラベルを使ったかを記録する**
- **較正標本が少ないと較正評価が丸ごと失敗していた。** 高い名目水準を有限標本で
  保証できないと例外にしていたため。評価できた水準だけで曲線を引き、**落とした水準を
  `unsupported_levels` に記録する** (黙って落とすと右端が消えて ECE が実勢より
  小さく出る)
- **収集ログの `policy` がコマンドラインの既定値をそのまま記録していた。** 実際に
  配信されていたのは `pi05_libero` だが `pi0_libero` と書かれていた。policy server の
  メタデータから取り、得られなければ推測せず `"unknown"` を記録する
- **`state` 8 次元の意味の記述が誤っていた。** 「7 関節 + グリッパ」ではなく
  「エンドエフェクタ位置 3 + 姿勢 (軸角) 3 + グリッパ 2」。次元数は合っていたため
  見落としやすかった
- `chunk_horizon` の扱いを明確にした。`pi0_libero` は 50、`pi05_libero` は 10、同梱の
  合成データは 16。`RolloutDataset` がフィールドで持つため同じスキーマで共存する
  (`action_horizon` の記述は本リリース中に二度訂正した。経緯は `docs/design.md` 10 節)

### Performance

- `diagnose` が実効更新行列の固有値を 2 度求めていたのを 1 度に (P2)。N=500 で
  0.615 秒から 0.493 秒 (20% 短縮)。報告される値は変わらない

### Documented

実データで確かめたことを `docs/design.md` 10.5〜10.14 に記録した。

- **被覆率保証は実データで成立する。** 4 回の収集すべてで 0.898〜0.904
  (`within_task`)。エピソード数を 10 倍にすると分散が約 1/5 になり、「有効標本数は
  ステップ数ではなくエピソード数」という 9.3 節の予測が裏付けられた
- **タスク間 split は保証が弱い。** `libero_10` では `across_task` の被覆率が 0.779
  まで落ち、ECE は `within_task` の 17 倍。6.3 節が理屈として書いていたことに数値が
  付いた
- **失敗検知の仮説は実データで否定された。** 合成データは「失敗でチャンク分散が
  上がる」と作られているが実データは無相関。途中で「逆相関」と結論したが、それも
  失敗 6 本・4 本が同一タスクという偏りによる見かけの信号で、失敗を 23 本に増やすと
  消えた。**符号を反転させなかった判断が結果的に正しかった**
- **openpi からの収集は再現しない。** pi0 は flow matching で行動をサンプリング
  するため、同じ `--seed` でも軌道が変わる。収集済みログを入力にした解析は再現する

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
[0.2.0]: https://github.com/Kotton-MAS/esn-vla-uq/compare/v0.1.0...v0.2.0
[unreleased]: https://github.com/Kotton-MAS/esn-vla-uq/compare/v0.2.0...main
