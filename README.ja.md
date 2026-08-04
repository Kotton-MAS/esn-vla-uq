# esn-vla-uq

Echo State Network とリッジ read-out、split conformal prediction による、
VLA (vision-language-action) ポリシーの**閉形式・アンサンブル不要な較正済み予測区間**。

被覆率保証は実 `openpi` ロールアウトで検証済みです。**失敗検知は本リリースの主張には
含みません** ([スコープ](#%E3%82%B9%E3%82%B3%E3%83%BC%E3%83%97) を参照)。

[English README](README.md)

> **本リポジトリが出す数値はすべて同梱の合成ロールアウトデータ由来です**
> (`source: "synthetic"`)。実 LIBERO の評価結果ではありません。合成データ生成器は
> 「単純なベースラインでは成否を完全には分離できない」よう意図的に調整してあります
> ([`docs/design.md`](docs/design.md) 7 節)。

## クイックスタート

```bash
uv sync
uv run esn-vla-uq calibrate     # 同梱の合成データで被覆率 / ECE
uv run esn-vla-uq diagnose      # スペクトル半径 / ESP / メモリ容量
```

サブコマンドは 4 つあります。

| コマンド          | 内容                                               |
| ----------------- | -------------------------------------------------- |
| `calibrate`       | conformal 予測区間・被覆率・reliability curve・ECE |
| `diagnose`        | リザバー診断 (スペクトル半径・ESP・メモリ容量)     |
| `gen-sample-data` | 同梱の合成ロールアウトを再生成する                 |
| `demo`            | 上のアニメーション (`[viz]` が要ります)            |

`calibrate` と `demo` は `--input` でデータセットを読みます。省略すると同梱の合成
データ、`.npz` を渡すと保存済みデータセット、**ディレクトリ**を渡すと収集した openpi
ロールアウトとして読みます。`diagnose` は `ESNConfig` からリザバーを構築するので
データセットを取らず、`gen-sample-data` は書き出す側です。

## 実 openpi ロールアウトを使う

openpi の評価スクリプトはロールアウトを保存しません (replay 動画を書くだけ) ので、
収集は別の手順になります。`scripts/collect_openpi_rollouts.py` が openpi の LIBERO
評価ループをなぞって state / action / action chunk を記録します。

```bash
# 1. openpi 側: policy server を起動 (openpi 側の環境)
cd path/to/openpi && uv run scripts/serve_policy.py --env LIBERO

# 2. LIBERO クライアント側 (openpi の examples/libero/.venv、Python 3.8)
python path/to/esn-vla-uq/scripts/collect_openpi_rollouts.py \
    --output-dir outputs/openpi_logs --task-suite-name libero_10

# 3. こちらへ戻る (Python 3.12)
uv run esn-vla-uq calibrate --input outputs/openpi_logs --split within_task
```

このスクリプト**だけ**が openpi と LIBERO を必要とし、wheel にも sdist にも含めて
いません。パッケージ本体は収集済みログを読むだけで、依存は numpy のみです。

## デモ

![失敗開始後に不確実性が立ち上がる](docs/assets/uncertainty_demo.gif)

```bash
uv sync --extra viz
uv run esn-vla-uq demo --output outputs/demo.gif
```

下段が conformal 予測区間の半幅、すなわちステップ単位の不確実性スコアです。
図のエピソードでは失敗開始の前後で 1.13 倍になります。

**これは合成データであり、この上昇は生成器の作りを反映したものです。** 実 openpi
ロールアウトでは同じ関係が成り立ちません ([スコープ](#%E3%82%B9%E3%82%B3%E3%83%BC%E3%83%97))。デモが示すのは
出力の見え方であって、スコアが失敗を予測することではありません。

跳ね幅が控えめなのは**設計上そうしている**ためです。幅は最大 2 倍の範囲に収まるよう
有界にしてあり、観測量のレンジに引きずられないようにしています (下記)。失敗検知が
使うのは不確実性スコアの**順序**であり、この有界化では順序は変わりません。

> **この不確実性は失敗への「反応」であって「予兆」ではありません。**
> 立ち上がるのは失敗開始の **15 ステップ後**で、直前ではありません。不確実性の材料
> であるチャンク分散は推論ステップ (16 ステップ間隔) でしか更新されないため、遅れは
> チャンク周期で上限が決まります。そもそも失敗条件が始まる前のチャンクには手がかりが
> ありません。`demo` コマンドは実行のたびに実測の `detection_lag_steps` を出力します。

## しくみ

ロールアウトから固有受容感覚・実行された行動・ポリシーの action chunk が得られます。
ESN がその履歴をリザバー状態へ写像し、リッジ read-out が次ステップの行動を予測し、
split conformal が残差を有限標本の被覆率保証つき予測区間に変えます。区間の半幅が
そのまま不確実性スコアになります。

リザバーは固定のランダム行列で、学習するのは線形 read-out だけ (閉形式)。アンサンブルも
勾配学習も使いません。これが将来の物理リザバーへの移植可能性を支えています。

## 同梱の合成データでの結果

名目被覆率 90%、20 通りの較正/テスト分割の平均:

| スコア       | 区間幅             | 被覆率        | 平均半幅   |
| ------------ | ------------------ | ------------- | ---------- |
| `absolute`   | 全ステップで一定   | 0.903 ± 0.027 | 0.0525     |
| `normalized` | ステップごとに変化 | 0.903 ± 0.026 | **0.0486** |

`normalized` は `absolute` と同等の被覆率をより狭い平均幅で達成するので既定です。

## 実 openpi ロールアウトでの結果

4 回の収集すべてを `scripts/collect_openpi_rollouts.py` で実際の policy server から
取得しました。

| スイート / ポリシー     | split         | 被覆率              | ECE    |
| ----------------------- | ------------- | ------------------- | ------ |
| `libero_spatial` / pi05 | `within_task` | **0.9033 ± 0.0102** | 0.0029 |
| `libero_10` / pi05      | `within_task` | 0.8984 ± 0.0114     | 0.0012 |
| `libero_10` / pi0       | `within_task` | 0.9035 ± 0.0232     | 0.0051 |
| `libero_10` / pi0       | `across_task` | 0.7789 ± 0.1062     | 0.0885 |

**被覆率は 4 回の収集すべてで保たれています。** 10 エピソードでは 0.881 ± 0.049
でしたが、100 エピソードで分散が約 1/5 に縮みました。「有効標本数はエピソード数」
という議論の予測どおりです。

**`across_task` は長期タスクで崩れます。** 較正集合とテスト集合が別のタスク分布から
来るため保証が転移しない、という交換可能性の議論どおりの挙動です。`libero_10` では
0.779 まで下がり ECE は 17 倍になります。既定を `within_task` にしている理由です。

**収集は再現しません。** pi0 は行動をサンプリングする (flow matching) ため、同じ
`--seed` でも軌道が変わります。`--seed` が固定するのは LIBERO の初期状態だけです。
収集済みログを入力にした解析は再現します。

## スコープ

**本リリースが確立したもの**: 有限標本の被覆率保証を持つ予測区間 (実 openpi
ロールアウトで検証済み)、およびリザバー診断。

**確立していないもの**: 不確実性スコアが失敗を検知すること。`calibrate` は失敗検知
AUROC を出力し続けますが、**探索的な診断値**です。実 openpi ロールアウトでは偶然と
同じ水準 (0.457〜0.477) で、合成データの高い値 (0.87) は生成器がその関係を組み込んで
作られているために出るものです。代替の観測量も評価しましたが、タスク内で見ると
どれも判定できませんでした。失敗 23 本を 8 タスクに分けるとタスクあたり 1〜3 本しか
なく、タスクごとの AUROC が 0.000 と 1.000 の間で振れます。

再開にはタスクあたり失敗 5〜10 本と、タイムアウト以外の失敗が起きる環境が要ります。
LIBERO には早期終了条件が無く、観測された失敗はすべて「時間内に完了できなかった」
ものでした。`docs/design.md` 10.14 節を参照してください。

## リザバー診断

診断は付け足しではなく第一級の出力です。予測区間を信用してよいかを判断する前に、
リザバーの挙動を検査可能にするためのものです。

```bash
uv run esn-vla-uq diagnose --output-dir outputs/
```

```text
spectral: spectral_radius=0.900000 effective_spectral_radius=0.900000
esp: verdict=esp_holds sufficient(sigma_max<1)=False[1.784905] necessary(rho<1)=True[0.900000] ...
memory_capacity: total_mc=14.8294 mc_per_neuron=0.0741 memory_horizon=20 n_delays=200 ...
```

Echo State Property は単一の数値ではなく、**3 指標と総合判定**で報告します
(十分条件 sigma_max < 1、必要条件 rho < 1、経験的収束)。既定設定は必要条件を満たし
十分条件を満たしませんが、これは正常であり、3 つすべてを併記する理由でもあります。

## 動作要件

- Python 3.12 以上と [uv](https://docs.astral.sh/uv/)
- ランタイム依存は **numpy のみ**
- `esn-vla-uq[viz]` で matplotlib が入り、reliability diagram とデモ GIF を出力できます。
  数値はすべて matplotlib 無しで計算されます。

## 未実装

- デモへの実 LIBERO 映像の取り込み (現状の映像パネルは合成データによる代替)。
- VLM 特徴量の注入 (要件書で v0.2 以降に延期)。
- 失敗検知 — 本リリースの対象外とした理由は [スコープ](#%E3%82%B9%E3%82%B3%E3%83%BC%E3%83%97) を参照。

## 開発

検証は `Makefile` に一元化してあり、ローカル・stop フック・GitHub Actions が同じ
ターゲットを呼びます。

```bash
make ci      # lock + gitignore + version + secrets + audit + lint + format + type + test
make test    # pytest のみ
make fmt     # ruff format (ファイルを書き換える)
```

## ディレクトリ構成

```
src/esn_vla_uq/
├── linalg.py        # スペクトル量の共有実装 (最下層)
├── provenance.py    # DataSource (最下層)
├── logging_paths.py # ログ用のパス表記 (最下層)
├── esn/             # リザバー・リッジ read-out・モデル
├── diagnostics/     # スペクトル半径 / ESP / メモリ容量
├── data/            # スキーマ・不変条件・sources/ (合成 + openpi)・入出力・特徴量
├── uncertainty/     # 予測タスク・分割・非適合度・split conformal
├── calibration/     # 被覆率 / ECE / reliability diagram
├── demo/            # デモアニメーション (フレームデータと描画を分離)
└── cli/             # argparse エントリポイント・型付きオプション・--input の解決

scripts/
└── collect_openpi_rollouts.py   # openpi と LIBERO を必要とする唯一のファイル
```

## ドキュメント

- [`docs/design.md`](docs/design.md) — 設計書。判断を変えた実測値をすべて記録しています
- [`docs/plans/`](docs/plans/) — スプリントごとの承認済み仕様書
- [`docs/next-pr-candidates.md`](docs/next-pr-candidates.md) — 既知の積み残し
- [`CHANGELOG.md`](CHANGELOG.md)

## ライセンス

Apache-2.0。[LICENSE](LICENSE) を参照してください。引用する場合は
[CITATION.cff](CITATION.cff) を参照してください。
