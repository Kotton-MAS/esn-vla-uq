# esn-vla-uq

Echo State Network とリッジ read-out、split conformal prediction による、
VLA (vision-language-action) ポリシーの**閉形式・アンサンブル不要**な不確実性定量化。

[English README](README.md)

> **本リポジトリが出す数値はすべて同梱の合成ロールアウトデータ由来です**
> (`source: "synthetic"`)。実 LIBERO の評価結果ではありません。合成データ生成器は
> 「単純なベースラインでは成否を完全には分離できない」よう意図的に調整してあります
> ([`docs/design.md`](docs/design.md) 7 節)。

## クイックスタート

```bash
uv sync
uv run esn-vla-uq calibrate     # 被覆率 / ECE / 失敗検知 AUROC
uv run esn-vla-uq calibrate --input <openpi ログのディレクトリ>   # 収集した実ログ
uv run esn-vla-uq diagnose      # スペクトル半径 / ESP / メモリ容量
```

## デモ

![失敗開始後に不確実性が立ち上がる](docs/assets/uncertainty_demo.gif)

```bash
uv sync --extra viz
uv run esn-vla-uq demo --output outputs/demo.gif
```

下段が conformal 予測区間の半幅、すなわちステップ単位の不確実性スコアです。
図のエピソードでは失敗開始の前後で **1.13 倍**になります。

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

| スコア       | 区間幅             | 被覆率        | 平均半幅   | 失敗検知 AUROC    |
| ------------ | ------------------ | ------------- | ---------- | ----------------- |
| `absolute`   | 全ステップで一定   | 0.903 ± 0.027 | 0.0525     | **0.500 ± 0.000** |
| `normalized` | ステップごとに変化 | 0.903 ± 0.026 | **0.0486** | 0.871             |

`absolute` の 0.5 は**定義上そうなります**。区間幅が定数なら不確実性スコアも定数で、
全ステップが同順位になるためです。`normalized` は同等の被覆率をより狭い平均幅で
達成し、かつステップを区別できるので既定にしています。

## 実 openpi ロールアウトでの結果

`libero_spatial` を 1 タスク 10 試行 x 10 タスク = **100 エピソード**収集。
`scripts/collect_openpi_rollouts.py` で実際の `pi0_libero` policy server から取得した
ものです。

| split         | 被覆率              | ECE    | 平均半幅 |
| ------------- | ------------------- | ------ | -------- |
| `within_task` | **0.9033 ± 0.0102** | 0.0029 | 0.250    |
| `across_task` | 0.8977 ± 0.0397     | 0.0020 | 0.297    |

**被覆率は実データでも保たれています。** 10 エピソードでは 0.881 ± 0.049 でしたが、
100 エピソードで分散が約 1/5 に縮みました。「有効標本数はエピソード数」という議論の
予測どおりです。`within_task` の分散が `across_task` の 1/4 なのも交換可能性の議論と
整合します。

**失敗検知は依然として判定できていません。** 100 エピソード中の失敗が 1 本だけ
(`pi0_libero` は libero_spatial で約 99% 成功) のため、AUROC (約 0.46) は 1 本の
エピソードに乗った数値で、どちらの結論も支持しません。判定するには同じスイートの
試行を増やすのではなく、より難しいスイートか弱いポリシーが要ります。

**収集は再現しません。** pi0 は行動をサンプリングする (flow matching) ため、同じ
`--seed` でも軌道が変わります (2 回の収集で失敗したエピソードが違いました)。
`--seed` が固定するのは LIBERO の初期状態だけです。収集済みログを入力にした解析は
再現します。

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

- **openpi の実ロールアウトはまだ収集していません。** `OpenpiLogSource` と収集
  スクリプトは openpi の実装を読んで書いてありますが、policy server を実際に動かして
  ログを取る作業は未了です (GPU と LIBERO のセットアップが要ります)。テストは openpi
  実出力と同じ形状 (`chunk_horizon=50`、推論間隔 5) のフィクスチャで行っています。
- デモへの実 LIBERO 映像の取り込み (現状の映像パネルは合成データによる代替)。
- VLM 特徴量の注入 (要件書で v0.2 以降に延期)。

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
├── linalg.py       # スペクトル量の共有実装 (最下層)
├── provenance.py   # DataSource (最下層)
├── esn/            # リザバー・リッジ read-out・モデル
├── diagnostics/    # スペクトル半径 / ESP / メモリ容量
├── data/           # スキーマ・不変条件・sources/・合成生成・入出力・特徴量
├── uncertainty/    # 予測タスク・分割・非適合度・split conformal
├── calibration/    # 被覆率 / ECE / reliability diagram
├── demo/           # デモアニメーション (フレームデータと描画を分離)
└── cli/            # argparse エントリポイント
```

## ドキュメント

- [`docs/design.md`](docs/design.md) — 設計書。判断を変えた実測値をすべて記録しています
- [`docs/plans/`](docs/plans/) — スプリントごとの承認済み仕様書
- [`docs/next-pr-candidates.md`](docs/next-pr-candidates.md) — 既知の積み残し
- [`CHANGELOG.md`](CHANGELOG.md)

## ライセンス

Apache-2.0。[LICENSE](LICENSE) を参照してください。引用する場合は
[CITATION.cff](CITATION.cff) を参照してください。
