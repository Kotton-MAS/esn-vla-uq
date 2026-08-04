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
uv run esn-vla-uq diagnose      # スペクトル半径 / ESP / メモリ容量
```

## デモ

![失敗開始後に不確実性が立ち上がる](docs/assets/uncertainty_demo.gif)

```bash
uv sync --extra viz
uv run esn-vla-uq demo --output outputs/demo.gif
```

下段が conformal 予測区間の半幅、すなわちステップ単位の不確実性スコアです。
図のエピソードでは失敗開始の前後で **3.6 倍**になります。

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

| スコア       | 区間幅             | 被覆率        | ECE    | 失敗検知 AUROC    |
| ------------ | ------------------ | ------------- | ------ | ----------------- |
| `absolute`   | 全ステップで一定   | 0.903 ± 0.027 | 0.0022 | **0.500 ± 0.000** |
| `normalized` | ステップごとに変化 | 0.864 ± 0.068 | 0.0416 | 0.869 ± 0.075     |

`absolute` の 0.5 は**定義上そうなります**。区間幅が定数なら不確実性スコアも定数で、
全ステップが同順位になるためです。較正の正確さでは `absolute` が優れ、ステップを
区別できるのは `normalized` です。用途で選んでください (既定は `normalized`)。

**被覆率はステップ数から想像されるより不安定です。** 同一エピソード内のステップは強く
相関するため、有効標本数はステップ数 (約 1,500) ではなく**エピソード数** (較正 8 本)
です。単一分割の被覆率は 0.63〜1.00 まで振れ、30 分割の平均が 0.896 になります。
そのため `calibrate` は既定で 20 分割を評価し、1 つの数字ではなく平均と散らばりを
報告します。

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

- **openpi 接続**。`OpenpiLogSource` は書いていません。実ログが入手できず、フィールドの
  マッピングを推測で書くと「誰も正しさを確認していないコード」になるためです。アダプタ
  境界 (`data/sources/`、`data/invariants.py`) は用意済みで、既存コードを触らずに追加
  できます。
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
