# Sprint 1 実装仕様書 — esn-vla-uq

対象要件: `docs/要件_Phase0リポジトリ化_v0.1.md`
ステータス: ユーザー承認済み（2026-08-02）

## 1. ゴール

`uv sync && uv run esn-vla-uq diagnose` で、同梱合成サンプルデータ上の自前 ESN の
リザバー診断（スペクトル半径・ESP・メモリ容量）が再現可能に出力される状態にする。

## 2. スコープ

含む: `docs/design.md` 再作成、ESN コア実装、リザバー診断モジュール、
サンプルデータ同梱、パッケージ scaffold 整備、LICENSE / CITATION.cff。

含まない（次スプリント）: openpi ロールアウト収集スクリプト、conformal prediction 実装、
較正評価コマンド（reliability diagram / ECE）、デモ GIF 生成、README 英/日の本格整備、
v0.1 タグ公開。ただし将来スプリントの決定事項は `docs/design.md` に記録する。

## 3. ユーザー確定事項

01. 実装スコープ = Sprint 1 相当のみ
02. conformal 較正データ分割 = タスク内 split を既定、タスク間 split は設定で選択可（Sprint 2 実装）
03. README 言語 = 英語主体 + `README.ja.md` 併設（Sprint 3）
04. デモ GIF = 「操作映像 + 不確実性バー + 失敗直前にバーが跳ねる」。実 LIBERO 映像が無いため
    v0.1 では同梱サンプルデータからの合成プロット動画で代替し、実映像に差し替え可能な構造にする（Sprint 3）
05. リポジトリ名 = `esn-vla-uq` 確定
06. `main.py` / `tests/test_main.py` は削除し CLI エントリに置換。CLAUDE.md の構成・コマンド一覧も実態に更新
07. LICENSE (Apache-2.0) と CITATION.cff は Sprint 1 に含める。公開名義は個人名義（所属表記なし）
08. build backend = hatchling。Sprint 1 のランタイム依存は numpy のみ
09. 同梱サンプルデータ置き場 = `src/esn_vla_uq/assets/samples/`
10. リッジ read-out の入力パススルーは有効化を既定（design.md に記録）

## 4. 現状認識（実測）

| パス                               | 該当箇所                                              | Sprint 1 での扱い                                     |
| ---------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| `pyproject.toml`                   | L1-2 `name = "template"` / L7 `dependencies = []`     | 変更（名前・依存・build-system 追加）                 |
| 同上                               | L18-20 `[tool.pytest.ini_options] pythonpath = ["."]` | 削除（src レイアウト化）                              |
| 同上                               | L22-38 ruff（`T20` = `print()` 禁止）                 | 維持（ハード制約）                                    |
| 同上                               | L40-44 mypy `strict` + `disallow_any_explicit`        | 維持（ハード制約）                                    |
| `main.py`                          | L1-12 Hello world                                     | 削除し CLI へ置換                                     |
| `tests/test_main.py`               | L5 `from main import main`                            | `tests/test_cli.py` へ差し替え                        |
| `Makefile`                         | L42 `ci: lock-check lint fmt-check type test`         | 変更しない（検証の単一の真実）                        |
| `.gitignore`                       | L169-170 `outputs/` `data/` を無視                    | サンプルデータは `src/.../assets/` に置くため変更不要 |
| `.pre-commit-config.yaml`          | L16 `check-added-large-files`（既定 500kB）           | サンプルデータのサイズ上限                            |
| `.github/workflows/python-ci.yaml` | `pull_request` のみ                                   | Sprint 1 では触らない                                 |
| `.devcontainer/Dockerfile`         | `python:3.12-slim-bookworm`                           | 維持                                                  |

### 実装前に必ず認識する落とし穴

1. `pyproject.toml` に `[build-system]` が無い。現状 uv は「非パッケージプロジェクト」として扱っており、
   `[project.scripts]` による CLI 登録も `src/` レイアウトも成立しない。T1 で追加必須。
2. `.gitignore` が `data/` を無視している。サンプルデータは `src/esn_vla_uq/assets/samples/` に置き、
   `importlib.resources` で参照する。
3. `pythonpath = ["."]` と src レイアウトは共存すると重複モジュール解決で事故る。T1 で削除。
4. mypy `strict` の `warn_return_any` と numpy の相性。numpy の一部戻り値は `Any` 扱いになるため、
   `float(np.max(...))` のような明示変換をコーディング規約として最初に決める。

### 既存の慣習

- ロギングは `logger = logging.getLogger(__name__)`。`print()` は ruff T20 で禁止。
- テストは `tests/test_{モジュール名}.py`、pytest fixture にも型注釈を付ける。
- 検証は `make ci` に一元化（ローカル・Stop フック・GitHub Actions が同じコマンドを呼ぶ）。

## 5. 制約

### ハード制約（変えない）

- Python 3.12+。`.python-version` / `requires-python` / Dockerfile の 3.12 一致を崩さない。
- 全関数・メソッドに型アノテーション。`Any` の明示使用禁止（`object` / `Protocol` / `TypeVar` を使う）。
  mypy `strict = true` + `disallow_any_explicit = true` を弱めない。
- `print()` 禁止。結果本体はファイル出力、サマリは `logging` 経由。
- 新規コードには必ずテスト。`make ci`（lock-check → lint → fmt-check → type → test）が緑。
- `# type: ignore` / `# noqa` / `# nosec` を理由コメント無しで使わない。
- 依存追加は `uv add` 経由（`uv.lock` を手編集しない）。本番ロジックの依存を `dev` グループに入れない。
- `Makefile` / `.github/workflows/python-ci.yaml` / `.pre-commit-config.yaml` / `.devcontainer/` は本スプリントで変更しない。
- openpi をランタイム依存に入れない（疎結合設計）。
- 合成データから得た数値を実 LIBERO の結果として提示しない。全出力・全ドキュメントに `source: "synthetic"` を明記。

### ソフト制約（理由があれば変えてよい。ただし design.md に記録）

- Sprint 1 のランタイム依存は numpy のみ。matplotlib / PyTorch / ReservoirPy は入れない。
- CLI は stdlib `argparse`。
- 活性化関数は `tanh` 固定（差し替え口だけ残す）。
- 疎行列は密行列 + マスクで表現（scipy を入れない）。スペクトル半径は `np.linalg.eigvals` の密計算。
  N の実用上限を design.md に明記。

## 6. タスク分解

依存関係: **T1 → T2 →（T3 ∥ T5）→ T4**

### T1: パッケージ scaffold 整備

1. `pyproject.toml`:
   - `name = "esn-vla-uq"`、`version = "0.1.0.dev0"`、`description` を実態に更新
   - `[build-system]` に hatchling を追加（`packages = ["src/esn_vla_uq"]` を明示）
   - `uv add numpy` でランタイム依存を追加
   - `[project.scripts] esn-vla-uq = "esn_vla_uq.cli:main"`
   - `[tool.pytest.ini_options]` から `pythonpath = ["."]` を削除、`addopts = ["--strict-markers", "--strict-config"]` を追加
   - mypy が src レイアウトを解決できる設定を追加（`strict` / `disallow_any_explicit` は維持）
2. ディレクトリ作成（`__init__.py` と docstring のみ、ロジックは各タスク）:
   `src/esn_vla_uq/{__init__.py, py.typed, esn/, diagnostics/, data/, assets/samples/, cli/}`
3. CLI は argparse のサブコマンド骨格（`diagnose` / `gen-sample-data`）。
   共通オプション `--seed` `--output-dir`（既定 `outputs/`）`--log-level`。
   `logging.basicConfig` は CLI エントリでのみ呼ぶ。
4. `main.py` / `tests/test_main.py` を削除、`tests/test_cli.py` を新設
5. LICENSE（Apache-2.0 全文）と CITATION.cff を追加（個人名義、所属表記なし）
6. CLAUDE.md のプロジェクト構成・コマンド一覧を実態に更新
7. `uv lock` を更新し `make ci` を通す

受け入れ基準:

- `uv run esn-vla-uq --version` が `0.1.0.dev0` を出力し exit 0
- `uv run esn-vla-uq --help` が `diagnose` と `gen-sample-data` を列挙し exit 0
- 引数なし実行で usage を出して exit code 2
- `make ci` の 5 工程すべて緑。`uv lock --check` 成功
- `grep -rn "template" pyproject.toml README.md` が 0 件
- 新規 `# type: ignore` が 0 件
- LICENSE が Apache-2.0 全文、CITATION.cff が `cffconvert` 相当のスキーマに適合

### T2: `docs/design.md` 再作成

以下 8 節を必ず含める:

1. 目的・非目標（Phase 1 以降・四足歩行への横展開・他 VLA・リアルタイム介入は非目標）
2. アーキテクチャ: openpi 疎結合、レイヤ構成（data → esn → diagnostics → uncertainty → calibration）とモジュール境界
3. ESN の数学仕様: 状態更新式、リザバー生成手順、リッジ read-out の閉形式、記法と既定ハイパーパラメータ表
   （T3 の実装が参照する唯一の真実）
4. 診断指標の定義: スペクトル半径 / ESP / メモリ容量の定義式・算出手順・判定閾値と依拠文献
   （Jaeger 2001 のメモリ容量、Yildiz et al. 2012 の ESP 再検討、Lukoševičius 2012 の実務ガイド）。
   複数定義が存在する指標はどれを採るかを明記
5. データスキーマ v0.1: T5 のフィールド仕様と openpi ログへの差し替え手順（アダプタ Protocol の契約）
6. 将来スプリントの決定事項: 上記「ユーザー確定事項」2〜4、および Sprint 1 で
   matplotlib / PyTorch / ReservoirPy を入れない判断とその理由。
   conformal のタスク内/タスク間 split については交換可能性の仮定が崩れる点を 1 段落で記述
7. 合成データの位置づけに関する誠実性宣言（v0.1 の全数値は合成データ由来であり実 LIBERO 評価ではない）
8. 未解決の設計論点（Sprint 2 持ち越し）

受け入れ基準:

- 上記 8 節がすべて存在する
- 要件書「未確定事項」の 4 項目すべてに確定回答と担当スプリントが書かれている
- T3/T4 の実装者が design.md だけを読んで既定ハイパーパラメータと診断指標の算出手順を一意に決められる
- `uv run pre-commit run --all-files` が緑
- ADR 級の設計判断（build backend 選定、依存を numpy のみに絞る判断）は `docs/adr/` に切り出すか
  design.md 内に理由付きで記載

### T3: ESN コア実装

- `esn/config.py`: `@dataclass(frozen=True) ESNConfig`
  フィールド: `n_reservoir`, `spectral_radius`, `input_scaling`, `bias_scaling`, `leak_rate`,
  `density`, `ridge_alpha`, `washout`, `input_passthrough: bool`, `seed: int`。
  `__post_init__` で範囲検証（`0 < leak_rate <= 1`, `0 < density <= 1`, `n_reservoir >= 1`,
  `spectral_radius > 0`, `ridge_alpha >= 0`）、違反時は `ValueError` にパラメータ名と実値を含める。
- `esn/reservoir.py`:
  - `np.random.default_rng(seed)` で `W_in`（Uniform(-1,1) × input_scaling）、
    `b`（Uniform(-1,1) × bias_scaling）、`W`（density マスク × Uniform(-1,1)）を生成
  - `W` を `spectral_radius / max|eigvals(W)|` でスケール。`max|eigvals(W)| == 0` は `ValueError`
  - 状態更新: `x[t] = (1 - a) * x[t-1] + a * tanh(W_in @ u[t] + W @ x[t-1] + b)`、初期状態は零ベクトル
  - `run(inputs, initial_state) -> NDArray[np.float64]` で状態行列 `[T, N]` を返す。
    washout の破棄は呼び出し側の責務とし、ヘルパを別途提供
- `esn/readout.py`: `RidgeReadout`
  - 閉形式 `W_out = (X^T X + λI)^{-1} X^T Y` を `np.linalg.solve` で解く（`inv` を使わない）
  - バイアス項は正則化対象外
  - `input_passthrough=True`（既定）のとき設計行列を `[1, u, x]` にする
- `esn/model.py`: `ESN`（`fit(u, y)` / `predict(u)` / `transform(u)`）。教師強制なし。
  未 fit で `predict` を呼んだら `RuntimeError`
- 全公開関数の入出力は `numpy.typing.NDArray[np.float64]` で注釈。`Any` 禁止。
  数値スカラーは `float(...)` で明示変換

受け入れ基準:

- 生成した `W` の実測スペクトル半径が設定値と相対誤差 `< 1e-8` で一致（N=100, density=0.1, ρ=0.9）
- 同一 seed で 2 回構築した `W_in`/`W`/`b`/状態系列/予測が `np.array_equal` で完全一致、異なる seed では不一致
- `leak_rate=1.0` のとき状態更新が非リーク型と `np.allclose(atol=0)` で一致
- `ridge_alpha` を 0→大 に増やすと `‖W_out‖_F` が単調非増加（3 点以上）
- 線形課題で `ridge_alpha=1e-10` の閉形式解が `np.linalg.lstsq` の解と `np.allclose(rtol=1e-6)` で一致
- 遅延再現課題（`y[t] = u[t-5]`, N=100）で test NRMSE < 0.15
- 不正パラメータ 5 種以上が `ValueError`、未 fit `predict` が `RuntimeError`
- `make ci` 緑。ESN 関連テストの実行時間合計 < 10 秒

### T4: リザバー診断モジュール

- `diagnostics/spectral.py`
  - `spectral_radius(W) -> float`
  - `effective_spectral_radius(W, leak_rate) -> float`: 実効行列 `(1-a)I + aW` の ρ
  - `largest_singular_value((1-a)I + aW) -> float`（ESP 十分条件用）
- `diagnostics/esp.py`: `check_esp(...) -> EspResult`。3 指標を**同時に**返し片方だけで判定しない:
  1. `sufficient_condition_met`: σ_max((1-a)I + aW) < 1
  2. `necessary_condition_met`: ρ((1-a)I + aW) < 1
  3. `empirical_converged`: 同一入力列を K(=8) 個の異なるランダム初期状態から駆動し、
     最終状態間の最大 L2 距離 `d(T)` が `tol`(=1e-6) 未満か。`log d(t)` の線形回帰傾きを `decay_rate` として返す
  - ESP は入力に依存するため、テスト入力列を引数で受け取り既定分布を design.md に記載
  - `verdict: Literal["esp_holds", "esp_likely", "esp_violated"]` を 3 指標の組み合わせで決定（判定表を design.md に記載）
- `diagnostics/memory_capacity.py`: `linear_memory_capacity(...) -> MemoryCapacityResult`
  - i.i.d. Uniform(-0.8, 0.8) スカラー入力、既定 `n_train=3000` / `n_test=1000` / `washout=200`
  - 遅延 k = 1..K（既定 `K = min(2 * n_reservoir, 200)`）ごとに read-out を学習し `MC_k = corr(ŷ_k, u[t-k])^2`
  - `total_mc = Σ MC_k`、`per_delay: list[float]`、`memory_horizon`（`MC_k < 0.1` となる最小 k）、
    `mc_per_neuron = total_mc / N` を返す
  - 微小リッジ（既定 1e-8）。MC が正則化強度に敏感である旨を docstring と design.md に明記。
    負の `MC_k` は 0 にクリップせず生値を返す
- `diagnostics/report.py`: `@dataclass(frozen=True) DiagnosticsReport` + `to_dict() -> dict[str, object]`
  - 収録: `schema_version`, UTC ISO8601 タイムスタンプ, パッケージ version, numpy version,
    `ESNConfig` の全フィールド, seed, 3 指標の結果, `data_source`
  - JSON 書き出し（既定 `outputs/diagnostics/<timestamp>.json`）と `logging.info` による人間可読サマリ（1 指標 1 行）
- CLI `diagnose` サブコマンドを配線（`--n-reservoir` `--spectral-radius` `--leak-rate` `--seed`
  `--output-dir` `--skip-memory-capacity`）

受け入れ基準:

- `spectral_radius` が既知行列で理論値と一致（diag(0.3, -0.7) → 0.7）
- ρ=0.9 / leak=1.0 / 零入力で `verdict == "esp_holds"` かつ `decay_rate < 0`。
  ρ=1.5 で `necessary_condition_met is False` かつ `verdict == "esp_violated"`
- `total_mc <= n_reservoir`（理論上界）を N ∈ {20, 40} で満たす。N=40 で `total_mc > 5.0`
- `per_delay` の長さが K と一致し `MC_1 > MC_K`
- 同一 seed で 2 回実行した `DiagnosticsReport.to_dict()` がタイムスタンプ以外完全一致
- `uv run esn-vla-uq diagnose --n-reservoir 50 --output-dir <tmp> --seed 0` が exit 0、
  JSON が生成され `json.load` 可能、キー欠落なし
- 診断テスト全体の実行時間 < 20 秒。`make ci` 緑

### T5: サンプルデータのスキーマ設計・合成生成・同梱・ローダ

- `data/schema.py`: `Episode` / `RolloutDataset`。フィールド:
  - エピソード: `episode_id: str`, `task_name: str`, `success: bool`, `n_steps: int`
  - ステップ配列: `state: float32[T, D_state]`（7 関節 + グリッパ = 8）、
    `action: float32[T, D_action]`（6 DoF デルタ + グリッパ = 7）、
    `action_chunk: float32[T, H, D_action]`（H = 16 既定、非推論ステップは NaN）、`is_inference_step: bool[T]`
  - メタデータ: `schema_version: str`（`"0.1.0"`）, `source: Literal["synthetic", "openpi"]`,
    `policy: str`, `seed: int`, `control_hz: float`
  - `validate()` で shape / dtype / NaN 配置 / `episode_starts` の整合を検証し、
    違反時にどのフィールドがどう不正かを含む `ValueError`
- `data/source.py`: `class RolloutSource(Protocol)` に `load() -> RolloutDataset`。
  `SyntheticRolloutSource` を実装し、Sprint 2 の `OpenpiLogSource` を同 Protocol で差し替え可能にする
- `data/synthetic.py`: 決定論的な合成ロールアウト生成
  - 成功: 目標姿勢へ向かう滑らかな軌道（最小躍度風）+ AR(1) ノイズ。所定フェーズでグリッパ閉
  - `action` = 状態の差分 + 観測ノイズ。`action_chunk` = 将来 H ステップ行動の分散を持つ予測
    （flow matching のサンプリングばらつきを模す）
  - 失敗: ランダムな `failure_onset` 以降で分布シフト（目標ドリフト / chunk 分散増大 / グリッパ滑り）を注入し
    `success=False`。`failure_onset` はメタデータに保存
  - 既定: `n_episodes=40`, `success_rate≈0.7`, `T ∈ [150, 250]`, `seed` 必須
- `data/io.py`: 単一 `.npz`（エピソード連結 + `episode_starts`/`episode_lengths`）+ `metadata.json` サイドカー。float32 保存
- 同梱: `src/esn_vla_uq/assets/samples/libero_synthetic_v0.1.npz`（+ `.json`）。500kB 未満。
  `importlib.resources` 経由の `load_bundled_sample()` を提供
- CLI `gen-sample-data`（`--seed` `--n-episodes` `--output`）を配線
- 全レポート・全出力に `source="synthetic"` が伝播すること

受け入れ基準:

- save → load ラウンドトリップで全配列が `np.array_equal`、全メタデータが `==` で一致
- 同一 seed の 2 回生成が完全一致、異なる seed では不一致
- 不正データ 4 種以上（dtype 不一致 / shape 不一致 / `episode_starts` 不整合 / `schema_version` 不明）が `ValueError`
- 同梱 `.npz` が `load_bundled_sample()` で読め `validate()` を通り、`n_episodes == 40`、成功/失敗の両方を含む。
  ファイルサイズ < 500kB（テストで実測アサート）
- 失敗エピソードで `failure_onset` 以降の `action_chunk` 分散が以前より統計的に大きい（比 > 1.5）
- かつ、単純ベースライン（chunk 分散のしきい値判定）でエピソード成否 AUROC が 1.0 未満（課題が自明でない）。
  1.0 になる場合は難易度調整して再生成
- `uv run esn-vla-uq gen-sample-data --seed 0 --output <tmp>/s.npz` が exit 0 かつファイルを生成
- `make ci` 緑

## 7. 評価軸

### 機能

- クリーン環境（`rm -rf .venv` 後）で `uv sync --locked` → `uv run esn-vla-uq diagnose` →
  `uv run esn-vla-uq gen-sample-data` の 3 コマンドがすべて exit 0
- `diagnose` の JSON に `spectral_radius` / `esp.verdict` / `memory_capacity.total_mc` が存在し有限値
- 既知の理論値との照合（ρ の解析解、MC ≤ N の上界、`leak_rate=1` での更新式退化）

### 性能

- `make test` 全体 < 60 秒（CI の `timeout-minutes: 5` に余裕）
- `uv run esn-vla-uq diagnose --n-reservoir 500` が CPU 単体で 60 秒以内（GPU 前提にしない）
- 同梱サンプルデータ < 500kB、リポジトリ全体の増分 < 1MB
- `uv sync` 後の `.venv` サイズが現状 + 100MB 以内

### 安全性・整合性

- `make ci` の 5 工程すべて緑（特に `uv lock --check`）
- 新規 `# type: ignore` / `# noqa` / `# nosec` が 0 件、または各々に理由コメント
- `git diff` にシークレット・`.env`・個人情報が含まれない
- 合成データ由来の数値が実験結果として誤読されない（README / design.md / JSON メタデータの 3 箇所に `synthetic` 表記）
- `.python-version` / `requires-python` / Dockerfile の Python バージョン整合が維持されている
- 本番ロジックの依存（numpy）が `dev` グループに入っていない

### テスト

- 新規: `tests/test_{config,reservoir,readout,model,spectral,esp,memory_capacity,report,schema,synthetic,io,cli}.py`
- 既存 `tests/test_main.py` は `tests/test_cli.py` へ置換（テストを減らしたまま完了としない）
- 種類: 決定性テスト、理論値照合テスト、異常系テスト（`pytest.raises`）、
  CLI スモークテスト（`tmp_path` 使用、リポジトリを汚さない）
- `uv run pytest --cov` で `src/esn_vla_uq` のカバレッジ 85% 以上を目標

## 8. 想定リスク（発生したら止まって人間に相談）

1. **mypy strict + `disallow_any_explicit` × numpy の摩擦**
   `# type: ignore` が 3 件を超える見込みになった時点で停止し、
   「明示的 `float()` 変換で吸収」「`disallow_any_explicit` を緩める」「専用の型エイリアスを導入」の
   いずれを採るか人間に確認する。**設定を独断で緩めない**。
2. **合成データの難易度が不適切**
   注入した失敗シグナルが強すぎると（単純しきい値で AUROC 1.0）Sprint 2 の較正評価が無意味になり、
   弱すぎると ESN が何も学習できない。調整が 2 回失敗したら合成モデルの設計自体を人間と再検討する。
   合成データで良い数値を出すためにパラメータを探索する行為はしない。
3. **ESP / メモリ容量の定義選択の恣意性**
   既定設定（ρ=0.9）で十分条件と経験的判定が食い違う可能性が高い（σ_max < 1 は保守的）。
   食い違った場合に「都合の良い指標だけ報告する」ことはせず、3 指標併記の設計を維持したまま
   判定表の妥当性を人間に確認する。
