# ADR 0001: build backend に hatchling を採用し、Sprint 1 のランタイム依存を numpy のみに限定する

- ステータス: 承認済み（`docs/plans/sprint1_v0.1.md` ユーザー確定事項 8）
- 日付: Sprint 1（2026-08-02 承認）
- 関連: `docs/design.md` 6.5 節、`docs/plans/sprint1_v0.1.md` T1

## コンテキスト

`pyproject.toml` には当初 `[build-system]` が定義されておらず、uv は本プロジェクトを
「非パッケージプロジェクト」として扱っていた（`docs/plans/sprint1_v0.1.md` 落とし穴 1）。
このままでは `[project.scripts]` による CLI エントリポイント登録も、`src/` レイアウトの
インストールも成立しない。パッケージ化のためには build backend を選定する必要があった。

また要件書では ESN を NumPy または PyTorch で自前実装する方針が示されており、
Sprint 1 のスコープ（ESN コア + リザバー診断 + 同梱合成データ）に対して依存を
どこまで持つべきかを決める必要があった。

## 決定

### build backend: hatchling

- `[build-system] requires = ["hatchling"]` / `build-backend = "hatchling.build"` を採用。
- `[tool.hatch.build.targets.wheel] packages = ["src/esn_vla_uq"]` で `src/` レイアウトを
  明示する。

検討した代替:

- **setuptools**: 枯れているが `src/` レイアウト・`pyproject.toml` ベースの設定が
  hatchling に比べて冗長になりやすい。
- **PDM-backend / flit-core**: いずれも軽量だが、uv の推奨・実績という観点で hatchling
  ほどの情報量が無く、チームの既存知見（uv-template との親和性）を優先した。

hatchling は uv との組み合わせでの実績が広く、`src/` レイアウトの設定が簡潔であるため
採用する。

### ランタイム依存: numpy のみ（Sprint 1）

- ESN の状態更新・リッジ read-out は閉形式で、自動微分や GPU 学習を必要としない
  （`docs/design.md` 3 節）。numpy の密行列演算のみで T3/T4 の受け入れ基準
  （理論値照合・決定性・性能要件）を満たせる。
- matplotlib（可視化）・PyTorch（学習フレームワーク）・ReservoirPy（参照実装）は
  Sprint 1 のスコープ（`uv sync && uv run esn-vla-uq diagnose` が動くこと）に対して
  過剰であり、`.venv` サイズ増加や CI 時間増加のコストに見合わない。

検討した代替:

- **PyTorch を最初から入れる**: 将来 uncertainty 層（Sprint 2 conformal prediction）や
  学習ベースの手法拡張に備える案。今回は不採用。閉形式実装で十分なうちは依存を
  増やさず、必要になった時点（Sprint 2 以降）で再検討する。
- **ReservoirPy を dev 依存に追加し参照実装と突き合わせる**: 検証強化になるが、
  Sprint 1 は理論値照合テスト（既知の解析解・NRMSE 基準）で正しさを担保する方針とし、
  見送った。

## 結果

- `pyproject.toml` の `dependencies = ["numpy>=2.5.1"]` のみ、`dev` グループに
  mypy / pre-commit / pytest / pytest-cov / ruff。
- 本番ロジックの依存（numpy）が `dev` グループに入っていないことを Sprint 1 の
  評価軸として維持する。
- Sprint 2 以降で matplotlib（reliability diagram）を追加する可能性が高い。追加時は
  本 ADR を更新するのではなく、新規 ADR を切ってこの決定を上書きする。

## 未解決事項

- `N > 500` でのスケーラビリティ要求が出た場合、scipy（疎行列・反復固有値計算）の
  追加を検討する必要があるかもしれない（`docs/design.md` 8 節）。本 ADR の対象外。
