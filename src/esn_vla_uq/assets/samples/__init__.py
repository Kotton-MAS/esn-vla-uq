"""同梱サンプルデータ置き場。

収録物:

- `libero_synthetic_v0.1.npz`: 連結された合成ロールアウト配列
- `libero_synthetic_v0.1.json`: メタデータのサイドカー (`source: "synthetic"`)

同梱データはすべて **合成データ** であり、実 LIBERO のロールアウトでも実 VLA
ポリシーの出力でもない。ここから得た数値を実験結果として提示してはならない。

再生成手順 (決定論的に同一内容になる)::

    uv run esn-vla-uq gen-sample-data --seed 0 \\
        --output src/esn_vla_uq/assets/samples/libero_synthetic_v0.1.npz

`.npz` は zip 形式でエントリのタイムスタンプを含むためバイト列は再現しないが、
格納される配列とメタデータは `--seed` が同じなら完全に一致する。読み込みは
`esn_vla_uq.data.load_bundled_sample()` を使う。
"""
