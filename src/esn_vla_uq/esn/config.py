"""ESN のハイパーパラメータ定義と範囲検証。

既定値は `docs/design.md` の「ESN の数学仕様」節に記載する既定ハイパーパラメータ表
と対応する。値の意味は以下のとおり。

- ``n_reservoir``: リザバーのニューロン数 N
- ``spectral_radius``: 再帰行列 W のスペクトル半径 rho
- ``input_scaling``: 入力行列 W_in の一様分布の振幅
- ``bias_scaling``: バイアス b の一様分布の振幅
- ``leak_rate``: リーク率 a (1.0 で非リーク型に退化する)
- ``density``: W の非零要素の割合 (疎行列は密行列 + マスクで表現する)
- ``ridge_alpha``: リッジ read-out の正則化強度 lambda
- ``washout``: 学習・評価から捨てる先頭ステップ数
- ``input_passthrough``: read-out の設計行列に入力 u を含めるか (既定で有効)
- ``use_reservoir``: read-out の設計行列にリザバー状態 x を含めるか (既定で有効)
- ``seed``: `numpy.random.default_rng` に渡す唯一の乱数シード

``input_passthrough`` と ``use_reservoir`` の組が read-out の設計行列を決める。
両方を False にすると設計行列がバイアス列だけになり、入力にも履歴にも依存しない
定数予測になるため `__post_init__` で拒否する。3 通りの有効な組み合わせは
`READOUT_FEATURE_FLAGS` に名前を付けてあり、リザバーの寄与を測るアブレーション
(`docs/next-research-directions.md` ①) の条件名として CLI から指定できる。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal, get_args

DEFAULT_N_RESERVOIR = 200
DEFAULT_SPECTRAL_RADIUS = 0.9
DEFAULT_INPUT_SCALING = 1.0
DEFAULT_BIAS_SCALING = 0.0
DEFAULT_LEAK_RATE = 1.0
DEFAULT_DENSITY = 0.1
DEFAULT_RIDGE_ALPHA = 1e-6
DEFAULT_WASHOUT = 100
DEFAULT_INPUT_PASSTHROUGH = True
DEFAULT_USE_RESERVOIR = True
DEFAULT_SEED = 0

ReadoutFeatures = Literal["input_reservoir", "reservoir_only", "input_only"]
"""read-out の設計行列に何を載せるかの名前。

- ``"input_reservoir"``: ``[1, u, x]`` (既定)。
- ``"reservoir_only"``: ``[1, x]``。パススルーを外す。
- ``"input_only"``: ``[1, u]``。**リザバーを外す**。リザバー無しの baseline で
  あり、時間記憶がどれだけ区間の幅を買っているかを測るための対照条件。
"""

SUPPORTED_READOUT_FEATURES: Final[tuple[ReadoutFeatures, ...]] = get_args(
    ReadoutFeatures
)
"""`ReadoutFeatures` が許可する値の実行時タプル (CLI の ``choices`` 用)。"""

DEFAULT_READOUT_FEATURES: Final[ReadoutFeatures] = "input_reservoir"
"""既定の設計行列。既存の既定値 (パススルー有効・リザバー有効) と一致する。"""

READOUT_FEATURE_FLAGS: Final[dict[ReadoutFeatures, tuple[bool, bool]]] = {
    "input_reservoir": (True, True),
    "reservoir_only": (False, True),
    "input_only": (True, False),
}
"""条件名から ``(input_passthrough, use_reservoir)`` への対応。

両方 False の組は載せない (`ESNConfig.__post_init__` が拒否する組であり、名前を
与えると呼び出し側から到達できてしまう)。
"""


@dataclass(frozen=True)
class ESNConfig:
    """ESN のハイパーパラメータ (不変)。

    `__post_init__` で範囲検証を行い、違反時は違反したパラメータ名と実値を含む
    `ValueError` を送出する。
    """

    n_reservoir: int = DEFAULT_N_RESERVOIR
    spectral_radius: float = DEFAULT_SPECTRAL_RADIUS
    input_scaling: float = DEFAULT_INPUT_SCALING
    bias_scaling: float = DEFAULT_BIAS_SCALING
    leak_rate: float = DEFAULT_LEAK_RATE
    density: float = DEFAULT_DENSITY
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA
    washout: int = DEFAULT_WASHOUT
    input_passthrough: bool = DEFAULT_INPUT_PASSTHROUGH
    use_reservoir: bool = DEFAULT_USE_RESERVOIR
    seed: int = DEFAULT_SEED

    @classmethod
    def readout_flags(cls, features: ReadoutFeatures) -> tuple[bool, bool]:
        """条件名を ``(input_passthrough, use_reservoir)`` へ展開する。

        Raises:
            ValueError: 未知の条件名の場合。
        """
        if features not in READOUT_FEATURE_FLAGS:
            raise ValueError(
                f"readout: 未知の条件です (actual={features!r}, "
                f"supported={list(SUPPORTED_READOUT_FEATURES)})"
            )
        return READOUT_FEATURE_FLAGS[features]

    @property
    def readout_features(self) -> ReadoutFeatures:
        """現在の設計行列に対応する条件名 (レポート・ログ用)。"""
        flags = (self.input_passthrough, self.use_reservoir)
        for name, candidate in READOUT_FEATURE_FLAGS.items():
            if candidate == flags:
                return name
        raise AssertionError(f"到達しないはずの組み合わせです: {flags}")

    def __post_init__(self) -> None:
        """範囲外のパラメータを検出して `ValueError` を送出する。"""
        if self.n_reservoir < 1:
            raise ValueError(
                f"n_reservoir は 1 以上である必要があります (実値: {self.n_reservoir})"
            )
        if not self.spectral_radius > 0.0:
            raise ValueError(
                "spectral_radius は 0 より大きい必要があります "
                f"(実値: {self.spectral_radius})"
            )
        if not 0.0 < self.leak_rate <= 1.0:
            raise ValueError(
                f"leak_rate は 0 < leak_rate <= 1 の範囲である必要があります "
                f"(実値: {self.leak_rate})"
            )
        if not 0.0 < self.density <= 1.0:
            raise ValueError(
                f"density は 0 < density <= 1 の範囲である必要があります "
                f"(実値: {self.density})"
            )
        if self.ridge_alpha < 0.0:
            raise ValueError(
                f"ridge_alpha は 0 以上である必要があります (実値: {self.ridge_alpha})"
            )
        if self.washout < 0:
            raise ValueError(
                f"washout は 0 以上である必要があります (実値: {self.washout})"
            )
        if not self.input_passthrough and not self.use_reservoir:
            raise ValueError(
                "input_passthrough と use_reservoir を同時に False にはできません "
                "(設計行列がバイアス列だけになり定数予測に退化します)"
            )

    def to_dict(self) -> dict[str, object]:
        """JSON シリアライズ可能な辞書へ変換する (診断レポート用)。

        フィールドは `dataclasses.asdict` で列挙する。以前は
        `diagnostics/report.py` がフィールド名を手書きで並べており、
        ここにハイパーパラメータを 1 つ足しても診断レポート JSON からは
        黙って欠落した (A2)。全フィールドが JSON 互換のスカラーであるため
        `asdict` の戻り値をそのまま使える。
        """
        return asdict(self)
