"""Echo State Property (ESP) の判定。

`docs/design.md` の 4.2 節に従い、**3 指標を必ず同時に計算**して `EspResult` に
併記する。単一の指標だけで `verdict` を決めない (Sprint 1 の想定リスク 3)。

実効更新行列を ``A = (1 - a) I + a W`` (``a = leak_rate``) として:

1. 十分条件 ``sufficient_condition_met``: ``sigma_max(A) < 1``。成立すれば任意の
   有界入力に対して ESP が数学的に保証される。**保守的**な条件であり、既定設定
   (``rho = 0.9``) でも満たされないことが多い (一般に ``sigma_max >= rho``)。
2. 必要条件 ``necessary_condition_met``: ``rho(A) < 1``。原点近傍の線形化が漸近
   安定であるための必要条件。不成立でも tanh の飽和により駆動系では経験的に
   収束しうる。
3. 経験的収束 ``empirical_converged`` / ``decay_rate``: 同一のテスト入力列を
   ``K`` 個の異なるランダム初期状態から駆動し、各時刻の状態間の最大ペアワイズ
   L2 距離 ``d(t)`` を見る。``d(T) < tol`` を収束とし、``log(d(t) + eps)`` の
   最小二乗の傾きを ``decay_rate`` として返す。

判定表 (`docs/design.md` 4.2 節) の実装は `_decide_verdict` を参照。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.diagnostics.spectral import (
    effective_update_matrix,
    largest_singular_value,
    spectral_radius,
)
from esn_vla_uq.esn.reservoir import Reservoir

logger = logging.getLogger(__name__)

DEFAULT_ESP_SEED: Final[int] = 0
"""テスト入力と初期状態の既定シード。

同モジュールの他の既定値が `Final` 定数になっているのに合わせる (C2)。
"""

DEFAULT_ESP_N_STEPS: Final[int] = 500
"""既定のテスト系列長 T (`docs/design.md` 4.2 節)。"""

DEFAULT_ESP_N_INITIAL_STATES: Final[int] = 8
"""既定の初期状態数 K。"""

DEFAULT_ESP_TOL: Final[float] = 1e-6
"""``d(T) < tol`` を経験的収束とみなす閾値。"""

LOG_EPSILON: Final[float] = 1e-300
"""``log(0)`` を避けるための微小値。"""

EspVerdict = Literal["esp_holds", "esp_likely", "esp_violated"]
"""ESP の総合判定。3 指標の組み合わせから `_decide_verdict` が導く。"""


@dataclass(frozen=True)
class EspResult:
    """ESP 判定の結果 (3 指標を必ず併記する)。

    Attributes:
        sufficient_condition_met: ``sigma_max(A) < 1`` (十分条件)。
        necessary_condition_met: ``rho(A) < 1`` (必要条件)。
        empirical_converged: ``d(T) < tolerance`` (経験的収束)。
        decay_rate: ``log(d(t) + eps)`` の最小二乗の傾き。収束していれば負。
        verdict: 判定表 (`docs/design.md` 4.2 節) による総合判定。
        largest_singular_value: 十分条件の実測値 ``sigma_max(A)``。
        effective_spectral_radius: 必要条件の実測値 ``rho(A)``。
        final_distance: 最終時刻の状態間最大ペアワイズ L2 距離 ``d(T)``。
        tolerance: 経験的収束の判定に使った ``tol``。
        n_initial_states: 初期状態数 K。
        n_steps: テスト系列長 T。
        zero_input: テスト入力が恒等的に零だったか (最も厳しいテストに相当)。
    """

    sufficient_condition_met: bool
    necessary_condition_met: bool
    empirical_converged: bool
    decay_rate: float
    verdict: EspVerdict
    largest_singular_value: float
    effective_spectral_radius: float
    final_distance: float
    tolerance: float
    n_initial_states: int
    n_steps: int
    zero_input: bool

    def to_dict(self) -> dict[str, object]:
        """JSON シリアライズ可能な辞書へ変換する (診断レポート用)。

        フィールドは `dataclasses.asdict` で列挙する。以前は
        `diagnostics/report.py` が 12 個のフィールド名を手書きで並べており、
        「3 指標を必ず併記する」という 4.2 節の要求が、レポート側の列挙が
        この dataclass に追随し続けることに依存していた。指標を 1 つ足しても
        mypy も pytest も落ちないまま JSON から欠落しうる状態だったため、
        列挙をここへ寄せる (A2)。全フィールドが JSON 互換のスカラー。
        """
        return asdict(self)


def default_test_inputs(
    rng: np.random.Generator, n_steps: int, n_inputs: int
) -> NDArray[np.float64]:
    """既定のテスト入力 i.i.d. ``Uniform(-1, 1)^{D_u}`` を `[T, D_u]` で返す。"""
    if n_steps < 1:
        raise ValueError(f"n_steps は 1 以上である必要があります (実値: {n_steps})")
    return rng.uniform(-1.0, 1.0, size=(n_steps, n_inputs))


def _pairwise_max_distances(
    trajectories: NDArray[np.float64],
) -> NDArray[np.float64]:
    """`[K, T, N]` の軌道から各時刻の最大ペアワイズ L2 距離 `[T]` を求める。"""
    n_states = trajectories.shape[0]
    distances = np.zeros(trajectories.shape[1], dtype=np.float64)
    for i in range(n_states):
        for j in range(i + 1, n_states):
            pair = np.linalg.norm(trajectories[i] - trajectories[j], axis=1)
            distances = np.maximum(distances, pair)
    return distances


def _decay_rate(distances: NDArray[np.float64]) -> float:
    """``{(t, log(d(t) + eps)) : d(t) > 0}`` に対する最小二乗の傾きを返す。

    有効点が 2 点未満 (= ほぼ全時刻で ``d(t) == 0``) のときは傾きが定まらないため
    警告ログを出して 0.0 を返す。この場合でも `EspResult.final_distance` と
    `EspResult.empirical_converged` が収束の実測値を保持する。
    """
    steps = np.arange(1, distances.shape[0] + 1, dtype=np.float64)
    positive = distances > 0.0
    if int(np.count_nonzero(positive)) < 2:
        logger.warning(
            "decay_rate: 正の距離が %d 点しかないため傾きを定義できません "
            "(0.0 として報告する)",
            int(np.count_nonzero(positive)),
        )
        return 0.0
    log_distances = np.log(distances[positive] + LOG_EPSILON)
    slope, _intercept = np.polyfit(steps[positive], log_distances, 1)
    return float(slope)


def _decide_verdict(
    *, sufficient: bool, necessary: bool, empirical: bool
) -> EspVerdict:
    """`docs/design.md` 4.2 節の判定表をそのまま実装する。

    | S | N | E | verdict |
    |---|---|---|---|
    | True | True | - | ``esp_holds`` (#1) |
    | True | False | - | ``esp_likely`` (#6: 理論上生じない。数値誤差として警告) |
    | False | True | True | ``esp_holds`` (#2) |
    | False | True | False | ``esp_likely`` (#3) |
    | False | False | True | ``esp_likely`` (#4) |
    | False | False | False | ``esp_violated`` (#5) |
    """
    if sufficient and necessary:
        return "esp_holds"
    if sufficient:
        logger.warning(
            "ESP: 十分条件が成立したのに必要条件が不成立です "
            "(rho(A) <= sigma_max(A) より理論上生じない組み合わせ)。"
            "浮動小数点誤差とみなし esp_likely にフォールバックします"
        )
        return "esp_likely"
    if necessary:
        return "esp_holds" if empirical else "esp_likely"
    return "esp_likely" if empirical else "esp_violated"


def check_esp(
    reservoir: Reservoir,
    *,
    inputs: NDArray[np.float64] | None = None,
    n_steps: int = DEFAULT_ESP_N_STEPS,
    n_initial_states: int = DEFAULT_ESP_N_INITIAL_STATES,
    tol: float = DEFAULT_ESP_TOL,
    seed: int = DEFAULT_ESP_SEED,
    effective_spectral_radius: float | None = None,
) -> EspResult:
    """ESP の 3 指標を計算して `EspResult` を返す。

    Args:
        reservoir: 診断対象のリザバー (``W`` / ``W_in`` / ``b`` と `ESNConfig`)。
        inputs: テスト入力 `[T, D_u]`。省略時は i.i.d. ``Uniform(-1, 1)^{D_u}``。
            ESP は入力に依存するため、零入力など別の分布を試したい場合は明示的に
            渡す (零入力は必要条件の検証に近い最も厳しいテストに相当する)。
        n_steps: `inputs` 省略時のテスト系列長 T。
        n_initial_states: ランダム初期状態の個数 K (2 以上)。
        tol: 経験的収束の判定閾値。
        seed: テスト入力と初期状態を生成する `np.random.default_rng` の種。
        effective_spectral_radius: ``rho(A)`` を外から渡す (P2)。省略時はここで
            計算する。`diagnostics/runner.py` は同じ ``A`` のスペクトル半径を
            `summarize_spectral` でも使うため、渡して O(N^3) の固有値計算の
            重複を避ける。**値の意味は変わらない**ので、単独で呼ぶときは省略
            してよい。

    Returns:
        3 指標と `verdict` を併記した `EspResult`。

    Raises:
        ValueError: `n_initial_states` が 2 未満、`tol` が非正、または `inputs` の
            shape がリザバーと整合しない場合。
    """
    if n_initial_states < 2:
        raise ValueError(
            f"n_initial_states は 2 以上である必要があります (実値: {n_initial_states})"
        )
    if tol <= 0.0:
        raise ValueError(f"tol は 0 より大きい必要があります (実値: {tol})")

    rng = np.random.default_rng(seed)
    test_inputs = (
        default_test_inputs(rng, n_steps, reservoir.n_inputs)
        if inputs is None
        else np.asarray(inputs, dtype=np.float64)
    )

    leak_rate = reservoir.config.leak_rate
    update_matrix = effective_update_matrix(reservoir.W, leak_rate)
    sigma_max = largest_singular_value(update_matrix)
    rho = (
        spectral_radius(update_matrix)
        if effective_spectral_radius is None
        else effective_spectral_radius
    )

    initial_states = rng.uniform(
        -1.0, 1.0, size=(n_initial_states, reservoir.n_reservoir)
    )
    trajectories = np.stack(
        [reservoir.run(test_inputs, initial_state) for initial_state in initial_states]
    )
    distances = _pairwise_max_distances(trajectories)
    final_distance = float(distances[-1])

    sufficient = sigma_max < 1.0
    necessary = rho < 1.0
    empirical = final_distance < tol
    verdict = _decide_verdict(
        sufficient=sufficient, necessary=necessary, empirical=empirical
    )
    logger.debug(
        "esp checked: sigma_max=%g rho=%g d_final=%g verdict=%s",
        sigma_max,
        rho,
        final_distance,
        verdict,
    )
    return EspResult(
        sufficient_condition_met=sufficient,
        necessary_condition_met=necessary,
        empirical_converged=empirical,
        decay_rate=_decay_rate(distances),
        verdict=verdict,
        largest_singular_value=sigma_max,
        effective_spectral_radius=rho,
        final_distance=final_distance,
        tolerance=float(tol),
        n_initial_states=int(n_initial_states),
        n_steps=int(test_inputs.shape[0]),
        zero_input=bool(np.all(test_inputs == 0.0)),
    )
