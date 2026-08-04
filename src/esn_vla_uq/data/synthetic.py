"""決定論的な合成ロールアウト生成。

ここで生成されるデータは **すべて合成データ** であり (`source == "synthetic"`)、
実 LIBERO のロールアウトでも実 VLA ポリシーの出力でもない。openpi 由来のログを
扱えるようになるまでの代替であり、数値を実験結果として提示してはならない。

生成モデル (1 エピソード):

1. 7 関節が最小躍度風のプロファイル (``10t^3 - 15t^4 + 6t^5``) で開始姿勢から
   目標姿勢へ移動し、AR(1) のプロセスノイズが乗る。
2. グリッパは所定フェーズ (``tau = 0.55``) で滑らかに閉じる。
3. ``action`` は状態差分に観測ノイズを加えたもの (6 DoF デルタの合成プロキシ) と
   グリッパ指令。実機の順運動学ではない。
4. ``action_chunk`` は ``CHUNK_HORIZON`` ステップ先までの将来行動に、チャンク単位の
   バイアスと要素ごとのノイズを加えた予測。flow matching のサンプリングばらつきを模す。
5. 失敗エピソードでは ``failure_onset`` 以降に分布シフトを注入する
   (目標ドリフト / チャンク分散の増大 / グリッパ滑り)。

合成データ固有の不変条件 (`validate_synthetic_dataset`) は `data/invariants.py`
にある。不変条件の実装をこちら側に置くと、それを読み込み境界でも掛けたい
`data/io.py` が本モジュール (将来は openpi ログパーサも) を import せざるを
えなくなるため、依存の向きを逆にしている (S7)。

難易度に関する設計上の注意 (仕様書 §8 リスク 2):
チャンクノイズの基準スケールはエピソードごとに対数正規で散らしてあり、その散らばりは
失敗時の分散増大より広い。したがって「チャンク分散のしきい値判定」という単純ベースライン
では成功/失敗を完全には分離できず (AUROC < 1.0)、一方で失敗区間の分散比は 1.5 を十分
上回る。この 2 条件を両立させることが本モジュールの設計意図である。

## 失敗モデルは実データで否定されている

**この生成器は「失敗すればチャンクの分散が上がる」を仮定して書かれており、実データは
その関係を示さなかった。** 実収集した LIBERO ロールアウト約 310 本での失敗検知
AUROC は 0.457〜0.477 (偶然と区別できない) であり、合成データで得られた 0.87 は
**生成器がその関係を組み込んであるから出た数字**である (`docs/design.md` 10.11 節)。
この経緯により失敗検知は v0.2.0 で対象外にした。

生成器はそのまま残してある。較正 (被覆率・区間幅) の検証には使えるためで、実際に
実データでも被覆率は保たれた。**使ってよいのは較正の検証まで**であり、
`failure_onset` や `success` を当てる能力の評価に使うと、生成器の仮定を測り直す
だけになる。
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
from numpy.typing import NDArray

from esn_vla_uq.data.invariants import validate_synthetic_dataset
from esn_vla_uq.data.schema import (
    ACTION_DIM,
    CHUNK_HORIZON,
    STATE_DIM,
    Episode,
    RolloutDataset,
)

logger = logging.getLogger(__name__)

DEFAULT_N_EPISODES: Final[int] = 40
"""既定のエピソード数。"""

DEFAULT_SUCCESS_RATE: Final[float] = 0.7
"""既定の成功率。実際の成功数は四捨五入で決定論的に決まる。"""

DEFAULT_MIN_STEPS: Final[int] = 150
"""エピソード長の下限。"""

DEFAULT_MAX_STEPS: Final[int] = 250
"""エピソード長の上限。"""

CONTROL_HZ: Final[float] = 20.0
"""合成ロールアウトの制御周波数 [Hz]。"""

POLICY_NAME: Final[str] = "synthetic-chunked-policy-v0.1"
"""合成ポリシー名。実在の VLA ポリシーではないことが名前から分かるようにする。"""

TASK_NAMES: Final[tuple[str, ...]] = (
    "synthetic_pick_up_bowl",
    "synthetic_place_bowl_on_plate",
    "synthetic_open_drawer",
    "synthetic_put_mug_in_drawer",
)
"""合成タスク名。LIBERO のタスクを模した命名だが実タスクではない。"""

N_JOINTS: Final[int] = STATE_DIM - 1
"""関節数 (状態次元からグリッパ 1 次元を除く)。"""

N_DELTA_DIM: Final[int] = ACTION_DIM - 1
"""行動のうちデルタ成分の次元 (6 DoF)。"""

_AR1_PHI: Final[float] = 0.85
# 状態のプロセスノイズと行動の観測ノイズは、行動系列そのものが持つ高周波成分の
# 大きさを決める。この「ノイズ床」がチャンク予測ノイズ (_CHUNK_SIGMA_BASE) を
# 上回ると、失敗時のチャンク分散増大が床に埋もれて観測できなくなるため、
# 床はチャンクノイズより十分小さく設定する。
_AR1_SIGMA: Final[float] = 0.0012
_OBS_SIGMA: Final[float] = 0.0003
_GRIPPER_CLOSE_PHASE: Final[float] = 0.55
_GRIPPER_CLOSE_WIDTH: Final[float] = 0.06
_GRIPPER_SIGMA: Final[float] = 0.002

_CHUNK_SIGMA_BASE: Final[float] = 0.004
_CHUNK_SIGMA_LOG_SPREAD: Final[float] = 0.35
_CHUNK_BIAS_RATIO: Final[float] = 0.5

_FAILURE_ONSET_MIN_PHASE: Final[float] = 0.50
_FAILURE_ONSET_MAX_PHASE: Final[float] = 0.85
_FAILURE_DRIFT_MIN: Final[float] = 0.05
_FAILURE_DRIFT_MAX: Final[float] = 0.15
_FAILURE_SLIP_MIN: Final[float] = 0.4
_FAILURE_SLIP_MAX: Final[float] = 0.9
_FAILURE_CHUNK_SIGMA_GAIN_MIN: Final[float] = 1.8
_FAILURE_CHUNK_SIGMA_GAIN_MAX: Final[float] = 3.0

STATE_QUANTUM: Final[float] = 2.0**-13
"""状態の量子化幅 (合成エンコーダ分解能)。約 1.2e-4。"""

ACTION_QUANTUM: Final[float] = 2.0**-16
"""行動・行動チャンクの量子化幅 (合成アクチュエータ分解能)。約 1.5e-5。"""


def _quantize(values: NDArray[np.float64], quantum: float) -> NDArray[np.float32]:
    """2 の冪の格子へ量子化して float32 にする。

    量子化幅を 2 の冪にすることで float32 変換が丸め誤差なく行え、仮数部の下位
    ビットが 0 になるため同梱アセットの圧縮率が上がる (500kB 制限への対応)。
    """
    quantized = np.round(values / quantum) * quantum
    return quantized.astype(np.float32)


def _ar1_noise(
    rng: np.random.Generator, n_steps: int, n_dims: int, phi: float, sigma: float
) -> NDArray[np.float64]:
    """AR(1) 過程のノイズ系列 `[n_steps, n_dims]` を返す。"""
    innovations = rng.normal(0.0, sigma, size=(n_steps, n_dims))
    noise = np.empty((n_steps, n_dims), dtype=np.float64)
    noise[0] = innovations[0]
    for t in range(1, n_steps):
        noise[t] = phi * noise[t - 1] + innovations[t]
    return noise


def _smoothstep(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """[0, 1] にクリップした滑らかな立ち上がり (3t^2 - 2t^3)。"""
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _minimum_jerk_profile(n_steps: int) -> NDArray[np.float64]:
    """最小躍度風の 0 -> 1 プロファイル `10t^3 - 15t^4 + 6t^5`。"""
    phase = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    return 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5


def _failure_ramp(n_steps: int, onset: int) -> NDArray[np.float64]:
    """`onset` 以降で 0 -> 1 に立ち上がるランプを返す。"""
    steps = np.arange(n_steps, dtype=np.float64)
    span = float(max(n_steps - 1 - onset, 1))
    return np.clip((steps - float(onset)) / span, 0.0, 1.0)


def _generate_state(
    rng: np.random.Generator, n_steps: int, onset: int | None
) -> NDArray[np.float64]:
    """状態系列 `[n_steps, STATE_DIM]` を生成する。"""
    start = rng.uniform(-0.5, 0.5, size=N_JOINTS)
    goal = start + rng.uniform(-0.9, 0.9, size=N_JOINTS)
    profile = _minimum_jerk_profile(n_steps)
    joints = start + (goal - start) * profile[:, None]
    joints += _ar1_noise(rng, n_steps, N_JOINTS, _AR1_PHI, _AR1_SIGMA)

    phase = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    gripper = _smoothstep((phase - _GRIPPER_CLOSE_PHASE) / _GRIPPER_CLOSE_WIDTH)
    gripper = gripper + rng.normal(0.0, _GRIPPER_SIGMA, size=n_steps)

    if onset is not None:
        ramp = _failure_ramp(n_steps, onset)
        direction = rng.normal(0.0, 1.0, size=N_JOINTS)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        drift = float(rng.uniform(_FAILURE_DRIFT_MIN, _FAILURE_DRIFT_MAX))
        joints = joints + direction * (drift * ramp**2)[:, None]
        slip = float(rng.uniform(_FAILURE_SLIP_MIN, _FAILURE_SLIP_MAX))
        gripper = gripper * (1.0 - slip * ramp)

    return np.column_stack((joints, gripper))


def _generate_action(
    rng: np.random.Generator,
    state: NDArray[np.float64],
    gripper_command: NDArray[np.float64],
) -> NDArray[np.float64]:
    """行動系列 `[n_steps, ACTION_DIM]` を生成する。

    デルタ成分は状態差分の合成プロキシであり、実機の順運動学に基づく 6 DoF
    エンドエフェクタ変位ではない。グリッパ成分は指令値 (滑りを含まない) を使う。
    """
    n_steps = state.shape[0]
    # 差分は T-1 本しか得られないため、最終ステップは直前の値を保持する。
    delta = np.diff(state[:, :N_DELTA_DIM], axis=0)
    delta = np.vstack((delta, delta[-1:]))
    delta = delta + rng.normal(0.0, _OBS_SIGMA, size=(n_steps, N_DELTA_DIM))
    return np.column_stack((delta, gripper_command))


def _generate_action_chunks(
    rng: np.random.Generator,
    action: NDArray[np.float64],
    inference_steps: NDArray[np.int64],
    chunk_sigma: float,
    onset: int | None,
    sigma_gain: float,
) -> NDArray[np.float64]:
    """推論ステップごとの行動チャンク `[n_inference, H, ACTION_DIM]` を生成する。"""
    n_steps = action.shape[0]
    horizon = np.arange(CHUNK_HORIZON, dtype=np.int64)
    future_index = np.clip(inference_steps[:, None] + horizon[None, :], 0, n_steps - 1)
    base = action[future_index]

    scales = np.full(inference_steps.shape[0], chunk_sigma, dtype=np.float64)
    if onset is not None:
        scales = np.where(inference_steps >= onset, chunk_sigma * sigma_gain, scales)

    element_noise = rng.normal(0.0, 1.0, size=base.shape) * scales[:, None, None]
    chunk_bias = (
        rng.normal(0.0, 1.0, size=(inference_steps.shape[0], 1, ACTION_DIM))
        * (scales * _CHUNK_BIAS_RATIO)[:, None, None]
    )
    chunks: NDArray[np.float64] = base + element_noise + chunk_bias
    return chunks


def _generate_episode(
    rng: np.random.Generator,
    episode_id: str,
    task_name: str,
    success: bool,
    min_steps: int,
    max_steps: int,
) -> Episode:
    """1 エピソードを生成する。"""
    n_steps = int(rng.integers(min_steps, max_steps + 1))
    onset: int | None = None
    if not success:
        phase = float(rng.uniform(_FAILURE_ONSET_MIN_PHASE, _FAILURE_ONSET_MAX_PHASE))
        onset = round(phase * (n_steps - 1))
    sigma_gain = float(
        rng.uniform(_FAILURE_CHUNK_SIGMA_GAIN_MIN, _FAILURE_CHUNK_SIGMA_GAIN_MAX)
    )
    chunk_sigma = _CHUNK_SIGMA_BASE * float(
        np.exp(rng.normal(0.0, _CHUNK_SIGMA_LOG_SPREAD))
    )

    state = _generate_state(rng, n_steps, onset)
    phase_grid = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    gripper_command = _smoothstep(
        (phase_grid - _GRIPPER_CLOSE_PHASE) / _GRIPPER_CLOSE_WIDTH
    )
    action = _generate_action(rng, state, gripper_command)

    inference_steps = np.arange(0, n_steps, CHUNK_HORIZON, dtype=np.int64)
    chunks = _generate_action_chunks(
        rng, action, inference_steps, chunk_sigma, onset, sigma_gain
    )

    is_inference_step = np.zeros(n_steps, dtype=np.bool_)
    is_inference_step[inference_steps] = True
    action_chunk = np.full(
        (n_steps, CHUNK_HORIZON, ACTION_DIM), np.nan, dtype=np.float32
    )
    action_chunk[inference_steps] = _quantize(chunks, ACTION_QUANTUM)

    return Episode(
        episode_id=episode_id,
        task_name=task_name,
        success=success,
        n_steps=n_steps,
        state=_quantize(state, STATE_QUANTUM),
        action=_quantize(action, ACTION_QUANTUM),
        action_chunk=action_chunk,
        is_inference_step=is_inference_step,
        failure_onset=onset,
    )


def _success_flags(
    rng: np.random.Generator, n_episodes: int, success_rate: float
) -> NDArray[np.bool_]:
    """成功フラグ列を返す。成功数は決定論的、並びのみ乱数で決める。"""
    n_success = round(success_rate * n_episodes)
    n_success = min(max(n_success, 0), n_episodes)
    flags = np.zeros(n_episodes, dtype=np.bool_)
    flags[:n_success] = True
    permuted: NDArray[np.bool_] = rng.permutation(flags)
    return permuted


def _validate_arguments(
    n_episodes: int, success_rate: float, min_steps: int, max_steps: int
) -> None:
    """生成パラメータを検証する。"""
    if n_episodes < 1:
        raise ValueError(
            f"n_episodes: 1 以上である必要があります (actual={n_episodes})"
        )
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError(f"success_rate: [0, 1] の範囲外です (actual={success_rate})")
    if min_steps < CHUNK_HORIZON:
        raise ValueError(
            f"min_steps: {CHUNK_HORIZON} 以上である必要があります (actual={min_steps})"
        )
    if max_steps < min_steps:
        raise ValueError(
            "max_steps: min_steps 以上である必要があります "
            f"(min_steps={min_steps}, max_steps={max_steps})"
        )


def generate_dataset(
    seed: int,
    n_episodes: int = DEFAULT_N_EPISODES,
    success_rate: float = DEFAULT_SUCCESS_RATE,
    min_steps: int = DEFAULT_MIN_STEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> RolloutDataset:
    """合成ロールアウトデータセットを決定論的に生成する。

    Args:
        seed: 乱数シード。同じ引数なら常に同一のデータセットを返す。
        n_episodes: エピソード数。
        success_rate: 成功エピソードの割合。
        min_steps: エピソード長の下限。
        max_steps: エピソード長の上限。

    Returns:
        `source == "synthetic"` の `RolloutDataset` (`RolloutDataset.validate()`
        と `validate_synthetic_dataset()` を通過済み)。

    Raises:
        ValueError: 生成パラメータが不正な場合、または生成された不変条件が
            破れている場合 (通常は発生しない防御的チェック)。
    """
    _validate_arguments(n_episodes, success_rate, min_steps, max_steps)
    rng = np.random.default_rng(seed)
    flags = _success_flags(rng, n_episodes, success_rate)

    episodes = [
        _generate_episode(
            rng,
            episode_id=f"synthetic_{index:04d}",
            task_name=TASK_NAMES[index % len(TASK_NAMES)],
            success=bool(flag),
            min_steps=min_steps,
            max_steps=max_steps,
        )
        for index, flag in enumerate(flags)
    ]

    dataset = RolloutDataset(
        episodes=episodes,
        source="synthetic",
        policy=POLICY_NAME,
        seed=seed,
        control_hz=CONTROL_HZ,
        # schema.py の既定値に暗黙に委ねず、本モジュールが実際に生成へ使った
        # 次元 (STATE_DIM/ACTION_DIM/CHUNK_HORIZON) を明示する。schema.py の
        # 既定値が将来変わっても、生成配列とメタデータが黙って食い違わない
        # ようにするため (挙動は変わらない、既定値と同一の値)。
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_horizon=CHUNK_HORIZON,
    )
    dataset.validate()
    validate_synthetic_dataset(dataset)
    logger.debug(
        "generated synthetic dataset: source=%s seed=%d n_episodes=%d total_steps=%d",
        dataset.source,
        dataset.seed,
        dataset.n_episodes,
        dataset.total_steps,
    )
    return dataset
