"""openpi の LIBERO 評価ロールアウトを収集して本リポジトリのログ形式で保存する。

**このスクリプトだけが openpi と LIBERO を必要とする。** パッケージ本体
(`esn_vla_uq`) は numpy のみに依存し、収集済みログを
`esn_vla_uq.data.sources.openpi.OpenpiLogSource` で読む。実行環境も分ける想定で、
本スクリプトは `scripts/` に置き、wheel にも sdist にも含めない。

## なぜ収集層が要るのか

openpi の評価スクリプト (`examples/libero/main.py`) は**ロールアウトを保存しない**。
replay 動画を書くだけで、`state` / `action` / `action_chunk` の時系列はループを
抜けた時点で捨てられる。したがって「openpi のログを食う」には、まず記録する側を
用意する必要がある。本スクリプトは openpi の評価ループをなぞりながら、各ステップの
観測・実行行動・推論したチャンクを記録する。

## 使い方

openpi 側の環境で policy server を起動しておく:

    # openpi のリポジトリで
    uv run scripts/serve_policy.py policy:checkpoint \\
        --policy.config=pi0_libero --policy.dir=<checkpoint>

その上で LIBERO 環境を持つ Python から実行する:

    python scripts/collect_openpi_rollouts.py \\
        --output-dir outputs/openpi_logs \\
        --task-suite-name libero_spatial \\
        --num-trials-per-task 10

openpi / LIBERO が入っていない環境では起動時に明示的なエラーになる。

## 再現性について

**このスクリプトの出力は実行ごとに再現しない。** `--seed` が決めるのは LIBERO 環境の
初期状態だけで、policy server 側のサンプリングは制御しない。pi0 は flow matching で
行動をサンプリングする確率的モデルなので、同じ初期状態でも軌道が変わる (実測: 2 回の
収集で失敗したエピソードが違った)。

収集済みログを入力にした解析 (`calibrate` / `demo`)は同一 seed で再現する。
再現しないのは収集そのものである。

## 記録するもの

各ステップで以下を記録する (形式の定義は `OpenpiLogSource` の docstring)。

- `state`: policy server へ送った `observation/state` (8 次元)
- `action`: 環境へ渡した行動 (7 次元)
- `action_chunk`: 推論したチャンク全体 (`action_horizon` 分。実行するのは先頭
  `replan_steps` だけだが、**捨てずに全部残す**。チャンク内のばらつきが
  不確実性の材料であり、実行分だけでは分散が測れない)
- `inference_steps`: チャンクを推論したステップ番号
- `object_state`: シミュレータが返す物体の状態 (`object-state`)。**失敗様式の
  事後分類にのみ使い、ESN の入力には渡さない**

観測画像は記録しない。要件書の入力は「action chunk 系列と固有受容感覚」であり、
画像は v0.2 以降の VLM 特徴量注入の話になる。画像を貯めるとログが桁違いに重くなる。

## なぜ物体の状態を記録するのか

LIBERO の `step` は `done = self._check_success()` であり、**成功以外に終了条件が
無い**。したがって失敗は定義上すべてタイムアウトになり、試行を増やしても難しい
スイートに変えても失敗様式は増えない (実測: `pi05_libero` の 6 本も `pi0_libero` の
23 本もすべてタイムアウト。`docs/design.md` 10.13 節)。

一方、観測には物体の位置・姿勢と、グリッパから物体までの相対位置が含まれる。これを
記録しておけば、同じ「タイムアウト」でも

- 一度も物体に近づけなかった
- 掴んだが落とした
- 掴んで運んだが目標位置に置けなかった

を事後に区別できる。**終了条件を増やさずに失敗様式を得るための記録**である。
物体の高さが下がった時刻から `failure_onset` を定義できる可能性もある。

**解釈は加えずに生の値を残す。** ここで分類まで済ませてしまうと、分類の基準を
変えたくなったときに再収集が要る。分類は読み込み側で行う。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

LOG_SCHEMA_VERSION: Final[str] = "0.1.0"
"""書き出すログ形式のバージョン (`OpenpiLogSource` と一致させる)。"""

DEFAULT_HOST: Final[str] = "0.0.0.0"
DEFAULT_PORT: Final[int] = 8000
DEFAULT_REPLAN_STEPS: Final[int] = 5
"""openpi の `examples/libero/main.py` と同じ既定値。"""

DEFAULT_RESIZE_SIZE: Final[int] = 224
DEFAULT_NUM_STEPS_WAIT: Final[int] = 10
"""シミュレータが物体を落ち着かせるまで待つステップ数 (openpi と同じ)。"""

CONTROL_HZ: Final[float] = 20.0
"""LIBERO の制御周波数。"""

MAX_STEPS_BY_SUITE: Final[dict[str, int]] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
"""タスクスイートごとの最大ステップ数 (openpi の評価スクリプトと同じ)。"""

MISSING_DEPENDENCY_HINT: Final[str] = (
    "このスクリプトは openpi と LIBERO を必要とします。"
    "openpi 側の環境 (LIBERO をインストール済み) で実行してください。"
    "パッケージ本体 `esn_vla_uq` はこれらに依存しません。"
)


@dataclasses.dataclass
class EpisodeRecord:
    """1 エピソード分の記録バッファ。"""

    episode_id: str
    task_name: str
    states: list[NDArray[np.float64]] = dataclasses.field(default_factory=list)
    actions: list[NDArray[np.float64]] = dataclasses.field(default_factory=list)
    chunks: list[NDArray[np.float64]] = dataclasses.field(default_factory=list)
    inference_steps: list[int] = dataclasses.field(default_factory=list)
    object_states: list[NDArray[np.float64]] = dataclasses.field(default_factory=list)
    success: bool = False

    def n_steps(self) -> int:
        """記録したステップ数。"""
        return len(self.states)

    def save(self, episodes_dir: Path) -> None:
        """エピソードを npz として書き出す。"""
        episodes_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            episodes_dir / f"{self.episode_id}.npz",
            state=np.asarray(self.states, dtype=np.float32),
            action=np.asarray(self.actions, dtype=np.float32),
            action_chunk=np.asarray(self.chunks, dtype=np.float32),
            inference_steps=np.asarray(self.inference_steps, dtype=np.int64),
            object_state=np.asarray(self.object_states, dtype=np.float32),
        )


def build_parser() -> argparse.ArgumentParser:
    """引数パーサを組み立てる。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial")
    parser.add_argument("--num-trials-per-task", type=int, default=10)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--replan-steps", type=int, default=DEFAULT_REPLAN_STEPS)
    parser.add_argument("--resize-size", type=int, default=DEFAULT_RESIZE_SIZE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--policy-label",
        type=str,
        default=None,
        help=(
            "ポリシー名を明示指定する。既定では policy server が申告する "
            "メタデータから取る (推奨。取り違えを防ぐ)"
        ),
    )
    return parser


def write_manifest(
    output_dir: Path,
    records: list[EpisodeRecord],
    *,
    args: argparse.Namespace,
    state_dim: int,
    action_dim: int,
    chunk_horizon: int,
    policy: str,
    server_metadata: dict[str, object],
    object_state_dim: int,
    object_keys: list[str],
) -> Path:
    """マニフェストを書き出す。

    Args:
        policy: ポリシー名。policy server が申告した値を優先する。
        server_metadata: policy server が接続時に送ってくるメタデータ全体。
            **加工せずそのまま残す**。何が配信されていたかを後から検証できる
            唯一の記録であり、こちらで解釈して削ると取り違えを検出できなくなる。
    """
    manifest = {
        "schema_version": LOG_SCHEMA_VERSION,
        "policy": policy,
        "server_metadata": server_metadata,
        "task_suite": str(args.task_suite_name),
        "seed": int(args.seed),
        "control_hz": CONTROL_HZ,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "object_state_dim": object_state_dim,
        "object_keys": object_keys,
        "chunk_horizon": chunk_horizon,
        "replan_steps": int(args.replan_steps),
        "episodes": [
            {
                "episode_id": record.episode_id,
                "task_name": record.task_name,
                "success": record.success,
                "n_steps": record.n_steps(),
            }
            for record in records
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """収集を実行する。"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = build_parser().parse_args(argv)

    # openpi / LIBERO はこの関数の中でだけ使う。入っていない環境で import しても
    # 「何をすればよいか」が分かるメッセージにする。openpi の評価ループ
    # (examples/libero/main.py) と同じ順序・同じ前処理をなぞる。
    try:
        from collections import deque

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools, websocket_client_policy
    except ImportError as error:
        raise ImportError(MISSING_DEPENDENCY_HINT) from error

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = MAX_STEPS_BY_SUITE[args.task_suite_name]
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    # **ポリシー名は利用者の申告ではなくサーバの申告を使う。** 実際に
    # `serve_policy.py --env LIBERO` が配信するのは pi05_libero だが、以前は
    # コマンドラインの既定値 "pi0_libero" をそのまま記録しており、収集ログの
    # 出自が事実と食い違っていた (chunk_horizon が 10 だったことから発覚)。
    server_metadata = dict(client.get_server_metadata())
    policy = _resolve_policy_name(args, server_metadata)
    logger.info("policy server: policy=%s metadata=%s", policy, server_metadata)

    records: list[EpisodeRecord] = []
    chunk_horizon = 0
    # 物体の状態は `object-state` に連結されて返る。個々のキー名も残しておくと、
    # 後から「どの物体か」を辿れる。
    object_keys: list[str] = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_bddl = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl), camera_heights=256, camera_widths=256
        )
        env.seed(args.seed)

        for episode_index in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[episode_index])
            if not object_keys:
                object_keys = sorted(
                    k
                    for k in obs
                    if k.endswith(("_pos", "_quat")) and "robot0" not in k
                )
            record = EpisodeRecord(
                episode_id=f"openpi_{task_id:03d}_{episode_index:03d}",
                task_name=str(task.language),
            )
            action_plan: deque[NDArray[np.float64]] = deque()
            step = 0
            while step < max_steps + DEFAULT_NUM_STEPS_WAIT:
                if step < DEFAULT_NUM_STEPS_WAIT:
                    obs, _reward, done, _info = env.step([0.0] * 6 + [-1.0])
                    step += 1
                    continue

                state: NDArray[np.float64] = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                if not action_plan:
                    element = {
                        "observation/image": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                np.ascontiguousarray(
                                    obs["agentview_image"][::-1, ::-1]
                                ),
                                args.resize_size,
                                args.resize_size,
                            )
                        ),
                        "observation/wrist_image": image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                np.ascontiguousarray(
                                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                                ),
                                args.resize_size,
                                args.resize_size,
                            )
                        ),
                        "observation/state": state,
                        "prompt": str(task.language),
                    }
                    action_chunk: NDArray[np.float64] = np.asarray(
                        client.infer(element)["actions"], dtype=np.float64
                    )
                    # 実行するのは先頭 replan_steps だけだが、チャンク全体を残す。
                    record.chunks.append(action_chunk)
                    record.inference_steps.append(record.n_steps())
                    chunk_horizon = int(action_chunk.shape[0])
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                record.states.append(state)
                record.actions.append(np.asarray(action, dtype=np.float64))
                record.object_states.append(
                    np.asarray(obs["object-state"], dtype=np.float64)
                )
                obs, _reward, done, _info = env.step(np.asarray(action).tolist())
                if done:
                    record.success = True
                    break
                step += 1

            if record.n_steps() > 0:
                record.save(Path(args.output_dir) / "episodes")
                records.append(record)
                logger.info(
                    "episode done: id=%s success=%s n_steps=%d n_inference=%d",
                    record.episode_id,
                    record.success,
                    record.n_steps(),
                    len(record.inference_steps),
                )
        env.close()

    if not records:
        logger.error("エピソードを 1 件も収集できませんでした")
        return 1

    path = write_manifest(
        Path(args.output_dir),
        records,
        args=args,
        state_dim=int(records[0].states[0].shape[0]),
        action_dim=int(records[0].actions[0].shape[0]),
        chunk_horizon=chunk_horizon,
        policy=policy,
        server_metadata=server_metadata,
        object_state_dim=(
            int(records[0].object_states[0].shape[0]) if records[0].object_states else 0
        ),
        object_keys=object_keys,
    )
    n_success = sum(1 for record in records if record.success)
    logger.info(
        "collected: n_episodes=%d n_success=%d manifest=%s",
        len(records),
        n_success,
        path.name,
    )
    return 0


def _resolve_policy_name(
    args: argparse.Namespace, server_metadata: dict[str, object]
) -> str:
    """記録するポリシー名を決める。

    `--policy-label` が明示されていればそれを使い、無ければサーバのメタデータ
    から探す。どちらも無ければ `"unknown"` にする。**推測で名前を埋めない。**
    """
    label = getattr(args, "policy_label", None)
    if isinstance(label, str) and label:
        return label
    for key in ("policy_name", "config", "name", "policy"):
        value = server_metadata.get(key)
        if isinstance(value, str) and value:
            return value
    logger.warning(
        "policy server がポリシー名を申告しませんでした。"
        "manifest には 'unknown' を記録します (--policy-label で明示できます)"
    )
    return "unknown"


def _quat2axisangle(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """クォータニオンを軸角表現へ変換する (openpi の実装と同じ)。

    openpi は `examples/libero/main.py` にこの関数を持つ。ここで再実装するのは
    openpi をこのスクリプトの import 依存に加えないため (収集時に必要なのは
    policy server との通信と LIBERO 環境だけにしたい)。
    """
    quaternion: NDArray[np.float64] = np.asarray(quat, dtype=np.float64).copy()
    quaternion[3] = float(np.clip(quaternion[3], -1.0, 1.0))
    density = float(np.sqrt(1.0 - quaternion[3] * quaternion[3]))
    if np.isclose(density, 0.0):
        return np.zeros(3, dtype=np.float64)
    axis_angle: NDArray[np.float64] = (
        quaternion[:3] * 2.0 * float(np.arccos(quaternion[3]))
    ) / density
    return axis_angle


if __name__ == "__main__":
    raise SystemExit(main())
