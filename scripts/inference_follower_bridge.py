"""
Use case 4: 컨테이너의 GR00T 정책 서버를 사용해 Windows 호스트에 연결된 SO-101 Follower 암을 원격 제어한다.

흐름:
    1. lerobot 의 SO101Follower 로 follower 시리얼 + 카메라 연결
    2. follower.get_observation() 으로 joint state + 카메라 프레임 동시 획득
    3. 컨테이너 gr00t-server (ZMQ REQ-REP @ tcp://<host>:5555) 에 GR00T N1.6 obs 형식으로 요청
    4. action chunk (16 step) 를 받아 follower.send_action() 으로 모터 적용

전제:
    * Windows 호스트 .venv (Python 3.11) 에서 `uv run` 으로 실행
    * gr00t-server 컨테이너가 5555 포트로 listen 중 (`docker compose up -d gr00t-server`)
    * SO-101 follower 가 시리얼 포트(예: COM8) 에 연결
    * 카메라 2 대 (front, wrist) 가 PC USB 포트에 직접 연결 (USB 허브 금지 — README 의 카메라 대역폭 섹션 참조)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from leisaac.policy.base import ZMQServicePolicy

# SO-101 의 lerobot 측 모터 순서 (so101_follower.py 의 motors dict 와 동일)
LEROBOT_MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def build_obs_dict(
    state_vec: np.ndarray,
    front_img: np.ndarray,
    wrist_img: np.ndarray,
    task_description: str,
) -> dict:
    """GR00T N1.6 서버가 받는 obs 형식 구성.

    state_vec: shape (6,) — lerobot 순서. single_arm[5], gripper[1] 로 분할된다.
    front_img / wrist_img: (H, W, 3) uint8 RGB.
    """
    state_vec = state_vec.astype(np.float32)
    return {
        "observation": {
            "video": {
                "front": front_img[None, None, ...].astype(np.uint8),
                "wrist": wrist_img[None, None, ...].astype(np.uint8),
            },
            "state": {
                "single_arm": state_vec[:5].reshape(1, 1, 5),
                "gripper":    state_vec[5:6].reshape(1, 1, 1),
            },
            "language": {
                "annotation.human.task_description": [[task_description]],
            },
        }
    }


def parse_action_chunk(action_chunk: list | dict) -> np.ndarray:
    """GR00T 서버 응답을 (horizon, 6) 형태로 정규화."""
    if isinstance(action_chunk, list):
        action_chunk = action_chunk[0]
    single_arm = np.asarray(action_chunk["single_arm"]).squeeze(0)  # (T, 5)
    gripper    = np.asarray(action_chunk["gripper"]).squeeze(0)     # (T, 1)
    return np.concatenate([single_arm, gripper], axis=-1)            # (T, 6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy_host", default="localhost")
    ap.add_argument("--policy_port", type=int, default=5555)
    ap.add_argument("--policy_timeout_ms", type=int, default=15000)
    ap.add_argument("--follower_port", default="COM8")
    ap.add_argument("--follower_id", default="follower_arm")
    ap.add_argument("--front_cam_index", type=int, default=0)
    ap.add_argument("--wrist_cam_index", type=int, default=1)
    ap.add_argument("--cam_width",  type=int, default=640)
    ap.add_argument("--cam_height", type=int, default=480)
    ap.add_argument("--cam_fps",    type=int, default=30)
    ap.add_argument(
        "--task",
        default="Pick up the orange and place it on the plate",
        help="GR00T 정책에 전달할 자연어 task description.",
    )
    ap.add_argument("--step_hz", type=int, default=30)
    ap.add_argument("--max_seconds", type=float, default=0.0,
                    help="0 = 무한 루프, 양수 = 해당 초 후 자동 종료.")
    args = ap.parse_args()

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=args.front_cam_index,
            width=args.cam_width, height=args.cam_height, fps=args.cam_fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam_index,
            width=args.cam_width, height=args.cam_height, fps=args.cam_fps,
        ),
    }
    follower = SO101Follower(
        SO101FollowerConfig(port=args.follower_port, id=args.follower_id, cameras=cameras)
    )
    follower.connect(calibrate=True)

    print(f"[bridge] connecting to gr00t-server tcp://{args.policy_host}:{args.policy_port}")
    policy = ZMQServicePolicy(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
        ping_endpoint="ping",
    )

    period = 1.0 / args.step_hz
    started = time.time()
    iteration = 0

    try:
        while True:
            t_loop = time.time()
            obs = follower.get_observation()

            # joint state (lerobot 순서) → numpy
            state_vec = np.asarray(
                [obs[f"{m}.pos"] for m in LEROBOT_MOTOR_ORDER], dtype=np.float32,
            )

            # 카메라 프레임 (lerobot async_read 는 RGB ndarray 반환)
            front = np.asarray(obs["front"])
            wrist = np.asarray(obs["wrist"])

            obs_dict = build_obs_dict(state_vec, front, wrist, args.task)
            action_chunk = parse_action_chunk(policy.call_endpoint("get_action", obs_dict))

            # action_chunk: (horizon, 6) — 시점별로 모터에 적용
            for step_action in action_chunk:
                action_payload = {f"{m}.pos": float(v) for m, v in zip(LEROBOT_MOTOR_ORDER, step_action)}
                follower.send_action(action_payload)
                # rate-limit 모터 적용
                slack = period - (time.time() - t_loop)
                if slack > 0: time.sleep(slack)
                t_loop = time.time()

            iteration += 1
            if args.max_seconds > 0 and (time.time() - started) >= args.max_seconds:
                print(f"[bridge] reached --max_seconds={args.max_seconds}, exiting after {iteration} chunks")
                break

    except KeyboardInterrupt:
        print("\n[bridge] interrupted, cleaning up …")
    finally:
        try:
            follower.disconnect()
        except Exception as e:
            print(f"[bridge] follower.disconnect failed: {e}")


if __name__ == "__main__":
    main()
