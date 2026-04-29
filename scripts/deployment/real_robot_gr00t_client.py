"""
Real robot GR00T closed-loop client.

H100 서버에서 GR00T 정책 서버를 띄우고, 이 스크립트가 실행되는 Windows PC의
SO101 Follower Arm + 카메라로부터 observation을 수집한 뒤, ZMQ REQ로 정책
서버에 전달하고 받은 action을 실제 암에 적용한다.

사전 조건:
  - H100 서버에서 run_gr00t_server.py 가 0.0.0.0:5555 에 바인딩되어 있어야 함
  - SO101 Follower Arm이 Windows COM 포트에 연결되어 있어야 함 (기본 COM7)
  - 카메라 2개 (front, wrist)가 OpenCV 인덱스로 연결되어 있어야 함

실행 예시 (Windows PowerShell):
  uv run python scripts/deployment/real_robot_gr00t_client.py `
    --policy_host 10.10.40.254 `
    --robot_port COM7 `
    --camera_front_id 0 `
    --camera_wrist_id 1 `
    --task_description "Grab orange and place into plate"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Isaac-GR00T의 PolicyClient를 임포트하기 위해 sys.path에 추가한다.
# gr00t 패키지는 Isaac-GR00T/ 하위에 있고, 별도 설치 없이 소스 경로를 추가한다.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "Isaac-GR00T"))

from gr00t.policy.server_client import PolicyClient  # noqa: E402

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

# SO101 모터 이름 순서 (leisaac/assets/robots/lerobot.py 와 동일하게 유지)
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# 실제 모터 안전 한계 (motor normalized range, SO101_FOLLOWER_MOTOR_LIMITS 기준)
MOTOR_LIMITS = {
    "shoulder_pan":  (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex":    (-100.0, 100.0),
    "wrist_flex":    (-100.0, 100.0),
    "wrist_roll":    (-100.0, 100.0),
    "gripper":       (  0.0, 100.0),
}


def build_observation(obs: dict, task_description: str) -> dict:
    """SO101Follower.get_observation() 결과를 GR00T PolicyClient 형식으로 변환."""
    # get_observation()은 카메라 키(설정한 이름)를 사용해 (H, W, 3) uint8 RGB 이미지를 반환.
    front = obs["front"]  # (H, W, 3)
    wrist = obs["wrist"]  # (H, W, 3)

    # 모터 state: single_arm(5) + gripper(1)
    state = np.array(
        [obs[f"{m}.pos"] for m in MOTOR_NAMES],
        dtype=np.float32,
    )

    return {
        "video": {
            "front": front[None, None].astype(np.uint8),   # (1, 1, H, W, 3)
            "wrist": wrist[None, None].astype(np.uint8),
        },
        "state": {
            "single_arm": state[None, None, :5],  # (1, 1, 5)
            "gripper":    state[None, None, 5:6], # (1, 1, 1)
        },
        "language": {
            "annotation.human.task_description": [[task_description]],
        },
    }


def clip_action(action_vec: np.ndarray) -> np.ndarray:
    """각 조인트 action을 모터 한계 안으로 클램핑한다."""
    clipped = action_vec.copy()
    for i, name in enumerate(MOTOR_NAMES):
        lo, hi = MOTOR_LIMITS[name]
        clipped[i] = float(np.clip(clipped[i], lo, hi))
    return clipped


def parse_args():
    p = argparse.ArgumentParser(
        description="Real robot GR00T closed-loop client — H100 server + Windows PC arm"
    )
    p.add_argument(
        "--policy_host", required=True,
        help="GR00T 정책 서버가 실행 중인 H100의 IP 주소 (예: 192.168.1.100)"
    )
    p.add_argument("--policy_port", type=int, default=5555, help="GR00T 서버 ZMQ 포트 (기본값: 5555)")
    p.add_argument("--policy_timeout_ms", type=int, default=15000, help="ZMQ 응답 타임아웃 (ms)")
    p.add_argument(
        "--robot_port", default="COM7",
        help="SO101 Follower 직렬 포트 (Windows: COM7 등, Linux: /dev/ttyACM0)"
    )
    p.add_argument("--camera_front_id", type=int, default=0, help="Front 카메라 OpenCV 인덱스")
    p.add_argument("--camera_wrist_id", type=int, default=1, help="Wrist 카메라 OpenCV 인덱스")
    p.add_argument("--fps", type=float, default=30.0, help="제어 루프 목표 주파수")
    p.add_argument(
        "--action_horizon", type=int, default=16,
        help="GR00T가 한 번에 예측하는 step 수"
    )
    p.add_argument(
        "--exec_horizon", type=int, default=8,
        help="재예측 전에 실행할 step 수 (≤ action_horizon). 낮을수록 반응이 빠르지만 서버 부하 증가."
    )
    p.add_argument(
        "--max_relative_target", type=float, default=5.0,
        help="한 step에서 허용되는 최대 모터 이동량 (motor normalized units). 낮을수록 안전."
    )
    p.add_argument(
        "--no_calibrate", action="store_true",
        help="캘리브레이션 스킵 (기존 캘리브레이션 파일 재사용)"
    )
    p.add_argument(
        "--task_description",
        default="Grab orange and place into plate",
        help="언어 지시문 (학습 시 사용한 task 문구와 일치시킬 것)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.exec_horizon > args.action_horizon:
        raise ValueError(
            f"--exec_horizon ({args.exec_horizon}) must be ≤ --action_horizon ({args.action_horizon})"
        )

    # ── 1. GR00T 정책 서버 연결 ────────────────────────────────────────────────
    print(f"[1/2] Connecting to GR00T policy server at tcp://{args.policy_host}:{args.policy_port} ...")
    policy = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
    )
    if not policy.ping():
        print(
            f"ERROR: Cannot reach GR00T server at {args.policy_host}:{args.policy_port}\n"
            "H100 서버에서 run_gr00t_server.py 가 실행 중이고 포트 5555가 열려 있는지 확인하세요.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("       GR00T policy server connected.")

    # ── 2. SO101 Follower Arm + 카메라 연결 ────────────────────────────────────
    print(f"[2/2] Connecting SO101 Follower on {args.robot_port} ...")
    robot_config = SO101FollowerConfig(
        port=args.robot_port,
        max_relative_target=args.max_relative_target,
        cameras={
            "front": OpenCVCameraConfig(
                index=args.camera_front_id,
                fps=int(args.fps),
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                index=args.camera_wrist_id,
                fps=int(args.fps),
                width=640,
                height=480,
            ),
        },
    )
    robot = SO101Follower(config=robot_config)
    robot.connect(calibrate=not args.no_calibrate)
    print(f"       SO101 Follower connected. (calibrated={robot.is_calibrated})")

    dt = 1.0 / args.fps
    print(
        f"\n>> Closed-loop inference started\n"
        f"   policy_host    : {args.policy_host}:{args.policy_port}\n"
        f"   task           : \"{args.task_description}\"\n"
        f"   action_horizon : {args.action_horizon}  exec_horizon : {args.exec_horizon}\n"
        f"   fps            : {args.fps}  max_relative_target : {args.max_relative_target}\n"
        f"   Press Ctrl+C to stop.\n"
    )

    step_count = 0
    try:
        # ── 3. 제어 루프 ──────────────────────────────────────────────────────
        while True:
            loop_start = time.perf_counter()

            # 3a. SO101에서 카메라 + 관절 상태 읽기
            obs = robot.get_observation()
            obs_dict = build_observation(obs, args.task_description)

            # 3b. H100 서버에 ZMQ REQ 전송 → action_horizon 예측값 수신
            t_inf = time.perf_counter()
            action_dict, _ = policy.get_action(obs_dict)
            inf_ms = (time.perf_counter() - t_inf) * 1e3

            # 3c. action 디코딩: (1, action_horizon, dim) → (action_horizon, dim)
            single_arm = action_dict["single_arm"][0]  # (action_horizon, 5)
            gripper    = action_dict["gripper"][0]     # (action_horizon, 1)
            actions = np.concatenate([single_arm, gripper], axis=-1)  # (action_horizon, 6)

            # 3d. exec_horizon 스텝 실행 후 재예측
            for t in range(args.exec_horizon):
                step_start = time.perf_counter()

                action_vec = clip_action(actions[t])
                cmd = {
                    f"{name}.pos": float(action_vec[i])
                    for i, name in enumerate(MOTOR_NAMES)
                }
                robot.send_action(cmd)
                step_count += 1

                elapsed = time.perf_counter() - step_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            total_loop_ms = (time.perf_counter() - loop_start) * 1e3
            print(
                f"  step={step_count:5d} | inf={inf_ms:5.0f}ms | "
                f"loop={total_loop_ms:5.0f}ms | "
                f"gripper={float(actions[0, -1]):6.1f}"
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        robot.disconnect()
        print("SO101 Follower disconnected.")


if __name__ == "__main__":
    main()
