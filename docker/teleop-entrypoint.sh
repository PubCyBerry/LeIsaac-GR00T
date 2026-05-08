#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — LeRobot 0.4.2 SO-101 Container Entry
#
# ■ 실행 모드 (CMD 첫 번째 인자)
#   teleop          : lerobot-teleoperate  — 실시간 원격 조작
#   record          : lerobot-record       — 텔레옵 기반 데이터 수집
#   replay          : lerobot-replay       — 녹화 에피소드 재실행
#   calibrate       : lerobot-calibrate    — 로봇/텔레옵 보정
#   setup-motors    : lerobot-setup-motors — 모터 ID/Baud 초기 설정
#   find-joint-limits: lerobot-find-joint-limits — 관절 가동 범위 탐색
#   find-cameras    : lerobot-find-cameras — 연결된 카메라 검색 및 캡처 확인
#   find-port       : lerobot-find-port    — MotorsBus USB 포트 자동 감지
#   dataset-viz     : lerobot-dataset-viz  — 데이터셋 시각화 (Rerun)
#   train           : lerobot-train        — Policy 학습 (인자 완전 위임)
#   eval            : lerobot-eval         — Policy 평가 (인자 완전 위임)
#   edit-dataset    : lerobot-edit-dataset — 데이터셋 편집 (인자 완전 위임)
#   info            : lerobot-info         — LeRobot/시스템 정보 출력
#   bash | shell    : 인터랙티브 Bash 쉘
#   python <args>   : python 직접 실행
#   <기타>           : 명령 그대로 exec
#
# ■ 환경 변수 요약 (docker-compose.yaml ↔ .env 에서 주입)
#   하드웨어 : LEADER_PORT  LEADER_ID  FOLLOWER_PORT  FOLLOWER_ID
#             BELLY_CAM_DEV  WRIST_CAM_DEV  BELLY_CAM_INDEX  WRIST_CAM_INDEX
#             CAM_WIDTH  CAM_HEIGHT  CAM_FPS
#   record  : HF_DATASET_REPO_ID  SINGLE_TASK  NUM_EPISODES
#             EPISODE_TIME_S  RESET_TIME_S  RECORD_FPS  PUSH_TO_HUB
#             RECORD_EXTRA_ARGS
#   replay  : HF_DATASET_REPO_ID  EPISODE_INDEX  REPLAY_EXTRA_ARGS
#   기기유틸: ROBOT_TYPE  TELEOP_TYPE  CALIBRATE_TARGET  TELEOP_TIME_S
#   viz     : HF_DATASET_REPO_ID  EPISODE_INDEX  VIZ_MODE  VIZ_WS_PORT
# =============================================================================
set -euo pipefail

# ── 하드웨어 환경 변수 기본값 ──────────────────────────────────────────────────
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
LEADER_ID="${LEADER_ID:-konan_teleop}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-konan_robot}"
BELLY_CAM_DEV="${BELLY_CAM_DEV:-/dev/video0}"
WRIST_CAM_DEV="${WRIST_CAM_DEV:-/dev/video2}"
BELLY_CAM_INDEX="${BELLY_CAM_INDEX:-0}"
WRIST_CAM_INDEX="${WRIST_CAM_INDEX:-2}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
CAM_FPS="${CAM_FPS:-25}"

# ── record 환경 변수 ────────────────────────────────────────────────────────
# HF_DATASET_REPO_ID: '{hf_username}/{dataset_name}' 형식 (필수 — 비어 있으면 오류)
DATASET_REPO_ID="${HF_DATASET_REPO_ID:-}"
# 에피소드당 작업 설명 (자유 문자열)
SINGLE_TASK="${SINGLE_TASK:-pick and place}"
# 수집할 에피소드 수
NUM_EPISODES="${NUM_EPISODES:-50}"
# 에피소드 1회 녹화 시간(초)
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
# 에피소드 간 환경 초기화 대기 시간(초)
RESET_TIME_S="${RESET_TIME_S:-30}"
# 데이터셋 저장 fps (teleop fps 와 독립적으로 설정 가능)
RECORD_FPS="${RECORD_FPS:-30}"
# HuggingFace Hub 업로드 여부 (true / false)
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
# 로컬 저장 루트 디렉터리 (비어 있으면 HF 캐시 폴더 사용)
DATASET_ROOT="${DATASET_ROOT:-/workspace/data}"
# 추가 lerobot-record 인자 (예: "--dataset.video=false --resume=true")
RECORD_EXTRA_ARGS="${RECORD_EXTRA_ARGS:-}"

# ── replay 환경 변수 ────────────────────────────────────────────────────────
# 재생할 에피소드 인덱스 (0-based)
EPISODE_INDEX="${EPISODE_INDEX:-0}"
# 추가 lerobot-replay 인자
REPLAY_EXTRA_ARGS="${REPLAY_EXTRA_ARGS:-}"

# ── 기기 유틸 공통 환경 변수 ──────────────────────────────────────────────────
# 로봇 타입 (calibrate / setup-motors / find-joint-limits 에서 사용)
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
# 텔레옵 타입 (calibrate / find-joint-limits 에서 사용)
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
# calibrate / setup-motors 대상: "robot" 또는 "teleop"
CALIBRATE_TARGET="${CALIBRATE_TARGET:-robot}"
# find-joint-limits 텔레옵 조작 시간(초)
TELEOP_TIME_S="${TELEOP_TIME_S:-30}"

# ── dataset-viz 환경 변수 ───────────────────────────────────────────────────
# 시각화 모드: local | distant (distant = WebSocket 서버 모드)
VIZ_MODE="${VIZ_MODE:-local}"
# distant 모드 WebSocket 포트
VIZ_WS_PORT="${VIZ_WS_PORT:-9087}"

# ── 레거시 (teleop 전용, 하위 호환) ────────────────────────────────────────
TELEOP_EXTRA_ARGS="${TELEOP_EXTRA_ARGS:-}"

# ── 색상 출력 유틸 ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 직렬 포트 존재 확인 ───────────────────────────────────────────────────────
check_port() {
    local port="$1" label="$2"
    if [[ ! -e "$port" ]]; then
        error "${label} 포트 '${port}' 를 찾을 수 없습니다."
        error "  → PowerShell: usbipd attach --wsl --busid <BUS_ID>"
        error "  → WSL2:       ls /dev/ttyACM*"
        return 1
    fi
    if [[ ! -r "$port" || ! -w "$port" ]]; then
        warn "${label} 포트 '${port}' 읽기/쓰기 권한 없음 (--privileged 확인)"
    else
        info "${label} 직렬 포트 OK: ${port}"
    fi
}

# ── 카메라 디바이스 존재 및 권한 확인 ────────────────────────────────────────
check_camera() {
    local dev="$1" label="$2"
    if [[ ! -e "$dev" ]]; then
        warn "${label} 디바이스 '${dev}' 를 찾을 수 없습니다."
        warn "  → PowerShell: usbipd attach --wsl --busid <BUS_ID>"
        warn "  카메라 없이 텔레옵은 동작하지만 데이터 수집 시 이미지 누락됩니다."
        return 0
    fi
    if [[ ! -r "$dev" ]]; then
        warn "${label} 디바이스 '${dev}' 읽기 권한 없음 (--privileged 확인)"
    else
        if command -v v4l2-ctl &>/dev/null; then
            local card
            card=$(v4l2-ctl -d "$dev" --info 2>/dev/null | grep "Card type" | sed 's/.*: //' | xargs)
            info "${label} 카메라 OK: ${dev} — ${card:-unknown}"
        else
            info "${label} 카메라 OK: ${dev}"
        fi
    fi
}

# ── HF_DATASET_REPO_ID 존재 확인 ──────────────────────────────────────────────
check_dataset_repo_id() {
    if [[ -z "$DATASET_REPO_ID" ]]; then
        error "HF_DATASET_REPO_ID 가 설정되지 않았습니다."
        error "  → .env:           HF_DATASET_REPO_ID=your_username/dataset_name"
        error "  → docker compose: -e HF_DATASET_REPO_ID=your_username/dataset_name"
        exit 1
    fi
    info "Dataset repo ID: ${DATASET_REPO_ID}"
}

# ── GPU 확인 ──────────────────────────────────────────────────────────────────
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        info "NVIDIA GPU 감지됨:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | \
            while IFS= read -r line; do info "  GPU: $line"; done
    else
        warn "nvidia-smi 를 찾을 수 없습니다. CPU 전용으로 실행됩니다."
    fi
}

# ── LeRobot 버전 및 CLI entry point 확인 ──────────────────────────────────────
check_lerobot() {
    local ver
    ver=$(python -c "import lerobot; print(lerobot.__version__)" 2>/dev/null || echo "unknown")
    info "LeRobot 버전: ${ver}"

    if ! command -v lerobot-teleoperate &>/dev/null; then
        error "lerobot CLI entry point 를 찾을 수 없습니다."
        error "  /opt/venv/bin/lerobot-* 가 존재하는지 확인하세요."
        exit 1
    fi
}

# ── 메인 ──────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  LeRobot 0.4.2 SO-101 Container"
echo "========================================================"

check_gpu
check_lerobot

CMD="${1:-teleop}"

case "$CMD" in

  # ────────────────────────────────────────────────────────────────────────────
  # teleop — 실시간 원격 조작
  #
  # [env var → CLI arg 매핑]
  #   FOLLOWER_PORT  → --robot.port
  #   FOLLOWER_ID    → --robot.id
  #   LEADER_PORT    → --teleop.port
  #   LEADER_ID      → --teleop.id
  #   WRIST_CAM_DEV  → --robot.cameras wrist index_or_path
  #   BELLY_CAM_DEV  → --robot.cameras belly index_or_path
  #   CAM_WIDTH/HEIGHT/FPS → cameras 해상도·FPS
  #
  # [주요 CLI 인자 — TELEOP_EXTRA_ARGS 로 추가 전달 가능]
  #   --robot.type=so101_follower   : 팔로워 로봇 타입
  #   --robot.port=<path>           : 팔로워 직렬 포트
  #   --robot.id=<str>              : 팔로워 ID (캘리브레이션 파일명)
  #   --robot.cameras=<json>        : 카메라 설정 (type/index_or_path/width/height/fps)
  #   --teleop.type=so101_leader    : 리더 텔레옵 타입
  #   --teleop.port=<path>          : 리더 직렬 포트
  #   --teleop.id=<str>             : 리더 ID
  #   --fps=<int>                   : 제어 주기 (기본 60)
  #   --teleop_time_s=<float>       : 최대 동작 시간(초, 기본 무제한)
  #   --display_data=true|false     : 카메라 영상 화면 출력 여부
  # ────────────────────────────────────────────────────────────────────────────
  teleop)
    info "── 장치 점검 ─────────────────────────────────────"
    check_port   "$LEADER_PORT"   "Leader Arm"
    check_port   "$FOLLOWER_PORT" "Follower Arm"
    check_camera "$BELLY_CAM_DEV" "BELLY"
    check_camera "$WRIST_CAM_DEV" "WRIST"

    info "── Teleoperation 시작 ────────────────────────────"
    info "  Leader   → ID: ${LEADER_ID}, PORT: ${LEADER_PORT}"
    info "  Follower → ID: ${FOLLOWER_ID}, PORT: ${FOLLOWER_PORT}"
    info "  belly cam → ${BELLY_CAM_DEV} (index=${BELLY_CAM_INDEX})"
    info "  wrist cam → ${WRIST_CAM_DEV} (index=${WRIST_CAM_INDEX})"

    lerobot-teleoperate \
      --robot.type=so101_follower \
      --robot.port=${FOLLOWER_PORT} \
      --robot.cameras="{
          wrist: {type: opencv, index_or_path: ${WRIST_CAM_DEV}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}},
          belly: {type: opencv, index_or_path: ${BELLY_CAM_DEV}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}},
          }" \
      --robot.id=${FOLLOWER_ID} \
      --teleop.type=so101_leader \
      --teleop.port=${LEADER_PORT} \
      --teleop.id=${LEADER_ID} \
      ${TELEOP_EXTRA_ARGS}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # record — 텔레오퍼레이션 기반 데이터 수집
  #
  # [env var → CLI arg 매핑]
  #   FOLLOWER_PORT/ID        → --robot.port / --robot.id
  #   LEADER_PORT/ID          → --teleop.port / --teleop.id
  #   WRIST/BELLY_CAM_DEV     → --robot.cameras (teleop 와 동일)
  #   CAM_WIDTH/HEIGHT/FPS    → cameras 해상도·FPS
  #   HF_DATASET_REPO_ID      → --dataset.repo_id        (필수)
  #   SINGLE_TASK             → --dataset.single_task
  #   NUM_EPISODES            → --dataset.num_episodes
  #   EPISODE_TIME_S          → --dataset.episode_time_s
  #   RESET_TIME_S            → --dataset.reset_time_s
  #   RECORD_FPS              → --dataset.fps
  #   PUSH_TO_HUB             → --dataset.push_to_hub
  #   DATASET_ROOT            → --dataset.root
  #
  # [주요 CLI 인자 전체 — RECORD_EXTRA_ARGS 로 추가 전달 가능]
  #   --robot.type=so101_follower
  #   --robot.port=<path>
  #   --robot.id=<str>
  #   --robot.cameras=<json>
  #   --teleop.type=so101_leader
  #   --teleop.port=<path>
  #   --teleop.id=<str>
  #   --dataset.repo_id=<str>           : '{hf_user}/{name}' (필수)
  #   --dataset.single_task=<str>       : 에피소드 작업 설명
  #   --dataset.root=<path>             : 로컬 저장 루트 (기본: HF 캐시)
  #   --dataset.fps=<int>               : 저장 fps (기본 30)
  #   --dataset.episode_time_s=<int>    : 에피소드 녹화 시간(초, 기본 60)
  #   --dataset.reset_time_s=<int>      : 초기화 대기 시간(초, 기본 60)
  #   --dataset.num_episodes=<int>      : 수집할 에피소드 수 (기본 50)
  #   --dataset.video=true|false        : 비디오 인코딩 여부 (기본 true)
  #   --dataset.push_to_hub=true|false  : HF Hub 업로드 여부 (기본 true)
  #   --dataset.private=true|false      : Hub 비공개 업로드 (기본 false)
  #   --dataset.tags=<list>             : Hub 태그
  #   --dataset.num_image_writer_processes=<int>   : 이미지 저장 프로세스 수 (기본 0=스레드)
  #   --dataset.num_image_writer_threads_per_camera=<int> : 카메라당 스레드 수 (기본 4)
  #   --dataset.video_encoding_batch_size=<int>    : 배치 인코딩 단위 (기본 1)
  #   --dataset.rename_map=<dict>       : 관측 키 이름 변경 매핑
  #   --display_data=true|false         : 카메라 영상 화면 출력
  #   --play_sounds=true|false          : 음성 이벤트 알림 (기본 true)
  #   --resume=true|false               : 기존 데이터셋에 이어서 수집 (기본 false)
  # ────────────────────────────────────────────────────────────────────────────
  record)
    info "── 장치 점검 ─────────────────────────────────────"
    check_port   "$LEADER_PORT"   "Leader Arm"
    check_port   "$FOLLOWER_PORT" "Follower Arm"
    check_camera "$BELLY_CAM_DEV" "BELLY"
    check_camera "$WRIST_CAM_DEV" "WRIST"
    check_dataset_repo_id

    info "── Record 시작 ───────────────────────────────────"
    info "  Leader   → ID: ${LEADER_ID}, PORT: ${LEADER_PORT}"
    info "  Follower → ID: ${FOLLOWER_ID}, PORT: ${FOLLOWER_PORT}"
    info "  Dataset  → ${DATASET_REPO_ID} (${NUM_EPISODES} episodes, ${EPISODE_TIME_S}s each)"
    info "  Task     → ${SINGLE_TASK}"

    lerobot-record \
      --robot.type=${ROBOT_TYPE} \
      --robot.port=${FOLLOWER_PORT} \
      --robot.id=${FOLLOWER_ID} \
      --robot.cameras="{
          wrist: {type: opencv, index_or_path: ${WRIST_CAM_DEV}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}},
          belly: {type: opencv, index_or_path: ${BELLY_CAM_DEV}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}},
          }" \
      --teleop.type=${TELEOP_TYPE} \
      --teleop.port=${LEADER_PORT} \
      --teleop.id=${LEADER_ID} \
      --dataset.repo_id=${DATASET_REPO_ID} \
      --dataset.single_task="${SINGLE_TASK}" \
      --dataset.root=${DATASET_ROOT} \
      --dataset.fps=${RECORD_FPS} \
      --dataset.episode_time_s=${EPISODE_TIME_S} \
      --dataset.reset_time_s=${RESET_TIME_S} \
      --dataset.num_episodes=${NUM_EPISODES} \
      --dataset.push_to_hub=${PUSH_TO_HUB} \
      --display_data=true \
      ${RECORD_EXTRA_ARGS}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # replay — 녹화된 에피소드를 로봇에서 재실행
  #
  # [env var → CLI arg 매핑]
  #   FOLLOWER_PORT/ID    → --robot.port / --robot.id
  #   HF_DATASET_REPO_ID  → --dataset.repo_id   (필수)
  #   EPISODE_INDEX       → --dataset.episode
  #   RECORD_FPS          → --dataset.fps
  #   DATASET_ROOT        → --dataset.root
  #
  # [주요 CLI 인자 전체 — REPLAY_EXTRA_ARGS 로 추가 전달 가능]
  #   --robot.type=so101_follower
  #   --robot.port=<path>
  #   --robot.id=<str>
  #   --dataset.repo_id=<str>    : '{hf_user}/{name}' (필수)
  #   --dataset.episode=<int>    : 재생할 에피소드 인덱스 (0-based, 기본 0)
  #   --dataset.root=<path>      : 로컬 저장 루트 (기본: HF 캐시)
  #   --dataset.fps=<int>        : 재생 fps (기본 30)
  #   --play_sounds=true|false   : 음성 이벤트 알림 (기본 true)
  # ────────────────────────────────────────────────────────────────────────────
  replay)
    info "── 장치 점검 ─────────────────────────────────────"
    check_port "$FOLLOWER_PORT" "Follower Arm"
    check_dataset_repo_id

    info "── Replay 시작 ───────────────────────────────────"
    info "  Follower → ID: ${FOLLOWER_ID}, PORT: ${FOLLOWER_PORT}"
    info "  Dataset  → ${DATASET_REPO_ID}, episode=${EPISODE_INDEX}"

    lerobot-replay \
      --robot.type=${ROBOT_TYPE} \
      --robot.port=${FOLLOWER_PORT} \
      --robot.id=${FOLLOWER_ID} \
      --dataset.repo_id=${DATASET_REPO_ID} \
      --dataset.episode=${EPISODE_INDEX} \
      --dataset.root=${DATASET_ROOT} \
      --dataset.fps=${RECORD_FPS} \
      ${REPLAY_EXTRA_ARGS}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # calibrate — 로봇 또는 텔레옵 보정
  #
  # [env var → 동작]
  #   CALIBRATE_TARGET=robot  (기본) → 팔로워 암 보정
  #   CALIBRATE_TARGET=teleop         → 리더 텔레옵 보정
  #
  # [주요 CLI 인자 전체]
  #   [robot 타깃]
  #     --robot.type=<str>     : 로봇 타입 (so101_follower, so100_follower, ...)
  #     --robot.port=<path>    : 직렬 포트
  #     --robot.id=<str>       : 캘리브레이션 파일 ID
  #   [teleop 타깃]
  #     --teleop.type=<str>    : 텔레옵 타입 (so101_leader, so100_leader, ...)
  #     --teleop.port=<path>   : 직렬 포트
  #     --teleop.id=<str>      : 캘리브레이션 파일 ID
  #
  # ※ robot 과 teleop 중 하나만 지정해야 함 (동시 지정 불가)
  # ────────────────────────────────────────────────────────────────────────────
  calibrate)
    if [[ "$CALIBRATE_TARGET" == "teleop" ]]; then
        info "── 장치 점검 ─────────────────────────────────────"
        check_port "$LEADER_PORT" "Leader Arm (teleop)"
        info "── Calibrate (teleop) 시작 ─────────────────────"
        info "  Teleop → ID: ${LEADER_ID}, PORT: ${LEADER_PORT}"
        lerobot-calibrate \
          --teleop.type=${TELEOP_TYPE} \
          --teleop.port=${LEADER_PORT} \
          --teleop.id=${LEADER_ID}
    else
        info "── 장치 점검 ─────────────────────────────────────"
        check_port "$FOLLOWER_PORT" "Follower Arm (robot)"
        info "── Calibrate (robot) 시작 ──────────────────────"
        info "  Robot → ID: ${FOLLOWER_ID}, PORT: ${FOLLOWER_PORT}"
        lerobot-calibrate \
          --robot.type=${ROBOT_TYPE} \
          --robot.port=${FOLLOWER_PORT} \
          --robot.id=${FOLLOWER_ID}
    fi
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # setup-motors — Feetech 모터 ID 및 Baud Rate 초기 설정
  #
  # [env var → 동작]
  #   CALIBRATE_TARGET=robot  (기본) → 팔로워 암 모터 설정
  #   CALIBRATE_TARGET=teleop         → 리더 텔레옵 모터 설정
  #
  # [주요 CLI 인자 전체]
  #   [robot 타깃]
  #     --robot.type=<str>    : 로봇 타입
  #     --robot.port=<path>   : 직렬 포트
  #   [teleop 타깃]
  #     --teleop.type=<str>   : 텔레옵 타입
  #     --teleop.port=<path>  : 직렬 포트
  #
  # ※ robot 과 teleop 중 하나만 지정해야 함 (동시 지정 불가)
  # ────────────────────────────────────────────────────────────────────────────
  setup-motors)
    if [[ "$CALIBRATE_TARGET" == "teleop" ]]; then
        info "── 장치 점검 ─────────────────────────────────────"
        check_port "$LEADER_PORT" "Leader Arm (teleop)"
        info "── Setup Motors (teleop) 시작 ──────────────────"
        lerobot-setup-motors \
          --teleop.type=${TELEOP_TYPE} \
          --teleop.port=${LEADER_PORT}
    else
        info "── 장치 점검 ─────────────────────────────────────"
        check_port "$FOLLOWER_PORT" "Follower Arm (robot)"
        info "── Setup Motors (robot) 시작 ───────────────────"
        lerobot-setup-motors \
          --robot.type=${ROBOT_TYPE} \
          --robot.port=${FOLLOWER_PORT}
    fi
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # find-joint-limits — 텔레오퍼레이션으로 관절 가동 범위 탐색
  #
  # robot + teleop 양쪽 모두 필요 (calibrate 와 달리 둘 다 필수)
  #
  # [env var → CLI arg 매핑]
  #   FOLLOWER_PORT/ID  → --robot.port / --robot.id
  #   LEADER_PORT/ID    → --teleop.port / --teleop.id
  #   TELEOP_TIME_S     → --teleop_time_s
  #
  # [주요 CLI 인자 전체]
  #   --robot.type=<str>         : 로봇 타입
  #   --robot.port=<path>        : 팔로워 직렬 포트
  #   --robot.id=<str>           : 팔로워 ID
  #   --teleop.type=<str>        : 텔레옵 타입
  #   --teleop.port=<path>       : 리더 직렬 포트
  #   --teleop.id=<str>          : 리더 ID
  #   --teleop_time_s=<float>    : 탐색 제한 시간(초, 기본 30)
  #   --display_data=true|false  : 카메라 영상 화면 출력
  # ────────────────────────────────────────────────────────────────────────────
  find-joint-limits)
    info "── 장치 점검 ─────────────────────────────────────"
    check_port "$LEADER_PORT"   "Leader Arm"
    check_port "$FOLLOWER_PORT" "Follower Arm"

    info "── Find Joint Limits 시작 ────────────────────────"
    info "  Leader   → ID: ${LEADER_ID}, PORT: ${LEADER_PORT}"
    info "  Follower → ID: ${FOLLOWER_ID}, PORT: ${FOLLOWER_PORT}"
    info "  시간: ${TELEOP_TIME_S}s"

    lerobot-find-joint-limits \
      --robot.type=${ROBOT_TYPE} \
      --robot.port=${FOLLOWER_PORT} \
      --robot.id=${FOLLOWER_ID} \
      --teleop.type=${TELEOP_TYPE} \
      --teleop.port=${LEADER_PORT} \
      --teleop.id=${LEADER_ID} \
      --teleop_time_s=${TELEOP_TIME_S}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # find-cameras — 시스템에 연결된 카메라 검색 및 캡처 확인
  #
  # 하드웨어 체크 없음 (자동 스캔)
  #
  # [주요 CLI 인자 전체]
  #   [위치 인자]
  #   opencv|realsense            : 카메라 타입 지정 (생략 시 전체 스캔)
  #   [옵션]
  #   --output-dir=<path>         : 캡처 이미지 저장 경로 (기본: outputs/captured_images)
  #   --record-time-s=<float>     : 캡처 시도 시간(초, 기본 6.0)
  #
  # 예시:
  #   docker compose run --rm teleop find-cameras
  #   docker compose run --rm teleop find-cameras opencv
  #   docker compose run --rm teleop find-cameras realsense --output-dir=/workspace/data/cams
  # ────────────────────────────────────────────────────────────────────────────
  find-cameras)
    info "── Find Cameras 시작 ─────────────────────────────"
    shift
    exec lerobot-find-cameras "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # find-port — MotorsBus USB 포트 자동 감지
  #
  # 하드웨어 체크 없음; 사용자 인터랙션 필요
  # (USB 케이블 분리 후 Enter 키 입력 안내)
  #
  # [CLI 인자 없음 — 인터랙티브 전용]
  #
  # 예시:
  #   docker compose run --rm teleop find-port
  # ────────────────────────────────────────────────────────────────────────────
  find-port)
    info "── Find Port 시작 ────────────────────────────────"
    exec lerobot-find-port
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # dataset-viz — 데이터셋 시각화 (Rerun 기반)
  #
  # [env var → CLI arg 매핑]
  #   HF_DATASET_REPO_ID  → --repo-id        (필수)
  #   EPISODE_INDEX       → --episode-index
  #   VIZ_MODE            → --mode           (local | distant)
  #   VIZ_WS_PORT         → --ws-port
  #   DATASET_ROOT        → --root
  #
  # [주요 CLI 인자 전체]
  #   --repo-id=<str>          : HF Hub 데이터셋 ID (필수)
  #   --episode-index=<int>    : 시각화할 에피소드 인덱스 (필수)
  #   --root=<path>            : 로컬 저장 루트 (생략 시 HF 캐시)
  #   --output-dir=<path>      : .rrd 파일 저장 경로 (--save 1 시 사용)
  #   --batch-size=<int>       : DataLoader 배치 크기 (기본 32)
  #   --num-workers=<int>      : DataLoader 프로세스 수 (기본 4)
  #   --mode=local|distant     : 로컬 뷰어 or 원격 WebSocket 서버 (기본 local)
  #   --web-port=<int>         : Rerun 웹 포트 (distant 모드, 기본 9090)
  #   --ws-port=<int>          : Rerun WebSocket 포트 (distant 모드, 기본 9087)
  #   --save=0|1               : 1이면 .rrd 파일로 저장 (뷰어 미실행)
  #   --tolerance-s=<float>    : 타임스탬프 허용 오차(초, 기본 1e-4)
  #
  # distant 모드 이용법:
  #   서버 쪽 : VIZ_MODE=distant 로 실행
  #   클라이언트: rerun ws://HOST:9087
  # ────────────────────────────────────────────────────────────────────────────
  dataset-viz)
    check_dataset_repo_id

    info "── Dataset Viz 시작 ──────────────────────────────"
    info "  Dataset → ${DATASET_REPO_ID}, episode=${EPISODE_INDEX}"
    info "  Mode    → ${VIZ_MODE}"

    lerobot-dataset-viz \
      --repo-id=${DATASET_REPO_ID} \
      --episode-index=${EPISODE_INDEX} \
      --root=${DATASET_ROOT} \
      --mode=${VIZ_MODE} \
      --ws-port=${VIZ_WS_PORT}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # train — Policy 학습 (모든 인자를 lerobot-train 에 완전 위임)
  #
  # 하드웨어 없음; GPU 권장
  #
  # [주요 CLI 인자 전체]
  #   --dataset.repo_id=<str>         : 학습 데이터셋 HF Hub ID (필수)
  #   --dataset.local_files_only=true : 오프라인 모드
  #   --dataset.root=<path>           : 로컬 저장 루트
  #   --policy.type=<str>             : 모델 타입 (act, diffusion, smolvla, ...)
  #   --policy.path=<str>             : 사전학습 체크포인트 경로/Hub ID
  #   --output_dir=<path>             : 체크포인트·로그 출력 디렉터리
  #   --job_name=<str>                : 실행 이름 (WandB 표시)
  #   --resume=true|false             : 기존 output_dir 에서 재개 (기본 false)
  #   --seed=<int>                    : 난수 시드 (기본 1000)
  #   --batch_size=<int>              : 배치 크기 (기본 8)
  #   --steps=<int>                   : 총 학습 스텝 수 (기본 100000)
  #   --num_workers=<int>             : DataLoader 워커 수 (기본 4)
  #   --eval_freq=<int>               : 평가 주기(스텝, 기본 20000)
  #   --save_freq=<int>               : 체크포인트 저장 주기(스텝, 기본 20000)
  #   --log_freq=<int>                : 로그 출력 주기(스텝, 기본 200)
  #   --save_checkpoint=true|false    : 체크포인트 저장 여부 (기본 true)
  #   --use_policy_training_preset=true|false : 정책별 학습 프리셋 사용 (기본 true)
  #   --rename_map=<dict>             : 관측 키 이름 변경 매핑
  #   --wandb.enable=true|false       : WandB 로깅 여부 (기본 false)
  #   --wandb.project=<str>           : WandB 프로젝트 이름
  #   --wandb.entity=<str>            : WandB 엔티티
  #   --wandb.run_id=<str>            : 재개 시 WandB run ID
  #   --env.type=<str>                : 시뮬레이션 평가 환경 (pusht, aloha, xarm, ...)
  #
  # 예시:
  #   docker compose run --rm teleop train \
  #     --dataset.repo_id=my_user/so101_pick \
  #     --policy.type=act \
  #     --output_dir=/workspace/data/outputs/act \
  #     --steps=50000 \
  #     --batch_size=16 \
  #     --wandb.enable=true
  # ────────────────────────────────────────────────────────────────────────────
  train)
    info "── Train 시작 ────────────────────────────────────"
    shift
    exec lerobot-train "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # eval — Policy 평가 및 롤아웃 (모든 인자를 lerobot-eval 에 완전 위임)
  #
  # 하드웨어 없음; 시뮬레이션 환경 필요
  #
  # [주요 CLI 인자 전체]
  #   --policy.path=<str>           : Hub ID 또는 로컬 체크포인트 경로 (필수)
  #   --env.type=<str>              : 평가 환경 타입 (pusht, aloha, xarm, ...)
  #   --env.task=<str>              : 환경 내 세부 태스크
  #   --eval.n_episodes=<int>       : 평가 에피소드 수 (기본 50)
  #   --eval.batch_size=<int>       : 동시 병렬 롤아웃 수 (기본 50)
  #   --eval.use_async_envs=true    : 비동기 다중 환경 사용 (기본 false)
  #   --output_dir=<path>           : 결과 저장 경로
  #   --job_name=<str>              : 실행 이름
  #   --seed=<int>                  : 난수 시드 (기본 1000)
  #   --rename_map=<dict>           : 관측 키 이름 변경 매핑
  #
  # 예시:
  #   docker compose run --rm teleop eval \
  #     --policy.path=my_user/act_so101 \
  #     --env.type=pusht \
  #     --eval.n_episodes=20 \
  #     --eval.batch_size=10
  # ────────────────────────────────────────────────────────────────────────────
  eval)
    info "── Eval 시작 ─────────────────────────────────────"
    shift
    exec lerobot-eval "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # edit-dataset — 데이터셋 편집 (모든 인자를 lerobot-edit-dataset 에 완전 위임)
  #
  # 하드웨어 없음
  #
  # [주요 CLI 인자 전체]
  #   --repo_id=<str>              : 편집 대상 데이터셋 Hub ID (필수)
  #   --new_repo_id=<str>          : 결과 데이터셋 Hub ID (생략 시 원본 덮어쓰기)
  #   --root=<path>                : 로컬 저장 루트
  #   --push_to_hub=true|false     : 결과를 HF Hub 에 업로드 (기본 false)
  #   --operation.type=<str>       : 작업 종류 (아래 참조)
  #
  #   [delete_episodes 작업]
  #     --operation.type=delete_episodes
  #     --operation.episode_indices=[0,1,3]  : 삭제할 에피소드 인덱스 목록
  #
  #   [split 작업]
  #     --operation.type=split
  #     --operation.splits='{"train":0.8,"val":0.2}'  : 비율 or 인덱스 리스트로 분할
  #
  #   [merge 작업]
  #     --operation.type=merge
  #     --operation.repo_ids='["user/ds1","user/ds2"]'  : 병합할 데이터셋 ID 목록
  #
  #   [remove_feature 작업]
  #     --operation.type=remove_feature
  #     --operation.feature_names='["observation.images.wrist"]'  : 제거할 특성 키 목록
  #
  # 예시:
  #   docker compose run --rm teleop edit-dataset \
  #     --repo_id=my_user/so101_pick \
  #     --operation.type=delete_episodes \
  #     --operation.episode_indices=[0,5,9]
  # ────────────────────────────────────────────────────────────────────────────
  edit-dataset)
    info "── Edit Dataset 시작 ─────────────────────────────"
    shift
    exec lerobot-edit-dataset "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # info — LeRobot / Python / 시스템 정보 출력
  #
  # [CLI 인자 없음]
  #
  # 예시:
  #   docker compose run --rm teleop info
  # ────────────────────────────────────────────────────────────────────────────
  info)
    exec lerobot-info
    ;;

  # ── 인터랙티브 쉘 / 직접 실행 ────────────────────────────────────────────────
  bash|shell)
    info "인터랙티브 쉘로 진입합니다."
    exec /bin/bash
    ;;

  python)
    shift
    exec python "$@"
    ;;

  *)
    exec "$@"
    ;;

esac
