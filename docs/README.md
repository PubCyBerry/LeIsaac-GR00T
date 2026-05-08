# SO-101 Teleoperation — Docker 환경 구성 가이드

LeRobot 0.4.2 기반 SO-101 Leader/Follower Arm 텔레오퍼레이션을  
Windows 11 + WSL2 + Docker 환경에서 구성하는 가이드입니다.

---

## 목차

1. [환경 요구사항](#1-환경-요구사항)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [사전 설치](#3-사전-설치)
4. [하드웨어 연결 및 USB 포워딩](#4-하드웨어-연결-및-usb-포워딩)
5. [WSL2 장치 확인](#5-wsl2-장치-확인)
6. [환경 변수 설정](#6-환경-변수-설정)
7. [Docker 이미지 빌드](#7-docker-이미지-빌드)
8. [실행](#8-실행)
9. [실행 모드](#9-실행-모드)
10. [트러블슈팅](#10-트러블슈팅)

---

## 1. 환경 요구사항

### 소프트웨어

| 항목 | 버전 | 비고 |
|------|------|------|
| Windows | 11 (22H2 이상) | WSL2 커널 6.6+ 권장 |
| WSL2 | Ubuntu 24.04 | `wsl --install` |
| Docker Desktop | 최신 | WSL2 backend 활성화 필수 |
| usbipd-win | 5.0.0 이상 | USB 장치 WSL2 포워딩 |
| NVIDIA Driver | 525 이상 | CUDA 컨테이너 실행용 |
| NVIDIA Container Toolkit | 최신 | Docker GPU 지원 |

### 하드웨어

| 장치 | 수량 | 비고 |
|------|------|------|
| SO-101 Leader Arm | 1 | Feetech STS3215 서보 |
| SO-101 Follower Arm | 1 | Feetech STS3215 서보 |
| USB-Serial 어댑터 | 2 | CH343 칩 (COM 포트) |
| 카메라 | 2 | belly cam (전면), wrist cam (손목) |

---

## 2. 프로젝트 구조

```
프로젝트 루트/
├── Dockerfile.teleop          # Docker 이미지 빌드 파일
├── docker-compose.yaml        # 컨테이너 실행 정의
├── pyproject.toml             # Python 의존성 (그룹화)
├── .env.example               # 환경 변수 템플릿
├── .env                       # 실제 환경 변수 (git 제외)
├── docker/
│   ├── entrypoint.sh          # 컨테이너 진입점 스크립트
│   └── 99-feetech.rules       # USB 장치 udev 규칙
├── configs/
│   └── so101_teleop.yaml      # 카메라 포함 로봇 설정
├── data/                      # 녹화 데이터 (자동 생성)
└── logs/                      # 로그 (자동 생성)
```

---

## 3. 사전 설치

### 3-1. usbipd-win 설치

PowerShell (관리자)에서 실행합니다.

```powershell
winget install usbipd
```

설치 후 PowerShell을 재시작합니다.

### 3-2. Docker Desktop 설정

Docker Desktop → Settings → Resources → WSL Integration에서  
사용 중인 WSL2 배포판(Ubuntu-24.04)을 활성화합니다.

### 3-3. NVIDIA Container Toolkit (WSL2)

WSL2 터미널에서 실행합니다.

```bash
# GPG 키 등록
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 저장소 추가
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 설치
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 4. 하드웨어 연결 및 USB 포워딩

### 4-1. 장치 연결 순서

1. SO-101 Leader Arm USB-Serial 어댑터 → PC USB 포트 연결
2. SO-101 Follower Arm USB-Serial 어댑터 → PC USB 포트 연결
3. belly cam (전면부 카메라) → PC USB 포트 연결
4. wrist cam (손목 카메라) → PC USB 포트 연결

### 4-2. 장치 목록 확인

PowerShell (관리자)에서 실행합니다.

```powershell
usbipd list
```

출력 예시:

```
BUSID  VID:PID    DEVICE                              STATE
1-9    1a86:55d3  USB-Enhanced-SERIAL CH343 (COM6)    Not shared
3-4    1a86:55d3  USB-Enhanced-SERIAL CH343 (COM7)    Not shared
1-7    0c45:64ab  Integrated Webcam                   Not shared
5-2    0c45:64ab  Integrated Webcam                   Not shared
```

> **Leader/Follower 구분**: 어댑터를 하나씩 뽑아가며 COM 번호로 확인합니다.  
> COM 번호가 낮은 쪽(COM6)을 Leader, 높은 쪽(COM7)을 Follower로 지정합니다.

### 4-3. WSL2로 USB 포워딩

```powershell
# 최초 1회: bind (관리자 권한 필요)
usbipd bind --busid 1-9   # Leader Arm
usbipd bind --busid 3-4   # Follower Arm
usbipd bind --busid 1-7   # belly cam
usbipd bind --busid 5-2   # wrist cam

# 매 세션마다: attach
usbipd attach --wsl --busid 1-9
usbipd attach --wsl --busid 3-4
usbipd attach --wsl --busid 1-7
usbipd attach --wsl --busid 5-2
```

> `bind`는 최초 1회만 실행하면 됩니다. PC 재시작 후에는 `attach`만 다시 실행합니다.

---

## 5. WSL2 장치 확인

### 5-1. Arm 직렬 포트 확인

WSL2 터미널에서 실행합니다.

```bash
ls /dev/ttyACM*
# 예상 출력: /dev/ttyACM0  /dev/ttyACM1
```

> **주의**: SO-101의 CH343 칩은 `cdc_acm` 드라이버로 동작하므로  
> `ttyUSB`가 아닌 `ttyACM`으로 잡힙니다.

### 5-2. 카메라 장치 확인

```bash
ls /dev/video*
# 예상 출력: /dev/video0  /dev/video1  /dev/video2  /dev/video3
```

UVC 카메라는 장치당 노드 2개가 생성됩니다.

| 노드 | 역할 | OpenCV 사용 여부 |
|------|------|-----------------|
| `/dev/video0` | belly cam 캡처 노드 | ✅ 사용 |
| `/dev/video1` | belly cam 메타데이터 노드 | ❌ 미사용 |
| `/dev/video2` | wrist cam 캡처 노드 | ✅ 사용 |
| `/dev/video3` | wrist cam 메타데이터 노드 | ❌ 미사용 |

### 5-3. 카메라 동작 확인

```bash
# MJPEG 포맷으로 단일 프레임 캡처
timeout 5 ffmpeg \
  -f v4l2 -input_format mjpeg \
  -video_size 640x480 -framerate 25 \
  -i /dev/video0 -frames:v 1 -update 1 /tmp/belly.jpg 2>&1 | tail -3

timeout 5 ffmpeg \
  -f v4l2 -input_format mjpeg \
  -video_size 640x480 -framerate 25 \
  -i /dev/video2 -frames:v 1 -update 1 /tmp/wrist.jpg 2>&1 | tail -3

ls -lh /tmp/belly.jpg /tmp/wrist.jpg
# 두 파일이 생성되면 성공
```

---

## 6. 환경 변수 설정

`.env.example`을 복사해 `.env`를 생성합니다.

```bash
cp .env.example .env
```

`.env` 파일에서 실제 환경에 맞게 값을 수정합니다.

```bash
# Arm 직렬 포트
LEADER_PORT=/dev/ttyACM0
FOLLOWER_PORT=/dev/ttyACM1

# 카메라 디바이스
BELLY_CAM_DEV=/dev/video0
BELLY_CAM_META_DEV=/dev/video1
WRIST_CAM_DEV=/dev/video2
WRIST_CAM_META_DEV=/dev/video3

# OpenCV 인덱스 (videoN → N)
BELLY_CAM_INDEX=0
WRIST_CAM_INDEX=2

# 카메라 해상도
CAM_WIDTH=640
CAM_HEIGHT=480
CAM_FPS=25
```

---

## 7. Docker 이미지 빌드

### 의존성 그룹 구조

`pyproject.toml`은 환경별로 의존성을 그룹화합니다.  
Docker 이미지 빌드 시 `teleop` 그룹만 설치되어 이미지 크기를 최소화합니다.

```
[project].dependencies   → h5py, hf-xet, pyzmq         (공용)
[dependency-groups]
  teleop                 → torch, torchvision, lerobot  (텔레옵 전용)
  isaac                  → isaacsim, leisaac             (Isaac Sim 전용)
  dev                    → ipykernel                    (개발 도구)
```

### 빌드 전 lockfile 생성 (권장)

재현 가능한 빌드를 위해 lockfile을 먼저 생성합니다.

```bash
# 로컬 환경에서 (uv 설치 필요)
uv lock
# uv.lock 생성됨 → git에 커밋
```

### 이미지 빌드

```bash
docker compose --env-file .env build teleop
```

레이어 캐시 구조 덕분에 재빌드 시 변경된 레이어만 다시 빌드됩니다.

| 레이어 | 내용 | 재빌드 조건 |
|--------|------|------------|
| base | apt 시스템 패키지 | 거의 없음 |
| uv | uv 바이너리 | uv 버전 변경 시 |
| python | Python 3.11 + venv | 거의 없음 |
| torch-layer | torch 2.7.0 + torchvision | PyTorch 버전 변경 시 |
| teleop-deps | lerobot[feetech] 0.4.2 | lerobot 버전 변경 시 |
| app | 스크립트 / 설정 파일 | 자주 |

---

## 8. 실행

### 매 세션 시작 시 체크리스트

```powershell
# 1. PowerShell (관리자) — USB 포워딩
usbipd attach --wsl --busid 1-9   # Leader Arm
usbipd attach --wsl --busid 3-4   # Follower Arm
usbipd attach --wsl --busid 1-7   # belly cam
usbipd attach --wsl --busid 5-2   # wrist cam
```

```bash
# 2. WSL2 — 장치 확인
ls /dev/ttyACM* /dev/video*
# /dev/ttyACM0  /dev/ttyACM1
# /dev/video0  /dev/video1  /dev/video2  /dev/video3
```

### 텔레오퍼레이션 실행

```bash
docker compose --env-file .env up teleop
```

정상 시작 시 출력 예시:

```
[INFO]  LeRobot 버전: 0.4.2
[INFO]  NVIDIA GPU 감지됨:
[INFO]    GPU: NVIDIA GeForce RTX 3080, 10240 MiB
[INFO]  로봇 설정 파일 OK: /workspace/configs/so101_teleop.yaml
[INFO]  Leader Arm 직렬 포트 OK: /dev/ttyACM0
[INFO]  Follower Arm 직렬 포트 OK: /dev/ttyACM1
[INFO]  BELLY 카메라 OK: /dev/video0 — Integrated_Webcam_HD
[INFO]  WRIST 카메라 OK: /dev/video2 — Integrated_Webcam_HD
[INFO]  Teleoperation 시작
```

종료는 `Ctrl+C`입니다.

---

## 9. 실행 모드

`entrypoint.sh`는 여러 실행 모드를 지원합니다.

### teleop — 텔레오퍼레이션 (기본)

```bash
docker compose --env-file .env up teleop
```

Leader Arm을 움직이면 Follower Arm이 동일하게 따라옵니다.

### record — 데이터 녹화하며 텔레오퍼레이션

`.env`에 HuggingFace 설정 추가 후 실행합니다.

```bash
# .env
HF_TOKEN=hf_xxxxxxxxxxxx
HF_DATASET_REPO_ID=your-username/so101-dataset
MAX_EPISODE_STEPS=1000
RECORD_FPS=25
```

```bash
docker compose --env-file .env run teleop record
```

belly cam + wrist cam 이미지가 에피소드 데이터에 함께 저장됩니다.

### replay — 녹화 에피소드 재생

```bash
docker compose --env-file .env run teleop replay
```

### bash — 디버깅용 쉘 진입

```bash
docker compose --env-file .env run teleop bash
```

---

## 10. 트러블슈팅

### `/dev/ttyACM*` 가 없을 때

CH343 칩은 `cdc_acm` 드라이버를 사용합니다. `usbipd attach` 후 WSL2에서 확인합니다.

```bash
dmesg | grep -i "cdc_acm\|ttyACM" | tail -10
# "cdc_acm: ... ttyACM0" 로그가 있어야 정상
```

로그가 없으면 attach를 다시 시도합니다.

```powershell
usbipd detach --busid 1-9
Start-Sleep 2
usbipd attach --wsl --busid 1-9
```

### `/dev/video*` 는 있는데 ffmpeg이 hang될 때

Windows 카메라 앱, Teams, Zoom 등이 카메라를 점유 중입니다.

```powershell
# 카메라 사용 앱 모두 종료 후
usbipd detach --busid 1-7
usbipd detach --busid 5-2
Start-Sleep 5
usbipd attach --wsl --busid 1-7
usbipd attach --wsl --busid 5-2
```

### 카메라 캡처 시 포맷 오류

지원 포맷을 먼저 확인합니다.

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

`CAM_FPS`를 지원하는 값으로 맞춥니다. 이 카메라(`0c45:64ab`)의 경우:

- MJPEG: 640×480 @ 25fps, 1280×720 @ 30fps
- YUYV: 640×480 @ 25fps, 1280×720 @ 10fps

### GPU를 인식하지 못할 때

```bash
# WSL2에서 GPU 확인
nvidia-smi

# Docker에서 GPU 확인
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

NVIDIA Container Toolkit이 설치되지 않은 경우 [3-3절](#3-3-nvidia-container-toolkit-wsl2)을 참고합니다.

### PC 재시작 후 장치 재연결

PC 재시작 후에는 `bind`는 유지되므로 `attach`만 다시 실행합니다.

```powershell
usbipd attach --wsl --busid 1-9
usbipd attach --wsl --busid 3-4
usbipd attach --wsl --busid 1-7
usbipd attach --wsl --busid 5-2
```

매번 반복이 번거롭다면 PowerShell 스크립트로 저장해 사용합니다.

```powershell
# attach_devices.ps1
usbipd attach --wsl --busid 1-9
usbipd attach --wsl --busid 3-4
usbipd attach --wsl --busid 1-7
usbipd attach --wsl --busid 5-2
Write-Host "모든 장치 연결 완료"
```

```powershell
# 실행 (관리자 PowerShell)
.\attach_devices.ps1
```
