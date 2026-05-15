# AGENTS.md

## 프로젝트 개요

SO-ARM101 6축 로봇 팔용 **Docker 기반 LeRobot 파이프라인**. `docker/docker-compose.yaml` 의 두 서비스가 각자의 진입점을 사용한다: `lerobot` (Dockerfile.lerobot + `lerobot-entrypoint.sh`, teleop deps) 가 텔레오퍼레이션·데이터 수집·정책 학습·시각화를, `lerobot-policy-server` (Dockerfile.smolvla + `server-entrypoint.sh`, smolvla+async deps) 가 async inference gRPC 서버를 담당. SmolVLA 가 기본 정책이며 GR00T N1.5(flash-attn 필요) 등은 정책 서버 이미지에만 의존성을 추가한다. LeRobot 호환 모델이라면 어느 것이든 학습·추론 가능.

> 시뮬레이션 경로(LeIsaac on Isaac Sim 5.1)는 임시 비활성 상태(`Dockerfile.leisaac` 만 보존, docker-compose 에 미연결). 현재 활성 워크플로는 실기기(`lerobot-*`) 경로뿐이다.

운영 환경: Windows 워크스테이션과 Linux 원격 서버. 자세한 사양은 §환경 사양 참조.

자세한 사용법은 `README.md`에, 트러블슈팅은 `docs/TROUBLESHOOTING.md`에 정리되어 있다. 본 문서는 **README에 없는** 내부 구조·규칙과 자주 쓰는 명령만 다룬다.

## 환경 사양

| | Windows 워크스테이션 | Linux 학습 서버 |
|---|---|---|
| **OS** | Windows 11 Pro | Ubuntu 22.04, **Incus 컨테이너**  |
| **CPU** | Intel Xeon W-2245 @ 3.90GHz (8 cores / 16 threads, L3 16.5 MB) | Intel Xeon Platinum 8480C, 224 logical CPUs |
| **RAM** | 64 GB | 2.0 TiB |
| **Storage** | NVMe SSD 512 GB + SATA HDD 1 TB | `/dev/md127` RAID, 28 TB |
| **GPU** | NVIDIA RTX A4000 16 GB (driver 596.36, CUDA 13.2, compute_cap 8.6 Ampere) | NVIDIA H100 80GB HBM3 ×2 (driver 580.126.20, compute_cap 9.0 Hopper) |

테스트 스위트나 lint config는 현재 정의되어 있지 않다 (`tests/`, `ruff.toml`, `pre-commit-config.yaml` 등 없음). 변경 검증은 컨테이너 빌드 + 실기기 실행으로 수행한다.

## Docker 컨테이너 구조

- **활성 서비스**:
  - `lerobot` (이미지 `lerobot-so101:0.4.4`, `docker/Dockerfile.lerobot`) — teleop / record / replay / train / eval / dataset-viz. `docker compose -f docker/docker-compose.yaml build lerobot`.
  - `lerobot-policy-server` (이미지 `lerobot-policy-server:0.4.4`, `docker/Dockerfile.smolvla`) — async inference gRPC 서버 (`entrypoint.sh policy-server`). `docker compose -f docker/docker-compose.yaml build lerobot-policy-server`. teleop 이미지와 의존성 격리: GR00T 의 flash-attn / 원격 inference(H100 ↔ Windows) 확장 대비.
- **휴면 Dockerfile**: `Dockerfile.leisaac` / `Dockerfile.gr00t` — `docker-compose.yaml` 에 연결되어 있지 않으며 필요 시 수동 빌드. 시뮬 경로 복원 시 leisaac 부터 재연결.
- **빌드 스테이지** (`Dockerfile.lerobot` / `Dockerfile.smolvla` 가 Stage 1–4 동일 → BuildKit 캐시 공유): base(`nvidia/cuda:12.8.0-runtime-ubuntu24.04` + apt) → uv → python 3.11 venv → torch 2.7.0/torchvision 0.22.0 (cu128) → `uv sync --group <teleop|smolvla async> --no-install-project` → app(entrypoint, teleop 만 udev rules).
- **디바이스 마운트**: `${TELEOP_PORT}` `${ROBOT_PORT}` (직렬 암), `${BELLY_CAM_PORT}` `${BELLY_CAM_META_PORT}` `${WRIST_CAM_PORT}` `${WRIST_CAM_META_PORT}` `${TOP_CAM_PORT}` `{$TOP_CAM_META_PORT}`(UVC 캡처/메타 노드 쌍).
- **호스트 볼륨**: `./datasets`, `./logs`, `./outputs` → 컨테이너 `/workspace/*`. 명명 볼륨 `lerobot_hf_cache` → `/root/.cache/huggingface` (두 서비스 공유). 다른 머신으로 옮길 때는 `docker run -v lerobot_hf_cache:/cache alpine tar czf ...` 로 export 후 전송.
- **권한·네트워크**: `privileged: true` (udev/USB 접근), `network_mode: host` (rerun 뷰어·ROS 브릿지), `ipc: host`. GPU 1장 예약 (`deploy.resources.reservations.devices`).
- **서비스별 진입점**:
  - `docker/lerobot-entrypoint.sh` (lerobot 서비스): `teleop` / `record` / `replay` / `calibrate` / `setup-motors` / `find-port` / `find-cameras` / `find-joint-limits` / `dataset-viz` / `policy-client` / `edit-dataset` / `info` / `bash` / `python`. `policy-client` 는 `lerobot.async_inference.robot_client` 로 정책 서버에 gRPC 접속해 SO-101 follower 를 구동 — `async` 의존성 그룹(grpcio + protobuf) 이 teleop 이미지에도 함께 설치된다.
  - `docker/server-entrypoint.sh` (lerobot-policy-server 서비스): `prepare-model` / `policy-server` / `train` / `eval` / `info` / `bash` / `python`. CMD 기본값 `policy-server`. **train/eval 은 이쪽**: SmolVLA 학습이 필요로 하는 transformers / accelerate / num2words 가 `smolvla` 그룹에만 있고 lerobot 이미지에 미설치이기 때문.
  - 모드별 env var 매핑은 각 스크립트 상단 `${VAR:-default}` 블록과 case 분기 주석에 정리되어 있다.
- **`.env` 주입 경로**: `docker compose --env-file .env` 가 컨테이너에 환경변수로 주입하고, `entrypoint.sh` 가 기본값을 채워 `lerobot-*` CLI 인자로 매핑.

## 의존성 호환성 규칙

`pyproject.toml`은 ABI 호환성 때문에 다음 핀들이 의도적으로 걸려 있다. 임의 업그레이드 / `uv lock --upgrade` 금지.

| 핀 | 이유 | 어기면 |
|---|---|---|
| `numpy==1.26.0` (override) | Isaac Sim 5.1.0의 `isaacsim_kernel`이 강제 | uv 설치 자체가 실패 |
| `pyarrow<19` (override) | numpy 1.x C-API 호환 마지막 메이저. PyArrow 19+는 numpy 2.x ABI 전용이라 numpy 1.26과 segfault | Isaac Sim 시작 후 ~30초 silent crash (`arrow.dll!arrow_vendored::date::current_zone` 백트레이스) |
| `h5py<3.16` | Isaac Sim 번들 HDF5 1.14.x와 ABI 일치. h5py 3.16+는 HDF5 2.0 번들 | `Windows fatal exception: code 0xc0000139` |
| `torch==2.7.0+cu128` | Isaac Sim 5.1 번들 CUDA 12.8과 일치 | 기동 시 CUDA 호출 실패 |
| `packaging>=24.2,<26.0` (override) | 다른 패키지 메타데이터 검증 충돌 회피 | `uv sync` resolve 실패 |
| `setuptools<82` (build-constraint) | 일부 의존성의 `pkg_resources` 호환 | sdist 빌드 실패 |

`override-dependencies`는 transitive 제약을 강제로 무시한다. 예: `datasets 4.x`가 `pyarrow>=21`을 요구하지만 override로 `pyarrow<19` 설치 가능 — 본 레포의 검증된 워크플로(HDF5 → isaaclab2lerobot 변환)에서는 런타임에도 정상 동작.

## 시뮬레이션 환경 제약

**RT 코어 없는 GPU(H100/A100)는 NVIDIA가 Isaac Sim 5.1 공식 미지원으로 명시.** 시스템 요구사항 문서가 *"GPUs without RT Cores (A100, H100) are not supported."*라고 못박음. 카메라 sensor가 raytracing pipeline 생성 실패 → CUDA illegal memory access. 데이터 수집·학습은 NVIDIA 권장(RTX 4080+) 또는 RT 코어·16 GB VRAM 충족 GPU(RTX A4000/A5000/A6000, L40/L40S, RTX 6000 Ada, GeForce RTX 40/50)에서만

## 사용자 환경 컨벤션

- Windows USB 포워딩은 PowerShell(`usbipd bind/attach`), 컨테이너 내부 실행과 호스트 측 보조 명령은 bash 사용
- HF/W&B 토큰은 `.env`에서 읽음. `.env.example`이 템플릿. `docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot <mode>` 가 표준 실행 패턴
- 외부 CLI 호출은 가능한 한 비대화형 모드 (`--yes`, `--quiet`, `--json`)

## 운영 규칙

### 에러 수정 후 docs/TROUBLESHOOTING.md에 기록

새로운 종류의 에러를 진단하고 **수정에 성공**했을 때, 그 경험을 다음 세션·다른 작업자가 활용할 수 있도록 `docs/TROUBLESHOOTING.md`에 항목을 추가한다.

- 양식은 기존 항목과 동일: **현상 → 오류 메시지(코드 블록) → 원인 → 해결 방법 → 확인 방법** 5블록
- 같은 종류의 에러(ABI 불일치, GPU/드라이버 호환, 의존성 핀 충돌 등)는 인접 섹션에 배치해 흐름을 맞출 것
- 필요하면 §핵심 의존성 표나 §실제로 주의해야 할 로그 리스트도 함께 갱신
- 수정이 실패한 경우도 README에는 올리지 않는다
