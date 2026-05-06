# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

SO-ARM101 6축 로봇 팔에 대한 VLA 학습·배포 파이프라인. LeRobot 호환 모델이라면 어느 것이든 학습·추론 가능. **시뮬레이션 경로**(LeIsaac on Isaac Sim 5.1 → HDF5 기록 → LeRobot 변환)와 **실기기 경로**(`lerobot-record` → 학습 → 팔로워 암 추론) 양쪽을 지원한다.

운영 환경: Windows 워크스테이션과 Linux 원격 서버. 자세한 사양은 §환경 사양 참조.

자세한 사용법·트러블슈팅은 `README.md`에 정리되어 있다. 본 문서는 **README에 없는** 내부 구조·규칙과 자주 쓰는 명령만 다룬다.

## 환경 사양

| | Windows 워크스테이션 | Linux 학습 서버 |
|---|---|---|
| **OS** | Windows 11 Pro | Ubuntu 22.04, **Incus 컨테이너**  |
| **CPU** | Intel Xeon W-2245 @ 3.90GHz (8 cores / 16 threads, L3 16.5 MB) | Intel Xeon Platinum 8480C, 224 logical CPUs |
| **RAM** | 64 GB | 2.0 TiB |
| **Storage** | NVMe SSD 512 GB + SATA HDD 1 TB | `/dev/md127` RAID, 28 TB |
| **GPU** | NVIDIA RTX A4000 16 GB (driver 596.36, CUDA 13.2, compute_cap 8.6 Ampere) | NVIDIA H100 80GB HBM3 ×2 (driver 580.126.20, compute_cap 9.0 Hopper) |

테스트 스위트나 lint config는 현재 정의되어 있지 않다 (`tests/`, `ruff.toml`, `pre-commit-config.yaml` 등 없음). 변경 검증은 실제 시뮬레이터 / 실기기 실행으로 수행한다.

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

- 셸 명령은 PowerShell 표기 (`$env:VAR`, 백틱 line continuation) 기본. Linux 세션에서는 bash 표기 사용
- HF/W&B 토큰은 `.env`에서 읽음. `.env.example`이 템플릿
- 외부 CLI 호출은 가능한 한 비대화형 모드 (`--yes`, `--quiet`, `--json`)

## 운영 규칙

### 에러 수정 후 README Troubleshooting에 기록

새로운 종류의 에러를 진단하고 **수정에 성공**했을 때, 그 경험을 다음 세션·다른 작업자가 활용할 수 있도록 `README.md`의 §Troubleshooting에 항목을 추가한다.

- 양식은 기존 항목과 동일: **현상 → 오류 메시지(코드 블록) → 원인 → 해결 방법 → 확인 방법** 5블록
- 같은 종류의 에러(ABI 불일치, GPU/드라이버 호환, 의존성 핀 충돌 등)는 인접 섹션에 배치해 흐름을 맞출 것
- 필요하면 §핵심 의존성 표나 §실제로 주의해야 할 로그 리스트도 함께 갱신
- 단순 일회성 환경 문제(타이포, 권한 누락 등)는 기록하지 않는다 — 다른 사람이 다시 마주칠 가능성이 있는 클래스의 문제만
- **워크어라운드만 발견하고 근본 해결은 못 했다면 README에 올리지 않는다** (회피책이 정설로 굳어지는 걸 방지). 메모리에만 임시 기록하거나, 이슈 트래킹 메커니즘을 따로 마련
- 수정이 실패한 경우도 README에는 올리지 않는다
