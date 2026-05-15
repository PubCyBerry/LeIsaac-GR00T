# lerobot-so101:0.4.4 이미지 용량 분석 & 절감 방안

## Context

`lerobot-so101:0.4.4` 이미지(25.7 GB)가 base 인 `nvidia/cuda:13.0.0-devel-ubuntu24.04`(11.4 GB) 대비 **+14.3 GB** 증가했다. 본 문서는 (1) 레이어별로 어디서 용량이 늘었는지, (2) 어떤 파일/패키지가 중복·과잉인지, (3) 어떤 변경이 어느 정도 절감을 줄 수 있는지를 정리한다. 현 시점에서는 **분석 + 권장 방안 제시**까지만 다루고, 어떤 방안을 실행할지는 추후 결정한다.

---

## 1. 레이어별 용량 증가 분석

`docker history lerobot-so101:0.4.4` 결과를 토대로 한 증가분 (위에서 아래로, Dockerfile stage 순):

| # | Stage / 명령 | 크기 | 비고 |
|---|---|---:|---|
| 1 | Stage base — `apt-get install` (build tools / ffmpeg / vulkan / usb / speech) | **852 MB** | 빌드용 -dev 패키지가 상당수 |
| 2 | Stage uv — `curl … uv install` | 60 MB | uv 바이너리 |
| 3 | Stage python-setup — `uv python install 3.11 + venv` | 121 MB | managed CPython |
| 4 | Stage torch-layer — `uv pip install torch==2.7.0 torchvision==0.22.0 (cu128)` | **7.01 GB** | **최대 단일 레이어** |
| 5 | Stage teleop-deps — `uv sync --group teleop --no-install-project` | **1.44 GB** | lerobot, opencv, rerun, pyarrow, triton 등 |
| 6 | Stage app — udev rules / entrypoint.sh copy | ~140 kB | 무시 가능 |
| | **소계 (base 위에 추가된 분량)** | **≈ 9.6 GB** | docker history 직접 합산 |

> 참고: `docker images` 상의 25.7 GB와 history 합 사이 차이는 layer 압축/공유로 인한 표시값 차이. 핵심은 **torch 단일 레이어 7.0 GB + teleop deps 1.44 GB + apt 852 MB** 가 사실상 모든 증가분이라는 점.

### Base 이미지 내부 구성 (참고)

`/usr/local/cuda-13.0` = **5.1 GB**, 그중 `targets`(headers + dev libs) = **4.4 GB**, `nvvm` = 134 MB. base 이미지의 11.4 GB 중 절반 이상이 **CUDA 13.0 devel 툴체인**(nvcc, headers, stubs).

---

## 2. 실제 venv 내부 분포 (`/opt/venv` = 7.8 GB)

상위 패키지 (컨테이너 내부 `du -sh` 결과):

| 패키지 | 크기 | 용도 / 비고 |
|---|---:|---|
| `nvidia/*` (PyTorch 번들 CUDA 라이브러리들) | **3.6 GB** | 아래 표 참조 |
| `torch` | 2.0 GB | PyTorch 본체 |
| `triton` | 545 MB | torch.compile 백엔드 (런타임 JIT) |
| `cusparselt` | 227 MB | sparse linear algebra |
| `rerun_sdk` + `rerun_bindings` | 286 MB | 시각화/디버깅 뷰어 |
| `pyarrow` | 131 MB | HuggingFace datasets 의존 |
| `cmake` | 84 MB | 런타임에는 불필요 |
| `wandb` / `av.libs` / `sympy` / `imageio_ffmpeg` / `pandas` | 각 76–82 MB | |
| `cv2` + `opencv_python_headless.libs` | 137 MB | 카메라 캡처 |
| `diffusers` | 39 MB | lerobot 의존 |
| `torchvision` | 18 MB | |

### PyTorch 번들 CUDA libs 상세 (`site-packages/nvidia/`)

| 라이브러리 | 크기 | 비고 |
|---|---:|---|
| `cudnn` | **1.1 GB** | 학습/추론 필수 |
| `cublas` | **858 MB** | 선형대수 필수 |
| `cusparse` | 377 MB | torch sparse 연산 |
| `cusolver` | 377 MB | |
| `cufft` | 269 MB | FFT (오디오/spectral 처리에만) |
| `nccl` | 261 MB | 멀티-GPU 통신 (SO-101 단일 GPU 환경에서는 미사용) |
| `cuda_nvrtc` | 212 MB | 런타임 컴파일 (triton 사용 시 필요) |
| `curand` | 133 MB | 난수 |
| `nvjitlink` | 90 MB | |
| `cuda_cupti` | 41 MB | 프로파일링 |
| 기타 (`cuda_runtime`, `cufile`, `nvtx`) | < 10 MB | |

→ **핵심 관찰**: PyTorch wheel 이 **CUDA 12.8 런타임 전체를 자체 번들**(3.6 GB)한다. 따라서 base 이미지의 **CUDA 13.0 devel toolkit(5.1 GB) 은 torch 가 거의 사용하지 않는다**. 두 CUDA 스택(12.8 + 13.0)이 공존하면서 5+ GB 가량 중복.

---

## 3. 용량이 큰 직접 원인 (요약)

1. **PyTorch CUDA wheel 자체가 ~7 GB.** cu128 wheel 은 cuDNN/cuBLAS/cuSOLVER/cuFFT/NCCL 등을 모두 self-contained 로 번들하기 때문이다. base CUDA 와 무관하게 추가됨.
2. **Base 이미지가 `devel` 변종이라 4.4 GB 의 CUDA headers/targets 가 포함된다.** torch wheel 은 이를 사용하지 않으므로 사실상 dead weight. 컴파일이 필요한 native extension(예: feetech-servo-sdk, h5py 등)은 헤더가 필요하지만, **빌드가 끝나면 더 이상 필요 없음**.
3. **두 CUDA 메이저(12.8 ↔ 13.0) 가 공존.** torch는 12.8 wheel 을 가져오는데 base 는 13.0. 둘은 서로 다른 디렉토리에 설치되어 중복 부담만 만든다.
4. **apt -dev 패키지가 빌드 후에도 잔존** (`build-essential`, `python3-dev`, `lib*-dev`, `libavcodec-dev`, `libavformat-dev`, `libswscale-dev`, `libffi-dev`, `libssl-dev`, `libusb-1.0-0-dev`, `libevdev-dev`(중복 선언됨)). 모두 빌드 시점에만 필요한 헤더/툴.
5. **잠재적으로 미사용 가능한 패키지가 venv 에 포함**: `cmake`(84 MB), `jedi`(34 MB), `debugpy`(21 MB) 같은 dev/IDE 도구. NCCL(261 MB) 은 단일 GPU 환경에서 미사용. `rerun_sdk`/`rerun_bindings`(286 MB) 도 모드에 따라 사용 안 할 수 있음.

---

## 4. 절감 방안 (효과 큰 순)

### 방안 A — Base 를 `runtime` 변종으로 변경 (권장, 효과 ~5–6 GB)

`FROM nvidia/cuda:13.0.0-devel-ubuntu24.04` → `FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04`

- **이유**: PyTorch cu128 wheel 이 자체 CUDA 런타임을 번들하므로 devel toolkit(nvcc, headers, stubs) 거의 불필요. CUDA 버전도 torch 와 정합(12.8).
- **예상 절감**: **5–6 GB** (devel→runtime 변경으로 4.4 GB targets 제거 + nvvm 134 MB + nsight 등).
- **위험**: native extension(`feetech-servo-sdk`, `h5py` 등)이 빌드 시점에 헤더를 필요로 함. → **방안 D(멀티스테이지)** 와 결합.

### 방안 B — Multi-stage 분리: builder(devel) → runtime(runtime/base) (권장, 효과 ~5–7 GB)

```
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder
  # apt -dev, build-essential, uv, torch wheel, uv sync (deps 빌드)
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime
  COPY --from=builder /opt/venv /opt/venv
  # 런타임 apt 패키지만 (ffmpeg, libusb-1.0-0, udev, libgl1, libglib2.0-0 등 -dev 제외)
```

- **이유**: devel 은 빌드용으로만 쓰고 최종 이미지에 안 남김.
- **예상 절감**: **5–7 GB** (devel toolkit + -dev apt 패키지 + 빌드 캐시 모두 final 에서 제거).
- **위험**: venv 안의 native lib 가 builder 의 시스템 .so 에 동적 링크된 경우 runtime 에 해당 패키지 없으면 ImportError. 검증 필요 항목: `lerobot`, `pyav`, `opencv`, `h5py`, `feetech-servo-sdk`. 실제로는 PyTorch wheel 처럼 wheels 안에 .so 가 자급되는 경우가 대부분이라 큰 문제 없을 가능성이 높음.
- **변경 파일**: `docker/Dockerfile.lerobot` (전체 stage 재배치).

### 방안 C — 미사용 PyTorch 번들 CUDA libs 제거 (효과 ~250–500 MB, 위험 중)

cu128 wheel 에 번들된 `nvidia/*` 중 SO-101 워크플로(단일 GPU teleop·record·train·eval)에서 거의 미사용:

- `nccl` (261 MB) — 멀티-GPU 통신, 단일 GPU 환경에선 안 부름 (단, `torch.distributed` import 만으로도 검사 통과 필요할 수 있음 → 검증 필요)
- `cufft` (269 MB) — vision/torch 본체에서 FFT 거의 호출 안 함
- `cuda_cupti` (41 MB) — 프로파일러
- `cusparselt` (227 MB, 별도 패키지) — sparse linear algebra, 모델 따라 미사용

**위험**: torch import 시 lazy load 가 아니라 일부는 실행 시점에 로드 실패. 학습/추론 코드 경로마다 검증 필요. **추천 X** (절감 대비 안정성 손해 큼). 다만 train/eval 모드를 별도 이미지로 분리한다면 teleop 전용 이미지에서는 제거 가능.

### 방안 D — apt -dev 패키지를 builder 단계에서만 사용 (방안 B 의 일부, 효과 ~300–600 MB)

런타임 이미지에서 제거 후보:
- `build-essential`, `python3-dev`, `libffi-dev`, `libssl-dev`
- `libavcodec-dev`, `libavformat-dev`, `libswscale-dev` (ffmpeg 패키지만 두고 dev 제거)
- `libusb-1.0-0-dev`, `libevdev-dev`(중복 선언도 정리)

런타임에 남길 것: `ffmpeg`, `libsm6`, `libxext6`, `libgl1`, `libglib2.0-0`, `libusb-1.0-0`, `libevdev2`(런타임 패키지명 확인), `udev`, `ca-certificates`.

### 방안 E — 사용 여부 확인 후 제거 후보 (효과 ~200–400 MB)

- `vulkan-tools`, `mesa-vulkan-drivers`, `libxkbcommon-x11-0` — teleop/record 모드에서 실제 사용 안 하면 제거. (현 활성 워크플로에서는 시뮬레이션 미사용이므로 Vulkan 거의 불필요)
- `speech-dispatcher`, `espeak-ng` — LeRobot TTS 알림용. 미사용이면 제거 (~80 MB).
- venv 안 `cmake`, `jedi`, `debugpy` — 명시적으로 안 쓰면 `--exclude` 또는 사후 삭제로 정리(`cmake` 84 MB 즉시 절감, 나머지는 transitive 라 의존성 확인 필요).

### 방안 F — venv 내 `.py` 소스 제거하고 `.pyc` 만 보존 (효과 ~수백 MB, 비권장)

`UV_COMPILE_BYTECODE=1` 이미 켜져 있어 .pyc 가 미리 생성됨. .py 를 지우면 절감되지만 traceback/inspect 가 어려워져 디버깅 비용 증가. **비추천**.

---

## 5. 권장 실행 순서

효과 대비 위험이 낮은 순서로 단계적 적용을 권장:

1. **방안 B (멀티스테이지)** + **방안 A (CUDA 12.8 runtime base)** 를 한 번에 적용 → **~6–7 GB 절감 기대**. 검증 포인트: 각 모드(`teleop`, `record`, `replay`, `train`, `eval`, `dataset-viz`) 의 `python -c "import lerobot"` + 실제 1회 dry-run.
2. 1번이 안정화되면 **방안 D (런타임 -dev 패키지 제거)** + **방안 E (Vulkan/espeak 등 미사용 패키지 제거)** 적용 → 추가 ~500 MB–1 GB.
3. 방안 C 는 권장하지 않음. 굳이 더 줄이려면 teleop-only 이미지를 별도 분리해 NCCL 등 제거하는 게 안전.

### 검증 방법

```bash
docker compose -f docker/docker-compose.yaml build lerobot
docker images | grep lerobot-so101            # 새 크기 확인
docker compose -f docker/docker-compose.yaml run --rm lerobot info
docker compose -f docker/docker-compose.yaml run --rm lerobot \
    python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
docker compose -f docker/docker-compose.yaml run --rm lerobot \
    python -c "import lerobot, cv2, h5py, av; print('ok')"
# Feetech 통신 확인
docker compose -f docker/docker-compose.yaml run --rm lerobot find-port
# 실제 teleop 1회
docker compose -f docker/docker-compose.yaml run --rm lerobot teleop
```

각 단계에서 위 확인 통과 시 다음 방안으로 진행.

---

## 6. 핵심 변경 파일 (실행 단계에서 건드릴 곳)

- `docker/Dockerfile.lerobot` — base 이미지 변경, multi-stage 재구성, apt 패키지 분리
- `docker/docker-compose.yaml` — `image: lerobot-so101:0.4.4` 태그는 그대로 두되 빌드 검증 후 의미상 minor bump 검토 (운영 규칙 외)

별도 새 파일은 만들지 않는다. `pyproject.toml` 은 의존성 핀(§의존성 호환성 규칙)이 걸려있어 손대지 않는다.
