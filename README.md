# SO-ARM101 VLA Control System

SO-ARM101 6축 로봇 팔에 대한 VLA(Vision-Language-Action) 기반 제어 시스템. Hugging Face **LeRobot** + **LeIsaac** (NVIDIA Isaac Sim) 조합으로 시뮬레이션 학습과 실기기 텔레오퍼레이션을 모두 지원한다.

- **기본 태스크**: `pick_and_place`
- **지원 정책**: SmolVLA, GR00T N1.6

## 아키텍처

### 시뮬레이션 경로 (LeIsaac + Isaac Sim)

```
[SO-ARM101 리더 암] → teleop_se3_agent.py → Isaac Sim 씬 제어
                                ↓
                        HDF5 에피소드 기록 (./datasets/dataset.hdf5)
                                ↓
                          isaaclab2lerobot 변환
                                ↓
                    LeRobot 호환 데이터셋 → HF Hub
```

### 실기기 경로 (LeRobot)

```
[SO-ARM101 리더 암] → lerobot-record → [LeRobot 데이터셋] → HF Hub
                                           ↓
                                    lerobot-train (ACT / SmolVLA)
                                           ↓
                           outputs/train/*/checkpoints/last/
                                           ↓
                                    hf upload
                                           ↓
                                [HF Hub 모델 저장소]
                                           ↓
                                  lerobot-eval (추론)
                                           ↓
                               [SO-ARM101 팔로워 암 실행]
```

## 환경 요구사항

### 소프트웨어

| 항목 | 버전 | 비고 |
|------|------|------|
| Windows | 11 (22H2 이상) 또는 Ubuntu 24.04 | WSL2 커널 6.6+ 권장 |
| Docker | 최신 | WSL2 backend 활성화 필수 |
| usbipd-win | 5.0.0 이상 | USB 장치 WSL2 포워딩 |
| NVIDIA Driver | >=580 | CUDA 컨테이너 실행용 |
| NVIDIA Container Toolkit | 12.8>= | Docker GPU 지원 |
| Hugging Face 계정 | - | 데이터셋·모델 업로드·다운로드용 |
| uv | 최신 | Python 환경·의존성 관리자 (conda / pip 대체) |

### 하드웨어

| 장치 | 수량 | 비고 |
|------|------|------|
| NVIIDA GPU | >= 1 | (Isaac Sim)RT 코어 필수. Isaac Sim 5.1은 카메라 sensor·뷰포트 렌더링 모두 RTX raytracing pipeline 위에서만 동작. NVIDIA 공식 문서가 *"GPUs without RT Cores (A100, H100) are not supported."* 라고 H100/A100을 명시적으로 제외 ([Isaac Sim 5.1 System Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)). 권장 GPU는 GeForce RTX 4080(최소) / RTX 5080(양호) / RTX PRO 6000 Blackwell. VRAM 최소 16 GB |
| SO-101 Leader Arm | 1 | Feetech STS3215 서보 |
| SO-101 Follower Arm | 1 | Feetech STS3215 서보 |
| USB-Serial 어댑터 | 2 | CH343 칩 (COM 포트) |
| 카메라 | 2 | belly cam (전면), wrist cam (손목) |

### 핵심 의존성

버전은 `pyproject.toml` 에 고정되어 있으며, ABI·CUDA 호환성 때문에 임의 업그레이드는 피한다.

| 패키지 | 버전 | 비고 |
|--------|------|------|
| Python | 3.11 | Isaac Sim 5.1 지원 범위 |
| torch / torchvision | 2.7.0 / 0.22.0 (cu128) | Isaac Sim 번들 CUDA 와 맞춤 |
| isaacsim | 5.1.0 | `[all,extscache]` extras 포함 |
| isaaclab | 2.3.0 | `leisaac[isaaclab]` 로 간접 설치 |
| leisaac | 0.4.0 | `pyproject.toml` 의 `[tool.uv.sources]` 가 git tag `v0.4.0` 에서 설치 |
| lerobot | 0.4.2 | `leisaac[lerobot]` 로 간접 설치 |
| h5py | `<3.16` | Isaac Sim 의 HDF5 1.14.x 와 ABI 일치 ([상세](#h5py와-isaac-sim의-hdf5-abi-불일치-windows)) |
| pyarrow | `<19` | `tool.uv.override-dependencies` 로 강제. Isaac Sim 의 numpy 1.26 핀과 ABI 일치 ([상세](#pyarrow와-numpy의-c-api-abi-불일치-windows)) |

### 설치 단계

1. **의존성 동기화**

    ```powershell
    uv sync
    ```

    `pyproject.toml` 에 선언된 모든 패키지와 git 소스(`leisaac`) 가 `.venv` 에 설치된다. 최초 실행 시 Isaac Sim 번들이 커서 수 GB 다운로드가 발생한다.

2. **설치 검증**

    ```powershell
    uv run python -c "import isaacsim, lerobot, leisaac, h5py; print('isaacsim', isaacsim.__version__); print('h5py HDF5', h5py.version.hdf5_version)"
    ```

    `h5py HDF5` 가 `1.14.x` 로 출력되면 Isaac Sim 과 ABI 가 맞는 상태다. `2.x` 가 나오면 [Troubleshooting](#h5py와-isaac-sim의-hdf5-abi-불일치-windows) 섹션을 참조.

3. **Hugging Face 로그인**

    ```powershell
    hf auth login
    ```

    데이터셋·체크포인트 업로드와 `lerobot/smolvla_base` 등 게이티드 모델 다운로드에 사용한다. `uv sync` 이후에는 `hf` CLI 가 경로에 잡히므로 `uv run` 접두사 없이 직접 호출 가능.

### 선택 사항

- **W&B 로그인** — `--wandb.enable=true` 로 학습 로그를 보낼 경우 `wandb login`.
- **환경 재생성** — 캐시가 꼬이면 `.venv/` 를 삭제하고 `uv sync` 를 다시 실행.

## 사용법

### (LeIsaac)Teleoperation

LeIsaac scene 에서 SO-101 리더 암으로 가상 팔로워를 조작하고, 에피소드를 HDF5 로 기록한다.

```powershell
uv run scripts/environments/teleoperation/teleop_se3_agent.py `
    --task=LeIsaac-SO101-PickOrange-v0 `
    --teleop_device=so101leader `
    --port=COM7 `
    --num_envs=1 `
    --device=cuda `
    --enable_cameras `
    --record `
    --dataset_file=./datasets/dataset.hdf5
```

기동 후 표시되는 컨트롤:

| 키 | 동작 |
|----|------|
| `B` | 텔레오퍼레이션 시작 |
| `R` | 리셋 (실패로 기록) |
| `N` | 리셋 (성공으로 기록) |
| `Ctrl+C` | 종료 |

팔로워가 리더를 제대로 따라가지 못하면 `--recalibrate` 를 추가해 재보정.

### (LeIsaac)Remote Teleoperation

Follower Arm: 5556 포트로 ZMQ 통신, `pyzmq` 필요

```bash
uv sync
uv run scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=so101leader \
    --remote_endpoint=tcp://10.10.40.254:5556 \
    --num_envs=1 --device=cuda --enable_cameras
```

Leader Arm:

```bash
uv sync
uv run scripts/environments/teleoperation/so101_joint_state_server.py `
    --port COM7 --id leader_arm --rate 50
```

### (LeRobot)실기기 텔레오퍼레이션 + 데이터 수집

```powershell
uv run lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM<팔로워> `
    --teleop.type=so101_leader `
    --teleop.port=COM<리더> `
    --dataset.repo_id=<username>/SoArm_pick_and_place `
    --dataset.num_episodes=50
```

### Hugging Face Hub 업로드

```powershell
uv run hf upload <username>/<repo_name> `
    outputs/train/<job>/checkpoints/last/pretrained_model
```

#### 출력 디렉토리 구조

```markdown
outputs/train/<job_name>/
├── checkpoints/
│   ├── <step>/
│   └── last/
│       └── pretrained_model/   ← Hub 업로드 대상
└── logs/
```

## Troubleshooting

### 카메라 대역폭 제한

**현상**: `lerobot-find-cameras` 실행 시 카메라가 탐지는 되지만 일부만 캡처에 성공함

**오류 메시지**:

```
Failed to connect or configure OpenCV camera 1: Failed to open OpenCVCamera(1)
Failed to connect or configure OpenCV camera 2: Failed to open OpenCVCamera(2)
```

**카메라 모델**: Microdia Integrated_Webcam_HD — USB 2.0 전용(추정)

**지원 해상도 프로파일**: `1280×720`, `640×480` 두 가지만 존재 (그 외 해상도 설정 불가)

#### 원인

탐지 단계(`find_cameras`)에서는 카메라를 1대씩 열고 즉시 닫으므로 전체가 보이지만,
연결·스트리밍을 동시에 유지하면 일부 카메라가 열리지 않는다.

USB 2.0 카메라 1대의 YUY2 전송량:

```
640 × 480 × 2 bytes × 30 fps = 18.4 MB/s
```

#### 테스트 결과

| 구성 | 결과 |
|------|------|
| USB 허브 + YUY2 | 1대만 성공 |
| USB 허브 + MJPEG | 1대만 성공 |
| PC 포트 직접 연결 (각각) | 2대 이상 성공 ✅ |

USB 허브 자체의 하드웨어 한계로, MJPEG로 전송량을 줄여도 허브에서는 동시에 1대만 스트리밍된다.
USB 3.2 허브도 내부적으로 USB 2.0 카메라는 HS 경로(480 Mbps 공유)를 사용하므로 허브 교체로는 해결되지 않는다.

#### 해결 방법

**카메라마다 PC USB 포트에 직접 연결** (유일하게 확인된 해결책)

현재 PC(ThinkStation) 기준 사용 가능한 포트:
```
전면: 4× USB 3.2 Gen 1
후면: 4× USB 3.2 Gen 1
     2× USB 2.0
```

카메라 3대를 허브 없이 전부 직접 꽂을 수 있다.

#### MJPG 실제 적용 여부 확인 방법

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
actual = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
print(f"실제 포맷: {actual}")   # MJPG 또는 YUY2
cap.release()
```

#### USB 버전 확인 방법

카메라의 USB 버전 확인

```powershell
# 1. 카메라 InstanceId 조회 (Status OK인 항목 확인)
Get-PnpDevice -Class Camera | Select-Object Status, InstanceId

# 2. ACPI 경로에서 포트 접두사 확인 (<InstanceId>에 위 결과 붙여넣기)
(Get-PnpDeviceProperty -InstanceId "USB\VID_0C45&PID_64AB&MI_00\<InstanceId>" |
  Where-Object { $_.KeyName -eq "DEVPKEY_Device_LocationPaths" }).Data |
  Where-Object { $_ -match "ACPI" }
```

출력 예시:

```markdown
ACPI(_SB_)#ACPI(PC00)#ACPI(XHCI)#ACPI(RHUB)#ACPI(HS09)#USB(2)#USBMI(0)
                                              ^^^^^^^^^^
                                              여기를 본다
```

| 접두사 | USB 버전 | 최대 속도 |
|--------|---------|---------|
| `HS##` | USB 2.0 | 480 Mbps |
| `SS##` | USB 3.0 | 5 Gbps |
| `SSP##` | USB 3.1/3.2 | 10+ Gbps |

USB 허브 버전 확인

```powershell
Get-WmiObject -Class Win32_USBHub | Select-Object DeviceID, Name
```

| 장치 이름 | USB 버전 |
|-----------|---------|
| `Generic USB Hub` | USB 2.0 |
| `Generic SuperSpeed USB Hub` | USB 3.0 |

### h5py와 Isaac Sim의 HDF5 ABI 불일치 (Windows)

**현상**: `teleop_se3_agent.py` 등 Isaac Sim 기반 스크립트 실행 시 프로세스가 비정상 종료되거나 `isaacsim.sensors.rtx` 확장 로드가 실패한다. 어떤 DLL 이 먼저 로드되느냐에 따라 오류 메시지가 둘 중 하나로 나타난다.

**오류 메시지 A** — Isaac Sim 판 `hdf5.dll` 이 먼저 로드된 경우:

```log
Windows fatal exception: code 0xc0000139
ImportError: DLL load failed while importing _errors: 지정된 프로시저를 찾을 수 없습니다.
```

`0xc0000139` 는 `STATUS_ENTRYPOINT_NOT_FOUND` — DLL 은 찾았지만 필요한 export 심볼이 없다는 뜻.

**오류 메시지 B** — h5py 판 `hdf5.dll` 이 먼저 로드된 경우:

```python
Could not load the dynamic library from .../generic_model_output/bin/generic_mo_io.dll.
ImportError: DLL load failed while importing _generic_model_output: 지정된 모듈을 찾을 수 없습니다.
```

Isaac Sim 내부 로그에서는 같은 메시지가 CP949 → UTF-8 오해독으로 깨져 나온다:

```log
吏?뺣맂 ?꾨줈?쒖?瑜?李얠쓣 ???놁뒿?덈떎..   = 지정된 프로시저를 찾을 수 없습니다.
吏?뺣맂 紐⑤뱢??李얠쓣 ???놁뒿?덈떎..       = 지정된 모듈을 찾을 수 없습니다.
```

#### 원인

h5py 3.16.0 부터는 **HDF5 2.0.0** (2025 메이저 릴리스) 을 번들한다. Isaac Sim 5.1 은 **HDF5 1.14.6** 을 번들하며, HDF5 2.0 과 1.14 는 메이저 버전이 달라 **ABI 가 호환되지 않는다**.

한 프로세스에 두 버전의 `hdf5.dll` 이 공존하면 먼저 로드된 쪽이 다른 쪽의 호출을 잠식하고, 어느 쪽이든 필요한 심볼/모듈을 찾지 못해 실패한다. 이 때문에 단순 `import` 순서를 조정하는 방식으로는 해결되지 않는다 — 방향만 바뀔 뿐 반드시 한쪽이 깨진다.

| 구성요소 | HDF5 번들 버전 |
|---------|--------------|
| h5py 3.16.0 이상 | **2.0.0** |
| h5py 3.15.x 이하 | 1.14.x |
| Isaac Sim 5.1 | **1.14.6** |

#### 해결 방법

h5py 를 HDF5 1.14.x 를 번들하는 버전으로 고정:

```powershell
uv add "h5py<3.16"
```

이렇게 하면 h5py 와 Isaac Sim 이 같은 ABI 의 HDF5 를 공유하므로 어느 쪽이 먼저 로드되어도 문제가 없다.

#### 확인 방법

```powershell
uv run python -c "import h5py; print(h5py.version.hdf5_version)"
```

`1.14.x` 가 출력되면 해결된 상태.

#### DLL 버전 비교 (디버깅용)

의심스러울 때 두 hdf5.dll 내부의 버전 문자열을 직접 꺼내볼 수 있다:

```powershell
uv run python -c "import re; data = open(r'.venv/Lib/site-packages/isaacsim/kit/dev/libs/sensors/generic_model_output/bin/hdf5.dll','rb').read(); [print(s.decode('ascii','replace')) for s in re.findall(rb'HDF5 library version:[^\x00]+', data)]"
```

### PyArrow와 NumPy의 C-API ABI 불일치 (Windows)

**현상**: `teleop_se3_agent.py` 등 Isaac Sim + LeRobot 스택을 함께 import 하는 스크립트가 Simulation App startup 직후 (대개 30초 부근) Python traceback 없이 silent crash 한다. stdout 에는 정상 로그만 찍히고, 종료 후 `.venv/lib/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/crash_*.txt` 와 `.venv/lib/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_*.log` 에만 backtrace 가 남는다.

**오류 메시지** — kit log 의 `[Fatal] [carb.crashreporter-breakpad.plugin]` 백트레이스:

```log
000: arrow.dll!arrow_vendored::date::current_zone+0x45aaab
001: arrow.dll!arrow::AllocateResizableBuffer+0x7bd
007: arrow.dll!arrow::ArrayBuilder::Reserve+0x52
008: arrow_python.dll!arrow::py::ConvertPySequence+0x2323
013: _compute.cp311-win_amd64.pyd
016: _dataset.cp311-win_amd64.pyd!PyInit__dataset
```

`current_zone` / `OutOfMemory` 같은 심볼은 nearest-symbol guess (huge `+offset`) 일 뿐, 실제 충돌 지점은 `arrow::py::ConvertPySequence` 가 numpy buffer interface 를 호출하는 시점이다.

#### 원인

Isaac Sim 5.1 의 `isaacsim_kernel` 패키지는 **`numpy == 1.26.0`** 을 하드 핀한다 (`.venv/Lib/site-packages/isaacsim_kernel-5.1.0.0.dist-info/METADATA`). 한편 PyArrow 는 **19.x 부터 numpy 1.x C-API 지원을 제거**했고 `pyarrow 24.0.0` cp311 wheel 은 numpy 2.x ABI 를 가정한다.

- `pyproject.toml` 에 pyarrow 핀이 없으면 uv 는 자동으로 최신 (24.x) 을 선택한다
- `lerobot 0.4.2` 가 의존하는 `datasets >= 4.0` 이 `pyarrow >= 21` 을 요구하므로 transitive 해석으로는 더 낮은 버전이 선택될 여지도 없다
- 결과적으로 `pyarrow 24` 가 `numpy 1.26` 위에서 로드되어, `pyarrow.dataset` 모듈 초기화 중 numpy buffer protocol 을 잘못된 ABI 로 접근하고 segfault

isaacsim 의 numpy 핀은 우회할 수 없으므로 **PyArrow 측을 numpy 1.x ABI 호환 버전으로 다운그레이드** 해야 한다. PyArrow 는 18.x 가 numpy 1.x / 2.x dual ABI 를 지원하는 마지막 메이저 라인이다.

| 구성요소 | NumPy ABI | 비고 |
|---------|----------|------|
| isaacsim 5.1.0 | numpy 1.x (`==1.26.0` 강제) | 우회 불가 |
| pyarrow ≥ 19.0 | numpy 2.x 전용 | numpy 1.26 과 ABI 불일치 |
| pyarrow 18.x | numpy 1.x / 2.x dual | numpy 1.26 호환 ✅ |
| datasets 4.x | `pyarrow >= 21` 선언 | runtime 은 pyarrow 18.1 으로도 import/동작 (선언만 위반) |

#### 해결 방법

`pyproject.toml` 의 `[tool.uv].override-dependencies` 에 `"pyarrow<19"` 를 추가:

```toml
[tool.uv]
override-dependencies = [
    "packaging>=24.2,<26.0",
    "numpy==1.26.0",
    "pyarrow<19",
]
```

`override-dependencies` 는 `datasets >= 21` 같은 transitive 제약을 강제로 무시하므로 설치 자체는 성공한다. 본 레포의 검증된 워크플로 (`--record` + HDF5 → `isaaclab2lerobot` 변환) 는 pyarrow 18.x 위에서 정상 동작한다. LeRobot Hub 데이터셋 (parquet) 을 직접 다루는 경로 (`--use_lerobot_recorder`, `lerobot-record`, `lerobot-train`) 에서 pyarrow 신규 API 가 호출되면 그때 별도 검증이 필요할 수 있다.

이후 환경 재동기화:

```powershell
uv sync --reinstall-package pyarrow
```

#### 확인 방법

```powershell
uv run python -c "import pyarrow as pa, numpy as np; print('pyarrow', pa.__version__, '| numpy', np.__version__)"
uv run python -c "import pyarrow.dataset as ds; print('pyarrow.dataset OK')"
```

각각 `pyarrow 18.x | numpy 1.26.0` 와 `pyarrow.dataset OK` 가 출력되면 해결된 상태. 이후 `teleop_se3_agent.py` 가 30초 부근 silent crash 없이 viewport 까지 진입한다.

### Docker 컨테이너에서 Vulkan 초기화 실패 (Linux)

**현상**: `docker compose up` 으로 컨테이너를 띄우면 Isaac Sim 이 다음 에러를 토하면서 GPU 가속을 잃고 software 로 fallback 된다. CUDA 자체는 동작하지만 (nvidia-smi 에서 컨테이너 안의 python 프로세스가 GPU 메모리를 점유) 렌더링·카메라·GPU PhysX 가 모두 죽는다.

**오류 메시지**:

```log
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed.
                Vulkan 1.1 is not supported, or your driver requires an update.
[Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices,
                including an attempt with compatibility mode.
[Error] [omni.physx.plugin] CUDA libs are present, but no suitable CUDA GPU was found!
[Warning] [omni.physx.plugin] PhysX warning: GPU solver pipeline failed,
                switching to software
```

#### 원인

호스트의 NVIDIA 드라이버가 `.run` 인스톨러로 **`--no-opengl-files`** 옵션과 함께 설치된 경우, `libGLX_nvidia.so.0` / `libnvidia-glcore.so.<ver>` / `libEGL_nvidia.so.0` 같은 그래픽스 유저 스페이스 라이브러리가 호스트에 통째로 빠져 있다. 이 상태에서는 다음이 모두 성립한다:

1. `/etc/vulkan/icd.d/nvidia_icd.json` 은 존재하지만 `library_path: libGLX_nvidia.so.0` 이 가리키는 실제 파일이 호스트에 없다 (dangling pointer).
2. `nvidia-container-cli list` 출력에 `GLX_nvidia` / `glcore` / `EGL_nvidia` 가 한 줄도 없다 → nvidia-container-runtime 이 컨테이너로 마운트할 라이브러리 자체가 호스트에 없다.
3. 컨테이너 안에서 `NVIDIA_DRIVER_CAPABILITIES=all` 을 줘도 마운트할 게 없으니 Vulkan ICD 가 동작 못 한다.

기존 설치 옵션은 `/var/log/nvidia-installer.log` 에서 확인할 수 있다:

```bash
head -15 /var/log/nvidia-installer.log
# nvidia-installer command line:
#     ./nvidia-installer
#     --no-kernel-module
#     --no-opengl-files       ← 이게 원인
#     --silent
```

추가로, docker-compose 의 `deploy.resources.reservations.devices` (`capabilities: [gpu]`) 방식은 `nvidia-container-toolkit ≥ 1.19` 의 일부 환경에서 graphics capability 를 트리거하지 않는다. 같은 호스트에서 legacy 방식 (`runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=all`) 으로 띄우면 Vulkan ICD JSON 은 마운트되지만, 위 1번 이유로 라이브러리 자체가 없어서 결국 동일하게 실패한다.

#### 해결 방법

같은 버전의 `.run` 인스톨러를 다시 받아서 **커널 모듈은 건드리지 않고 그래픽스 유저 스페이스만** 추가 설치한다.

```bash
# 1. 기존 컨테이너 정지 + GPU 사용 프로세스 종료 확인
docker compose down
nvidia-smi

# 2. 동일 버전 .run 다운로드 (Data Center / Tesla 경로에 호스팅됨)
cd /tmp
DRIVER_VER=$(cat /proc/driver/nvidia/version | awk '/NVRM/ {print $8}')
curl -fLO "https://us.download.nvidia.com/tesla/${DRIVER_VER}/NVIDIA-Linux-x86_64-${DRIVER_VER}.run"
chmod +x "NVIDIA-Linux-x86_64-${DRIVER_VER}.run"

# 3. --no-opengl-files 빼고 --install-libglvnd 추가, 커널 모듈은 그대로 둠
sudo sh "./NVIDIA-Linux-x86_64-${DRIVER_VER}.run" \
    --no-kernel-module \
    --install-libglvnd \
    --silent
```

`--no-kernel-module` 가 핵심이다. 커널 모듈은 이미 동작 중이므로 건드리지 않고, 빠져 있던 GL/Vulkan/EGL 유저 스페이스 라이브러리만 채워 넣는다.

또한 `docker-compose.yaml` 의 GPU 접근 방식은 legacy syntax 로 두는 편이 안정적이다:

```yaml
services:
  leisaac-debug:
    runtime: nvidia
    network_mode: host          # livestream WebRTC 동적 포트 협상에 유리
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: all
    volumes:
      - /etc/vulkan/icd.d:/etc/vulkan/icd.d:ro   # ICD JSON 안전망
    # deploy: 블록은 사용하지 않음 (graphics capability 트리거 불안정)
```

#### 확인 방법

설치 후 호스트에서:

```bash
ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
ls /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.${DRIVER_VER}
ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
nvidia-container-cli list | grep -E 'GLX_nvidia|glcore|EGL_nvidia'
```

세 파일이 모두 존재하고 `nvidia-container-cli list` 에 GLX/glcore/EGL 항목이 출력되면 호스트 측 준비 완료.

컨테이너 안에서:

```bash
docker compose run --rm leisaac-debug bash -c '
  ldconfig -p | grep -E "libGLX_nvidia|libvulkan|libnvidia-glcore" &&
  apt-get install -y vulkan-tools && vulkaninfo --summary
'
```

`vulkaninfo --summary` 가 NVIDIA GPU 의 `deviceName` 과 `apiVersion 1.4.x` 를 출력하면 컨테이너 안에서도 Vulkan 이 정상이다. 이후 `docker compose up` 시 위의 `ERROR_INCOMPATIBLE_DRIVER` / `Failed to create any GPU devices` / `no suitable CUDA GPU was found` / `switching to software` 메시지가 모두 사라진다.

#### Headless 서버에서 외부 PC 로 화면 송출

호스트에 디스플레이가 없는 경우 (서버 환경) Isaac Sim 은 `--headless --livestream=2` (사내망 WebRTC) 로 띄워 외부 PC 에서 Omniverse Streaming Client / 호환 WebRTC 클라이언트로 접속한다. 이때 컨테이너가 바인드하는 포트는 다음과 같다:

| 포트 | 프로토콜 | 용도 | 출처 |
|------|---------|------|------|
| 8011 | TCP | HTTP signaling | `omni.services.transport.server.http` |
| 48010 | TCP | livestream core | `omni.kit.livestream.core` |
| 49100 | TCP | WebRTC media | `omni.kit.livestream.webrtc` |
| 47998-48020 | UDP | 동적 미디어 범위 | `omni.services.livestream.nvcf` |

`network_mode: host` 면 별도 포트 매핑 없이 그대로 노출된다. WebRTC 동적 미디어 협상이 NAT 뒤에서 깨지는 경우가 있어 host network 가 가장 안정적이다.

### 카메라 sensor 가 raytracing pipeline 생성 실패 (RT 코어 없는 GPU)

> ⚠ **H100/A100은 Isaac Sim 5.1 공식 미지원이다.** NVIDIA 공식 [System Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)가 다음과 같이 명시:
> > *"GPUs without RT Cores (A100, H100) are not supported."*
>
> 즉 H100/A100은 시스템 요구사항 단계부터 제외되어 있고, 아래 증상은 그 결과물이다. 워크어라운드를 찾기보다 GPU를 교체하는 게 정답.

**현상**: 위 Vulkan 문제를 해결한 뒤 (`Driver Version: ... | Graphics API: Vulkan` 가 정상 출력되고 `Streaming server started.` 까지 도달) 그 직후, 환경 초기화 단계에서 다음 트레이스로 컨테이너가 즉시 종료된다.

**오류 메시지**:

```log
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INITIALIZATION_FAILED
[Error] [carb.graphics-vulkan.plugin] vkCreateRayTracingPipelinesKHR failed.
[Error] [omni.physx.fabric.plugin] CUDA error: an illegal memory access was encountered:
                                   .../DirectGpuHelper.cpp: 563

Traceback (most recent call last):
  File ".../teleop_se3_agent.py", line 226, in main
    env = gym.make(task_name, cfg=env_cfg).unwrapped
  File ".../isaaclab/envs/mdp/observations.py", line 404, in image
    images = sensor.data.output[data_type]
  File ".../isaaclab/sensors/sensor_base.py", line 362, in _update_outdated_buffers
    self._is_outdated[outdated_env_ids] = False
RuntimeError: CUDA error: an illegal memory access was encountered
```

#### 원인

NVIDIA가 시스템 요구사항 문서에서 H100/A100을 미지원으로 명시한 이유와 정확히 일치하는 메커니즘이다. 데이터센터 GPU인 **NVIDIA H100 / A100 (Hopper / Ampere-DC)** 은 **RT 코어를 탑재하지 않는다**. RT 코어는 RTX A/L 워크스테이션 시리즈와 GeForce RTX, 그리고 일부 데이터센터 GPU (L40 / L40S / A40 / RTX 6000 Ada) 에만 있다.

Isaac Sim 5.1 의 카메라 sensor (`isaaclab.sensors.camera.Camera` / `TiledCamera`) 는 무조건 RTX renderer (`RaytracedLighting` / `PathTracing`) 로 동작하도록 강제되어 있다 (`isaaclab/sensors/camera/camera_cfg.py:64`, `isaaclab/apps/isaaclab.python.rendering.kit:50-71` 에 raytracing 비활성화 옵션 부재). 그래서 RT 코어 없는 GPU 에서는 다음 흐름으로 죽는다:

1. `--enable_cameras` 로 카메라 sensor 등록
2. RTX renderer 가 `vkCreateRayTracingPipelinesKHR` 호출 → `ERROR_INITIALIZATION_FAILED`
3. `omni.physx.fabric` 가 비어 있는/유효하지 않은 GPU 버퍼를 참조 → CUDA illegal memory access
4. observation manager 가 `sensor.data.output[...]` 접근 → 이미 corrupt 된 CUDA context 라 `RuntimeError`

CUDA 자체는 정상이고 (`nvidia-smi` 에서 컨테이너의 python 프로세스가 GPU 메모리 점유), GPU 가 두 장 모두 인식되며 livestream 서버까지 정상 기동한 뒤 발생하기 때문에 위쪽 Vulkan 섹션의 증상과는 구분된다.

GPU 별 RT 코어 유무 빠른 가이드 (NVIDIA 공식 시스템 요구사항 기준):

| GPU | 아키텍처 | RT 코어 | Isaac Sim 5.1 지원 |
|------|---------|---------|------|
| H100 / H200 | Hopper | ✗ | **NVIDIA 공식 미지원** (문서 명시) |
| A100 | Ampere-DC | ✗ | **NVIDIA 공식 미지원** (문서 명시) |
| L40 / L40S / L4 | Ada-DC | ✓ | 동작 |
| A40 / A30 | Ampere-DC (visualization) | ✓ | 동작 |
| RTX A4000 / A5000 / A6000 | Ampere | ✓ | 동작 (RT 코어·16GB VRAM 충족) |
| RTX 6000 Ada / 5000 Ada | Ada | ✓ | 동작 |
| GeForce RTX 4080 (최소) / 5080 (양호) / PRO 6000 Blackwell (이상적) | 컨슈머·Pro | ✓ | NVIDIA **권장** |
| GeForce RTX 30 시리즈 | Ampere | ✓ | 권장 라인업 미만이지만 RT 코어·16GB(3080 12GB는 미달) 충족 시 동작 |

#### 해결 방법

**근본적으로 길은 하나다: RT 코어 있는 GPU로 이동 (옵션 C).** NVIDIA가 H100/A100을 공식 미지원으로 못박은 이상 회피책은 잘 해야 부분 동작이고 production 워크플로 운영은 불가능하다. 아래 §A·§B는 "왜 우회로도 안 되는지" 기록 차원으로 남긴다 — 새로 시도하지 말 것.

##### A. 카메라 sensor + livestream 비활성화 (실질 불가)

이론적으로는 두 옵션을 모두 끄면 raytracing 경로를 우회할 수 있다:

1. `--enable_cameras` 비활성 → camera sensor raytracing pipeline 실패 회피
2. `--livestream=2` 비활성 → viewport swapchain compositor 의 RTX shader pipeline `ERROR_DEVICE_LOST` 회피

대표 swapchain crash 메시지 (livestream 만 켜고 카메라는 꺼도 발생 — H100 검증):

```
[Warning] [gpu.foundation.plugin] Invalid sync scope for buffer resource 'shared swapchain buffer'
[Error] [carb.graphics-vulkan.plugin] aftermath reports no active shader during GPU crash
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_DEVICE_LOST
[Error] [carb.graphics-vulkan.plugin] submitToQueueCommon failed.
[Error] [gpu.foundation.plugin] A GPU crash occurred. Exiting the application...
```

**그러나 LeIsaac 의 데모 task (`LeIsaac-SO101-PickOrange-v0` 등) 는 환경 config 자체에 카메라 prim 이 등록되어 있어 `--enable_cameras` 를 강제한다**. 빼면 시작 시점에 다음 에러로 즉시 종료:

```
RuntimeError: A camera was spawned without the --enable_cameras flag.
              Please use --enable_cameras to enable rendering.
```

LeIsaac task 의 카메라 prim 을 코드 수준에서 제거하지 않는 한, H100/A100 에서 옵션 A 로는 실행 불가. **실질적으로 H100/A100 환경에서는 LeIsaac 텔레오퍼레이션·데이터 수집을 운영할 수 없다.** Docker 빌드 검증, 의존성/CUDA 동작 확인까지만 가능하며 실제 시뮬레이션은 옵션 C 환경에서 돌려야 한다.

##### B. raster 렌더 모드 강제 시도 (NVIDIA 비공식, 권장 안 함)

Isaac Sim 의 `/rtx/rendermode` carb 설정을 raster 계열로 강제. command 에 다음 인자 추가:

```yaml
command:
  - ...
  - --enable_cameras
  - --/rtx/rendermode=Raster
  # 또는 --/renderer=pxr
```

**NVIDIA가 시스템 요구사항 자체에서 RT 코어 없는 GPU를 지원하지 않는다고 명시한 이상**, 이 carb 플래그로 raytracing 호출 일부를 회피해도 다른 RTX 의존 코드 경로(머티리얼 컴파일, MDL, postprocessing 등)에서 결국 실패한다. PathTracing 기반 ground truth 채널은 당연히 동작 안 함. 진단·실험 외 용도로 쓰지 말 것.

##### C. RT 코어 있는 GPU 로 이동 (근본 해결)

데이터 수집·학습용 환경은 다음 GPU 중 하나로 옮긴다:

- 로컬 워크스테이션: **RTX A4000 / A5000 / A6000**, **RTX 6000 Ada / 5000 Ada**, GeForce RTX 30/40/50 시리즈
- 클라우드/데이터센터: **L40S / L40 / L4**, **A40**

H100/A100 두 장이 있는 환경이라도 카메라 sensor 가 필요한 학습/eval 단계에서는 위 GPU 가 한 장 이상 추가로 필요하다.

#### 확인 방법

GPU 의 RT 코어 유무는 `nvidia-smi` 로 직접 확인되지 않는다. 모델명으로 위 표 참조 또는 `vulkaninfo` 출력에서 raytracing 확장 지원 확인:

```bash
docker compose run --rm leisaac-debug bash -c '
  apt-get install -y vulkan-tools >/dev/null 2>&1 && \
  vulkaninfo --summary | grep -A1 "deviceName\|apiVersion" && \
  vulkaninfo 2>/dev/null | grep -E "VK_KHR_ray_tracing_pipeline|VK_KHR_acceleration_structure" | sort -u
'
```

`VK_KHR_ray_tracing_pipeline` 확장이 출력되어도 H100 처럼 RT 코어 없는 GPU 는 hardware 가속을 못 하므로 Isaac Sim 카메라가 내부적으로 실패한다. 결국 GPU 모델로 판단하는 게 가장 빠르다.

### 시뮬레이션 기동 시 무시해도 되는 로그

`teleop_se3_agent.py` 가 정상 기동한 상태에서도 수십~수백 줄의 `[Error]` / `[Warning]` 로그가 찍힌다. 대부분 **LeIsaac 제공 scene USD 에셋 자체의 품질 이슈**에서 유래하며, 시뮬레이션·텔레오퍼레이션 기능에는 영향이 없다.

기동 성공 판단 기준: 로그 하단에 다음이 출력되면 정상 동작 상태다.

```
SO101-Leader connected.
 Running calibration of SO101-Leader
...
+-------------------------------------------------+
|  Teleoperation Controls for so101_leader        |
|   B  | start control                            |
|   R  | reset simulation ...                     |
|   N  | reset simulation ...                     |
+-------------------------------------------------+
```

#### 로그 카테고리별 해석

| 로그 패턴 | 의미 | 대응 |
|---------|------|------|
| `[Error] [omni.physx.plugin] PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification) cannot be a part of a dynamic body, falling back to convexHull approximation` | 씬 속 가구(cabinet/drawer/handle 등) 의 collision geometry 가 dynamic body 에 쓸 수 없는 triangle mesh 로 authored 됨 → PhysX 가 자동으로 convex hull 근사로 대체 | 물리 근사 품질이 약간 떨어질 뿐. 무시 |
| `[Error] [omni.physx.plugin] PhysX error: Supplied PxGeometry is not valid. Shape creation method returns NULL.`<br>`PhysX Shape failed to be created on a prim: .../outlet_room/...`, `.../light_switch_room/...` | 씬 속 콘센트·전등스위치 prim 의 geometry 가 유효하지 않아 shape 생성 실패 | 단순 장식 요소 한정. pick-and-place 와 무관, 무시 |
| `[Error] [omni.physx.plugin] PhysicsUSD: CreateJoint - cannot create a joint between static bodies, joint prim: .../wall_*/world_fixed_joint` | 벽·바닥 등 static body 쌍 사이에 fixed joint 를 만들려다 실패 | static 끼리는 조인트가 불필요, 무시 |
| `[Warning] [omni.physx.plugin] ... possibly invalid inertia tensor of {1.0, 1.0, 1.0} and a negative mass, small sphere approximated inertia was used` | light_switch/outlet 등 일부 rigid body 의 mass property 가 불량 → 작은 구로 근사 | 장식요소 한정, 무시 |
| `[Warning] [omni.physx.cooking.plugin] UjitsoMeshCookingContext: cooking failure for .../cab_3_main_group/post_0_0` | cab_3 의 세로 기둥(post) 메시 쿠킹 실패 → 해당 prim 에 대해 triangle mesh collider 가 생성되지 않음 | 시각만 렌더링, 물리 충돌 없음 — 물건이 통과할 수 있으나 태스크엔 무관 |
| `[Warning] [gpu.foundation.plugin] ECC is enabled on physical device 0` | A4000 의 ECC 메모리가 켜진 상태 안내 | 정상 |
| `[Warning] [omni.isaac.dynamic_control] omni.isaac.dynamic_control is deprecated as of Isaac Sim 4.5` | 구 API 사용 안내 | Isaac Lab 2.3 내부 호출로 사용자가 손댈 일 없음, 무시 |
| `[Warning] [pxr.Semantics] pxr.Semantics is deprecated - please use Semantics instead` | USD 모듈 deprecation 안내 | 무시 |
| `[Warning] [omni.graph.core.plugin] Found duplicate of category 'Replicator'` | OGN 카테고리 중복 등록 | 무시 |
| `[Warning] [omni.replicator.core.scripts.extension] No material configuration file, adding configuration to material settings directly.` | Replicator 의 기본 머티리얼 config 파일 부재 | 무시 |
| `[Warning] [omni.fabric.plugin] Warning: attribute overrideClipRange not found for bucket id 9` | Fabric 내부 속성 lookup 실패 | 무시 |
| `[Warning] [omni.fabric.plugin] USD->Fabric: Unhandled array type string[]`<br>`[Warning] [usdrt.population.plugin] [UsdNoticeHandler] Unhandled attribute type VtArray<std::string> (prim attribute: omni:rtx:material:db:flattener:*)` | USD 의 string 배열 속성을 Fabric/USDRT 가 처리하지 못함 (RTX material db 관련) | 렌더링엔 영향 없음, 무시 |
| `[Warning] [omni.hydra] Parameter 'diffuse_texture_enable' of shade node ... not available in the MDL representation` | OmniPBR 머티리얼의 일부 파라미터가 MDL 변환본에 없음 | 렌더링 품질엔 영향 없음, 무시 |
| `[Warning] [rtx.postprocessing.plugin] DLSS increasing input dimensions: Render resolution of (371, 278) is below minimal input resolution of 300` | 뷰포트 해상도가 DLSS 최소치 미만이라 자동 상향 | 정상 |
| `[Warning] [omni.physx.plugin] Damping attribute is unsupported for articulation joints and will be ignored (.../sink_main_group/joints/handle)` | 싱크대 articulation joint 의 damping 속성은 PhysX 에서 무시됨 | 무시 |
| `[Warning] [omni.fabric.plugin] getAttributeCount/getTypes called on non-existent path .../Robot/wrist/visuals/wrist_roll_pitch_so101_v2` | SO-101 wrist visual prim 의 attribute 조회 시점 문제 | 로봇 제어엔 영향 없음, 무시 |
| `[Warning] [carb] Client gpu.foundation.plugin has acquired [gpu::unstable::IMemoryBudgetManagerFactory v0.1] 100 times. Consider accessing this interface with carb::getCachedInterface()` | Carb 인터페이스 획득 회수가 많다는 성능 권고 | 무시 |
| `[Warning] [omni.kit.notification_manager.manager] Physics USD Load: ...` (같은 메시지가 기동 후 수십 초 지나 다시 반복) | `R`/`N` 키로 reset 하면 씬이 재로드되면서 동일 경고들이 재출력 | 정상 동작 |

#### 실제로 주의해야 할 로그

위 표에 해당하지 **않는** 다음 유형이 나오면 조치가 필요하다:

- `Windows fatal exception: code 0xc0000139` → **HDF5 ABI 불일치** (앞선 섹션 참조)
- kit log 백트레이스에 `arrow.dll` / `arrow_python.dll` / `_dataset.cp311-win_amd64.pyd` → **PyArrow / NumPy ABI 불일치** (앞선 섹션 참조)
- `ConnectionError: Could not connect on port 'COMx'` → 리더 암 시리얼 연결 실패. 포트 번호 / 드라이버 확인
- `AssertionError: the dataset file already exists, please use '--resume' to resume recording` → 기존 데이터셋 파일 삭제하거나 `--resume` 플래그 추가
- `Crash detected in pid ... thread ...` + `carb.crashreporter-breakpad.plugin` → 실제 프로세스 크래시. 직전에 찍힌 Python traceback 을 분석해야 함

## Reference

- [Isaac Sim 5.1 + Isaac Lab 2.3 + LeIsaac on Windows](https://hackmd.io/@asierarranz/rkg1tvT93gx)
- [Installation | LeIsaac Document](https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation)
- [Teleoperation | LeIsaac Document](https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation)
- [Policy Training & Inference | LeIsaac Document](https://lightwheelai.github.io/leisaac/docs/getting_started/policy_support)
- [Post-Training Isaac GR00T N1.5 for LeRobot SO-101 Arm](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac — Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)
- [isaac-sim/Sim-to-Real-SO-101-Workshop: This code supports learning content to demonstrate an end-to-end Physical AI workflow with the SO-101 robot, Isaac Lab, and Isaac GR00T.](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
