# SO-ARM101 VLA Control System

SO-ARM101 6축 로봇 팔에 대한 VLA(Vision-Language-Action) 기반 제어 시스템. Hugging Face **LeRobot** + **LeIsaac** (NVIDIA Isaac Sim) 조합으로 시뮬레이션 학습과 실기기 텔레오퍼레이션을 모두 지원한다.

- **기본 태스크**: `pick_and_place`
- **지원 정책**: ACT, SmolVLA, GR00T N1.5

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

## 환경 설정

### 사전 요구사항

- **Windows 11** — Isaac Sim 5.1 Windows 빌드 기준
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — Python 환경·의존성 관리자 (conda / pip 대체)
- **NVIDIA GPU + CUDA 12.8 호환 드라이버** — torch 2.7.0+cu128 번들과 맞춤
- **Hugging Face 계정** — 데이터셋·모델 업로드·다운로드용

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

### 1. LeIsaac: Teleoperation

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

### 2. LeIsaac: Remote Teleoperation(Simulation)

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

### 2. 실기기 텔레오퍼레이션 + 데이터 수집

```powershell
uv run lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM<팔로워> `
    --teleop.type=so101_leader `
    --teleop.port=COM<리더> `
    --dataset.repo_id=<username>/SoArm_pick_and_place `
    --dataset.num_episodes=50
```

### 3. 정책 학습

#### ACT (경량·빠른 학습)

```powershell
uv run lerobot-train `
    --dataset.repo_id=<username>/SoArm_pick_and_place `
    --policy.type=act `
    --policy.device=cuda `
    --job_name=act_so101 `
    --output_dir=outputs/train/act_so101/pick_and_place `
    --wandb.enable=true `
    --steps=50_000 `
    --save_checkpoint=true `
    --save_freq=10_000
```

#### SmolVLA (VLA·언어 조건부)

```powershell
uv run lerobot-train `
    --policy.path=lerobot/smolvla_base `
    --dataset.repo_id=<username>/SoArm_pick_and_place `
    --batch_size=64 `
    --steps=20_000 `
    --output_dir=outputs/train/smolvla/pick_and_place `
    --policy.device=cuda `
    --wandb.enable=true
```

### 4. 파인튜닝 (사전 학습 체크포인트 기반)

```powershell
uv run lerobot-train `
    --dataset.repo_id=<username>/SoArm_pick_and_place `
    --policy.type=act `
    --policy.pretrained_path=<username>/<checkpoint_repo> `
    --policy.device=cuda `
    --output_dir=outputs/train/act_so101/finetune `
    --steps=50_000
```

### 5. Hub 업로드

```powershell
uv run hf upload <username>/<repo_name> `
    outputs/train/<job>/checkpoints/last/pretrained_model
```

## 정책 비교

| 정책 | 특징 | 권장 steps | A4000 예상 시간 |
|------|------|-----------|----------------|
| **ACT** | 경량, 빠른 학습 | 50,000 | ~45분 |
| **SmolVLA** | VLA, 언어 조건부 | 20,000 | ~2.5시간 |

## 출력 디렉토리 구조

```
outputs/train/<job_name>/
├── checkpoints/
│   ├── <step>/
│   └── last/
│       └── pretrained_model/   ← Hub 업로드 대상
└── logs/
```

## 하드웨어 구성

- **GPU**: NVIDIA RTX A4000 (16GB, WDDM, ECC on) — 테스트 기준
- **CPU**: Intel Xeon W-2245
- **리더/팔로워 암**: SO-ARM101, USB-to-serial 어댑터 각 1개 (1 Mbaud 지원 드라이버 필요)
- **카메라**: Microdia Integrated_Webcam_HD (USB 2.0 전용) × 3대. **USB 허브 경유 금지** — 각 카메라를 PC 포트에 직접 연결해야 한다. 상세는 아래 Troubleshooting 참조.

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

**카메라의 USB 버전 확인**

```powershell
# 1. 카메라 InstanceId 조회 (Status OK인 항목 확인)
Get-PnpDevice -Class Camera | Select-Object Status, InstanceId

# 2. ACPI 경로에서 포트 접두사 확인 (<InstanceId>에 위 결과 붙여넣기)
(Get-PnpDeviceProperty -InstanceId "USB\VID_0C45&PID_64AB&MI_00\<InstanceId>" |
  Where-Object { $_.KeyName -eq "DEVPKEY_Device_LocationPaths" }).Data |
  Where-Object { $_ -match "ACPI" }
```

출력 예시:
```
ACPI(_SB_)#ACPI(PC00)#ACPI(XHCI)#ACPI(RHUB)#ACPI(HS09)#USB(2)#USBMI(0)
                                              ^^^^^^^^^^
                                              여기를 본다
```

| 접두사 | USB 버전 | 최대 속도 |
|--------|---------|---------|
| `HS##` | USB 2.0 | 480 Mbps |
| `SS##` | USB 3.0 | 5 Gbps |
| `SSP##` | USB 3.1/3.2 | 10+ Gbps |

**USB 허브 버전 확인**

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
```
Windows fatal exception: code 0xc0000139
ImportError: DLL load failed while importing _errors: 지정된 프로시저를 찾을 수 없습니다.
```
`0xc0000139` 는 `STATUS_ENTRYPOINT_NOT_FOUND` — DLL 은 찾았지만 필요한 export 심볼이 없다는 뜻.

**오류 메시지 B** — h5py 판 `hdf5.dll` 이 먼저 로드된 경우:
```
Could not load the dynamic library from .../generic_model_output/bin/generic_mo_io.dll.
ImportError: DLL load failed while importing _generic_model_output: 지정된 모듈을 찾을 수 없습니다.
```

Isaac Sim 내부 로그에서는 같은 메시지가 CP949 → UTF-8 오해독으로 깨져 나온다:
```
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
- `ConnectionError: Could not connect on port 'COMx'` → 리더 암 시리얼 연결 실패. 포트 번호 / 드라이버 확인
- `AssertionError: the dataset file already exists, please use '--resume' to resume recording` → 기존 데이터셋 파일 삭제하거나 `--resume` 플래그 추가
- `Crash detected in pid ... thread ...` + `carb.crashreporter-breakpad.plugin` → 실제 프로세스 크래시. 직전에 찍힌 Python traceback 을 분석해야 함

## Reference

- [Isaac Sim 5.1 + Isaac Lab 2.3 + LeIsaac on Windows](https://hackmd.io/@asierarranz/rkg1tvT93gx)
- [Installation | LeIsaac Document](https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation)
- [Teleoperation | LeIsaac Document](https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation)
- [Policy Training & Inference | LeIsaac Document](https://lightwheelai.github.io/leisaac/docs/getting_started/policy_support)
- [Post-Training Isaac GR00T N1.5 for LeRobot SO-101 Arm](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
