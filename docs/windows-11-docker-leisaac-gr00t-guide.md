# Windows 11 Docker — LeIsaac PickOrange × GR00T N1.7 가이드

## Context

- **목표**: Windows 11 호스트 + Docker 컨테이너 조합으로 LeIsaac PickOrange task 와 GR00T N1.7 모델의 (1) 텔레오퍼레이션·원격 텔레오퍼레이션, (2) Fine-tuning, (3) Open-loop Evaluation, (4) Closed-loop Deployment 까지 한 번에 돌릴 수 있는 환경을 만든다.
- **현재 저장소 상태**: `D:\Workspaces\robotics_manipulation` 에 LeIsaac 운영용 Dockerfile/docker-compose 가 이미 있고, Isaac-GR00T 가 서브디렉토리로 클론돼 있으며, PickOrange 데이터셋이 `./datasets/leisaac-so101-pick-orange/` 에 다운로드돼 있다. 단, GR00T 학습/추론을 위한 컨테이너는 따로 없고, 데이터셋의 `meta/modality.json` 도 빠져 있다.
- **버전 고정**:
    - LeIsaac 컨테이너 : `leisaac==0.4.0`, `isaaclab==2.3.0`, `isaacsim==5.1.0`, `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128`, Python 3.11
    - GR00T 컨테이너 : `Isaac-GR00T (n1.7-release)`, `torch==2.7.1`, `torchvision==0.22.1`, Python 3.10, flash-attn 2.7.4.post1, TensorRT (선택)
- **GPU 제약**: 텔레오퍼레이션/sim eval 은 RT 코어가 있는 GPU 필수 (RTX A4000 / A5000 / A6000 / RTX 6000 Ada / GeForce RTX 30·40·50). H100·A100 은 카메라 sensor 의 raytracing pipeline 생성 실패로 LeIsaac sim 자체가 못 뜬다 (commit `b7a92bb` 에서 확인). Fine-tuning 은 H100·A100 모두 가능.
- **호환성 핵심 메모**: LeIsaac 0.4.0 의 `service_policy_clients.py` 는 GR00T N1.5/N1.6 ZMQ 클라이언트까지만 공식 내장한다. N1.7 정책 서버(`Isaac-GR00T/gr00t/eval/run_gr00t_server.py`) 와 LeIsaac 의 N1.6 클라이언트 페이로드가 일치하는지 가이드 9.3 단계에서 검증하고, 어긋나는 경우 9.4 절차로 패치한다.

---

## 0. 전제 조건 (한 번만)

### 0.1 호스트 (Windows 11)

| 항목 | 권장 / 최소 |
|------|-----------|
| OS | Windows 11 Pro / Pro for Workstations (x64) |
| GPU | RT 코어 있는 NVIDIA GPU 1 장 이상 (위 표 참조) |
| 드라이버 | CUDA 12.8 호환 (≥ 555.x) |
| Docker | Docker Desktop 최신 + WSL2 백엔드 + GPU 지원 활성화 |
| NVIDIA | NVIDIA Container Toolkit (Docker Desktop 설치 시 자동 포함) |
| 디스크 | 200 GB 이상 여유 (Isaac Sim·체크포인트 포함) |
| 시리얼 | SO-ARM101 리더 암을 인식하는 USB-to-Serial 드라이버 (1 Mbaud 지원) |
| 계정 | Hugging Face 계정 + access token (gated 모델용) |

### 0.2 GPU 가시성 확인

PowerShell 에서:
```powershell
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.04-py3 nvidia-smi
```
GPU 가 모두 출력되면 OK. 출력이 비어 있으면 Docker Desktop 의 *Settings → Resources → WSL Integration* / *General → Use GPU acceleration* 를 확인.

### 0.3 USB 시리얼 패스스루 정책 (중요)

Docker Desktop on Windows (WSL2 backend) 는 **USB-to-Serial 디바이스의 컨테이너 직접 패스스루를 표준으로 지원하지 않는다**. 따라서:

- **로컬 텔레오퍼레이션**: Windows 호스트 venv (`uv run`) 에서 직접 실행. 컨테이너 안에서 COM 포트를 잡으려고 시도하지 않는다.
- **원격 텔레오퍼레이션**: Windows 호스트가 SO101 리더 암을 들고 ZMQ PUB 으로 발행 → Docker 컨테이너의 sim 이 SUB 로 수신. 이 가이드의 5.2 절 참조.

---

## 1. 저장소 / 데이터셋 / 환경변수 준비

### 1.1 저장소 클론 + 서브디렉토리 확인

이미 `D:\Workspaces\robotics_manipulation` 에 작업 트리가 있으므로 추가 클론 불필요. 단 Isaac-GR00T 가 `n1.7-release` 브랜치인지 확인:
```powershell
cd D:\Workspaces\robotics_manipulation\Isaac-GR00T
git fetch origin
git checkout n1.7-release
git pull --ff-only
cd ..
```

### 1.2 .env 생성

```powershell
Copy-Item .env.example .env
notepad .env
```

다음 값으로 채운다 (HF 토큰은 https://huggingface.co/settings/tokens 에서 read 권한으로 발급):
```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GR00T_MODEL_PATH=nvidia/GR00T-N1.7-3B
GR00T_EMBODIMENT_TAG=NEW_EMBODIMENT
LEADER_ENDPOINT=tcp://host.docker.internal:5556
```

`GR00T_MODEL_PATH` 는 학습 후 산출물 경로(`/workspace/repo/outputs/gr00t_finetune/<run>/checkpoint-<step>`) 로 바꿔 쓰면 된다.

### 1.3 PickOrange 데이터셋 (이미 있다면 skip)

이미 `./datasets/leisaac-so101-pick-orange/` 에 받아져 있는 상태. 새로 받을 때만:
```powershell
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange `
    --local-dir ./datasets/leisaac-so101-pick-orange
```

데이터셋 사양 (확인됨):
- LeRobot v2.1, robot_type=`so101_follower`, fps=30, 60 episodes / 36293 frames
- Action·State : 6-D `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
- Video : `observation.images.front`, `observation.images.wrist` (av1, 480×640, 30fps)
- Tasks : `{"task_index": 0, "task": "Grab orange and place into plate"}`
- **`meta/modality.json` 누락** → 가이드 6 절에서 직접 추가

---

## 2. LeIsaac 컨테이너 (기존 자산 활용)

기존 `Dockerfile` (베이스 `nvcr.io/nvidia/pytorch:25.04-py3`) 과 `docker-compose.yaml` 의 `leisaac-debug` 서비스를 **그대로 사용**한다. `pyproject.toml` 의 `leisaac[isaaclab,lerobot,gr00t]` extras 가 이미 잡혀 있어 ZMQ·pydantic·msgpack 까지 모두 들어간다.

빌드 (최초 1 회):
```powershell
docker compose build leisaac-debug
```

Smoke test (그래픽 없이 import 만 검증):
```powershell
docker compose run --rm leisaac-debug `
    uv run python -c "import isaacsim, lerobot, leisaac, h5py; print('isaacsim', isaacsim.__version__); print('h5py HDF5', h5py.version.hdf5_version)"
```
`h5py HDF5` 가 `1.14.x` 로 출력되어야 한다 (README "h5py 와 Isaac Sim 의 HDF5 ABI 불일치" 섹션 참조).

---

## 3. GR00T 학습·추론 컨테이너 (신규 추가)

### 3.1 `Dockerfile.gr00t` 작성 (저장소 루트에 신규)

```dockerfile
# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:25.04-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/gr00t-venv \
    PATH=/opt/gr00t-venv/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/repo/.cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/repo/Isaac-GR00T

# Isaac-GR00T 의 pyproject 는 Python 3.10 / torch 2.7.1 을 요구.
# uv sync 로 lockfile 그대로 재현 (flash-attn / tensorrt 포함).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=Isaac-GR00T/pyproject.toml,target=/workspace/repo/Isaac-GR00T/pyproject.toml \
    --mount=type=bind,source=Isaac-GR00T/uv.lock,target=/workspace/repo/Isaac-GR00T/uv.lock,ro \
    uv sync --python 3.10 --frozen --no-install-project \
    && find /opt/gr00t-venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /opt/gr00t-venv -name '*.pyc' -delete

WORKDIR /workspace/repo
CMD ["bash"]
```

> 메모: `Isaac-GR00T/uv.lock` 이 저장소에 같이 들어 있는지 확인 (`ls Isaac-GR00T/uv.lock`). 없는 경우 `Isaac-GR00T` 레포의 README 안내대로 `uv lock` 을 먼저 한 번 돌려 생성.

### 3.2 `docker-compose.yaml` 에 service 추가

`leisaac-debug` 서비스 아래에 다음을 덧붙인다 (기존 leisaac-debug 블록은 건드리지 않는다):

```yaml
  gr00t-train:
    build:
      context: .
      dockerfile: Dockerfile.gr00t
    image: leisaac-gr00t:latest
    container_name: leisaac-gr00t
    runtime: nvidia
    network_mode: host          # leisaac-debug 와 ZMQ 통신을 host 네트워크로 묶기 위함
    ipc: host
    stdin_open: true
    tty: true
    working_dir: /workspace/repo/Isaac-GR00T
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: all
      HF_TOKEN: ${HF_TOKEN}
      HF_HUB_ENABLE_HF_TRANSFER: "1"
      WANDB_API_KEY: ${WANDB_API_KEY:-}
    volumes:
      - .:/workspace/repo
    ulimits:
      memlock: -1
      stack: 67108864
    shm_size: "32gb"
    command: ["bash"]            # 학습/추론 시 docker compose run 으로 별도 명령 주입
```

> 학습은 GPU 메모리를 많이 쓰므로 `shm_size: 32gb` 와 `memlock: -1` 을 둔다. multi-GPU 라면 `NVIDIA_VISIBLE_DEVICES` 를 `0,1,…` 로 명시해도 된다.

### 3.3 빌드

```powershell
docker compose build gr00t-train
```

검증:
```powershell
docker compose run --rm gr00t-train `
    bash -lc "python -c 'import torch, gr00t; print(torch.__version__, torch.cuda.is_available())'"
```
`2.7.1+cu128 True` 가 나오면 OK.

---

## 4. Teleoperation

### 4.1 로컬 텔레오퍼레이션 (Windows 호스트, 컨테이너 우회)

Docker Desktop 의 USB 시리얼 제약 때문에 로컬 모드는 **Windows 호스트 venv 에서 직접** 실행한다. 호스트에 `uv` 와 `pyproject.toml` 의존성을 동기화한 뒤:

```powershell
uv sync
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

키:
| 키 | 동작 |
|----|------|
| `B` | 텔레오퍼레이션 시작 |
| `R` | 리셋 (실패로 기록) |
| `N` | 리셋 (성공으로 기록) |
| `Ctrl+C` | 종료 |

캘리브레이션 다시 받으려면 `--recalibrate` 추가.

### 4.2 원격 텔레오퍼레이션 (Windows Host ↔ Docker)

> **이 모드가 사용자의 "Remote Teleoperation(Windows Host ↔ Docker)" 요건과 정확히 매칭된다.**

#### 4.2.1 (Windows 호스트) 리더 암 ZMQ PUB

```powershell
uv sync
uv run scripts/environments/teleoperation/so101_joint_state_server.py `
    --port COM7 `
    --id leader_arm `
    --rate 50 `
    --bind tcp://0.0.0.0:5556
```

처음 실행 시 캘리브레이션이 진행되며 결과가 `scripts/environments/teleoperation/.cache/leader_arm.json` 에 저장된다.

#### 4.2.2 (Docker 컨테이너) sim SUB

`docker-compose.yaml` 의 `leisaac-debug.command` 에서 원격 모드 인자를 풀어준다 (해당 줄의 주석 해제 후 `--remote_endpoint` 가 활성화되도록 수정):

```yaml
    command:
      - uv
      - run
      - scripts/environments/teleoperation/teleop_se3_agent.py
      - --task=LeIsaac-SO101-PickOrange-v0
      - --teleop_device=so101leader
      - --remote_endpoint=tcp://host.docker.internal:5556   # Windows 호스트
      - --num_envs=1
      - --device=cuda
      - --enable_cameras
      - --headless
      - --livestream=2
      - --record
      - --dataset_file=/workspace/repo/datasets/dataset.hdf5
```

> `network_mode: host` 가 켜져 있어 컨테이너에서 `host.docker.internal:5556` 으로 호스트의 ZMQ 발행자에 SUB 가능하다. 만약 host 네트워크가 의도와 안 맞으면 docker-compose 의 host 매핑을 `extra_hosts: ["host.docker.internal:host-gateway"]` 로 대체할 수 있다.

기동:
```powershell
docker compose up leisaac-debug
```

호스트 브라우저에서 Omniverse Streaming Client / WebRTC 클라이언트로 `http://localhost:8011` 접속하면 sim 영상이 송출된다.

데이터를 HDF5 가 아닌 LeRobot v2 로 직접 기록하려면 `teleop_se3_agent.py` 의 `--dataset_file` 대신 leisaac 의 LeRobot recorder 옵션(README 의 § 1 참조 또는 추후 7 절에서 변환) 을 쓴다. 지금 가이드는 HDF5 → LeRobot v2 변환 경로를 6.1 에서 다룬다.

---

## 5. PickOrange 데이터셋을 GR00T 학습용으로 정비

### 5.1 LeRobot v2 호환성

PickOrange 는 이미 v2.1 포맷이라 변환 불필요. **누락된 `meta/modality.json` 만 직접 추가**한다.

추가 위치: `./datasets/leisaac-so101-pick-orange/meta/modality.json`

```json
{
    "state": {
        "single_arm": { "start": 0, "end": 5 },
        "gripper":    { "start": 5, "end": 6 }
    },
    "action": {
        "single_arm": { "start": 0, "end": 5 },
        "gripper":    { "start": 5, "end": 6 }
    },
    "video": {
        "front": { "original_key": "observation.images.front" },
        "wrist": { "original_key": "observation.images.wrist" }
    },
    "annotation": {
        "human.task_description": { "original_key": "task_index" }
    }
}
```

> SO-ARM101 은 SO-ARM100 과 모터 구성이 동일(5 joints + gripper, 총 6-D)이라 SO100 템플릿(`Isaac-GR00T/examples/SO100/modality.json`) 을 그대로 재사용한다. 카메라 키 `front`/`wrist` 도 PickOrange 의 `observation.images.front` / `observation.images.wrist` 와 일치.

### 5.2 모달리티 config 작성

`./Isaac-GR00T/examples/SO100/so100_config.py` 를 그대로 복사해 새 파일을 만든다:

복사 위치: `./Isaac-GR00T/examples/leisaac_so101/leisaac_so101_config.py`

내용 (파일 그대로 복사 + 주석만 살짝 수정):
```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

leisaac_so101_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(leisaac_so101_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

> `single_arm` 은 RELATIVE/NON_EEF (joint-space delta), `gripper` 는 ABSOLUTE/NON_EEF (binary 가까운 동작이라 절대 위치가 안정적). 이 조합은 SO100/SO101 표준 권장값.

### 5.3 (선택) 텔레오퍼레이션 HDF5 → LeRobot v2 변환

본인이 직접 수집한 HDF5 데이터를 GR00T 학습에 추가로 쓰려면 LeIsaac 의 컨버터를 사용:
```powershell
docker compose run --rm leisaac-debug `
    uv run scripts/convert/isaaclab2lerobot.py `
    --hdf5_path /workspace/repo/datasets/dataset.hdf5 `
    --output_dir /workspace/repo/datasets/my_so101_lerobot `
    --repo_id local/my_so101 `
    --robot_type so101_follower
```
변환 후 위 5.1 의 `meta/modality.json` 을 같은 형태로 추가해 주면 학습 데이터로 쓸 수 있다.

---

## 6. GR00T N1.7 Fine-tuning (gr00t-train 컨테이너)

### 6.1 HF 캐시 / 토큰 / 사전 모델 다운로드

`docker-compose.yaml` 의 gr00t-train 서비스가 이미 `HF_TOKEN` 환경변수를 주입받는다. 처음 실행 시 베이스 모델이 자동 다운로드된다 (`/workspace/repo/.cache/huggingface` 에 캐시).

### 6.2 학습 실행

```powershell
docker compose run --rm gr00t-train bash -lc '
  CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run python \
    /workspace/repo/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /workspace/repo/datasets/leisaac-so101-pick-orange \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /workspace/repo/Isaac-GR00T/examples/leisaac_so101/leisaac_so101_config.py \
    --num-gpus 1 \
    --output-dir /workspace/repo/outputs/gr00t_finetune/leisaac_pick_orange \
    --max-steps 10000 \
    --save-steps 2000 \
    --save-total-limit 5 \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4
'
```

옵션 메모:
- `--use-wandb` 는 `WANDB_API_KEY` 가 있을 때만 추가.
- VRAM 이 부족하면 `--global-batch-size 16` + `--gradient-accumulation-steps 2` 로 등가 배치 32 유지.
- multi-GPU 는 `--num-gpus 2` + `CUDA_VISIBLE_DEVICES=0,1`.
- VLM 본체까지 학습하려면 `--tune-llm true --tune-visual true` (메모리 폭증 주의).

### 6.3 산출물

```
outputs/gr00t_finetune/leisaac_pick_orange/
├── checkpoint-2000/
├── checkpoint-4000/
└── checkpoint-10000/
    ├── config.json, model*.safetensors, tokenizer*, …
    └── experiment_cfg/   ← 평가/배포 시 모달리티 메타가 같이 필요함
```

---

## 7. Open-loop Evaluation (gr00t-train 컨테이너)

학습 데이터셋의 ground-truth action 시퀀스 vs 모델 예측을 비교해 MSE / MAE 와 시각화 PDF 를 받는다.

```powershell
docker compose run --rm gr00t-train bash -lc '
  uv run python /workspace/repo/Isaac-GR00T/gr00t/eval/open_loop_eval.py \
    --dataset-path /workspace/repo/datasets/leisaac-so101-pick-orange \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /workspace/repo/outputs/gr00t_finetune/leisaac_pick_orange/checkpoint-10000 \
    --traj-ids 0 1 2 \
    --action-horizon 16 \
    --steps 400 \
    --modality-keys single_arm gripper \
    --save-plot-path /workspace/repo/outputs/gr00t_finetune/leisaac_pick_orange/eval_plots
'
```

검증:
- `single_arm` 6 차원 trajectory 가 ground truth 와 시각적으로 따라붙고, `gripper` 가 0/1 근처에서 토글되면 학습이 잘 된 것.
- 너무 평탄하거나 ground truth 와 위상차가 크면 (1) modality.json index slice, (2) action_configs 의 RELATIVE/ABSOLUTE 설정, (3) 학습 step 수 / batch size 를 의심.

---

## 8. Closed-loop Deployment (sim 안에서 정책 평가)

### 8.1 GR00T 정책 서버 (gr00t-train 컨테이너)

```powershell
docker compose run --rm --service-ports gr00t-train bash -lc '
  uv run python /workspace/repo/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/repo/outputs/gr00t_finetune/leisaac_pick_orange/checkpoint-10000 \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 5555
'
```

> `network_mode: host` 라 `--service-ports` 는 사실상 불필요하지만, host 네트워크가 비활성화된 경우 위 명령으로 5555 를 노출.

### 8.2 LeIsaac 측에서 closed-loop 실행 (leisaac-debug 컨테이너)

다른 PowerShell 창에서:
```powershell
docker compose run --rm leisaac-debug bash -lc '
  uv run python /workspace/repo/scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=gr00tn1.6 \
    --policy_host=127.0.0.1 \
    --policy_port=5555 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Grab orange and place into plate" \
    --episode_length_s=60 \
    --eval_rounds=10 \
    --num_envs=1 --device=cuda --enable_cameras --headless --livestream=2
'
```

> `--policy_type=gr00tn1.6` 으로 두는 이유: LeIsaac 0.4.0 에 등록된 정책 클라이언트는 `gr00tn1.5` / `gr00tn1.6` 둘뿐이고, N1.6 클라이언트가 N1.7 정책 서버와 페이로드 호환되는지를 9.3 에서 검증한다. 호환되면 그대로, 안 되면 9.4 패치 후 다시 실행.

### 8.3 N1.7 ↔ LeIsaac N1.6 클라이언트 호환성 검증

서버를 띄운 상태에서 다음 셋이 모두 통과해야 한다.

1. **연결 확인**: `policy_inference.py` 가 startup 직후 `RuntimeError: ZMQ ...` 없이 첫 step 까지 진입.
2. **action shape 확인**: `single_arm` (5,) + `gripper` (1,) = (6,) per step, action_horizon = 16 → 응답 shape `(16, 6)` 또는 dict `{"single_arm": (16,5), "gripper": (16,1)}`.
3. **observation key 확인**: 정책 서버 측 로그에 들어오는 키가 `video.front`, `video.wrist`, `state.single_arm`, `state.gripper`, `annotation.human.task_description` 5 개와 일치.

검증 명령 (서버 stderr 에 들어온 페이로드를 한 번만 찍어 보기):
```powershell
docker compose run --rm gr00t-train bash -lc '
  GR00T_DEBUG_PAYLOAD=1 uv run python /workspace/repo/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/repo/outputs/gr00t_finetune/leisaac_pick_orange/checkpoint-10000 \
    --embodiment-tag NEW_EMBODIMENT --device cuda:0 --port 5555
'
```
(`GR00T_DEBUG_PAYLOAD=1` 은 Isaac-GR00T 코드에 따라 옵션이 다르다. 없으면 `gr00t/eval/run_gr00t_server.py` 의 inference handler 에 `print(observation.keys(), {k:v.shape for k,v in observation.items() if hasattr(v,'shape')})` 를 한 줄 추가한 뒤 재기동.)

### 8.4 N1.7 클라이언트 패치 (3 의 키 또는 2 의 shape 가 안 맞을 때)

- LeIsaac 측 클라이언트 위치 :
  `D:\Workspaces\robotics_manipulation\.venv\Lib\site-packages\leisaac\policy\service_policy_clients.py`
- 패치 절차 :
    1. `Gr00t16ServicePolicyClient` 클래스를 복사해 `Gr00t17ServicePolicyClient` 신설.
    2. `policy_inference.py` 의 `--policy_type` 분기에 `gr00tn1.7` 케이스 추가.
    3. observation/action 키 매핑을 N1.7 `run_gr00t_server.py` 의 ZMQ 헤더에 맞게 보정.
- 패치한 클라이언트는 venv site-packages 안이라 컨테이너 재빌드 시 사라진다. 정식으로 쓰려면 `leisaac` 를 fork 하고 `pyproject.toml` 의 `[tool.uv.sources]` 를 fork 의 git tag 로 돌리는 게 안전.

---

## 9. H100 원격 서버 + Windows PC 실제 로봇 배포

Fine-tuning / open-loop eval 이 H100 에서 완료된 상태에서, **실제 SO101 Follower Arm 이 연결된 Windows PC** 가 H100 의 GR00T 정책 서버에 ZMQ 로 접속해 closed-loop real-robot inference 를 수행하는 방법이다.

### 9.1 아키텍처 개요

```
┌──────────────────────────┐         TCP :5555          ┌────────────────────────────────┐
│   H100 서버              │  ◄──── ZMQ REQ-REP ──────►  │   Windows PC                  │
│   gr00t-train 컨테이너   │                             │   (컨테이너 밖, uv run)         │
│   GR00T policy server    │                             │   real_robot_gr00t_client.py   │
│   0.0.0.0:5555 바인딩    │                             │   ├─ OpenCV 카메라 ×2           │
└──────────────────────────┘                             │   └─ SO101 Follower Arm (COM)  │
                                                         └────────────────────────────────┘
```

- **GR00T server** (`run_gr00t_server.py`): 기본값 `host=0.0.0.0`, `port=5555` → 원격 연결 허용
- **ZMQ 패턴**: REQ-REP (msgpack 직렬화)
- **좌표계**: SO101 Follower 의 motor normalized range (±100) 를 그대로 사용. sim ↔ radian 변환 불필요.

### 9.2 H100 측: GR00T 정책 서버 기동

H100 서버의 PowerShell 또는 bash:

```bash
docker compose run --rm --service-ports gr00t-train bash -lc '
  uv run python /workspace/repo/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/repo/outputs/gr00t_finetune/leisaac_pick_orange/checkpoint-10000 \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 5555
'
```

서버가 준비되면 `Server is ready and listening on tcp://0.0.0.0:5555` 가 출력된다.

서버 IP 확인 (같은 터미널 또는 별도 터미널):

```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
# 또는
hostname -I
```

### 9.3 네트워크 연결 확인

H100 서버의 방화벽에서 TCP 5555 를 허용해야 한다:

```bash
# Ubuntu (ufw 사용 시)
sudo ufw allow 5555/tcp

# 또는 iptables 직접
sudo iptables -A INPUT -p tcp --dport 5555 -j ACCEPT
```

Windows PC 에서 연결 테스트:

```powershell
# PowerShell (nc 없으면 Test-NetConnection 사용)
Test-NetConnection -ComputerName 192.168.1.100 -Port 5555
# TcpTestSucceeded : True 가 나오면 OK
```

### 9.4 Windows PC 측: 의존성 확인

```powershell
cd D:\Workspaces\robotics_manipulation

# 의존성 동기화 (처음 1회)
uv sync

# 필수 패키지 확인
uv run python -c "import zmq, msgpack, lerobot, numpy; print('OK')"
```

### 9.5 SO101 Follower + 카메라 인덱스 확인

**카메라 인덱스 탐색** (Windows Device Manager → 이미징 장치 에서 번호 확인 또는 스캔):

```powershell
uv run python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera index {i}: available')
        cap.release()
"
```

**COM 포트 확인**: Windows 장치 관리자 → 포트(COM & LPT) 에서 SO101 연결 포트 확인.

### 9.6 real_robot_gr00t_client.py 실행

```powershell
uv run python scripts/deployment/real_robot_gr00t_client.py `
    --policy_host 192.168.1.100 `
    --policy_port 5555 `
    --robot_port COM7 `
    --camera_front_id 0 `
    --camera_wrist_id 1 `
    --task_description "Grab orange and place into plate" `
    --action_horizon 16 `
    --exec_horizon 8 `
    --max_relative_target 5.0
```

주요 파라미터:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--policy_host` | (필수) | H100 서버 IP |
| `--robot_port` | `COM7` | SO101 직렬 포트 |
| `--camera_front_id` | `0` | Front 카메라 OpenCV 인덱스 |
| `--camera_wrist_id` | `1` | Wrist 카메라 OpenCV 인덱스 |
| `--exec_horizon` | `8` | 재예측 전 실행할 step 수. 낮을수록 반응 빠름, 서버 부하 증가. |
| `--max_relative_target` | `5.0` | 한 step 최대 모터 이동량 (motor normalized units). 처음엔 낮게 설정. |
| `--no_calibrate` | (미설정) | 기존 캘리브레이션 파일 재사용 시 추가 |

터미널 출력 예시:
```
  step=    8 | inf=  210ms | loop=  480ms | gripper=  12.3
  step=   16 | inf=  195ms | loop=  470ms | gripper=  45.7
```

`inf` 는 H100 추론 레이턴시, `loop` 는 exec_horizon 실행 전체 시간.

### 9.7 안전 주의사항

1. **첫 실행 시 `--max_relative_target 2.0` 으로 낮게 설정** — 모터가 예상 밖의 큰 움직임을 보이면 즉시 Ctrl+C.
2. **팔 주변 반경 30 cm 이내 접근 금지** — 첫 inference 결과가 올바른지 확인 전까지.
3. **Ctrl+C** 시 `robot.disconnect()` 가 자동 호출돼 토크가 해제된다. 암이 아래로 떨어질 수 있으니 손으로 받칠 준비.
4. **캘리브레이션 파일 공유**: 텔레오퍼레이션 시 생성된 `.cache/follower_arm.json` 이 있으면 `--no_calibrate` 로 재사용 가능. 없으면 첫 실행 시 캘리브레이션 진행.
5. 모델이 학습 배포(PickOrange)와 다른 장면·조명에서는 성공률이 크게 떨어질 수 있다. Open-loop eval (Section 7) 에서 MSE 가 수렴한 checkpoint 를 사용할 것.

---

## 10. 검증 (End-to-End Smoke)

각 단계가 끝났을 때 다음을 한 번씩 확인한다.

| 단계 | 검증 방법 | 기대 출력 |
|------|----------|-----------|
| 0.2 | `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.04-py3 nvidia-smi` | GPU 모델명 + 드라이버 표 |
| 2 | `docker compose run --rm leisaac-debug uv run python -c "import isaacsim; print(isaacsim.__version__)"` | `5.1.0` |
| 3.3 | `docker compose run --rm gr00t-train bash -lc "python -c 'import gr00t, torch; print(torch.__version__)'"` | `2.7.1+cu128` |
| 4.2 | host 발행 + 컨테이너 sim 기동 후 키 `B` 누르면 sim 안 SO101 이 호스트 리더 암을 따라 움직임 | livestream 영상에서 추종 확인 |
| 5.1 | `Test-Path .\datasets\leisaac-so101-pick-orange\meta\modality.json` | `True` |
| 6.2 | `outputs/.../checkpoint-2000/` 에 `model*.safetensors` 생성 | 파일 존재 + W&B 로그 정상 |
| 7 | `eval_plots/*.pdf` 생성, MSE 가 처음 step 보다 감소 | 시각화에서 GT 와 따라붙음 |
| 8.2 | `policy_inference.py` 가 `eval_rounds` 만큼 episode 완주, 종료 시 success rate 출력 | 0% 가 아니어야 (학습 step 부족 시 처음엔 낮음) |

---

## 11. Critical Files (수정·생성 대상)

| 파일 | 상태 | 역할 |
|------|------|------|
| `Dockerfile` | 기존 | LeIsaac 컨테이너 베이스 |
| `Dockerfile.gr00t` | **신규 작성 (3.1)** | GR00T 학습/추론 컨테이너 |
| `docker-compose.yaml` | **수정 (3.2 + 4.2.2)** | `gr00t-train` 서비스 추가, `leisaac-debug.command` 에 `--remote_endpoint` 활성화 |
| `.env` | **신규 생성 (1.2)** | HF/W&B 토큰, GR00T 모델 경로 |
| `datasets/leisaac-so101-pick-orange/meta/modality.json` | **신규 생성 (5.1)** | GR00T LeRobot 모달리티 인덱스 |
| `Isaac-GR00T/examples/leisaac_so101/leisaac_so101_config.py` | **신규 생성 (5.2)** | NEW_EMBODIMENT 모달리티 등록 |
| `outputs/gr00t_finetune/leisaac_pick_orange/checkpoint-*` | **자동 생성 (6.2)** | 학습 산출물 |
| `.venv/Lib/site-packages/leisaac/policy/service_policy_clients.py` | **선택 패치 (8.4)** | N1.7 호환 클라이언트 추가 (검증 결과에 따라) |
| `scripts/deployment/real_robot_gr00t_client.py` | **신규 작성 (9.6)** | 실제 로봇 closed-loop client (Windows PC 실행) |

재사용하는 기존 자산 (수정 불필요):
- `scripts/environments/teleoperation/teleop_se3_agent.py`, `so101_joint_state_server.py`
- `scripts/evaluation/policy_inference.py`
- `scripts/convert/isaaclab2lerobot.py`
- `Isaac-GR00T/gr00t/experiment/launch_finetune.py`, `gr00t/eval/open_loop_eval.py`, `gr00t/eval/run_gr00t_server.py`
- `Isaac-GR00T/examples/SO100/so100_config.py` (5.2 의 템플릿 출처)
- `Isaac-GR00T/gr00t/policy/server_client.py` (`PolicyClient` — 9.6 스크립트에서 사용)

---

## 12. Troubleshooting (이미 실측된 함정)

- **`Windows fatal exception: code 0xc0000139` / `_errors DLL load failed`** → `h5py >= 3.16` 이 HDF5 2.0 을 끌어와서 Isaac Sim (HDF5 1.14.6) 과 ABI 충돌. `pyproject.toml` 의 `h5py<3.16` 핀이 풀린 상태가 아닌지 확인.
- **`vkCreateRayTracingPipelinesKHR failed` / `CUDA error: an illegal memory access`** → H100/A100 등 RT 코어 없는 GPU. 텔레오퍼레이션·sim eval 은 RT 코어 있는 GPU 로 옮긴다 (학습은 그대로 H100/A100 가능).
- **`ConnectionError: Could not connect on port 'COMx'`** → 호스트 USB 시리얼이 사용 중이거나 드라이버 누락. Windows 장치 관리자에서 포트 번호 / 사용 중 프로세스 확인.
- **`AssertionError: dataset file already exists`** → `--dataset_file` 경로의 HDF5 가 이미 있음. 삭제하거나 `--resume` 추가.
- **livestream 클라이언트 연결 실패** → `network_mode: host` 인지, Windows 방화벽이 8011/48010/49100 을 막지 않는지 확인.
- **gr00t-train 빌드 시 flash-attn 컴파일 에러** → uv lock 안의 prebuilt wheel 을 쓰고 있는지 확인. 새로 빌드되는 경우 `MAX_JOBS=4` 환경변수로 메모리 OOM 방지.
- **`Invalid zip file structure / Encountered an unexpected header (actual: 0x73726576)`** → `scripts/deployment/dgpu/wheels/flash_attn-...-linux_aarch64.whl` 이 Git LFS 포인터(텍스트) 상태일 때 `uv run` 이 해당 파일을 zip 으로 파싱하려다 실패하는 것. 볼륨 마운트 후 컨테이너 런타임에서 발생. `docker-compose.yaml` 의 `gr00t-train` 서비스에 `UV_NO_SYNC: "1"` 과 `PYTHONPATH: /workspace/repo/Isaac-GR00T` 를 추가하면 해결된다 (이미 3.2 절에 반영됨).
- **`Cannot reach GR00T server at <IP>:5555`** (9.6 실행 시) → H100 방화벽 확인 (`sudo ufw status`), `docker compose run --rm gr00t-train` 이 `--service-ports` 옵션 없이 실행됐는지 확인. `network_mode: host` 가 활성화돼 있으면 컨테이너 포트가 자동 노출됨.
- **real_robot_gr00t_client.py 에서 `ModuleNotFoundError: No module named 'gr00t'`** → `Isaac-GR00T/` 서브디렉토리가 존재하는지 확인 (`Test-Path .\Isaac-GR00T\gr00t\policy\server_client.py`). 스크립트가 실행 시 `sys.path` 에 자동으로 추가하므로 별도 설치는 불필요.
- **`zmq.error.Again: Resource temporarily unavailable`** (타임아웃) → H100 의 GR00T 추론이 `--policy_timeout_ms` (기본 15000 ms) 안에 완료되지 않음. 모델 로딩 중이거나 batch 처리 지연. `--policy_timeout_ms 30000` 으로 늘리거나 서버 측 로그 확인.
- **암이 갑자기 크게 움직임** → `--max_relative_target` 값이 너무 크거나 캘리브레이션 불일치. `--max_relative_target 2.0` 으로 낮추고, 캘리브레이션을 다시 진행 (`.cache/follower_arm.json` 삭제 후 재실행).

---

## 13. Reference

- `D:\Workspaces\robotics_manipulation\README.md` (특히 § Troubleshooting)
- `https://github.com/NVIDIA/Isaac-GR00T/blob/n1.7-release/getting_started/finetune_new_embodiment.md`
- `https://github.com/NVIDIA/Isaac-GR00T/blob/n1.7-release/getting_started/data_preparation.md`
- `https://github.com/NVIDIA/Isaac-GR00T/blob/n1.7-release/getting_started/data_config.md`
- `https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation`
- `https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange`
