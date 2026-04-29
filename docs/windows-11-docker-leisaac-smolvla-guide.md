# Windows 11 Docker — LeIsaac PickOrange × SmolVLA 가이드

## Context

- **목표**: Windows 11 호스트 + Docker 컨테이너 조합으로 LeIsaac PickOrange task 와 SmolVLA 모델의 (1) 텔레오퍼레이션·원격 텔레오퍼레이션, (2) Fine-tuning, (3) Open-loop Evaluation, (4) Closed-loop Deployment 까지 한 번에 돌릴 수 있는 환경을 만든다.
- **기존 GR00T 가이드와의 관계**: 이 가이드는 `windows-11-docker-leisaac-gr00t-guide.md` 의 SmolVLA 대안이다. GR00T 대비 설정이 훨씬 단순하므로 처음 시도하는 경우 이 가이드를 우선 권장.
- **현재 저장소 상태**: `D:\Workspaces\robotics_manipulation` 에 LeIsaac 운영용 Dockerfile/docker-compose 가 있고, PickOrange 데이터셋이 `./datasets/leisaac-so101-pick-orange/` 에 있다.

### GR00T 대비 핵심 차이

| 항목 | GR00T N1.7 | SmolVLA |
|------|-----------|---------|
| `--policy_type` | `gr00tn1.6` | `lerobot-smolvla` |
| 서버 프로토콜 | ZMQ (포트 5555) | gRPC (포트 8080) |
| 서버 실행 도구 | `gr00t/eval/run_gr00t_server.py` | `python -m lerobot.scripts.server.policy_server` |
| `action_horizon` | 16 | 50 |
| `meta/modality.json` | 수동 작성 필요 | **불필요** |
| Embodiment 등록 | `register_modality_config` 필요 | **불필요** |
| 학습 명령 | `gr00t/experiment/launch_finetune.py` | `python -m lerobot.scripts.train` |
| 학습 base 모델 | `nvidia/GR00T-N1.7-3B` (HF gated) | `lerobot/smolvla_base` (공개) |
| leisaac extra | `leisaac[gr00t]` | `leisaac[lerobot-async]` |
| 학습 소요 (A100) | 수십 시간 | ~5시간 (20k steps) |
| 체크포인트 경로 | `checkpoint-XXXX/` | `checkpoints/last/pretrained_model/` |

- **버전 고정**:
    - LeIsaac 컨테이너 : `leisaac==0.4.0`, `isaacsim==5.1.0`, `torch==2.7.0+cu128`, Python 3.11
    - SmolVLA 컨테이너 : `lerobot==0.3.3`, `torch==2.7.0+cu128` (pytorch base 이미지 제공), Python 3.10
- **GPU 제약**: 텔레오퍼레이션/sim eval 은 RT 코어가 있는 GPU 필수 (RTX A4000/A5000/A6000/RTX 6000 Ada/GeForce RTX 30·40·50). H100·A100 은 카메라 sensor 의 raytracing pipeline 생성 실패로 LeIsaac sim 자체가 못 뜬다. **SmolVLA 학습·추론 서버는 H100·A100 가능**.

---

## 0. 전제 조건 (한 번만)

GR00T 가이드 0절과 동일. 요약:

| 항목 | 권장 |
|------|------|
| OS | Windows 11 Pro (x64) |
| GPU | RT 코어 GPU (텔레오퍼레이션·sim) / H100·A100 (SmolVLA 학습) |
| 드라이버 | CUDA 12.8 호환 (≥ 555.x) |
| Docker | Docker Desktop 최신 + WSL2 백엔드 + GPU 지원 |
| 시리얼 | SO-ARM101 리더 암 USB-to-Serial 드라이버 |
| 계정 | Hugging Face 계정 + access token |

> SmolVLA base 모델(`lerobot/smolvla_base`)은 HF gated 가 아니라 HF_TOKEN 없이도 다운로드 가능. 단 파인튜닝 결과를 Hub 에 push 할 때는 토큰이 필요.

GPU 가시성 확인:
```powershell
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.04-py3 nvidia-smi
```

---

## 1. 저장소 / 환경변수 준비

### 1.1 .env 생성

```powershell
Copy-Item .env.example .env
notepad .env
```

SmolVLA 에 필요한 값:
```dotenv
HF_TOKEN=hf_xxxxxxxxxx          # Hub push 용 (추론만이면 비워도 됨)
SMOLVLA_CHECKPOINT_PATH=lerobot/smolvla_base   # 파인튜닝 후엔 로컬 경로로 교체
LEADER_ENDPOINT=tcp://host.docker.internal:5556
```

`SMOLVLA_CHECKPOINT_PATH` 를 파인튜닝 완료 후에는 아래로 교체:
```dotenv
SMOLVLA_CHECKPOINT_PATH=/workspace/repo/outputs/smolvla/leisaac-pick-orange/checkpoints/last/pretrained_model
```

### 1.2 PickOrange 데이터셋 확인

```powershell
Test-Path .\datasets\leisaac-so101-pick-orange\meta\info.json
```

`True` 면 OK. 없는 경우:
```powershell
hf download --repo-type dataset LightwheelAI/leisaac-pick-orange `
    --local-dir ./datasets/leisaac-so101-pick-orange
```

> SmolVLA 는 `meta/modality.json` 이 **없어도** LeRobot 네이티브 데이터셋을 직접 사용. GR00T 가이드 5절의 modality.json 작성·embodiment 등록 절차 전부 불필요.

---

## 2. LeIsaac 컨테이너 (기존 그대로)

빌드:
```powershell
docker compose build leisaac-debug
```

Smoke test:
```powershell
docker compose run --rm leisaac-debug `
    uv run python -c "import isaacsim, lerobot; print('isaacsim', isaacsim.__version__)"
```

---

## 3. SmolVLA 학습·추론 컨테이너 (`smolvla-server`)

### 3.1 `Dockerfile.smolvla` (저장소 루트, 신규)

이미 저장소에 추가되어 있다:
```dockerfile
FROM nvcr.io/nvidia/pytorch:25.04-py3

# nvcr pytorch 25.04 의 PIP_CONSTRAINT 가 packaging==23.2 를 강제 → lerobot 0.3.3 과 충돌.
ENV PIP_CONSTRAINT=/dev/null

RUN apt-get update && apt-get install -y ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# lerobot v0.3.3 — smolvla(VLM 백본) + async(PolicyServer) extras.
# lerobot 설치가 NumPy 2.x 를 끌어오면서 base 이미지의 NumPy 1.x ABI 로 빌드된
# torch 2.7.0a0 / cv2 와 ABI 불일치 (`RuntimeError: Numpy is not available`).
# lerobot 설치 직후 numpy<2 로 강제 고정해 ABI 맞춤.
RUN git clone --branch v0.3.3 --depth 1 \
        https://github.com/huggingface/lerobot.git /opt/lerobot \
    && pip install -e "/opt/lerobot[smolvla,async]" \
    && pip install --no-deps "numpy<2"

WORKDIR /workspace/repo
CMD ["bash"]
```

### 3.2 `docker-compose.yaml` 추가 서비스 (`smolvla-server`)

이미 추가되어 있다:
```yaml
  smolvla-server:
    build:
      context: .
      dockerfile: Dockerfile.smolvla
    image: leisaac-smolvla:latest
    runtime: nvidia
    network_mode: host
    ipc: host
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
      HF_TOKEN: ${HF_TOKEN:-}
      WANDB_API_KEY: ${WANDB_API_KEY:-}
    volumes:
      - .:/workspace/repo
    shm_size: "16gb"
    command: ["bash"]
```

> `network_mode: host` 덕분에 leisaac-debug(LeIsaac sim) 가 `localhost:8080` 으로 smolvla-server 의 PolicyServer 에 바로 접속한다.

### 3.3 빌드

```powershell
docker compose build smolvla-server
```

Smoke test:
```powershell
docker compose run --rm smolvla-server bash -lc '
  python -c "
import torch, lerobot
from lerobot.scripts.server.policy_server import serve
print(\"torch:\", torch.__version__)
print(\"lerobot:\", lerobot.__version__)
print(\"CUDA:\", torch.cuda.is_available())
print(\"OK\")
"'
```
`torch: 2.7.x`, `lerobot: 0.3.3`, `CUDA: True` 가 나와야 한다.

---

## 4. 텔레오퍼레이션

GR00T 가이드 4절과 동일. 로컬·원격 텔레오퍼레이션 모두 LeIsaac 컨테이너에서 실행되며 SmolVLA 컨테이너와 무관.

요약:
- **로컬**: Windows 호스트 venv 에서 `teleop_se3_agent.py --teleop_device=so101leader --port=COM7 ...`
- **원격**: 호스트에서 ZMQ PUB → Docker leisaac-debug 컨테이너 SUB

수집 데이터는 `./datasets/dataset.hdf5` 또는 LeRobot v2 포맷으로 저장. 상세는 GR00T 가이드 4절·5.3절 참조.

---

## 5. PickOrange 데이터셋 확인 (SmolVLA 는 변환 불필요)

SmolVLA 학습 시 LeRobot 데이터셋을 **그대로** 사용. 데이터셋이 올바른지만 확인:

```powershell
docker compose run --rm smolvla-server bash -lc '
  python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(\"LightwheelAI/leisaac-pick-orange\",
                    root=\"/workspace/repo/datasets/leisaac-so101-pick-orange\")
print(\"episodes:\", ds.num_episodes)
print(\"frames:\", ds.num_frames)
print(\"keys:\", list(ds[0].keys())[:6])
"'
```

기대 출력:
```
episodes: 60
frames: 36293
keys: ['observation.images.front', 'observation.images.wrist', 'observation.state', 'action', ...]
```

---

## 6. SmolVLA Fine-tuning (`smolvla-server` 컨테이너)

### 6.1 HF 캐시 / 토큰

`.env` 의 `HF_TOKEN` 이 채워져 있으면 docker compose 가 자동 주입. base 모델은 공개이므로 토큰 없이도 다운로드 가능.

### 6.2 학습 실행

```powershell
docker compose run --rm \
  -e CUDA_VISIBLE_DEVICES=0 \
  smolvla-server bash -lc '
    python -m lerobot.scripts.train \
      --policy.path=lerobot/smolvla_base \
      --policy.push_to_hub=false \
      --dataset.repo_id=LightwheelAI/leisaac-pick-orange \
      --dataset.root=/workspace/repo/datasets/leisaac-so101-pick-orange \
      --batch_size=64 \
      --steps=20000 \
      --output_dir=/workspace/repo/outputs/smolvla/leisaac-pick-orange \
      --job_name=smolvla_pick_orange \
      --policy.device=cuda \
      --wandb.enable=false
  '
```

주요 옵션:
- `--policy.push_to_hub=false` 가 **필수**. v0.3.3 의 `TrainPipelineConfig.validate()` 는 `policy.push_to_hub` 기본 True 인 정책에서 `policy.repo_id` 가 비어 있으면 시작 직전에 `ValueError: 'policy.repo_id' argument missing` 으로 거부한다. Hub 에 체크포인트를 올릴 게 아니면 push 자체를 끄는 쪽이 깔끔. (Hub 에 push 하려면 `--policy.push_to_hub=true --policy.repo_id=<your-org/your-name>` 로 둘 다 지정.)
- `--wandb.enable=true` 는 `WANDB_API_KEY` 가 있을 때만.
- VRAM 부족 시 `--batch_size=32` 로 줄인다.
- **단일 GPU 핀이 필수**: `-e CUDA_VISIBLE_DEVICES=0` 누락 시, SmolVLA 가 끌어오는 SmolVLM2 백본이 HuggingFace `from_pretrained(..., device_map="auto")` 로 가중치를 보이는 모든 GPU 에 분산 배치 → lerobot 학습 루프(단일-GPU 가정) 가 첫 forward 에서 `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!` 로 죽는다. SmolVLA(450M params) 는 1장으로 충분.
- 멀티 GPU 데이터 병렬은 `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m lerobot.scripts.train ...` 으로 wrap (lerobot train.py 는 torchrun 지원). DDP 는 각 rank 가 독립 모델을 들고 있으므로 device_map 충돌이 없다.
- `--steps=20000` 은 A100 기준 약 5시간. 빠른 검증은 `--steps=1000`.
- **NumPy ABI 핀**: `Dockerfile.smolvla` 가 lerobot 설치 직후 `pip install --no-deps "numpy<2"` 로 NumPy 를 1.x 로 다운그레이드한다. nvcr pytorch 25.04 의 torch 2.7.0a0 / 사전 설치된 cv2 가 NumPy 1.x ABI 로 컴파일된 NVIDIA dev build 라, 이 핀이 없으면 학습이 시작 직전 `RuntimeError: Numpy is not available` 또는 `cv2 import 시 ImportError` 로 멈춘다. **이 핀이 빠진 옛 이미지로 빌드한 경우 `docker compose build smolvla-server --no-cache` 로 재빌드 필요**. 자세한 진단은 12 절 troubleshooting 의 NumPy 항목.

### 6.3 산출물

```
outputs/smolvla/leisaac-pick-orange/
└── checkpoints/
    ├── 010000/
    │   └── pretrained_model/   ← 중간 체크포인트
    └── last/
        └── pretrained_model/   ← 최종 로드 대상
            ├── config.json
            ├── model.safetensors
            └── ...
```

`.env` 의 `SMOLVLA_CHECKPOINT_PATH` 를 최종 체크포인트 경로로 교체:
```dotenv
SMOLVLA_CHECKPOINT_PATH=/workspace/repo/outputs/smolvla/leisaac-pick-orange/checkpoints/last/pretrained_model
```

---

## 7. Closed-loop Sim 평가 (배치/정량)

> **lerobot v0.3.3 에는 dataset GT 와 비교하는 별도 open-loop CLI 가 없다.** `lerobot.scripts.eval` 은 lerobot env registry (pusht/aloha 등)에 등록된 Gym 환경을 요구하는 **closed-loop** 평가이고, LeIsaac PickOrange 는 lerobot env 가 아닌 IsaacLab task 이므로 호출 대상이 아니다. 따라서 학습 결과 검증은 SmolVLA PolicyServer 를 띄운 뒤 LeIsaac sim 측 `policy_inference.py` 를 다회 episode 로 돌려 **성공률·에피소드 길이·완료 시간** 통계를 수집하는 방식으로 한다.
>
> 본 절은 "정량적 배치 평가" 시점, 8 절은 "단발 데모 / 운영 배포" 시점으로 구분해 사용. **명령 자체는 동일** 하므로 여기서는 핵심 파라미터만 짚고 상세 명령은 8.1 (서버) + 8.3 (LeIsaac 클라이언트) 을 참조.

평가 시 조정 권장 파라미터:

| 파라미터 | 의미 | 추천 |
|---|---|---|
| `--eval_rounds` | 반복할 episode 수 | 정량 평가 30~100, 빠른 검증 5~10 |
| `--episode_length_s` | 한 episode 최대 길이 (초) | 60 (학습 데이터셋 평균보다 1.5배 권장) |
| `--policy_action_horizon` | 1회 추론에서 받는 action 수 | 학습 시와 동일 (50) |
| `--policy_timeout_ms` | 첫 핸드셰이크 timeout | 60000 (모델 로드에 수십 초 소요) |

성공률은 `policy_inference.py` 가 stdout 으로 찍는 episode 별 종료 사유 (`success` / `timeout` / `failure`) 를 집계해 산출. 8.3 절 명령에 `--eval_rounds=30` 만 올려 그대로 사용.

---

## 8. Closed-loop Deployment (sim · 실물 공통 PolicyServer)

### 8.1 SmolVLA PolicyServer 기동 (`smolvla-server` 컨테이너)

PowerShell 창 1:
```powershell
docker compose run --rm smolvla-server bash -lc '
  python -m lerobot.scripts.server.policy_server \
    --host=0.0.0.0 \
    --port=8080
'
```

서버 동작 원리:
- 빈 컨테이너로 시작 — 모델을 미리 로드하지 않음.
- 클라이언트(LeIsaac sim 또는 실물 RobotClient)가 첫 번째 gRPC 핸드셰이크를 보내면, 핸드셰이크에 포함된 `policy_type`, `pretrained_name_or_path`, `device` 정보로 모델을 그 자리에서 로드.
- 이후 관측값을 받아 action chunk 를 반환하는 비동기 추론 루프에 진입.

> `network_mode: host` 이므로 leisaac-debug 또는 같은 호스트의 RobotClient 가 `localhost:8080` 으로 바로 접속 가능.

### 8.2 실물 SO-ARM101 Follower Arm 배포 (`smolvla-server` 컨테이너 재사용)

학습한 SmolVLA 정책을 **실물** SO-ARM101 follower 암에 적용한다. lerobot v0.3.3 가 제공하는
`lerobot.scripts.server.robot_client` 가 카메라·시리얼 포트를 직접 잡고 8.1 의 PolicyServer 와 gRPC 로 통신.

전제 조건:
- 8.1 절 PolicyServer 가 같은 호스트에서 8080 포트로 떠 있다.
- SO-ARM101 follower 가 USB 로 연결되어 호스트에서 `/dev/ttyACM0` (Linux) 또는 `COMx` (Windows) 로 인식.
- 학습 데이터셋의 카메라 키 (`observation.images.front`, `observation.images.wrist`) 와 **동일한 이름·해상도·fps** 의 실물 카메라가 준비.

PowerShell 창 2 (Linux 호스트 기준):
```bash
docker compose run --rm \
  --device=/dev/ttyACM0 \
  --device=/dev/video0 \
  --device=/dev/video2 \
  smolvla-server bash -lc '
    python -m lerobot.scripts.server.robot_client \
      --robot.type=so100_follower \
      --robot.port=/dev/ttyACM0 \
      --robot.id=so101_follower_01 \
      --robot.cameras='\''{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}'\'' \
      --server_address=127.0.0.1:8080 \
      --policy_type=smolvla \
      --pretrained_name_or_path=/workspace/repo/outputs/smolvla/leisaac-pick-orange/checkpoints/last/pretrained_model \
      --policy_device=cuda \
      --actions_per_chunk=50 \
      --chunk_size_threshold=0.5 \
      --task="Pick up the orange and place it on the plate"
  '
```

> `--robot.type=so100_follower` 가 SO-ARM100 / SO-ARM101 공용 드라이버 (lerobot 표준 명칭). USB 카메라 인덱스는 `v4l2-ctl --list-devices` 로 확인.

Windows 호스트 (Docker Desktop + WSL2):
- USB 시리얼은 호스트가 직접 보지 못하므로 `usbipd-win` 으로 WSL2 에 attach 한 뒤 WSL 측 디바이스 (`/dev/ttyUSB0` 또는 `/dev/ttyACM0`) 를 마운트.
  ```powershell
  usbipd list                          # BUSID 확인
  usbipd bind --busid 1-4
  usbipd attach --wsl --busid 1-4
  ```
  WSL 안에서 `dmesg | tail` 로 디바이스 노드 확인 후 위 명령의 `--device=` 경로를 그에 맞게 수정.
- 카메라는 Windows 호스트에서 직접 보이므로 `--robot.cameras` 의 `index_or_path` 를 호스트 카메라 번호로 지정.

안전 체크리스트 (첫 실행):
1. 정책이 학습된 적 없는 자세로 암을 시작하면 큰 1차 동작이 나올 수 있음 → home pose 로 정렬 후 시작.
2. `--actions_per_chunk=10`, `--chunk_size_threshold=0.3` 로 시작해 동작 안정성을 확인한 뒤 50 / 0.5 로 복귀.
3. `policy_device=cuda` 가 핸드셰이크 직후 모델 로드에 수십 초 소요 — 그 동안 로봇은 정지 상태 유지.

### 8.3 LeIsaac Closed-loop 실행 (`leisaac-debug` 컨테이너)

PowerShell 창 2 (PolicyServer 가 완전히 뜬 다음 실행):
```powershell
docker compose run --rm leisaac-debug bash -lc '
  uv run python /workspace/repo/scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=127.0.0.1 \
    --policy_port=8080 \
    --policy_timeout_ms=60000 \
    --policy_checkpoint_path=/workspace/repo/outputs/smolvla/leisaac-pick-orange/checkpoints/last/pretrained_model \
    --policy_action_horizon=50 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --episode_length_s=60 \
    --eval_rounds=10 \
    --num_envs=1 --device=cuda --enable_cameras --headless --livestream=2
'
```

동작 흐름:
1. LeIsaac 가 `LeRobotServicePolicyClient` 로 `localhost:8080` 에 gRPC 접속.
2. 핸드셰이크 — `policy_type=smolvla`, `pretrained_name_or_path=<path>` 전송.
3. SmolVLA PolicyServer 가 모델 로드 (최초 1회, 수십 초 소요).
4. sim 이 관측값을 전송 → PolicyServer 가 50-step action chunk 반환.
5. LeIsaac 가 action chunk 를 실행하면서 비동기로 다음 chunk 를 미리 요청.

---

## 9. 비동기 추론 파라미터 튜닝

SmolVLA 와 GR00T 의 가장 큰 차이는 **비동기 추론(async inference)** 이다. 두 파라미터가 성능을 결정한다.

| 파라미터 | 기본값 | 역할 |
|---------|--------|------|
| `policy_action_horizon` (`actions_per_chunk`) | 50 | 한 번에 예측하는 action 수. 클수록 빈 큐 위험 감소, 작을수록 반응성 향상. |
| `chunk_size_threshold` | 0.7 | 큐가 `threshold * chunk_size` 이하가 되면 새 관측값을 서버로 전송. 0에 가까울수록 동기 방식, 1에 가까울수록 매 step 추론 요청. |

조정 권장:
1. 처음엔 `--policy_action_horizon=50 chunk_size_threshold=0.5` 로 시작.
2. sim 이 action 이 없어 멈추는 현상(`empty queue`)이 발생하면 `--step_hz` 를 낮추거나 `actions_per_chunk` 를 늘린다.
3. 반응이 너무 느리면 `chunk_size_threshold` 를 0.7~0.8 로 올린다.

> `chunk_size_threshold` 노출 위치:
> - **sim (LeIsaac)**: `LeRobotServicePolicyClient` 내부에서 관리. `policy_inference.py` 에 현재 노출된 플래그가 없으면 `leisaac.policy.LeRobotServicePolicyClient` 생성자 인자로 직접 전달.
> - **실물 (8.2 절)**: `lerobot.scripts.server.robot_client` 가 `--chunk_size_threshold` CLI 플래그를 직접 받음.

---

## 10. 검증 (End-to-End Smoke)

| 단계 | 검증 방법 | 기대 출력 |
|------|----------|-----------|
| 0 | `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.04-py3 nvidia-smi` | GPU 모델명 + 드라이버 표 |
| 2 | `docker compose run --rm leisaac-debug uv run python -c "import isaacsim; print(isaacsim.__version__)"` | `5.1.0` |
| 3.3 | smolvla-server smoke test | `lerobot: 0.3.3`, `CUDA: True` |
| 5 | LeRobotDataset 로드 | `episodes: 60`, `frames: 36293` |
| 6.2 | `checkpoints/010000/pretrained_model/` 생성 | step 진행 후 체크포인트 파일 존재 |
| 8.1 | PolicyServer 기동 | `Listening on 0.0.0.0:8080` 로그 |
| 8.2 | `robot_client` 핸드셰이크 | `connected to policy server` 로그 + 첫 action chunk 수신 |
| 8.3 | `policy_inference.py` 실행 | `Evaluating episode 1...` 출력 후 sim 동작 |

---

## 11. Critical Files (수정·생성 대상)

| 파일 | 상태 | 역할 |
|------|------|------|
| `Dockerfile.smolvla` | **신규 작성** | SmolVLA PolicyServer·학습 컨테이너 |
| `docker-compose.yaml` | **수정** (`smolvla-server` 서비스 추가) | 컨테이너 오케스트레이션 |
| `.env` / `.env.example` | **수정** (`SMOLVLA_CHECKPOINT_PATH` 추가) | 체크포인트 경로 관리 |
| `outputs/smolvla/leisaac-pick-orange/` | **자동 생성** (6.2) | 학습 산출물 |

수정 불필요 (GR00T 대비 간소화):
- `datasets/leisaac-so101-pick-orange/meta/modality.json` — SmolVLA 는 불필요
- `Isaac-GR00T/examples/leisaac_so101/leisaac_so101_config.py` — SmolVLA 는 불필요
- `scripts/evaluation/policy_inference.py` — 이미 `lerobot-smolvla` 지원

---

## 12. Troubleshooting

- **빌드 중 `ResolutionImpossible: packaging==23.2 vs packaging>=24.2`** → base 이미지가 `PIP_CONSTRAINT=/etc/pip/constraint.txt` 로 `packaging==23.2` 를 고정. `--upgrade` 로도 우회 불가. `Dockerfile.smolvla` 에서 `ENV PIP_CONSTRAINT=/dev/null` 로 constraint 파일을 무력화해 해결 (이미 반영됨).
- **`python: can't open file '/opt/lerobot/lerobot/scripts/train.py'`** → 옛 lerobot 레이아웃 가정. v0.3.3 은 `src/lerobot/` 레이아웃이라 train.py 의 절대 경로는 `/opt/lerobot/src/lerobot/scripts/train.py`. 모듈 호출 형태 (`python -m lerobot.scripts.train ...`) 로 바꿔 사용.
- **`ValueError: 'policy.repo_id' argument missing. Please specify it to push the model to the hub.`** → v0.3.3 `TrainPipelineConfig.validate()` 가 `policy.push_to_hub=True` 면 `policy.repo_id` 를 강제. 6.2 절처럼 `--policy.push_to_hub=false` 를 추가하거나, Hub 에 올릴 거면 `--policy.repo_id=<org/name>` 까지 함께 지정.
- **`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!`** (학습 첫 forward) → 호스트에 GPU 가 2장 이상 보일 때 HF `from_pretrained` 의 `device_map="auto"` 가 SmolVLM2 가중치를 분산 배치하는데, lerobot v0.3.3 SmolVLA 학습 루프는 단일-GPU 가정이라 깨짐. `-e CUDA_VISIBLE_DEVICES=0` (또는 다른 단일 인덱스) 을 `docker compose run` 에 추가해 GPU 한 장만 노출. DDP 멀티-GPU 가 필요하면 `torchrun --nproc_per_node=N` 으로 감싸 각 rank 가 자기 GPU 에만 모델을 두도록 한다.
- **`RuntimeError: Numpy is not available` / `UserWarning: Failed to initialize NumPy` / `ImportError: ... compiled using NumPy 1.x ... NumPy 2.4.4`** → nvcr pytorch 25.04 의 torch 2.7.0a0 (NVIDIA dev build) 과 사전 설치된 cv2 가 NumPy 1.x ABI 로 컴파일되어 있는데, lerobot v0.3.3 설치가 NumPy 2.x 를 끌어와 ABI 가 어긋난다. **반드시 차단해야 학습이 진행됨** (단순 경고 아님 — `tensor.numpy()` 호출 지점에서 즉시 학습 중단). `Dockerfile.smolvla` 가 `pip install -e ...` 직후 `pip install --no-deps "numpy<2"` 로 1.x 로 핀하도록 되어 있다. 이미 옛 이미지로 빌드된 경우 `docker compose build smolvla-server --no-cache` 로 재빌드. 임시 우회는 컨테이너 안에서 `pip install --no-deps "numpy<2"` 직접 실행.
- **`ModuleNotFoundError: lerobot.async_inference` 또는 `lerobot.common.*`** → 두 경로 모두 v0.3.3 에서 사라졌다. `lerobot.async_inference.policy_server` → `lerobot.scripts.server.policy_server`, `lerobot.common.datasets.*` → `lerobot.datasets.*` 로 import 경로 정정.
- **PolicyServer 가 시작 직후 바로 종료** → `python -m lerobot.scripts.server.policy_server --host=0.0.0.0 --port=8080` 의 포트가 이미 점유 중. `netstat -an | grep 8080` 으로 확인.
- **클라이언트 핸드셰이크 타임아웃** (`policy_timeout_ms=5000` 초과) → SmolVLA 모델 로드 시간이 5초를 넘는 것. `--policy_timeout_ms=60000` 으로 늘린다 (최초 로드만 느림).
- **`ConnectionRefusedError: localhost:8080`** → PolicyServer 가 아직 안 뜬 것. 서버 로그에서 `Listening` 메시지 확인 후 클라이언트 재실행.
- **VRAM OOM (학습 중)** → `--batch_size` 를 32 → 16 으로 줄인다. SmolVLA base 추론은 약 2 GB.
- **`vkCreateRayTracingPipelinesKHR failed`** → H100/A100 에서 LeIsaac sim 실행 불가. 학습은 H100 에서, sim 평가는 RTX GPU 환경으로 분리.
- **`AssertionError: dataset file already exists`** (텔레오퍼레이션 녹화 시) → `--dataset_file` 경로의 HDF5 삭제 또는 `--resume` 추가.
- **smolvla-server 빌드 시 git clone 실패** → 네트워크 문제 또는 v0.3.3 태그 없음. `git ls-remote --tags https://github.com/huggingface/lerobot.git v0.3.3` 으로 태그 존재 확인.
- **실물 8.2: `serial.SerialException: could not open port /dev/ttyACM0`** → 컨테이너에 시리얼 디바이스 미마운트 또는 권한 부족. `docker compose run --rm --device=/dev/ttyACM0 ...` 인자 누락 여부 확인. 권한 문제면 호스트에서 `sudo chmod a+rw /dev/ttyACM0` 또는 사용자가 `dialout` 그룹에 속해 있는지 확인.
- **실물 8.2: `KeyError: 'observation.images.front'` (RobotClient 핸드셰이크)** → `--robot.cameras` 의 카메라 이름이 학습 데이터셋 키와 불일치. `front`, `wrist` 두 채널을 학습 시와 동일한 해상도·fps 로 맞춰야 한다.

---

## 13. Reference

- `D:\Workspaces\robotics_manipulation\docs\windows-11-docker-leisaac-gr00t-guide.md` (GR00T 가이드, 텔레오퍼레이션·데이터 변환 상세)
- `https://huggingface.co/docs/lerobot/async` (SmolVLA 비동기 추론 공식 문서)
- `https://lightwheelai.github.io/leisaac/resources/available_policy/` (LeIsaac 지원 정책 목록)
- `https://github.com/huggingface/lerobot/tree/v0.3.3` (lerobot v0.3.3 소스)
- `https://github.com/huggingface/lerobot/tree/v0.3.3/src/lerobot/scripts/server` (v0.3.3 PolicyServer / RobotClient 모듈 위치)
- `https://huggingface.co/lerobot/smolvla_base` (SmolVLA base 모델)
- `https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange` (PickOrange 데이터셋)
