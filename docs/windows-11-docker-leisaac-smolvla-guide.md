# Windows 11 Docker — LeIsaac PickOrange × SmolVLA 가이드

## Context

- **목표**: Windows 11 호스트 + Docker 컨테이너 조합으로 LeIsaac PickOrange task 와 SmolVLA 모델의 (1) 텔레오퍼레이션·원격 텔레오퍼레이션, (2) Fine-tuning, (3) Open-loop Evaluation, (4) Closed-loop Deployment 까지 한 번에 돌릴 수 있는 환경을 만든다.
- **기존 GR00T 가이드와의 관계**: 이 가이드는 `windows-11-docker-leisaac-gr00t-guide.md` 의 SmolVLA 대안이다. GR00T 대비 설정이 훨씬 단순하므로 처음 시도하는 경우 이 가이드를 우선 권장.
- **현재 저장소 상태**: `D:\Workspaces\robotics_manipulation` 에 LeIsaac 운영용 Dockerfile/docker-compose 가 있고, PickOrange 데이터셋이 `./datasets/leisaac-so101-pick-orange/` 에 있다.

### GR00T 대비 핵심 차이

| 항목 | GR00T N1.7 | SmolVLA |
|------|-----------|---------|
| `--policy_type` | `gr00tn1.6` | `lerobot-smolvla` |
| 서버 프로토콜 | ZMQ (포트 5555) | HTTP (포트 8080) |
| 서버 실행 도구 | `gr00t/eval/run_gr00t_server.py` | `python -m lerobot.async_inference.policy_server` |
| `action_horizon` | 16 | 50 |
| `meta/modality.json` | 수동 작성 필요 | **불필요** |
| Embodiment 등록 | `register_modality_config` 필요 | **불필요** |
| 학습 명령 | `gr00t/experiment/launch_finetune.py` | `lerobot/scripts/train.py` |
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

RUN apt-get update && apt-get install -y ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# lerobot v0.3.3 — smolvla(VLM 백본) + async(PolicyServer) extras
RUN git clone --branch v0.3.3 --depth 1 \
        https://github.com/huggingface/lerobot.git /opt/lerobot \
    && pip install -e "/opt/lerobot[smolvla,async]"

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
from lerobot.async_inference.policy_server import serve
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
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
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
docker compose run --rm smolvla-server bash -lc '
  python /opt/lerobot/lerobot/scripts/train.py \
    --policy.path=lerobot/smolvla_base \
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
- `--wandb.enable=true` 는 `WANDB_API_KEY` 가 있을 때만.
- VRAM 부족 시 `--batch_size=32` 로 줄인다.
- 멀티 GPU 는 `torchrun --nproc_per_node=2 ...` 로 wrap (lerobot train.py 는 torchrun 지원).
- `--steps=20000` 은 A100 기준 약 5시간. 빠른 검증은 `--steps=1000`.

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

## 7. Open-loop 검증 (`smolvla-server` 컨테이너)

학습 데이터셋의 ground-truth 와 모델 예측을 비교. lerobot 의 `eval_policy.py` 를 사용:

```powershell
docker compose run --rm smolvla-server bash -lc '
  python /opt/lerobot/lerobot/scripts/eval_policy.py \
    --pretrained_policy_name_or_path=/workspace/repo/outputs/smolvla/leisaac-pick-orange/checkpoints/last/pretrained_model \
    --eval.n_episodes=5 \
    --eval.batch_size=1
'
```

> `eval_policy.py` 는 환경 없이 데이터셋에서 action 예측만 하는 open-loop 평가. closed-loop sim 평가는 8절 참조.

---

## 8. Closed-loop Deployment (sim 안에서 정책 평가)

### 8.1 SmolVLA PolicyServer 기동 (`smolvla-server` 컨테이너)

PowerShell 창 1:
```powershell
docker compose run --rm smolvla-server bash -lc '
  python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
'
```

서버 동작 원리:
- 빈 컨테이너로 시작 — 모델을 미리 로드하지 않음.
- LeIsaac 클라이언트가 첫 번째 HTTP 핸드셰이크를 보내면, 핸드셰이크에 포함된 `policy_type`, `pretrained_name_or_path`, `device` 정보로 모델을 그 자리에서 로드.
- 이후 관측값을 받아 action chunk 를 반환하는 비동기 추론 루프에 진입.

> `network_mode: host` 이므로 leisaac-debug 에서 `localhost:8080` 으로 바로 접속 가능.

### 8.2 LeIsaac Closed-loop 실행 (`leisaac-debug` 컨테이너)

PowerShell 창 2 (PolicyServer 가 완전히 뜬 다음 실행):
```powershell
docker compose run --rm leisaac-debug bash -lc '
  uv run python /workspace/repo/scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=127.0.0.1 \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_checkpoint_path=/workspace/repo/outputs/smolvla/leisaac-pick-orange/checkpoints/last/pretrained_model \
    --policy_action_horizon=50 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --episode_length_s=60 \
    --eval_rounds=10 \
    --num_envs=1 --device=cuda --enable_cameras --headless --livestream=2
'
```

동작 흐름:
1. LeIsaac 가 `LeRobotServicePolicyClient` 로 `localhost:8080` 에 연결.
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

> `chunk_size_threshold` 는 `LeRobotServicePolicyClient` 내부에서 관리. `policy_inference.py` 에 현재 노출된 플래그가 없으면 `leisaac.policy.LeRobotServicePolicyClient` 생성자 인자로 직접 전달.

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
| 8.2 | `policy_inference.py` 실행 | `Evaluating episode 1...` 출력 후 sim 동작 |

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
- **`ModuleNotFoundError: lerobot.async_inference`** → lerobot 이 `v0.3.3` 이전 버전. `pip show lerobot` 로 버전 확인 후 smolvla-server 이미지 재빌드.
- **PolicyServer 가 시작 직후 바로 종료** → `python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080` 의 포트가 이미 점유 중. `netstat -an | grep 8080` 으로 확인.
- **클라이언트 핸드셰이크 타임아웃** (`policy_timeout_ms=5000` 초과) → SmolVLA 모델 로드 시간이 5초를 넘는 것. `--policy_timeout_ms=60000` 으로 늘린다 (최초 로드만 느림).
- **`ConnectionRefusedError: localhost:8080`** → PolicyServer 가 아직 안 뜬 것. 서버 로그에서 `Listening` 메시지 확인 후 클라이언트 재실행.
- **VRAM OOM (학습 중)** → `--batch_size` 를 32 → 16 으로 줄인다. SmolVLA base 추론은 약 2 GB.
- **`vkCreateRayTracingPipelinesKHR failed`** → H100/A100 에서 LeIsaac sim 실행 불가. 학습은 H100 에서, sim 평가는 RTX GPU 환경으로 분리.
- **`AssertionError: dataset file already exists`** (텔레오퍼레이션 녹화 시) → `--dataset_file` 경로의 HDF5 삭제 또는 `--resume` 추가.
- **smolvla-server 빌드 시 git clone 실패** → 네트워크 문제 또는 v0.3.3 태그 없음. `git ls-remote --tags https://github.com/huggingface/lerobot.git v0.3.3` 으로 태그 존재 확인.

---

## 13. Reference

- `D:\Workspaces\robotics_manipulation\docs\windows-11-docker-leisaac-gr00t-guide.md` (GR00T 가이드, 텔레오퍼레이션·데이터 변환 상세)
- `https://huggingface.co/docs/lerobot/async` (SmolVLA 비동기 추론 공식 문서)
- `https://lightwheelai.github.io/leisaac/resources/available_policy/` (LeIsaac 지원 정책 목록)
- `https://github.com/huggingface/lerobot/tree/v0.3.3` (lerobot v0.3.3 소스)
- `https://huggingface.co/lerobot/smolvla_base` (SmolVLA base 모델)
- `https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange` (PickOrange 데이터셋)
